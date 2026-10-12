import math
import os
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "openai-community/gpt2"
DATASET_NAME = "nshah-fbcs/childes-engUK-conversational-pairs"

# Start small.
# Once everything works, increase these.
MAX_TRAIN_EXAMPLES = 3_200 #10_000
MAX_VAL_EXAMPLES = 1_000
MAX_TEST_EXAMPLES = 1_000

BATCH_SIZE = 4
NUM_EPOCHS = 1
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

# GPT-2 supports sequences up to 1024 tokens.
# We won't truncate in Experiment A unless an example exceeds this.
MAX_LENGTH = 1024

SEED = 42

OUTPUT_DIR = "./gpt2-childes-experiment-a"


# ============================================================
# 2. Reproducibility
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. Select device
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ============================================================
# 4. Load CHILDES dataset
# ============================================================

print("\nLoading dataset...")

dataset = load_dataset(DATASET_NAME)

print(dataset)

print("\nExample:")
print(dataset["train"][0])


# ============================================================
# 5. Take small subsets for the first experiment
# ============================================================

train_dataset = dataset["train"].select(
    range(min(MAX_TRAIN_EXAMPLES, len(dataset["train"])))
)

val_dataset = dataset["validation"].select(
    range(min(MAX_VAL_EXAMPLES, len(dataset["validation"])))
)

test_dataset = dataset["test"].select(
    range(min(MAX_TEST_EXAMPLES, len(dataset["test"])))
)

print("\nSubset sizes:")
print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")


# ============================================================
# 6. Load GPT-2 tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT-2 does not have a dedicated padding token.
# For dynamic padding, use EOS as PAD.
tokenizer.pad_token = tokenizer.eos_token

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"EOS token: {tokenizer.eos_token!r}")
print(f"PAD token: {tokenizer.pad_token!r}")


# ============================================================
# 7. Convert one CHILDES record into one text sequence
# ============================================================

def format_example(example):
    """
    Convert one CHILDES conversational pair into a single
    causal-language-modeling sequence.

    Experiment A deliberately trains on the ENTIRE sequence.

    Example:

        adult: what's that ?
        child: i don't know .
    """

    text = (
        f"{example['speaker_a']}: {example['utterance_a']}\n"
        f"{example['speaker_b']}: {example['utterance_b']}"
    )

    return text


# Show an example before tokenization.
print("\nFormatted example:")

example_text = format_example(train_dataset[0])
print(example_text)


# ============================================================
# 8. Tokenize
# ============================================================

def tokenize_example(example):
    text = format_example(example)

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


print("\nTokenizing datasets...")

train_tokenized = train_dataset.map(
    tokenize_example,
    remove_columns=train_dataset.column_names,
)

val_tokenized = val_dataset.map(
    tokenize_example,
    remove_columns=val_dataset.column_names,
)

test_tokenized = test_dataset.map(
    tokenize_example,
    remove_columns=test_dataset.column_names,
)


# ============================================================
# 9. Inspect tokenization
# ============================================================

print("\nTokenized example:")

print(train_tokenized[0]["input_ids"])

print("\nDecoded again:")

print(
    tokenizer.decode(
        train_tokenized[0]["input_ids"]
    )
)


# ============================================================
# 10. Custom PyTorch collate function
# ============================================================

def collate_batch(examples):
    """
    Dynamically pad examples in a batch.

    This is intentionally implemented ourselves instead of
    relying on a Trainer/data collator so we can see exactly
    what enters GPT-2.
    """

    input_ids = [
        torch.tensor(example["input_ids"], dtype=torch.long)
        for example in examples
    ]

    attention_masks = [
        torch.tensor(example["attention_mask"], dtype=torch.long)
        for example in examples
    ]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    attention_masks = torch.nn.utils.rnn.pad_sequence(
        attention_masks,
        batch_first=True,
        padding_value=0,
    )

    # For causal LM training, labels are the same token sequence.
    #
    # GPT-2's forward() internally shifts:
    #
    #   input:  x1 x2 x3 x4
    #   target: x2 x3 x4
    #
    # Padding positions should not contribute to the loss.
    labels = input_ids.clone()

    labels[attention_masks == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }


# ============================================================
# 11. PyTorch DataLoaders
# ============================================================

train_loader = DataLoader(
    train_tokenized,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch,
)

val_loader = DataLoader(
    val_tokenized,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
)

test_loader = DataLoader(
    test_tokenized,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
)


# ============================================================
# 12. Inspect one actual batch
# ============================================================

batch = next(iter(train_loader))

print("\nBatch shapes:")

print(
    "input_ids:",
    batch["input_ids"].shape
)

print(
    "attention_mask:",
    batch["attention_mask"].shape
)

print(
    "labels:",
    batch["labels"].shape
)

print("\nFirst example in batch:")
print(
    tokenizer.decode(
        batch["input_ids"][0],
        skip_special_tokens=False,
    )
)


# ============================================================
# 13. Load GPT-2
# ============================================================

print("\nLoading GPT-2...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.to(device)


# ============================================================
# 14. Inspect model
# ============================================================

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

num_trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"\nTotal parameters: "
    f"{num_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{num_trainable_parameters:,}"
)


# ============================================================
# 15. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# 16. Evaluation function
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader):
    """
    Calculate average causal-LM loss.

    No gradients are calculated here.
    """

    model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)

        loss = outputs.loss

        total_loss += loss.item()
        total_batches += 1

    average_loss = total_loss / total_batches

    perplexity = math.exp(average_loss)

    return average_loss, perplexity


# ============================================================
# 17. Evaluate BEFORE fine-tuning
# ============================================================

print("\nEvaluating pretrained GPT-2...")

initial_val_loss, initial_val_ppl = evaluate(
    model,
    val_loader,
)

print(
    f"Initial validation loss: "
    f"{initial_val_loss:.4f}"
)

print(
    f"Initial validation perplexity: "
    f"{initial_val_ppl:.2f}"
)


# ============================================================
# 18. Training loop
# ============================================================

print("\nStarting training...")

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
    )

    for step, batch in enumerate(progress_bar):

        # Move tensors to CPU / MPS / CUDA.
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(**batch)

        loss = outputs.loss

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        # ----------------------------------------------------
        # Parameter update
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        running_loss += loss.item()

        average_running_loss = (
            running_loss / (step + 1)
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg_loss=f"{average_running_loss:.4f}",
        )

    # --------------------------------------------------------
    # End-of-epoch validation
    # --------------------------------------------------------

    val_loss, val_ppl = evaluate(
        model,
        val_loader,
    )

    print(
        f"\nEpoch {epoch + 1}: "
        f"validation loss = {val_loss:.4f}, "
        f"perplexity = {val_ppl:.2f}"
    )


# ============================================================
# 19. Final test evaluation
# ============================================================

print("\nEvaluating on test set...")

test_loss, test_ppl = evaluate(
    model,
    test_loader,
)

print(
    f"Test loss: {test_loss:.4f}"
)

print(
    f"Test perplexity: {test_ppl:.2f}"
)


# ============================================================
# 20. Save fine-tuned model
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print(
    f"\nModel saved to: {OUTPUT_DIR}"
)


# ============================================================
# 21. Generation helper
# ============================================================

def generate_text(
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
):
    """
    Generate text from the fine-tuned GPT-2.
    """

    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )


# ============================================================
# 22. Try the fine-tuned model
# ============================================================

prompts = [
    "adult: what are you doing ?\nchild:",
    "adult: do you want to play ?\nchild:",
    "adult: what's that ?\nchild:",
    "adult: where are you going ?\nchild:",
]

print("\n" + "=" * 60)
print("GENERATIONS")
print("=" * 60)

for prompt in prompts:

    print("\nPrompt:")
    print(prompt)

    print("\nGeneration:")

    print(
        generate_text(prompt)
    )