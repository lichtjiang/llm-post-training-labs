import math
import os
import random

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "openai-community/gpt2"
DATASET_NAME = "nshah-fbcs/childes-engUK-conversational-pairs"

MAX_TRAIN_EXAMPLES = 2000 #10_000, #250757
MAX_VAL_EXAMPLES = 1_000
MAX_TEST_EXAMPLES = 1_000

BATCH_SIZE = 4
NUM_EPOCHS = 1
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

MAX_LENGTH = 1024

SEED = 42

OUTPUT_DIR = "./gpt2-childes-response-only"


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Device
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ============================================================
# Load dataset
# ============================================================

dataset = load_dataset(DATASET_NAME)

print(dataset)

train_dataset = dataset["train"].select(
    range(min(MAX_TRAIN_EXAMPLES, len(dataset["train"])))
)

val_dataset = dataset["validation"].select(
    range(min(MAX_VAL_EXAMPLES, len(dataset["validation"])))
)

test_dataset = dataset["test"].select(
    range(min(MAX_TEST_EXAMPLES, len(dataset["test"])))
)


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT-2 has no dedicated padding token.
tokenizer.pad_token = tokenizer.eos_token

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Pad token: {tokenizer.pad_token!r}")
print(f"EOS token: {tokenizer.eos_token!r}")


# ============================================================
# Helper: identify adult and child
# ============================================================

def get_adult_child(example):
    """
    The dataset stores two adjacent utterances.

    Either:
        speaker_a = adult, speaker_b = child

    or:
        speaker_a = child, speaker_b = adult

    Normalize both cases into:

        adult_utterance
        child_utterance
    """

    if example["speaker_a"] == "adult":
        adult_utterance = example["utterance_a"]
        child_utterance = example["utterance_b"]

    elif example["speaker_b"] == "adult":
        adult_utterance = example["utterance_b"]
        child_utterance = example["utterance_a"]

    else:
        raise ValueError(
            f"Could not find adult speaker in example: {example}"
        )

    return adult_utterance, child_utterance


# ============================================================
# Tokenize one example for response-only SFT
# ============================================================

def tokenize_example(example):
    adult_utterance, child_utterance = get_adult_child(example)

    # --------------------------------------------------------
    # Prompt
    #
    # This is what the model sees as context.
    # --------------------------------------------------------

    prompt = (
        f"Adult: {adult_utterance}\n"
        f"Child:"
    )

    # --------------------------------------------------------
    # Response
    #
    # Explicitly add EOS so the model learns when to stop.
    # --------------------------------------------------------

    response = (
        f" {child_utterance}"
        f"{tokenizer.eos_token}"
    )

    # Tokenize separately so that we know exactly where
    # the response begins.
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    response_ids = tokenizer(
        response,
        add_special_tokens=False,
    )["input_ids"]

    # --------------------------------------------------------
    # Combine them into one sequence.
    # --------------------------------------------------------

    input_ids = prompt_ids + response_ids

    # --------------------------------------------------------
    # Response-only labels
    #
    # -100 tells CrossEntropyLoss:
    #
    #     "ignore this position"
    #
    # Therefore:
    #
    # prompt tokens -> -100
    # response tokens -> actual token IDs
    # --------------------------------------------------------

    labels = (
        [-100] * len(prompt_ids)
        + response_ids
    )

    # --------------------------------------------------------
    # Truncate if necessary.
    #
    # GPT-2 has a maximum context length of 1024.
    # --------------------------------------------------------

    input_ids = input_ids[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


# ============================================================
# Pre-tokenize datasets
# ============================================================

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
# Collator
# ============================================================

def collate_batch(batch):

    input_ids = [
        torch.tensor(
            example["input_ids"],
            dtype=torch.long,
        )
        for example in batch
    ]

    labels = [
        torch.tensor(
            example["labels"],
            dtype=torch.long,
        )
        for example in batch
    ]

    # Dynamic padding:
    #
    # Each batch is padded only to the longest sequence
    # in that batch, rather than always to 1024.

    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100,
    )

    # 1 = real token
    # 0 = padding

    attention_mask = (
        input_ids != tokenizer.pad_token_id
    ).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# DataLoaders
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
# Inspect ONE batch before training
# ============================================================

print("\n" + "=" * 70)
print("INSPECTING ONE TRAINING BATCH")
print("=" * 70)

batch = next(iter(train_loader))

print("input_ids.shape:      ", batch["input_ids"].shape)
print("attention_mask.shape: ", batch["attention_mask"].shape)
print("labels.shape:         ", batch["labels"].shape)

print("\nFirst example:")

print("input_ids:")
print(batch["input_ids"][0])

print("\nattention_mask:")
print(batch["attention_mask"][0])

print("\nlabels:")
print(batch["labels"][0])

print("\nDecoded input:")
print(
    tokenizer.decode(
        batch["input_ids"][0],
        skip_special_tokens=False,
    )
)

print("\nDecoded response labels:")

response_label_ids = batch["labels"][0]
response_label_ids = response_label_ids[
    response_label_ids != -100
]

print(
    tokenizer.decode(
        response_label_ids,
        skip_special_tokens=False,
    )
)


# ============================================================
# Load GPT-2
# ============================================================

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.to(device)

print("\n" + "=" * 70)
print("MODEL")
print("=" * 70)

print(model)

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(f"\nTotal parameters: {num_parameters:,}")


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, data_loader):

    model.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(
        data_loader,
        desc="Evaluating",
        leave=False,
    ):

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)

        total_loss += outputs.loss.item()
        num_batches += 1

    average_loss = total_loss / num_batches

    perplexity = math.exp(average_loss)

    return average_loss, perplexity


# ============================================================
# Evaluate before fine-tuning
# ============================================================

print("\n" + "=" * 70)
print("BEFORE FINE-TUNING")
print("=" * 70)

val_loss, val_ppl = evaluate(
    model,
    val_loader,
)

print(f"Validation loss: {val_loss:.4f}")
print(f"Validation PPL:  {val_ppl:.4f}")


# ============================================================
# Training
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

for epoch in range(NUM_EPOCHS):

    model.train()

    total_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
    )

    for batch in progress_bar:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        # ----------------------------------------------------
        # Forward pass
        #
        # input_ids:
        #     [B, T]
        #
        # logits:
        #     [B, T, V]
        #
        # labels:
        #     [B, T]
        #
        # GPT-2 internally shifts logits/labels for causal
        # language modeling.
        #
        # Positions whose labels are -100 do not contribute
        # to the cross-entropy loss.
        # ----------------------------------------------------

        outputs = model(**batch)

        loss = outputs.loss

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_train_loss = (
        total_loss / len(train_loader)
    )

    print(
        f"\nEpoch {epoch + 1} "
        f"training loss: {average_train_loss:.4f}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loss, val_ppl = evaluate(
        model,
        val_loader,
    )

    print(
        f"Validation loss: {val_loss:.4f}"
    )

    print(
        f"Validation PPL:  {val_ppl:.4f}"
    )


# ============================================================
# Final test evaluation
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST")
print("=" * 70)

test_loss, test_ppl = evaluate(
    model,
    test_loader,
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test PPL:  {test_ppl:.4f}")


# ============================================================
# Save model
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(
    f"\nSaved model to: {OUTPUT_DIR}"
)


# ============================================================
# Generation: incorectly include the adult prompt
#   use generate_child_response2 from below instead
# ============================================================

def generate_child_response(
    model,
    adult_utterance,
    max_new_tokens=50,
):

    model.eval()

    prompt = (
        f"Adult: {adult_utterance}\n"
        f"Child:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=False,
    )

    return generated_text

# ===========================================================
#   Generate
# ===========================================================
def generate_child_response2(
    model,
    adult_utterance,
    max_new_tokens=50,
):
    model.eval()

    prompt = (
        f"Adult: {adult_utterance}\n"
        f"Child:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Number of tokens that were already supplied.
    prompt_length = inputs["input_ids"].shape[1]

    # Keep ONLY newly generated tokens.
    new_token_ids = generated_ids[0, prompt_length:]

    response = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=False,
    )

    return response

# ============================================================
# Try a few prompts
# ============================================================

print("\n" + "=" * 70)
print("GENERATED RESPONSES")
print("=" * 70)

test_prompts = [
    "what are you doing ?",
    "do you want to play ?",
    "what is that ?",
    "where are you going ?",
    "what did you do today ?",
]

for adult_utterance in test_prompts:

    result = generate_child_response2(
        model,
        adult_utterance,
    )

    print("\n" + "-" * 70)
    print(f"Adult: {adult_utterance}")
    print(result)
