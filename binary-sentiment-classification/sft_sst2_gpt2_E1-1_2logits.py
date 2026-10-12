import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Model

"""
This implementation intentionally does not use GPT2ForSequenceClassification, 
because I think manually building the head is more valuable for our current learning goal.
"""

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "openai-community/gpt2"

TRAIN_SIZE = 5000 #10_000
VAL_SIZE = 872

MAX_LENGTH = 128
BATCH_SIZE = 4
NUM_EPOCHS = 1

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01

SEED = 42

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(SEED)


# ============================================================
# Load SST-2
# ============================================================

dataset = load_dataset("stanfordnlp/sst2")

print(dataset)

# We use:
#
#   train      -> 10,000 examples
#   validation -> official SST-2 validation set
#
# The official test labels are hidden, so we don't use test here.

train_dataset = (
    dataset["train"]
    .shuffle(seed=SEED)
    .select(range(TRAIN_SIZE))
)

val_dataset = dataset["validation"]

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT-2 does not have a native padding token.
# For batching, use EOS as PAD.
tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Dataset preprocessing
# ============================================================

def tokenize_example(example):
    encoded = tokenizer(
        example["sentence"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

    encoded["label"] = example["label"]

    return encoded


train_dataset = train_dataset.map(
    tokenize_example,
    remove_columns=train_dataset.column_names,
)

val_dataset = val_dataset.map(
    tokenize_example,
    remove_columns=val_dataset.column_names,
)


# ============================================================
# Dynamic padding collator
# ============================================================

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]

    encoded = tokenizer.pad(
        {
            "input_ids": input_ids,
        },
        padding=True,
        return_tensors="pt",
    )

    labels = torch.tensor(
        [item["label"] for item in batch],
        dtype=torch.long,
    )

    encoded["labels"] = labels

    return encoded


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)


# ============================================================
# Model
# ============================================================

class GPT2SentimentClassifier(nn.Module):

    def __init__(self, model_name, num_labels=2):
        super().__init__()

        # GPT-2 WITHOUT the language-model head.
        self.gpt2 = GPT2Model.from_pretrained(model_name)

        hidden_size = self.gpt2.config.hidden_size

        # Binary classification head:
        #
        # [B, 768] -> [B, 2]
        self.classifier = nn.Linear(
            hidden_size,
            num_labels,
        )

    def forward(self, input_ids, attention_mask, labels=None):

        # ----------------------------------------------------
        # GPT-2
        #
        # last_hidden_state:
        # [B, T, 768]
        # ----------------------------------------------------

        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        # ----------------------------------------------------
        # Find the position of the last real token
        #
        # attention_mask:
        #
        #   1 1 1 1 1 0 0
        #
        # last real token = position 4
        # ----------------------------------------------------

        sequence_lengths = attention_mask.sum(dim=1) - 1

        batch_indices = torch.arange(
            hidden_states.size(0),
            device=hidden_states.device,
        )

        # [B, 768]
        pooled_hidden_states = hidden_states[
            batch_indices,
            sequence_lengths,
        ]

        # ----------------------------------------------------
        # Classification head
        #
        # [B, 768] -> [B, 2]
        # ----------------------------------------------------

        logits = self.classifier(
            pooled_hidden_states
        )

        # ----------------------------------------------------
        # Classification loss
        # ----------------------------------------------------

        loss = None

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }


model = GPT2SentimentClassifier(
    MODEL_NAME,
    num_labels=2,
)


# ============================================================
# Freeze GPT-2
# ============================================================

for parameter in model.gpt2.parameters():
    parameter.requires_grad = False


# Only classifier is trainable.
trainable_params = [
    p for p in model.parameters()
    if p.requires_grad
]

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_count = sum(
    p.numel()
    for p in trainable_params
)

print()
print("Total parameters:", total_params)
print("Trainable parameters:", trainable_count)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    trainable_params,
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
    total_correct = 0
    total_examples = 0

    for batch in data_loader:

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs["loss"]
        logits = outputs["logits"]

        predictions = logits.argmax(dim=-1)

        total_loss += loss.item() * labels.size(0)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_examples += labels.size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return avg_loss, accuracy


# ============================================================
# Training
# ============================================================

model.to(DEVICE)

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0

    for step, batch in enumerate(train_loader):

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs["loss"]

        # Backward
        optimizer.zero_grad()

        loss.backward()

        # Update ONLY the classification head
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % 100 == 0:

            avg_loss = running_loss / 100

            print(
                f"Epoch {epoch + 1} "
                f"Step {step + 1} "
                f"Loss {avg_loss:.4f}"
            )

            running_loss = 0.0

    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
    )

    print()
    print(
        f"Epoch {epoch + 1} "
        f"Validation Loss: {val_loss:.4f} "
        f"Validation Accuracy: {val_accuracy:.4f}"
    )
    print()


# ============================================================
# Example predictions
# ============================================================

model.eval()

examples = [
    "This movie was absolutely fantastic.",
    "The movie was boring and disappointing.",
    "I really enjoyed this film.",
    "I wouldn't recommend this movie.",
]

encoded = tokenizer(
    examples,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

encoded = {
    key: value.to(DEVICE)
    for key, value in encoded.items()
}

with torch.no_grad():

    outputs = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )

    probabilities = torch.softmax(
        outputs["logits"],
        dim=-1,
    )

    predictions = probabilities.argmax(dim=-1)

for text, prediction, probability in zip(
    examples,
    predictions,
    probabilities,
):

    label = (
        "positive"
        if prediction.item() == 1
        else "negative"
    )

    print()
    print("Text:", text)
    print("Prediction:", label)
    print("Probabilities:", probability.cpu().tolist())
