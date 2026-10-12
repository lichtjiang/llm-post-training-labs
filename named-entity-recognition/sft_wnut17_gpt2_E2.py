import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Model

"""
This will fail on old macos such as big sur 11.7.10 due to old version of datasets & its 
dependencies and maybe transformers pacakge, tec.
Run it on recent versions of all dependencies, e.g. run it on Google's colab!!!!!
"""

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "openai-community/gpt2"
DATASET_NAME = "flaitenberger/wnut_17"

BATCH_SIZE = 4
NUM_EPOCHS = 1

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", DEVICE)

torch.manual_seed(SEED)


# ============================================================
# Load WNUT 2017
# ============================================================

dataset = load_dataset(DATASET_NAME)

print(dataset)

print("Train size:", len(dataset["train"]))
print("Validation size:", len(dataset["validation"]))
print("Test size:", len(dataset["test"]))


# ============================================================
# Inspect labels
# ============================================================

#this only works in more recent version of datasets
#label_names = dataset["train"].features["ner_tags"].feature.names

#a workaround for datasets 2.19.2

label_names = [
"O",
"B-corporation",
"I-corporation",
"B-creative-work",
"I-creative-work",
"B-group",
"I-group",
"B-location",
"I-location",
"B-person",
"I-person",
"B-product",
"I-product"
]

NUM_LABELS = len(label_names)

print()
print("Number of labels:", NUM_LABELS)

for i, name in enumerate(label_names):
    print(i, name)


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

# GPT-2 doesn't have a native padding token.
tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Tokenization + NER label alignment
# ============================================================

def tokenize_and_align_labels(example):

    # The WNUT example contains something conceptually like:
    #
    # tokens:
    # ["Hello", "from", "New", "York"]
    #
    # ner_tags:
    # [0, 0, 7, 8]

    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=128,
        add_special_tokens=True,
    )

    word_ids = tokenized.word_ids()

    aligned_labels = []

    previous_word_id = None

    for word_id in word_ids:

        # Special tokens such as padding:
        #
        # word_id = None
        #
        # They should not contribute to the loss.

        if word_id is None:

            aligned_labels.append(-100)

        # First BPE subtoken of a word:
        elif word_id != previous_word_id:

            aligned_labels.append(
                example["ner_tags"][word_id]
            )

        # Additional BPE subtokens:
        else:

            # Ignore them for this first experiment.
            aligned_labels.append(-100)

        previous_word_id = word_id

    tokenized["labels"] = aligned_labels

    return tokenized


train_dataset = dataset["train"].map(
    tokenize_and_align_labels,
    remove_columns=dataset["train"].column_names,
)

val_dataset = dataset["validation"].map(
    tokenize_and_align_labels,
    remove_columns=dataset["validation"].column_names,
)


# ============================================================
# Dynamic padding collator
# ============================================================

def collate_fn(batch):

    # input_ids and attention_mask are dynamically padded.
    #
    # We cannot simply use tokenizer.pad() for labels because
    # labels need -100 padding rather than the pad token ID.

    input_ids = [
        item["input_ids"]
        for item in batch
    ]

    attention_masks = [
        item["attention_mask"]
        for item in batch
    ]

    labels = [
        item["labels"]
        for item in batch
    ]

    max_length = max(
        len(ids)
        for ids in input_ids
    )

    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []

    for ids, mask, label in zip(
        input_ids,
        attention_masks,
        labels,
    ):

        padding_length = (
            max_length - len(ids)
        )

        padded_input_ids.append(
            ids + [
                tokenizer.pad_token_id
            ] * padding_length
        )

        padded_attention_masks.append(
            mask + [0] * padding_length
        )

        padded_labels.append(
            label + [-100] * padding_length
        )

    return {
        "input_ids": torch.tensor(
            padded_input_ids,
            dtype=torch.long,
        ),

        "attention_mask": torch.tensor(
            padded_attention_masks,
            dtype=torch.long,
        ),

        "labels": torch.tensor(
            padded_labels,
            dtype=torch.long,
        ),
    }


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

class GPT2NERClassifier(nn.Module):

    def __init__(
        self,
        model_name,
        num_labels,
    ):
        super().__init__()

        # GPT-2 backbone WITHOUT the LM head.
        self.gpt2 = GPT2Model.from_pretrained(
            model_name
        )

        hidden_size = self.gpt2.config.hidden_size

        # Token classification head:
        #
        # [B, T, 768]
        #       ↓
        # Linear(768 → num_labels)
        #       ↓
        # [B, T, num_labels]
        self.classifier = nn.Linear(
            hidden_size,
            num_labels,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
    ):

        # ----------------------------------------------------
        # GPT-2
        #
        # [B, T] → [B, T, 768]
        # ----------------------------------------------------

        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        # ----------------------------------------------------
        # Token classification
        #
        # [B, T, 768]
        #       ↓
        # [B, T, K]
        # ----------------------------------------------------

        logits = self.classifier(
            hidden_states
        )

        loss = None

        if labels is not None:

            loss_fn = nn.CrossEntropyLoss(
                ignore_index=-100
            )

            # CrossEntropyLoss expects:
            #
            # input:
            #     [N, K]
            #
            # target:
            #     [N]
            #
            # So flatten the B and T dimensions.
            #
            # logits:
            #     [B, T, K]
            #
            # → [B*T, K]
            #
            # labels:
            #     [B, T]
            #
            # → [B*T]

            loss = loss_fn(
                logits.view(-1, NUM_LABELS),
                labels.view(-1),
            )

        return {
            "loss": loss,
            "logits": logits,
        }


model = GPT2NERClassifier(
    MODEL_NAME,
    NUM_LABELS,
)


# ============================================================
# Freeze GPT-2
# ============================================================

for parameter in model.gpt2.parameters():
    parameter.requires_grad = False


# ============================================================
# Parameter counts
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    [
        p
        for p in model.parameters()
        if p.requires_grad
    ],
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Evaluation
# ============================================================
#there is a bug down there in this function (c below)
@torch.no_grad()
def evaluate1(model, data_loader):

    model.eval()

    total_loss = 0.0

    total_correct = 0
    total_valid_tokens = 0

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

        predictions = logits.argmax(
            dim=-1
        )

        # Only evaluate positions that have
        # an actual NER label.
        valid_positions = (
            labels != -100
        )

        total_correct += (
            (
                predictions[valid_positions]
                == labels[valid_positions]
            )
            .sum()
            .item()
        )

        total_valid_tokens += (
            valid_positions
            .sum()
            .item()
        )
        #this seems to be wrong, because we already have loss for a batch! why we have this multiplication thing?
        total_loss += (
            loss.item()
            * total_valid_tokens
        )

    accuracy = (
        total_correct
        / total_valid_tokens
    )

    # For simplicity, calculate average loss
    # directly over batches here.
    #
    # Accuracy is the more intuitive metric for
    # understanding this first experiment.
    average_loss = (
        total_loss
        / len(data_loader)
        if len(data_loader) > 0
        else 0.0
    )

    return average_loss, accuracy

#here is the fixed evaluate function
@torch.no_grad()
def evaluate(model, data_loader):

    model.eval()

    total_loss = 0.0
    total_batches = 0

    total_correct = 0
    total_valid_tokens = 0

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

        valid_positions = labels != -100

        total_correct += (
            predictions[valid_positions]
            == labels[valid_positions]
        ).sum().item()

        total_valid_tokens += (
            valid_positions.sum().item()
        )

        total_loss += loss.item()
        total_batches += 1

    average_loss = (
        total_loss / total_batches
    )

    accuracy = (
        total_correct / total_valid_tokens
    )

    return average_loss, accuracy

# ============================================================
# Training
# ============================================================

model.to(DEVICE)

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0

    for step, batch in enumerate(
        train_loader
    ):

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs["loss"]

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % 100 == 0:

            average_loss = (
                running_loss / 100
            )

            print(
                f"Epoch {epoch + 1} "
                f"Step {step + 1} "
                f"Loss {average_loss:.4f}"
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
        f"Validation Accuracy: "
        f"{val_accuracy:.4f}"
    )
    print()


# ============================================================
# Example NER prediction
# ============================================================

def predict_entities(model, words):

    model.eval()

    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    logits = outputs["logits"]

    predictions = logits.argmax(
        dim=-1
    )[0]

    word_ids = encoded.word_ids(
        batch_index=0
    )

    results = []

    previous_word_id = None

    for token_index, word_id in enumerate(
        word_ids
    ):

        if word_id is None:
            continue

        # Only use the first subtoken of
        # each word.
        if word_id == previous_word_id:
            continue

        predicted_label_id = predictions[
            token_index
        ].item()

        results.append(
            (
                words[word_id],
                label_names[predicted_label_id],
            )
        )

        previous_word_id = word_id

    return results


examples = [
    ["I", "live", "in", "San", "Francisco", "."],
    ["Apple", "released", "a", "new", "iPhone", "."],
    ["Michael", "visited", "London", "last", "week", "."],
]

for words in examples:

    results = predict_entities(
        model,
        words,
    )

    print()
    print("Input:", " ".join(words))

    for word, label in results:
        print(
            f"{word:15s} {label}"
        )

