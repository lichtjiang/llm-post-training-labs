import random

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

"""
C1: LoRA on mlp.c_fc + mlp.c_proj in block 11 only
C2: LoRA on mlp.c_fc + mlp.c_proj in all 12 blocks

GPT-2
│
├── block 0
│    └── MLP
│         ├── c_fc   ← LoRA
│         └── c_proj ← LoRA
│
├── block 1
│    └── MLP
│         ├── c_fc   ← LoRA
│         └── c_proj ← LoRA
│
│       ...
│
└── block 11
     └── MLP
          ├── c_fc   ← LoRA
          └── c_proj ← LoRA

"""

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "openai-community/gpt2"
DATASET_NAME = "nshah-fbcs/childes-engUK-conversational-pairs"

MAX_TRAIN_EXAMPLES = 2000 #10_000
MAX_VAL_EXAMPLES = 1_000
MAX_TEST_EXAMPLES = 1_000

MAX_LENGTH = 1024

BATCH_SIZE = 4
NUM_EPOCHS = 1

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01

SEED = 42

OUTPUT_DIR = "./gpt2-childes-lora-c2"


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT-2 has no dedicated PAD token.
tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Dataset
# ============================================================

dataset = load_dataset(DATASET_NAME)

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
# Identify adult/child utterances
# ============================================================

def get_adult_child(example):
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
# Response-only SFT tokenization
# ============================================================

def tokenize_example(example):
    adult_utterance, child_utterance = get_adult_child(example)

    prompt = f"Adult: {adult_utterance}\nChild:"
    response = f" {child_utterance}{tokenizer.eos_token}"

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    response_ids = tokenizer(
        response,
        add_special_tokens=False,
    )["input_ids"]

    input_ids = prompt_ids + response_ids

    # Response-only loss:
    #
    # prompt tokens -> -100
    # response tokens -> actual token IDs
    labels = [-100] * len(prompt_ids) + response_ids

    # Truncate input and labels together.
    input_ids = input_ids[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


train_dataset = train_dataset.map(
    tokenize_example,
    remove_columns=train_dataset.column_names,
)

val_dataset = val_dataset.map(
    tokenize_example,
    remove_columns=val_dataset.column_names,
)

test_dataset = test_dataset.map(
    tokenize_example,
    remove_columns=test_dataset.column_names,
)


# ============================================================
# Collator
# ============================================================

def collate_fn(batch):
    input_ids = [
        torch.tensor(x["input_ids"], dtype=torch.long)
        for x in batch
    ]

    labels = [
        torch.tensor(x["labels"], dtype=torch.long)
        for x in batch
    ]

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

    attention_mask = (
        input_ids != tokenizer.pad_token_id
    ).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# Load base GPT-2
# ============================================================

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.config.pad_token_id = tokenizer.pad_token_id


# ============================================================
# C2: LoRA on the MLP of ALL 12 Transformer blocks
# ============================================================
#
# GPT-2 small has:
#
# transformer.h.0
# transformer.h.1
# ...
# transformer.h.11
#
# Each block has:
#
# mlp.c_fc
# mlp.c_proj
#
# We explicitly construct all target names so there is no
# ambiguity about which modules receive LoRA.
# ============================================================

target_modules = []

for block_idx in range(12):
    target_modules.append(
        f"transformer.h.{block_idx}.mlp.c_fc"
    )
    target_modules.append(
        f"transformer.h.{block_idx}.mlp.c_proj"
    )

print("\nC2 LoRA target modules:")

for name in target_modules:
    print(" ", name)


# ============================================================
# LoRA configuration
# ============================================================

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,

    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    target_modules=target_modules,

    bias="none",
)


# ============================================================
# Attach LoRA adapters
# ============================================================

model = get_peft_model(
    model,
    lora_config,
)


# ============================================================
# Inspect trainable parameters
# ============================================================

print("\nTrainable parameters:")

model.print_trainable_parameters()


# ============================================================
# Print every trainable parameter
# ============================================================

print("\nTrainable parameter names:")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(
            f"  {name}  "
            f"shape={tuple(param.shape)}"
        )


# ============================================================
# Sanity check:
# make sure only MLP c_fc/c_proj LoRA parameters are trainable
# ============================================================

trainable_names = [
    name
    for name, param in model.named_parameters()
    if param.requires_grad
]

for name in trainable_names:
    assert "lora_" in name, (
        f"Unexpected trainable parameter: {name}"
    )

    assert ".mlp." in name, (
        f"Unexpected non-MLP LoRA parameter: {name}"
    )

    assert (
        ".c_fc." in name or
        ".c_proj." in name
    ), (
        f"Unexpected MLP parameter: {name}"
    )


print(
    "\nSanity check passed: "
    "only MLP c_fc/c_proj LoRA parameters are trainable."
)


# ============================================================
# Inspect one training batch
# ============================================================

print("\nInspecting one training batch...")

sample_batch = collate_fn(
    [
        train_dataset[i]
        for i in range(min(BATCH_SIZE, len(train_dataset)))
    ]
)

print(
    "input_ids shape:     ",
    sample_batch["input_ids"].shape,
)

print(
    "attention_mask shape:",
    sample_batch["attention_mask"].shape,
)

print(
    "labels shape:        ",
    sample_batch["labels"].shape,
)


print("\nFirst input:")

print(
    tokenizer.decode(
        sample_batch["input_ids"][0],
        skip_special_tokens=False,
    )
)


response_label_ids = sample_batch["labels"][0]

response_label_ids = response_label_ids[
    response_label_ids != -100
]

print("\nFirst response labels:")

print(
    tokenizer.decode(
        response_label_ids,
        skip_special_tokens=False,
    )
)


# ============================================================
# Trainer configuration
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,

    # Transformers 4.39.x uses evaluation_strategy.
    evaluation_strategy="epoch",
    save_strategy="epoch",

    logging_steps=100,

    report_to="none",

    seed=SEED,

    # Keep this simple on the Intel Mac.
    fp16=False,
    bf16=False,
)


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    tokenizer=tokenizer,
    data_collator=collate_fn,
)


# ============================================================
# Train
# ============================================================

print("\nStarting C2 LoRA training...")

trainer.train()


# ============================================================
# Evaluation
# ============================================================

print("\nEvaluating...")

eval_results = trainer.evaluate()

print("\nEvaluation results:")

for key, value in eval_results.items():
    print(f"{key}: {value}")


if "eval_loss" in eval_results:
    perplexity = np.exp(eval_results["eval_loss"])

    print(
        f"eval_perplexity: {perplexity:.4f}"
    )


# ============================================================
# Save adapter
# ============================================================

print("\nSaving C2 LoRA adapter...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)


# ============================================================
# Generation
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

    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,

            max_new_tokens=max_new_tokens,

            do_sample=True,
            temperature=0.8,
            top_k=50,

            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Remove the prompt from the returned sequence.
    new_token_ids = generated_ids[
        0,
        prompt_length:
    ]

    response = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=False,
    )

    return response


# ============================================================
# Sample generations
# ============================================================

test_prompts = [
    "what did you do today ?",
    "what are you playing with ?",
    "where is mommy ?",
    "what did you eat ?",
    "do you want to go outside ?",
]


print("\nSample generations:\n")

for prompt in test_prompts:

    response = generate_child_response(
        model,
        prompt,
    )

    print(f"Adult: {prompt}")
    print(f"Child:{response}")
    print("-" * 60)

