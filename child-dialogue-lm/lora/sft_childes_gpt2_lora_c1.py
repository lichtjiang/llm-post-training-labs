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
"""

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "openai-community/gpt2"
DATASET_NAME = "nshah-fbcs/childes-engUK-conversational-pairs"

MAX_TRAIN_EXAMPLES = 10_000
MAX_VAL_EXAMPLES = 1_000
MAX_TEST_EXAMPLES = 1_000

MAX_LENGTH = 1024

BATCH_SIZE = 4
NUM_EPOCHS = 1

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01

SEED = 42

OUTPUT_DIR = "./gpt2-childes-lora-c1"


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

# GPT-2 does not have a separate PAD token.
# We use EOS as PAD, as in Experiment B.
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
# Convert each example to:
#
# Adult: ...
# Child: ...
#
# with response-only labels.
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

    # Response-only SFT:
    #
    # prompt tokens -> ignored by cross entropy
    # response tokens -> contribute to loss
    labels = [-100] * len(prompt_ids) + response_ids

    # Truncate both together.
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
# Custom collator
#
# We keep this from Experiment B because the response-only
# label construction is an important part of our experiment.
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
# Load GPT-2
# ============================================================

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Make sure GPT-2 knows its padding token.
model.config.pad_token_id = tokenizer.pad_token_id


# ============================================================
# C1: LoRA only on the MLP of the final Transformer block
#
# GPT-2 block 11:
#
# transformer.h.11.mlp.c_fc
# transformer.h.11.mlp.c_proj
#
# PEFT's target_modules matching is name-based. Because
# c_fc/c_proj occur in every block, we first verify the names
# and then use exact module names for block 11.
# ============================================================

target_modules = [
    "transformer.h.11.mlp.c_fc",
    "transformer.h.11.mlp.c_proj",
]

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=target_modules,
    bias="none",
)

model = get_peft_model(model, lora_config)

print("\nTrainable parameters:")
model.print_trainable_parameters()


# ============================================================
# Verify exactly which parameters are trainable
# ============================================================

print("\nTrainable parameter names:")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}  {tuple(param.shape)}")


# ============================================================
# Inspect one batch
# ============================================================

print("\nInspecting one training batch...")

sample_batch = collate_fn(
    [train_dataset[i] for i in range(min(BATCH_SIZE, len(train_dataset)))]
)

print("input_ids shape:     ", sample_batch["input_ids"].shape)
print("attention_mask shape:", sample_batch["attention_mask"].shape)
print("labels shape:        ", sample_batch["labels"].shape)

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
# Trainer
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,

    evaluation_strategy="epoch",
    save_strategy="epoch",

    logging_steps=100,

    report_to="none",

    seed=SEED,

    # Keep the experiment simple on the Mac.
    fp16=False,
    bf16=False,
)


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

print("\nStarting C1 LoRA training...")

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
    print(f"eval_perplexity: {perplexity:.4f}")


# ============================================================
# Save LoRA adapter
# ============================================================

print("\nSaving LoRA adapter...")

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

    prompt = f"Adult: {adult_utterance}\nChild:"

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

    # Remove the prompt from the generated sequence.
    new_token_ids = generated_ids[0, prompt_length:]

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
    response = generate_child_response(model, prompt)

    print(f"Adult: {prompt}")
    print(f"Child:{response}")
    print("-" * 60)
