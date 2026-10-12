import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "openai-community/gpt2"
DATASET_NAME = "nshah-fbcs/childes-engUK-conversational-pairs"

OUTPUT_DIR = "./gpt2-childes-qlora-d1"

MAX_LENGTH = 1024

TRAIN_SIZE = 10_000
VAL_SIZE = 1_000
TEST_SIZE = 1_000

SEED = 42


# ============================================================
# LoRA configuration
# ============================================================

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


# ============================================================
# Training configuration
# ============================================================

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 4
NUM_EPOCHS = 1


# ============================================================
# 4-bit QLoRA configuration
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    # QLoRA's recommended 4-bit quantization type.
    bnb_4bit_quant_type="nf4",

    # Double quantization:
    # quantize the quantization constants themselves.
    bnb_4bit_use_double_quant=True,

    # Compute in float32 on this simple experiment.
    bnb_4bit_compute_dtype=torch.float32,
)


# ============================================================
# Load tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# GPT-2 has no dedicated padding token.
tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Load dataset
# ============================================================

dataset = load_dataset(DATASET_NAME)

print(dataset)


# ============================================================
# Keep ONLY Adult -> Child examples
# ============================================================

def is_adult_to_child(example):
    speaker_a = str(example["speaker_a"]).strip().lower()
    speaker_b = str(example["speaker_b"]).strip().lower()

    return (
        speaker_a == "adult"
        and speaker_b == "child"
    )


dataset = dataset.filter(
    is_adult_to_child
)

print("\nAfter adult -> child filtering:")
print(dataset)


# ============================================================
# Shuffle and create fixed train/validation/test sets
# ============================================================

all_data = dataset["train"].shuffle(
    seed=SEED
)

required_size = (
    TRAIN_SIZE
    + VAL_SIZE
    + TEST_SIZE
)

if len(all_data) < required_size:
    raise ValueError(
        f"Not enough adult->child examples. "
        f"Need {required_size}, "
        f"but only {len(all_data)} are available."
    )


train_dataset = all_data.select(
    range(0, TRAIN_SIZE)
)

val_dataset = all_data.select(
    range(
        TRAIN_SIZE,
        TRAIN_SIZE + VAL_SIZE,
    )
)

test_dataset = all_data.select(
    range(
        TRAIN_SIZE + VAL_SIZE,
        TRAIN_SIZE + VAL_SIZE + TEST_SIZE,
    )
)


print("\nDataset sizes:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# ============================================================
# Response-only SFT tokenization
# ============================================================

def tokenize_example(example):
    adult_text = example["utterance_a"]
    child_text = example["utterance_b"]

    # Prompt
    prompt = (
        f"Adult: {adult_text}\n"
        f"Child:"
    )

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    # Response.
    #
    # Leading space preserves GPT-2's normal BPE behavior
    # for a word appearing after "Child:".
    response_ids = tokenizer(
        " " + child_text,
        add_special_tokens=False,
    )["input_ids"]

    # Explicit EOS target.
    response_ids = (
        response_ids
        + [tokenizer.eos_token_id]
    )

    # Complete causal-LM sequence.
    input_ids = (
        prompt_ids
        + response_ids
    )

    # Response-only loss.
    #
    # Prompt positions are ignored by CrossEntropyLoss.
    labels = (
        [-100] * len(prompt_ids)
        + response_ids
    )

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
# Dynamic padding collator
# ============================================================

def collate_fn(features):
    input_ids = [
        item["input_ids"]
        for item in features
    ]

    labels = [
        item["labels"]
        for item in features
    ]

    # Pad input_ids with GPT-2's pad token.
    batch_input_ids = tokenizer.pad(
        {
            "input_ids": input_ids,
        },
        padding=True,
        return_tensors="pt",
    )["input_ids"]

    max_length = batch_input_ids.shape[1]

    # Pad labels with -100 so padding does not contribute
    # to the loss.
    padded_labels = []

    for label in labels:
        padded = (
            label
            + [-100] * (max_length - len(label))
        )

        padded_labels.append(padded)

    batch_labels = torch.tensor(
        padded_labels,
        dtype=torch.long,
    )

    attention_mask = (
        batch_input_ids
        != tokenizer.pad_token_id
    ).long()

    return {
        "input_ids": batch_input_ids,
        "labels": batch_labels,
        "attention_mask": attention_mask,
    }


# ============================================================
# Load GPT-2 in 4-bit
# ============================================================

print("\nLoading GPT-2 in 4-bit NF4...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

model.config.pad_token_id = (
    tokenizer.pad_token_id
)


# ============================================================
# Prepare quantized model for k-bit training
# ============================================================

model = prepare_model_for_kbit_training(
    model
)


# ============================================================
# D1 LoRA target modules
#
# MLP only, all 12 GPT-2 blocks.
#
# Block:
#
#   Attention
#       c_attn
#       c_proj
#
#   MLP
#       c_fc    <-- LoRA
#       c_proj  <-- LoRA
# ============================================================

target_modules = []

for block_idx in range(12):
    target_modules.append(
        f"transformer.h.{block_idx}.mlp.c_fc"
    )

    target_modules.append(
        f"transformer.h.{block_idx}.mlp.c_proj"
    )


print("\nLoRA target modules:")

for module in target_modules:
    print(" ", module)


# ============================================================
# LoRA configuration
# ============================================================

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=target_modules,
)


# ============================================================
# Attach LoRA adapters
# ============================================================

model = get_peft_model(
    model,
    lora_config,
)


# ============================================================
# Trainable parameter inspection
# ============================================================

model.print_trainable_parameters()


# ============================================================
# Sanity checks
# ============================================================

trainable_names = [
    name
    for name, param in model.named_parameters()
    if param.requires_grad
]


# Only LoRA parameters should be trainable.
assert all(
    "lora_" in name
    for name in trainable_names
)


# D1 should only modify MLP layers.
assert all(
    ".mlp." in name
    for name in trainable_names
)


# Both MLP projections should have LoRA.
assert any(
    ".mlp.c_fc." in name
    for name in trainable_names
)

assert any(
    ".mlp.c_proj." in name
    for name in trainable_names
)


# ============================================================
# Training arguments
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

    # Keep this explicit.
    #
    # QLoRA's quantized weights are handled by bitsandbytes.
    # We do not additionally enable fp16/bf16 here.
    fp16=False,
    bf16=False,

    seed=SEED,
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

trainer.train()


# ============================================================
# Validation evaluation
# ============================================================

print("\nValidation evaluation:")

val_metrics = trainer.evaluate()

print(val_metrics)


# ============================================================
# Held-out test evaluation
# ============================================================

print("\nTest evaluation:")

test_metrics = trainer.evaluate(
    eval_dataset=test_dataset
)

print(test_metrics)


# ============================================================
# Save adapter
# ============================================================

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ============================================================
# Generation helper
# ============================================================

def generate_child_response(adult_text):
    prompt = (
        f"Adult: {adult_text}\n"
        f"Child:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    # device_map="auto" may place the model somewhere other
    # than CPU, so put inputs on the same device as the model's
    # input embedding layer.
    input_device = (
        model.get_input_embeddings()
        .weight
        .device
    )

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    prompt_length = (
        inputs["input_ids"].shape[1]
    )

    generated_ids = model.generate(
        **inputs,

        max_new_tokens=50,

        do_sample=True,
        temperature=0.8,
        top_p=0.9,

        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Keep only newly generated response tokens.
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
# Example generation
# ============================================================

print("\nExample generation:")

print(
    generate_child_response(
        "What are you doing?"
    )
)

