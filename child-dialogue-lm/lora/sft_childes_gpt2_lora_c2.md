# SFT CHILDES GPT-2 LoRA C2

LoRA on mlp.c_fc + mlp.c_proj in all 12 blocks, everything else stays the same as C1.

One small improvement from the C1 implementation: rather than relying on PEFT's name matching with exact full paths, we'll explicitly construct the target module names for blocks 0..11. This makes the experiment unambiguous.


## Learning Objectives

- Understand LoRA mechanics
- How to reference a target layer in the base model in order to attach LoRA
- Start using Trainer class to replace our training loop
- Start using `peft` package
- Add LoRA adapters and use LoraConfig


## Comparison to the previous experiment

Very little has changed from C1, which is exactly what we want.

The key difference is this:

C1

```python
target_modules = [
    "transformer.h.11.mlp.c_fc",
    "transformer.h.11.mlp.c_proj",
]
```

C2

```python
target_modules = []

for block_idx in range(12):
    target_modules.append(
        f"transformer.h.{block_idx}.mlp.c_fc"
    )
    target_modules.append(
        f"transformer.h.{block_idx}.mlp.c_proj"
    )
```

Everything else—response-only SFT, dataset, tokenizer, rank, alpha, dropout, LR, batch size, number of examples, epoch count—stays the same.

That makes C1 vs C2 a relatively clean experiment.

One thing we would notice

C1 had approximately:

```text
2 target modules
× 2 LoRA matrices
× ~30,720 parameters
≈ 61,440 trainable parameters
```

C2 has:

```text
12 blocks
× 2 target modules/block
× 2 LoRA matrices
```

so approximately:

```text
61,440 × 12
≈ 737,280 trainable parameters
```

That's still only around:

`737K / 124M ≈ 0.6%`

of GPT-2's parameters.

So C2 gives the model `12×` as many trainable LoRA parameters as C1, while still leaving `~99.4%` of the original parameters frozen.

That is a nice concrete illustration of the flexibility we get by choosing where to put LoRA.

Below is a quick comparison between C2 and C1.

```text

                     C1             C2
                    ─────          ─────
trainable params     ~61K          ~737K
target blocks        11             0-11
target modules       MLP            MLP
LoRA rank             8              8
training examples   10K            10K
epochs                1              1
```

## Training Data

### Dataset

Same as C1.

### Input and Label

Same as C1.

## Architecture

```text
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
```
No attention LoRA yet. That comes in C3.

## Loss Function

Same as C1.

## Evaluation Metrics

Same as C1.

## Things to Inspect Next

- Trainable parameter count
- Actual lora_A / lora_B shapes
- Adapter checkpoint size
- Training loss
- Validation loss
- A few generations
- Whether every block 0–11 actually has LoRA tensors

For the last one, this is a particularly nice check:

```python
for block_idx in range(12):
    layer = model.base_model.model.transformer.h[block_idx].mlp.c_fc

    A = layer.lora_A["default"].weight
    B = layer.lora_B["default"].weight

    print(
        f"block {block_idx}: "
        f"A={tuple(A.shape)}, "
        f"B={tuple(B.shape)}"
    )
```

We should see:

```text
block 0:  A=(8, 768),   B=(3072, 8)
block 1:  A=(8, 768),   B=(3072, 8)
...
block 11: A=(8, 768),   B=(3072, 8)
```

For `c_proj`, the corresponding shapes should be:

```text
A=(8, 3072)
B=(768, 8)
```

That will make C2's difference from C1 extremely concrete: we haven't changed what LoRA is; we've changed how many places in the Transformer are allowed to receive a low-rank update.

Once C2 is running, C3 can then make the next clean change: add attention c_attn + attention c_proj while keeping the MLP adapters, which lets us explore which parts of the Transformer we're adapting.