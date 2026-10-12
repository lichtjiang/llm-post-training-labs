# SFT CHILDES GPT-2 LoRA C3

LoRA on Attention + MLP, all 12 GPT-2 blocks

## Learning Objectives

- Understand LoRA mechanics
- How to reference a target layer in the base model in order to attach LoRA
- Start using Trainer class to replace our training loop
- Start using `peft` package
- Add LoRA adapters and use LoraConfig

## Comparison to the previous experiment

Compared with C2, there are now two changes:

**1. Dataset filtering: keep only _adult → child_ examples.**

So a row like:

Adult: "What are you doing?"

Child: "I'm playing."

is retained, while:

Child: "Look at this!"

Adult: "Oh, that's interesting."

is discarded.

The important implementation detail is that we should filter the entire dataset first, then shuffle and take the train/validation/test subsets. That way, the 10k/1k/1k examples in C3 all satisfy the new constraint.

**2. LoRA targets: add GPT-2 attention layers**

- attn.c_attn
- attn.c_proj

in addition to:

- mlp.c_fc
- mlp.c_proj


C2 → C3 parameter expansion

There's a nice mechanical difference worth observing.

For one GPT-2 block:

```text
Module	Input → output	LoRA A	LoRA B	LoRA params
attn.c_attn	768 → 2304	8×768	2304×8	24,576
attn.c_proj	768 → 768	8×768	768×8	12,288
mlp.c_fc	768 → 3072	8×768	3072×8	30,720
mlp.c_proj	3072 → 768	8×3072	768×8	30,720
Total/block				                98,304
```
Across 12 blocks:

`98,304 × 12 = 1,179,648 trainable parameters`

So C3 goes from roughly:

```text
C1:       61,440
C2:      737,280
C3:    1,179,648
```

while the underlying GPT-2 remains about 124M parameters and almost all of those base parameters remain frozen.

This gives us a particularly useful progression:

```text
C1
MLP, last block
        ↓
C2
MLP, all blocks
        ↓
C3
Attention + MLP, all blocks
```

And now the dataset is controlled consistently as:

                  Adult → Child
                        │
              ┌─────────┴─────────┐
              ↓                   ↓
         prompt tokens       response tokens
              │                   │
          label = -100       label = token ID
                                  │
                                  ↓
                              SFT loss

One thing we would not change yet is the 10k/1k/1k split methodology. Keeping the same sizes and seed while changing the filtering criterion makes C3 a much cleaner experiment. The filtering itself changes the data distribution, of course, but the training budget stays comparable to C1/C2.

## Training Data

### Dataset

Same as C2.

### Input and Label

The difference from C2 is described above.

## Architecture

```text
GPT-2
│
├── Block 0
│   ├── Attention
│   │   ├── c_attn ← LoRA
│   │   └── c_proj ← LoRA
│   └── MLP
│       ├── c_fc   ← LoRA
│       └── c_proj ← LoRA
│
├── Block 1
│   └── ...
│
...
│
└── Block 11
    ├── Attention ← LoRA
    └── MLP       ← LoRA
```

## Loss Function

Same as C2.

## Evaluation Metrics

Same as C2. 