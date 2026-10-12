# SFT CHILDES GPT-2 LoRA C1

In both of the previous two experiments, we do full fine-tuning, which means that we update all 124M pretrained parameters. Although 124M parameters are considered small compared to frontier models, generally speaking, we want a more efficient way to fine tune an LLM by not updating all its parameters. That's what we are going to do in this and the next two expeirments, freezing 124M parameters + learning a small number of low-rank matrices, which is called `LoRA`, or Low-Rank Adaptation of Large Language Models.

Specifically, C1 = GPT-2 + LoRA on mlp.c_fc and mlp.c_proj in block 11 only, with the base GPT-2 model frozen and Hugging Face Trainer handling the training loop.

One important detail: GPT-2 has 12 blocks indexed 0–11, so "11th" technically means transformer.h.11, the 12th/last block if counting from 1. I'll use transformer.h.11.

## Learning Objectives

- Understand LoRA mechanics
- How to reference a target layer in the base model in order to attach LoRA
- Start using Trainer class to replace our training loop
- Start using `peft` package
- Add LoRA adapters and use LoraConfig

## Why Current Experiment?

First, let's understand GPT-2's modules. GPT-2's block roughly looks like:

```text
                 ┌─── Attention ──────────────┐
x → LayerNorm →  │ c_attn → split Q,K,V       │
                 │                            │
                 │ attention → c_proj         │
                 └────────────────────────────┘
                         ↓
                    residual
                         ↓
                    LayerNorm
                         ↓
                 ┌─── MLP ─────────────────────┐
                 │ c_fc → activation → c_proj  │
                 └─────────────────────────────┘
                         ↓
                    residual

```

So there are different locations where we can target a LoRA adapter:

```text
attention:
    c_attn       # combined Q/K/V projection
    c_proj       # attention output projection

MLP:
    c_fc         # 768 → 3072
    c_proj       # 3072 → 768
```

Note that there are two different `c_proj` modules in each block, one under `attention` and one under `MLP`. So when configuring LoRA, we generally want to use the full module path or a carefully chosen target pattern.

If we ask "Can the MLP layer in the final Transformer block alone redirect the pretrained representation into CHILDES-style responses?", then we have:

```text
h
 │
 ▼
c_fc
768 → 3072
 │
 ▼
activation
 │
 ▼
c_proj
3072 → 768
 │
 ▼
residual
```

The LoRA modification is conceptually:

`W' = W + BA`

where:

```text
W  = frozen pretrained matrix
BA = small trainable low-rank update
```

```text
                 c_fc

       pretrained W
       3072 × 768
            │
            │ frozen
            │
            ├───────────────┐
            │               │
            │          LoRA delta
            │               │
            │             B @ A
            │               │
            └───────┬───────┘
                    ▼
             effective W'
```

For example, suppose:

`c_fc: 768 × 3072`

With rank `r = 8`, instead of learning:

`768 × 3072 = 2,359,296 parameters`

LoRA learns approximately:

```text
A: 8 × 3072
B: 768 × 8

= 24,576 + 6,144
= 30,720 parameters
```

That's a **huge reduction**.

And importantly:

```text
original W
   ↓
frozen

A, B
   ↓
trainable
```

So our experiment makes the LoRA mechanism very tangible.

That's exactly one way to perform a parameter efficient fine-tuning. By targeting different layers/blocks, we have these three LoRA experiments:

```text
C1: LoRA on mlp.c_fc + mlp.c_proj in block 11 (the top most) only
C2: LoRA on mlp.c_fc + mlp.c_proj in all 12 blocks
C3: LoRA on attn.c_attn, attn.c_proj + mlp.c_fc, mlp.c_proj in all 12 blocks
```

So, the progression looks like this:

```text
Experiment A
Full fine-tuning
    ↓
Experiment B
Response-only SFT
    ↓
Experiment C1 (current)
LoRA: block 11 MLP
    ↓
Experiment C2
LoRA: all MLP
    ↓
Experiment C3
LoRA: attention + MLP
```

## Training Data

### Dataset

Same as Experiment B

### Input and Label

Same as Experiment B


## Architecture

```text
GPT-2 124M
│
├── transformer.h.0      frozen
├── transformer.h.1      frozen
├── ...
├── transformer.h.10     frozen
│
└── transformer.h.11
      │
      └── mlp
           ├── c_fc       ← LoRA
           └── c_proj     ← LoRA

lm_head                  frozen
```

The original `c_fc` and `c_proj` weights remain frozen. LoRA adds trainable low-rank matrices to those two modules.

So C1 has roughly:

30,720 (see ["Why current experiment"](https://github.com/lichtjiang/llm-post-training-labs/blob/main/child-dialogue-lm/lora/sft_childes_gpt2_lora_c1.md#why-current-experiment) from above) × 2
= 61,440

trainable LoRA parameters, versus ~124M total GPT-2 parameters.

That's about 0.05% of the model.

We'll keep the same:

- 10,000 training examples
- 1,000 validation examples
- 1 epoch
- response-only labels
- maximum length 1024
- GPT-2 tokenizer
- CHILDES conversational pairs
- generation procedure

The main things that change from Experiment B are PEFT + Trainer.

## Loss Function

Same as Experiment B

## Evaluation Metrics

Same as Experiment B

## Things to Inspect

Since our goal is now understanding the mechanics, we would inspect the `C1` checkpoint fairly deeply before moving to `C2`. There are three especially useful things to learn: adapter-only loading, merging, and the actual LoRA tensors.

One version caveat: I was using `peft==0.10.0`, so the examples below were written to the older PEFT API style that matches the original development environment of mine. The core APIs—PeftModel.from_pretrained(), save_pretrained(), and merge_and_unload()—are the same concepts documented by PEFT.

**1. Load the base GPT-2 + saved LoRA adapter for inference**

Suppose our training produced:

```text
gpt2-childes-lora-c1/
├── adapter_config.json
├── adapter_model.safetensors    # or adapter_model.bin
├── ...
```

The important point is that this directory does not contain another copy of the 124M GPT-2 weights. It contains the LoRA adapter.

Load the base model first:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "openai-community/gpt2"
ADAPTER_DIR = "./gpt2-childes-lora-c1"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
base_model.config.pad_token_id = tokenizer.pad_token_id

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
)

model.eval()
```

That's it.

Conceptually:

```text
                   disk
                    │
          ┌─────────┴─────────┐
          │                   │
     GPT-2 124M           LoRA adapter
     pretrained           ~61K params
          │                   │
          └─────────┬─────────┘
                    ▼
               PeftModel
```

`PeftModel.from_pretrained()` takes an already-loaded base model and loads the adapter configuration and weights from the specified directory.

We can then use exactly the same generation code we used before:

```python
prompt = "Adult: what did you do today ?\nChild:"

inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=False))
```

We can load the exact same base model without the adapter:

```python
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
base_model.config.pad_token_id = tokenizer.pad_token_id

base_model.eval()
```

Then compare:

```text
base GPT-2
     vs.
GPT-2 + C1 LoRA
```

using exactly the same prompt and random seed.

That's probably the cleanest demonstration that the adapter is actually changing the model.

**2. Merge LoRA into the base model**

This is another important concept.

Before merging:

```text
W' x = W x + BA x
      ↑     ↑
    frozen  LoRA
```

After merging:

```text
W' = W + BA

W' x
```

So the LoRA machinery can disappear after the merge.

PEFT provides `merge_and_unload()` for precisely this purpose. It returns a normal base-model architecture with the adapter weights merged into the base weights.

For our C1 checkpoint:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "openai-community/gpt2"
ADAPTER_DIR = "./gpt2-childes-lora-c1"
MERGED_DIR = "./gpt2-childes-lora-c1-merged"

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
)

merged_model = model.merge_and_unload()

merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
```

Notice this:

`merged_model = model.merge_and_unload()`

rather than:

`model.merge_and_unload()`

The returned model is the important object. PEFT documents `merge_and_unload()` as returning the standalone base architecture with the adapter merged into it.

We can then load it normally:

```python
model = AutoModelForCausalLM.from_pretrained(
    "./gpt2-childes-lora-c1-merged"
)
```

No PEFT is required for inference anymore.

**Adapter vs merged model**

This distinction is worth remembering:

**Adapter checkpoint**

```text
gpt2-childes-lora-c1/
    adapter_config.json
    adapter_model.safetensors
```

Needs:

```text
base GPT-2
+
PEFT
+
adapter
```

Advantages:

- very small checkpoint
- can keep multiple adapters
- can disable/swap adapters
- preserves the separation between base model and adaptation

**Merged checkpoint**
```text
gpt2-childes-lora-c1-merged/
    config.json
    model.safetensors
    tokenizer files...
```

Contains:

`W' = W + BA`

Advantages:

- ordinary Transformers model
- no PEFT adapter machinery required
- convenient for deployment

The tradeoff is that once merged, we lose the ability to manipulate the LoRA adapter separately; PEFT explicitly notes this limitation.

**3. How to inspect the actual LoRA matrices**

This is probably the most educational part of C1.

After:
```python
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
)
```

inspect:

```python
for name, module in model.named_modules():
    if "lora" in name.lower():
        print(name, type(module))
```

We should see LoRA-related modules.

An even more useful inspection is:

```python
for name, param in model.named_parameters():
    if "lora_" in name:
        print(name, tuple(param.shape), param.requires_grad)
```

We should get something conceptually like:

```python
base_model.model.transformer.h.11.mlp.c_fc.lora_A.default.weight
base_model.model.transformer.h.11.mlp.c_fc.lora_B.default.weight

base_model.model.transformer.h.11.mlp.c_proj.lora_A.default.weight
base_model.model.transformer.h.11.mlp.c_proj.lora_B.default.weight
```

The exact prefixes can vary slightly with PEFT version, but the important part is lora_A / lora_B.

**4. What shapes should we expect?**

This is a particularly nice sanity check.

GPT-2's MLP:

```python
hidden = 768
intermediate = 3072
```

For:

`c_fc: 768 → 3072`

the original weight is conceptually:

`W_fc ∈ R^(3072 × 768)`

With rank:

`r = 8`

LoRA factorizes the update into:

```text
A ∈ R^(8 × 768)

B ∈ R^(3072 × 8)
```

so:

`B A`

has shape:

```text
(3072 × 8) @ (8 × 768)
= (3072 × 768)
```

exactly the same shape as `W_fc`.

For `c_proj`:

```text
W_proj ∈ R^(768 × 3072)

A ∈ R^(8 × 3072)

B ∈ R^(768 × 8)
```

Therefore:

```text
B A
=
(768 × 8) @ (8 × 3072)

=
768 × 3072
```

Again, exactly the same shape as the original weight.

So we can verify this programmatically.

```python
for name, param in model.named_parameters():
    if "lora_" in name:
        print(f"{name:90s} {tuple(param.shape)}")
```

We should see roughly:

```text
...c_fc.lora_A...       (8, 768)
...c_fc.lora_B...       (3072, 8)

...c_proj.lora_A...     (8, 3072)
...c_proj.lora_B...     (768, 8)
```

This is exactly the rank-8 bottleneck we discussed mathematically.

**5. Inspect the LoRA configuration itself**

We can also access the PEFT configuration:

`print(model.peft_config)`

For the default adapter:

```python
config = model.peft_config["default"]

print("r:", config.r)
print("alpha:", config.lora_alpha)
print("dropout:", config.lora_dropout)
print("target modules:", config.target_modules)
```

We should see approximately:

```text
r: 8
alpha: 16
dropout: 0.05
target modules: ...
```

This is useful because now we have two independent verification mechanisms:

```text
configuration
     ↓
r = 8
     ↓
actual tensors
     ↓
A/B shapes contain 8
```

**6. Calculate the actual update**

We can go one step deeper.

For `c_fc`, retrieve:

`layer = model.base_model.model.transformer.h[11].mlp.c_fc`

Then:

```python
A = layer.lora_A["default"].weight
B = layer.lora_B["default"].weight

print("A:", A.shape)
print("B:", B.shape)
```

And calculate the low-rank update:

```python
delta_W = B @ A

print("delta_W:", delta_W.shape)
```

We should get:

```text
A:        torch.Size([8, 768])
B:        torch.Size([3072, 8])
delta_W: torch.Size([3072, 768])
```

That's a beautiful concrete demonstration of:

$$ W' = W + \frac{\alpha}{r}BA $$

For our configuration:

```text
alpha = 16
r     = 8
```

so:

`alpha / r = 2`

and therefore the LoRA contribution is scaled accordingly.

The exact implementation details of how PEFT applies the scaling are worth inspecting too, but conceptually this is the equation we're learning.

**7. Compare base and adapted weights**

Before merging:

```python
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

peft_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
)
```

Take the original `c_fc`:

```python
base_weight = (
    base_model
    .transformer
    .h[11]
    .mlp
    .c_fc
    .weight
)
```

Then inspect the PEFT-wrapped layer:

```python
lora_layer = (
    peft_model
    .base_model
    .model
    .transformer
    .h[11]
    .mlp
    .c_fc
)
```

Then calculate:

```python
A = lora_layer.lora_A["default"].weight
B = lora_layer.lora_B["default"].weight

delta_W = (lora_layer.scaling["default"]) * (B @ A)

print("base weight:", base_weight.shape)
print("LoRA delta:", delta_W.shape)
```

This gives us the physical picture:
```text
                 c_fc

       pretrained W
       3072 × 768
            │
            │ frozen
            │
            ├───────────────┐
            │               │
            │          LoRA delta
            │               │
            │          B @ A
            │               │
            └───────┬───────┘
                    ▼
             effective W'
```

That's probably the single most useful C1 exercise we should do.


**8. Four quick measurements before proceeding to the next experiment**

**A. Trainable parameter count**

We already have:

`model.print_trainable_parameters()`

Record:

```text
total parameters
trainable parameters
percentage
```

For C1, we expect roughly 61K trainable parameters with r=8, versus ~124M total.

**B. Confirm every non-LoRA parameter is frozen**

```python
trainable = []

for name, param in model.named_parameters():
    if param.requires_grad:
        trainable.append(name)

print("\n".join(trainable))
```

We should see only LoRA parameters.

This is important because C1 is specifically testing:

    Can I adapt GPT-2 without modifying its pretrained weights?

**C. Confirm the target location**

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
```

Make sure everything says:

`transformer.h.11.mlp...`

and nothing says:

```text
h.0
h.1
...
h.10
```

This verifies that our "last block only" experiment really is what we intended.

**D. Compare parameter storage**

This is fun and gives us an intuition for why LoRA matters.

```python
total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Total:", total_params)
print("Trainable:", trainable_params)
print("Percentage:", 100 * trainable_params / total_params)
```

For C1, the ratio should be tiny.

Please refer to `sft_childes_gpt2_lora_c1_inspect.py` for more things to inspect, such as:

- adapter_config.json
- adapter checkpoint size
- model.print_trainable_parameters()
- actual lora_A / lora_B shapes
- B @ A shape
- alpha / r scaling
- which parameters are frozen/trainable
- adapter-only inference vs base-model inference

That turns C1 from "I successfully ran LoRA" into actually understanding what PEFT inserted into our GPT-2 model.