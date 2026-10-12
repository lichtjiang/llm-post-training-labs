# SFT CHILDES GPT-2 QLoRA 4Bit NF4 Base D1

D1 is a QLoRA experiment in our lab progression, while keeping everything else aligned with C3 so that the main new variable is quantization.

We'll use:

- GPT-2 124M
- 4-bit NF4 quantization
- double quantization
- LoRA only on the MLP
    - mlp.c_fc
    - mlp.c_proj
- all 12 Transformer blocks
- adult → child only
- response-only SFT
- 10k / 1k / 1k train/validation/test
- same r=8, alpha=16, dropout 0.05
- same LR 5e-4
- same batch size and 1 epoch

So the LoRA progression is now:

```text
C1:
FP32 base
MLP LoRA, on block 11

C2:
FP32 base
MLP LoRA, all blocks

C3:
FP32 base
Attention + MLP LoRA, all blocks

D1:
4-bit NF4 base
MLP LoRA, all blocks
```

For some of us who plan to run on CPUs locally, the first thing we should do is verify that our installed bitsandbytes actually supports our local backend. Standard 4-bit bitsandbytes historically depends heavily on CUDA, so D1 may not run on our machine even though the code is conceptually correct. Due to this limitation, run it in Google CoLab with a T4 GPUT Runtime is a good option.

## Learning Objectives

- Understand QLoRA mechanics
- 4-bit NF4 quantization
- Double quantization

## Comparison to the experiment C2

D1 is designed as a quantization version of C2. One important implementation difference is that the base model is loaded with BitsAndBytesConfig(load_in_4bit=True, ...), and PEFT's prepare_model_for_kbit_training() prepares the quantized model before attaching LoRA.

The important conceptual distinction is this:

```text
C2
┌───────────────────────────────┐
│ GPT-2 FP32 weights            │
│                               │
│ frozen W                      │
│       +                       │
│ LoRA ΔW = B @ A               │
└───────────────────────────────┘


D1
┌───────────────────────────────┐
│ GPT-2 4-bit NF4 weights       │
│                               │
│ frozen quantized Wq           │
│       +                       │
│ LoRA ΔW = B @ A               │
└───────────────────────────────┘
```

The base model is not trained in either case.

The difference is that in D1 the frozen base weights are stored in 4-bit quantized form. During computation, bitsandbytes dequantizes them as needed.

The LoRA matrices themselves remain trainable in higher precision.

For each MLP layer:

$$ W' \approx \operatorname{dequant}(W_q) + \frac{\alpha}{r}BA $$

with:

$$ r=8,\qquad \alpha=16 $$

so the LoRA scaling is still:

$$ \frac{\alpha}{r}=2 $$

D1 should have the same LoRA parameter count as C2

This is an important sanity check.

Quantization doesn't change the LoRA architecture.

Therefore:

C2:
737,280 trainable parameters

D1:
737,280 trainable parameters

The difference is the storage/compute representation of the frozen base model, not the number of trainable adapter parameters.

And because we're retaining exactly the same:

```text
adult → child filtering
10,000 train
1,000 validation
1,000 test
seed = 42
response-only loss
r = 8
alpha = 16
MLP-only targets
```

D1 gives us a reasonably clean comparison against C2:

What changes when the frozen GPT-2 base goes from ordinary precision to 4-bit NF4 + double quantization, while the trainable LoRA setup stays the same?

One caveat before running it: as we briefly mentioned, in our environment, if our installed bitsandbytes rejects load_in_4bit=True, that's a hardware/backend limitation rather than a problem with the D1 experiment. In that case, we should verify the exact bitsandbytes version/backend we're using before attempting to change anything in the experiment.

## What comes next?

So far, C1 → C2 → C3 teaches us where LoRA is applied.

C2 → D1 then isolates a different question:

    Can we quantize the frozen base model heavily without substantially sacrificing the adaptation we get from LoRA?

The next question to ask is whether adding LoRA to attention on top of the quantized base produces the same kind of change we saw from C2 → C3? Answer to this question leads us to our next experiment D2:

```text
4-bit NF4 base
Attention + MLP LoRA, all blocks
``

## Training Data

### Dataset

Same as C3.

### Input and Label

Same as C3.

## Architecture

4-bit NF4 base
MLP LoRA, all blocks

## Loss Function

Same as C3.

## Evaluation Metrics

Same as C3.

## Q & A

### What do we expect to see in D1 compared to C2?

The strongest expectation is lower memory usage, not necessarily faster training. And because our model is only GPT-2 124M, D1 may not show a dramatic practical advantage over C2.

```text
C2 vs D1
	                            C2              	D1
Base GPT-2	              Normal precision	    4-bit NF4
Base weights trainable?	        No	                No
LoRA                      MLP, all 12 blocks	   Same
Trainable params              ~737K                ~737K
Dataset                   Adult → Child            Same
SFT                  	  Response-only            Same
Expected quality            Baseline    	Hopefully comparable
Base-model memory       	Higher              Much lower
Training speed             Baseline     	Not necessarily faster
```

The reason D1 exists is primarily:

Quantize the frozen base model aggressively while keeping the small trainable LoRA adapters in higher precision.

### Why can quality remain comparable?

The important thing is that we're not quantizing the trainable LoRA matrices.

Conceptually, C2 does:

$$ y = (W + \Delta W)x $$

where

$$ \Delta W = \frac{\alpha}{r}BA $$

D1 does approximately:

$$ y = (\operatorname{dequant}(W_q) + \Delta W)x $$

where W_q is the 4-bit NF4 representation of the frozen GPT-2 weights.

So D1 introduces some quantization error into the base model, but the model can still learn the task through the LoRA update.

That's why QLoRA can achieve performance surprisingly close to ordinary LoRA/full-precision fine-tuning in many settings.

But don't expect D1 to necessarily be faster

This is the part we'd emphasize.

We might initially think:

`4-bit weights → less data moved → faster training.`

But training involves more than loading the frozen weights.

For each forward/backward pass, the quantized weights have to be dequantized for computation. That introduces additional work.

So there are competing effects:

```text
4-bit weights
     │
     ├── ↓ memory
     │
     ├── ↓ memory bandwidth requirements
     │
     └── ↑ dequantization overhead
```

Whether D1 is faster depends heavily on the hardware and implementation.

On a GPU with a well-supported bitsandbytes stack, QLoRA's big advantage is usually memory efficiency, allowing us to fine-tune models that otherwise wouldn't fit. It isn't fundamentally a "make training faster" technique.

## Things to Inspect

GPT-2 124M is a special case in that it is already small.

So, the approximate base-weight storage difference is enormous in percentage terms:

```text
FP32:
124M × 4 bytes ≈ 496 MB

4-bit:
124M × 0.5 bytes ≈ 62 MB
```

The actual memory footprint contains other cose such as quantization metadata, model buffers, activations, etc., but the basic point stands.

For a 7B or 13B model, this difference becomes much more consequential.

For a 124M model, the absolute savings may not be dramatic enough to compensate for quantization/dequantization overhead.

So D1 is actually a good mechanics experiment, even if it isn't necessarily the most useful optimization for this particular model. This is an important point for our experiment.

Since we're learning rather than chasing benchmark quality, we'd record four things:

1. Validation/test loss

Compare:

```text
C2 validation loss
D1 validation loss

C2 test loss
D1 test loss
```

2. Peak memory

This is where D1 should demonstrate its main advantage.

3. Training time

Measure:

```text
C2: total training time
D1: total training time
```

Don't assume D1 wins.

4. Generated responses

Use exactly the same handful of adult prompts:

```text
"What are you doing?"
"Where is your mommy?"
"Do you want some milk?"
"What did you see?"
"Are you hungry?"
```

Then qualitatively compare the C2 and D1 outputs.
