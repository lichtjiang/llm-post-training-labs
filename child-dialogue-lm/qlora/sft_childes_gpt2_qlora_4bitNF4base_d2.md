# SFT CHILDES QLORA 4Bit NF4 Base D2

D2 is essentially D1 + the attention LoRA targets from C3. All the other experimental conditions remain unchanged: adult→child only, response-only SFT, 10k/1k/1k, seed 42, NF4 + double quantization, r=8, alpha=16.

So we now have the clean progression:

```text
             Precision       LoRA targets
C2          normal/FP32          MLP
C3          normal/FP32      MLP + Attention

D1           4-bit NF4          MLP
D2           4-bit NF4       MLP + Attention
```

## Learning Objectives

- Understand QLoRA mechanics
- 4-bit NF4 quantization
- Double quantization

## Comparison to the previous experiment

The key comparison we now have

D1 has:

$$ \text{MLP LoRA} $$

while D2 has:

$$ \text{Attention LoRA}+\text{MLP LoRA} $$

The trainable parameter count should therefore jump from approximately:

$$ 737{,}280 $$

in D1 to:

$$ 1{,}179{,}648 $$

in D2.

The calculation is the same as C3:

```text
Component	Per block	× 12 blocks
attn.c_attn	24,576	294,912
attn.c_proj	12,288	147,456
mlp.c_fc	30,720	368,640
mlp.c_proj	30,720	368,640
Total	    98,304	1,179,648
```

So an especially nice result would be:

```text
D1  ~737K trainable
D2 ~1.180M trainable
```

while the quantized base remains frozen in both cases.

And D1/D2 should use essentially the same quantized base-model memory footprint. The additional D2 memory is primarily from its extra LoRA parameters and their optimizer states.


## Training Data

### Dataset

Same as D1.

### Input and Label

Same as D1.

## Architecture

```text
4-bit NF4 base
Attention + MLP LoRA, all blocks
```

## Loss Function

Same as D1.

## Evaluation Metrics

Same as D1.