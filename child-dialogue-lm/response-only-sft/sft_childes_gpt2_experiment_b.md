# SFT CHILDES GPT-2 Experiment B

This is response-only SFT. That is, we mask the prompt and calculate the loss only on the response (a child utterance). This makes us closer to training a Child Dialogue model.

## Learning Objectives

- response-only loss masking
- understand the difference between attention_mask and label/loss mask

## Why Experiment B

Remember that in the previous lab, "Experiment A", we train on the entire sequence (i.e. a pair of adjacent utterances). 

**Experiment A — simplest**

Train on the entire sequence:

```text
Adult: What are you doing?
Child: I'm playing.
```

and calculate causal LM loss over everything.

But there is an interesting SFT-specific refinement we are going to make.

Suppose the training example is:

```text
Adult: What are you doing?
Child: I'm playing.
```

Do we really want to train the model to predict:

```text
Adult
:
What
are
you
doing
?
Child
:
```
?

Not necessarily.

What we really care about is:

    Given the conversational context, generate the child's response.

So we can construct labels like:

```text
tokens:

Adult : What are you doing ? Child : I'm playing .
↑────────────── prompt ────────────↑↑── target ──↑

labels:

 -1   -1   -1  -1  -1  -1  -1  -1  -1  I'm playing .
```

where -1 represents "don't calculate loss here" conceptually; in PyTorch/Hugging Face this is normally represented by -100 for ignored positions.

So:
```text
                 loss?
Adult: What are you doing?        NO  NO  NO  NO  NO
Child: I'm playing with my toys.  YES YES YES YES YES
```
This is generally called response-only loss masking.

Then the loss becomes:

$$ L = -\sum_{t \in \text{child response}} \log P(x_t\mid x_{<t}) $$

That's much closer to what we mean by supervised fine-tuning for a conversational task. And this would make an excellent second experiment after we get the simplest version working.

That is exactly why we would like to progress to "Experiment B".

**Experiment B — response-only SFT**

Mask the prompt:

```text
Adult: What are you doing?
Child:
```

and calculate loss only on:

```text
I'm playing.
```

Then we can actually observe the difference between **continued language-model training** (Experiment A) and **response-targeted SFT** (Experiment B).

### Key Change

The important implementation idea is:

```text
                input_ids                         labels
Adult:          adult tokens                     -100
Child:          child tokens                     child token IDs
```

while _attention_mask_ remains `1` for both adult and child tokens.

So, quite little change in code for Experiment B.

In Experiment A, we effectively had:

```python
labels = input_ids.clone()
labels[attention_mask == 0] = -100
```

That means:

    Calculate loss on every real token.

For Experiment B, we need one additional piece of information`response_start`, i.e.,

    Where does the child response begin?

Then:

```python
labels = input_ids.clone()

# Ignore everything before the child response
labels[:, :response_start] = -100

# Also ignore padding
labels[attention_mask == 0] = -100
```

And importantly, the Transformer architecture, optimizer, cross-entropy loss, attention mask, batch processing, etc. don't fundamentally change.

We're simply changing which target tokens are allowed to contribute to the loss.

That makes Experiment B an excellent experiment because we can directly compare:

continued causal-LM training on the dialogue vs. instruction/response-style SFT where only the desired response is supervised

In addition, this gives us a particularly nice pedagogical Experiment B: the Transformer itself has not changed at all; we've changed the training objective simply by changing which label positions participate in cross-entropy.

## Training Data

### Dataset

    CHILDES Eng-UK Conversational Pairs Dataset on Hugging Face

### Input and Label

Unlike what we did in Experiment A, we only treat a child utterance as the label to learn.

## Architecture

Same as Experiment A.

## Loss Function

Same as Experiment A except for the label mask on prompt part.

## Evaluation Metrics

Same as Experiment A.


## Understand the Code

Since most code remains unchanged from the previous experiment, pay atttention to preprocessing and label masking part.

## Things to Inspect

Printing one actual training example's input_ids, labels, decoded text, and the exact positions where `labels == -100`. That would let us see our response-only SFT objective numerically from end to end.