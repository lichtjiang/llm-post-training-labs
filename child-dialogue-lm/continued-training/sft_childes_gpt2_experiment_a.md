# SFT CHILDES GPT-2 Experiment A

This is continued language-model training. That is, neither do we change the base model by adding new layer(s) nor do we change the loss function. We just continue to train the base model against a specific domain dataset, which is CHILDES in our case.

Conceptually:

```text
Pretrained GPT-2
       │
       │ continue training
       │
       ▼
CHILDES conversational distribution
       │
       ▼
child-response generation
```
For Experiment A, we would intentionally keep the implementation fairly "low level":

- datasets → download/load CHILDES
- transformers → load GPT-2 tokenizer/model
- PyTorch DataLoader
- PyTorch training loop
- PyTorch AdamW
- GPT-2's built-in causal-LM cross-entropy
- no Trainer
- no TRL
- no PEFT/LoRA yet

That gives us a very clean view of what SFT is doing.

## Learning Objectives

- Preprocessing and preparing conversational datasets
- Applying causal language modeling loss
- Construct Pytorch DataLoader with custom collate function
- Load and inspect the base model
- Understand and Build training loop
  
## Training Data

### Dataset

CHILDES Eng-UK Conversational Pairs Dataset on Hugging Face

### Input and Label

For Experiment A, we'll deliberately treat every pair as a plain causal-language-modeling sequence, regardless of who is speaking. Thus this first experiment is really continued causal-LM training on CHILDES-style dialogue, rather than response-only SFT.

So, in other words, each adjacent utterance pair (a mixture of an adult's utterance and a child's utterance, in any order) is both the input and the label. Exactly like how we train a base GPT model, we predict the next token based on the tokens we have seen in the current input sequence.


## Architecture

For this particular task, we can fine-tune the existing GPT-2 as-is. No new neural-network layer.

The Hugging Face openai-community/gpt2 model is already a causal language model with a language-modeling head. It has 124M parameters and was pretrained specifically to predict the next token.

Conceptually:

```text

                 GPT-2 124M
              ┌─────────────┐
context ────► │ Transformer │
              │   blocks    │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ LM head     │
              │ Linear(d,V) │
              └──────┬──────┘
                     │
                     ▼
                 logits
               [T × 50257]
```

The LM head is already there.

We don't need to add something like:

`GPT-2 → new Linear layer → child classifier`

because out task isn't classification.

Out task is still:

    Given the tokens so far, predict the next token.

The only thing we're changing is the distribution of training examples. This is an important conceptual distinction.

Suppose we give GPT-2:

```text
Adult: What are you doing?
Child:
```

The model produces:
```text
P("I'm")
P("I")
P("playing")
P("the")
...
```

Then we fine-tune its parameters using CHILDES.

After SFT, ideally:

```text
P("I'm")       ↑
P("playing")   ↑
P("dinosaurs") ↑
P("approximately") ↓
P("therefore") ↓
...
```

The architecture hasn't fundamentally changed. We're adapting the parameters of the existing language model to a particular data distribution. That's one of the most important ideas we want to take away from this exercise.

Our very first SFT pipeline looks very simple:


```text
Hugging Face CHILDES
        │
select conversational examples
        │
        ▼
construct prompt/response pairs
        │
        ▼
speaker_a + utterance_a
speaker_b + utterance_b
        │
        ▼
"adult: what's that?\nchild: i don't know."
        │
        ▼
GPT-2 tokenizer
        │
        ▼
input_ids (labels)
        │
        ▼
GPT-2 124M
        │
        ▼
logits [batch, sequence, vocab]
        │
        ▼
CrossEntropyLoss
        │
        ▼
loss.backward()
        │
        ▼
      AdamW
        │
        ▼
updated GPT-2 parameters
        │
        ▼
generate()
```

The GPT-2 model already contains its language-modeling head; we aren't adding anything. The new model is thus an 124M-parameter smallest GPT-2 variant.

### Output Shape

Our GPT-2 output is [B, T, V]

```text
input_ids:  [B, T]
                │
                ▼
          GPT-2 Transformer
                │
                ▼
logits:     [B, T, V]
```

where:

- B = batch size
- T = sequence length of this particular batch
- V = vocabulary size (GPT-2: 50,257)

So if a batch happens to be:

```text
B = 4
longest sequence = 73 tokens
V = 50257
```

then:

```python
input_ids.shape   # [4, 73]
logits.shape      # [4, 73, 50257]
```

There is a logit vector at every position.

## Loss Function

Again, the nice answer is:

    We don't need to invent a new loss function. Use causal language-modeling cross-entropy.

In fact, this is exactly the same basic objective GPT-2 was originally trained with. The GPT-2 model card explicitly describes the objective as predicting the next token from preceding tokens.

Suppose our training example is:

Adult: What are you doing?
Child: I'm playing.

After tokenization, imagine it becomes:

```text
[Adult, :, What, are, you, doing, ?, Child, :, I'm, playing, .]
```

The model predicts:
```text

        input                  target
        ─────────────────────────────────
        Adult                  :
        Adult :                What
        Adult : What           are
        Adult : What are       you
        ...
        Child :               I'm
        Child : I'm            playing
        Child : I'm playing    .
```

So the fundamental loss is simply:

```math
L = -\sum_t \log P(x_t\mid x_{<t})
```

which is standard cross-entropy loss.

In PyTorch terms, conceptually:

```python
outputs = model(
    input_ids=input_ids,
    labels=labels
)

loss = outputs.loss
```

And Hugging Face's GPT-2 implementation can calculate this loss for us.

This illustrates what we are calculating a cross entropy between.

```text
             openai-community/gpt2
                     │
                     │ 124M pretrained parameters
                     ▼
             ┌───────────────┐
             │    GPT-2      │
             │ Transformer   │
             │   + LM head   │
             └───────┬───────┘
                     │
             logits [B,T,V]
                     │
                     ▼
             Cross Entropy
                     ▲
                     │
             CHILDES utterance pairs
```

Here's an example to help understand what loss compares.

Suppose the sequence has four tokens:

`x = [I, love, cats, EOS]`

GPT-2 produces: 

`logits.shape = [B, 4, V]`

but the loss compares:

```python
logits[:, :-1, :]   # positions 0,1,2
        vs
labels[:, 1:]       # positions 1,2,3
```

Hence:

```text
logits[0] → love
logits[1] → cats
logits[2] → EOS
```

The final EOS is important because it teaches the model when the response/sequence should terminate.

## Evaluation Metrics

We're calculating both:

```text
loss
perplexity
```

Perplexity is:

$$ PPL = e^{loss} $$

For example:

loss = 4.0

gives:

$$ PPL=e^4\approx54.6 $$

The interpretation is roughly:

    How uncertain is the model about the next token, expressed as an equivalent number of choices?

It's not a perfect intuitive measure, but it's very useful for comparing:

```text
pretrained GPT-2
        ↓
fine-tuned GPT-2
```

## Understand the Code

**1. Why AutoModelForCausalLM?**

We load:

```python
model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2"
)
```

The GPT-2 model is already a causal language model. Its output is approximately:

```text
input_ids
    │
    ▼
GPT-2 Transformer
    │
    ▼
hidden states
    │
    ▼
LM head
    │
    ▼
logits
```

If:

```text
batch_size = 4
sequence_length = 50
vocab_size = 50257
```

then:

`outputs.logits.shape`

will be approximately:

`[4, 50, 50257]`

Every position gets a probability distribution over the vocabulary.

**2. Generation/Prediction**

This line of code:

`outputs = model(**batch)`

is doing considerably more than it might initially appear.

Our batch contains:

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...
}
```

Because we supplied labels, GPT-2 doesn't just return logits. It also calculates the causal language-modeling loss.

Conceptually, suppose:

```text
input_ids:

Adult : what are you doing ? Child : i'm playing .
```

The model internally creates the shifted relationship:

```text
input:    Adult  :  what  are  you  doing  ?  Child  :
target:     :   what  are   you  doing  ?  Child  :  i'm
```

and so on. This is exactly the next-token objective described in the GPT-2 model documentation.

So:

`loss = outputs.loss`

is essentially the average cross-entropy over the valid target tokens.

**3. Why are labels = input_ids?**

This is a subtle but important concept. We might initially think:

    "Shouldn't labels be different from inputs?"

For ordinary supervised classification, yes:

```text
input → "This movie is great"
label → positive
```

But causal language modeling is different. The target is simply the input shifted by one position:

```text
input:
I   love   cats

target:
love   cats   <EOS>
```

Therefore Hugging Face's causal LM interface expects:

`labels = input_ids`

and the GPT-2 implementation performs the shift internally.

That's why this:

`labels = input_ids.clone()`

is correct.

**4. Why -100?**

This line is particularly important:

`labels[attention_masks == 0] = -100`

Suppose we dynamically padded a batch:

```text
example 1:  Adult : hi      Child : hello
example 2:  Adult : hello   Child : yes
                         <PAD> <PAD> <PAD>
```

We don't want GPT-2 to learn `<PAD>` as meaningful language.

PyTorch's cross-entropy implementation uses `ignore_index=-100` by convention. So:

```text
real token → contributes to loss
padding     → ignored
```

This is another reason we prefer implementing our own collator for this first experiment rather than hiding it inside Trainer.

**5. What exactly are we training?**

This is worth being very precise about. Our first example might become:

```text
adult: what are you ?
child: a big boy .
```

The model is trained on the entire sequence:

```text
adult
:
what
are
you
?
child
:
a
big
boy
.
```

So it learns all of these:

```text
adult → :
: → what
what → are
are → you
you → ?
? → child
child → :
: → a
a → big
big → boy
boy → .
```

That's why I'm calling this Experiment A: continued causal-LM training, rather than response-only SFT.

Later, Experiment B will mask the first part:

```text
adult: what are you ?
child:
```

so that only:

`a big boy .`

contributes to the loss.

That will give us a very nice apples-to-apples comparison.

**6. One subtle issue with this particular dataset**

There's something we want to preserve rather than silently hide.

The dataset isn't exclusively:

`adult → child`

The dataset documentation says each record is an adjacent adult-child or child-adult utterance pair.

For example:

```text
child: i can do it .
adult: you can , can't you ?
```

and:

```text
adult: what are you ?
child: a big boy .
```

Therefore, if we train on all examples, the model learns both directions.

That's perfectly appropriate for Experiment A because we're intentionally asking:

    "What happens if I continue training GPT-2's causal LM objective on this entire CHILDES conversational distribution?"

But it means we shouldn't yet claim:

    "We have trained GPT-2 specifically to answer like a child."

We'll get closer to that with Experiment B.

**7. Why start with only 10,000 examples?**

The dataset actually contains `250,757` training pairs, so we're intentionally using only:

`MAX_TRAIN_EXAMPLES = 10_000`

The full dataset is about `37.8 MB` and has `278,458` examples overall. 

This is not because `10K` is sufficient for a good model. It's because our first objective is:

    Understand SFT.

We want a training run that completes reasonably quickly and lets us inspect:

```text
initial loss
      ↓
training loss
      ↓
validation loss
      ↓
test loss
      ↓
generated text
```

Once this works, change:

`MAX_TRAIN_EXAMPLES = 10_000`

to:

`MAX_TRAIN_EXAMPLES = 50_000`

and eventually:

`MAX_TRAIN_EXAMPLES = None`

with appropriate handling in the `.select()` code.

Note: if we decide to train our new model hard, be sure to train it with a GPU or learn to be good at being patient :-)

## Q & A

**1. Isn't training implemented as an autoregressive loop?**

That's the biggest misconception. It conflats what conceptually the objective means and how the Transformer is actually implemented to compute the training forward pass.

Suppose one training example is:

```text
Adult: What are you doing?
Child: I'm playing.
```

Imagine it tokenizes to only 10 tokens:

`x0 x1 x2 x3 x4 x5 x6 x7 x8 x9`

We might imagine GPT-2 doing this during training:

```text
x0 → predict x1
      ↓
x1 → predict x2
      ↓
x2 → predict x3
      ↓
...
x8 → predict x9
```

Although that's conceptually what the objective means, but it is NOT how the Transformer computes the training forward pass.

Instead, GPT-2 processes the whole sequence simultaneously:

```text
x0 x1 x2 x3 x4 x5 x6 x7 x8 x9
│  │  │  │  │  │  │  │  │
▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼
Transformer
│  │  │  │  │  │  │  │  │
▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼
l0 l1 l2 l3 l4 l5 l6 l7 l8 l9
```

where each `li` is a vector of logits over the entire vocabulary.

So if GPT-2's vocabulary is `50,257` tokens:

```text
input:   [10]
logits:  [10, 50257]
```

**2. How can it predict the future without seeing the future?**

This is where the causal attention mask becomes important.

Suppose:

```text
x0   x1   x2   x3
I    love cats .
```

GPT-2 processes all four positions simultaneously, but its causal attention mask prevents each position from looking into the future:

```text
             Can attend to
             ↓

x0           x0
x1           x0 x1
x2           x0 x1 x2
x3           x0 x1 x2 x3
```

Therefore:

```text
position 0 → predicts x1
position 1 → predicts x2
position 2 → predicts x3
position 3 → predicts x4
```

The final position would predict the token after the sequence.

So one forward pass gives us all these predictions simultaneously.

This is one of the beautiful tricks of Transformer training:

**Teacher forcing + causal masking allows all next-token predictions to be computed in parallel.**

There is one logit vector for every input position.

**3. Isn't there normally a softmax during training?**

This is another important correction.

The model produces:

`logits`

not necessarily:

`softmax → predicted token`

during the training forward pass.

Suppose at position `i` we have:

```python
logits[i] =
[
    1.2,
    0.3,
    5.7,
    -1.1,
    ...
]
```

We could calculate:

```python
probabilities = softmax(logits[i])
```

and then select:

```python
predicted_token = argmax(probabilities)
```

But we don't need to do that to calculate the loss.

Instead, cross-entropy works directly from the logits:

```text
logits + target token
       ↓
cross entropy
       ↓
loss
```

PyTorch's `CrossEntropyLoss` effectively combines the relevant log-softmax operation and negative log likelihood in a numerically stable way.

So the training flow is:

```text
input tokens
     │
     ▼
GPT-2 Transformer
     │
     ▼
logits [B, T, V]
     │
     ├──────────────┐
     │              │
     ▼              ▼
 target tokens    logits
                    │
                    ▼
             CrossEntropyLoss
                    │
                    ▼
                   loss
```

No token sampling is necessary. Sampling is something we do during generation/inference.

**4. Since we train in batch, does it mean that we always shift 1024 (MAX_LENGTH) steps regardless of varied lengths of examples in a batch?**

Short answer is "No".

In our code, we specified `MAX_LENGTH = 1024`. But that does not mean:

    Every training example is converted into 1024 tokens.

It means:

    An example is allowed to contain at most 1024 tokens.

Suppose one batch contains four examples:

```text
Example 1: 35 tokens
Example 2: 52 tokens
Example 3: 41 tokens
Example 4: 28 tokens
```

Our collator dynamically pads them to the longest example `52 tokens`. So the actual batch is: `input_ids.shape = [4, 52]` instead of `[4, 1024]`. This is because we're using `pad_sequence(...)` rather than padding every example to `MAX_LENGTH `.

But there is an even subtler point: there actually isn't a literal "shift 1024 steps" operation in the Transformer.

Instead, think of the sequence as:

`x0 x1 x2 x3 x4 ... xT-1`

The model produces:

`logits0 logits1 logits2 logits3 ... logitsT-1`

Then the causal-LM loss aligns them like:

```text
logits0 → target x1
logits1 → target x2
logits2 → target x3
...
logitsT-2 → target xT-1
logitsT-1 → target xT
```

So the "shift" is essentially an alignment operation for the loss, not a sequential operation inside the Transformer. That's a very important distinction.


**5. What happens to the padding?**

Conceptually:

```text
Example 1:
[t0 t1 t2 ... t34 PAD PAD PAD ... PAD]

Example 2:
[t0 t1 t2 ... t51]

Example 3:
[t0 t1 t2 ... t40 PAD ...]

Example 4:
[t0 t1 ... t27 PAD ...]
```

The attention mask is:

```text
Example 1:
[1 1 1 ... 1  0 0 0 ... 0]

Example 2:
[1 1 1 ... 1]

Example 3:
[1 1 1 ... 1  0 0 ... 0]

Example 4:
[1 1 1 ... 1  0 0 ... 0]
```

And we created:

```python
labels = input_ids.clone()

labels[attention_masks == 0] = -100
```

So the loss effectively says:

```text
real token → calculate loss
PAD token  → ignore
```

**6. So what determines the number of Transformer positions?**

For a particular batch:

$$ T_{\text{batch}} = \max(\text{sequence lengths in batch}) $$

subject to:

$$ T_{\text{batch}} \leq 1024 $$

For example:

```text
MAX_LENGTH = 1024

Batch A:
35, 52, 41, 28
       ↓
T = 52

Batch B:
83, 71, 95, 64
       ↓
T = 95

Batch C:
400, 512, 287, 350
       ↓
T = 512
```

Thus the actual tensors might be:

```text
Batch A: [4,  52]
Batch B: [4,  95]
Batch C: [4, 512]
```

This is called **dynamic padding**.

**7. Run me through a complete training example**

Let's use a tiny sequence:

`I love cats .  `

Token IDs:

`[10, 20, 30, 40]`

We feed:


`input_ids = [10, 20, 30, 40]`

GPT-2 produces:

```python
logits =
[
  logits_0,
  logits_1,
  logits_2,
  logits_3
]
```

where each `logits_i` has shape: `[50257]`. Then the causal LM loss effectively aligns:

```text
logits_0 → target 20
logits_1 → target 30
logits_2 → target 40
logits_3 → target next-token
```

The last target might be `EOS`, depending on how we've constructed the sequence.

So the important thing is:

```text
                 ONE forward pass
                       │
                       ▼
        ┌────────────────────────────┐
        │                            │
        ▼                            ▼
     position 0                  position 1
     predicts x1                 predicts x2

        ▼                            ▼
     position 2                  position 3
     predicts x3                 predicts x4
```

**All of these predictions are computed in parallel.**

**8. Where does the autoregressive loop actually happen?**

It happens during **generation**, not ordinary teacher-forced training.

Suppose after fine-tuning we give GPT-2:

```text
Adult: Do you want to play?
Child:
```

At inference time, we don't know the answer.

So we have to do:

```text
context
   ↓
GPT-2
   ↓
logits
   ↓
sample "Yeah"
   ↓
append "Yeah" to context
   ↓
GPT-2
   ↓
logits
   ↓
sample "!"
   ↓
append "!" to context
   ↓
GPT-2
   ↓
...
```

That's the autoregressive loop:
```text
          ┌──────────────┐
          │  GPT-2       │
          └──────┬───────┘
                 │
               logits
                 │
              sample
                 │
                 ▼
             new token
                 │
                 ▼
            append to
             context
                 │
                 └───────────► GPT-2
```

And that's where **softmax + sampling** becomes relevant.

**9. Compare Training to Inference in the encoder-decoder Transformer?**

The same fundamental idea applies here:

**Training**

```text
"I love cats"

        ↓

I     love     cats
│       │        │
▼       ▼        ▼
predict predict predict
 love    cats    <EOS>

          ↑
    teacher forcing
```

All positions are processed **in parallel** using causal masking.

**Inference**

```text
"I"
 ↓
predict "love"
 ↓
"I love"
 ↓
predict "cats"
 ↓
"I love cats"
 ↓
predict <EOS>
```

**Sequential**.

**10. Why is training Transformers computationally efficient?**

Imagine a 100-token sequence.

A naive sequential training approach would require:

`100 forward passes`

But Transformer training can do:

`1 forward pass`

producing:

`100 sets of logits`

simultaneously.

That's a huge part of why GPUs are so effective at Transformer training.

The _causal mask_ preserves the autoregressive constraint:

`position i cannot see positions > i`

while matrix operations let us calculate all positions concurrently.

There are really three different things that are easy to conflate:

**① Transformer forward pass**

```text
[B, T]
   ↓
Transformer
   ↓
[B, T, V]
```

Parallel across `T`.

**② Causal-LM target alignment**

```text
logits[:, :-1, :]
        ↓
targets[:, 1:]
```

Conceptually shifts predictions against the next token. No sequential loop.

**③ Autoregressive generation**

```text
prompt
 ↓
predict token
 ↓
append token
 ↓
predict token
 ↓
append token
 ↓
...
```

Sequential loop.

## Things to Inspect

As a bonus, manually inspect one actual batch numerically—input_ids, labels, logits, the shifted targets, and manually calculate the cross-entropy for one or two tokens. That will connect this SFT exercise directly to the Transformer/softmax/cross-entropy mechanics we may have been studying.