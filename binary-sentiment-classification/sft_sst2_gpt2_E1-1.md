# SFT SST2 GPT-2 E1-1

So far we've been doing:

```text
GPT-2
  ↓
LM head
  ↓
next-token prediction
```

and changing how the base model is adapted.

The next experiment could instead be:

```text
GPT-2
  ↓
Transformer representations
  ↓
NEW task-specific head
  ↓
task prediction
```
That teaches us the distinction between generative fine-tuning and discriminative/task-specific fine-tuning.

Towards this direction, we'd use sentiment analysis as the simplest possible task-head experiment, then move to Named Entity Recognition (NER).

Therefore, the next two experiments in the progression look like this:

```text
E1  GPT-2 + classification head
    └── Dataset: SST-2
    └── Task: Binary Sentiment Classification

E2  GPT-2 + token classification head
    └── Dataset: WNUT 2017
    └── Task: named entity recognition
```

A fundamental change from B/C/D experiments: we no longer use response-only SFT here Remember, response-only SFT is specific to our generative CHILDES task.

For E1, the training objective should instead be:

$$ \text{GPT-2 hidden states} \rightarrow \text{classification head} \rightarrow \text{cross-entropy over sentiment labels} $$

And E2:

$$ [B,T,768] \rightarrow \text{classification head} \rightarrow [B,T,K] $$

with token-level cross entropy, where K is the number of NER labels/ids.

That gives us a very clean progression from what we've already learned:

```text
B–D
Causal LM / response generation
        ↓
        ↓ change adaptation strategy
        ↓
E1
Sequence classification / Binary Sentiment
        ↓
        ↓ change prediction granularity
        ↓
E2
Token classification / NER
```

Therefore, in E1-1, GPT-2 stops being a generative model and becomes a classifier instead. This is exactly what SST-2 gives us, a binary sentiment classification, where `label 0 = negative` and `label 1 = positive`.

To make this study more fun, we create three experiments out of it. First of all, we can have one logit or two logits in the final output of the classification head. One logit can be used to obtain the probability of an input being "positive" (or "negative") while two logits can be used for both classes, one for "positive" and the ohter for "nagetive". 

Then, we also study the idea of treating this classification as a regression problem, that is, creating a model that fits an unknown function that maps an input to a real number such that it generats a positive real number that is close to 1.0 for a "positive" input and generats a small real number that is close to 0.0 for a "negative" input. At test time, we can use "0.5" as the class boundary: "negative" < 0.5 or "positive" otherwise.

E1-1 is therefore the right first implementation in this direction because it isolates the new concept: take a pretrained Transformer backbone, extract a sequence representation, and attach a task-specific head. The three experiments can be depicted as follows:

```text
                     SST-2 label
                         │
                         │
                    scalar 0/1
                         │
             ┌───────────┴───────────┐
             │                       │
       Two-logit model         One-logit model
             │                       │
         [z₀, z₁]                    z
             │                       │
          softmax                  sigmoid
             │                       │
      [P(0), P(1)]                 P(1)
             │                       │
       CrossEntropy              BCEWithLogits
             │                       │
          argmax                  threshold
             │                       │
           0 / 1                   0 / 1

and
                     scalar 0/1 target
                           │
                       regression
                           │
                    Linear(H → 1)
                           │
                          MSE
                           │
                    continuous value
```

## Learning Objectives

- Binary sentiment classification modeling
- Classification vs regression
- Add task-specific head
- 1 logit vs 2 logits in output layer
- BCELoss vs BCEWithLogitsLoss
- Prediction with raw logits vs probability
  
## Training Data

### Dataset

stanfordnlp/sst2

### Input and Label

```text
Input: a sentence in English
Label: 0="negative", 1="positive"
```

Example:

```text
"contains no wit , only labored gags", 0
"the greatest musicians", 1
```


## Architecture

Instead of:

```text
tokens
   ↓
GPT-2
   ↓
hidden states [B, T, 768]
   ↓
LM head
   ↓
logits [B, T, 50257]
```

For classification with 1 logit output, we will build:

```text
tokens
   ↓
GPT-2
   ↓
hidden states [B, T, 768]
   ↓
take last real token
   ↓
[ B, 768 ]
   ↓
Linear(768 → 1)
   ↓
[ B, 1 ]
scalar z
  ↓
sigmoid(z)
  ↓
probability p ∈ [0,1]
   ↓
BCELoss
```
Then

```text
p >= 0.5 → positive
p <  0.5 → negative
```

Or we calculate BCEWithLogitsLoss between labels and the raw logit without applying a sigmoid, and classify an input as positive if the logit is greather than 0, or negative otherwise.

We name this experiment **sft_sst2_gpt2_E1-1_1logits**.

For classficiation with 2 logits, we will build:

```text
tokens
   ↓
GPT-2
   ↓
hidden states [B, T, 768]
   ↓
take last real token
   ↓
[ B, 768 ]
   ↓
Linear(768 → 2)
   ↓
[ B, 2 ]
[negative_logit, positive_logit]
   ↓
CrossEntropyLoss
```

We name this experiment **sft_sst2_gpt2_E1-1_2logits**.


For regression, we use the same architecture as 1 logit classification but with a different loss function `MSELoss`:

```text
tokens
   ↓
GPT-2
   ↓
hidden states [B, T, 768]
   ↓
take last real token
   ↓
[ B, 768 ]
   ↓
Linear(768 → 1)
   ↓
[ B, 1 ]
scalar z ∈ R
   ↓
MSELoss
```

We name this experiment **sft_sst2_gpt2_E1-1_regression**.

As we can see from above, the only difference in these three experiments lies in the classification head and the loss function.

Lastly, in all three experiments, we **_freeze GPT-2 and train only the new classification head_**.

This gives us very clean experiments:

```text
GPT-2 parameters:          frozen
Classification head:       trainable

Trainable parameters = 768 × 1 + 1 = 769 for 1 logit
Trainable parameters = 768 × 2 + 2 = 1,538 for 2 logits
Trainable parameters = 768 × 1 + 1 = 769 for regression
```

## 1 logit or 2 logits?

In `1 logit` experiment, we calculate a single probability:

```text
p >= 0.5 → positive
p <  0.5 → negative
```

This is very reasonable.

In fact, this is the standard formulation of binary classification with one output neuron.

Binary classification gives us a better probabilistic interpretation With a single output, we'd normally do:

```python
z = Linear(768 → 1)

p = sigmoid(z)
```

where:

$$ p = \sigma(z)=\frac{1}{1+e^{-z}} $$

Now `p` naturally represents: estimated probability that the example belongs to the positive class.

For example:

```text
z = -2.2
    ↓ sigmoid
p = 0.10

→ 10% positive
→ predict negative
```

and:

```text
z = 2.2
    ↓ sigmoid
p = 0.90

→ 90% positive
→ predict positive
```

The corresponding loss is binary cross entropy:

```python
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(logits, labels.float())
```

Notice the particularly nice PyTorch detail:

Don't explicitly apply sigmoid before `BCEWithLogitsLoss`.

Instead:

```python
logits = classifier(hidden)
loss = BCEWithLogitsLoss()(logits, labels)
```

`BCEWithLogitsLoss` internally combines sigmoid + binary cross entropy in a numerically stable way.

In short, `1 logit` is

```text
Linear(768 → 1)
       ↓
     sigmoid
       ↓
 probability of positive
```

with `BCEWithLogitsLoss()`

Now, compare it to `2 logits` experiment:

```text
Linear(768 → 2)

        ┌→ negative
[768] ──┤
        └→ positive
```

with `CrossEntropyLoss()`

In fact, if we take a closer look at them, it turns out these two are mathematically very closely related.

For binary classification:

Two-logit formulation

```text
z_negative
z_positive
```

and `softmax` gives:

$$ P(positive)= \frac{e^{z_{positive}}} {e^{z_{negative}}+e^{z_{positive}}} $$


One-logit formulation

`z` and `sigmoid` gives:

$$ P(positive)=\frac{1}{1+e^{-z}} $$

If we define:

$$ z=z_{positive}-z_{negative} $$

then the two formulations produce the same positive-class probability.

So conceptually:

```text
2 logits + softmax
        ≈
1 logit + sigmoid
```

for binary classification.

However, `1 logit` version can be cheaper. Here's what I mean.

First, recall that a logit is the raw, unnormalized score output by a model's final linear layer — before any sigmoid or softmax. 

For a binary classifier with a single output unit:

A logit of 0 corresponds to a probability of 0.5 after sigmoid (sigmoid(0) = 0.5).

```text
            sigmoid
logit z ───────────────────→ probability
                             
z < 0       p < 0.5          negative
z = 0       p = 0.5          boundary
z > 0       p > 0.5          positive
```
Thus, we can do prediction as follows:

```text
Positive logits → probability > 0.5 → predict class 1.
Negative logits → probability < 0.5 → predict class 0.
```

So logits > 0 is mathematically equivalent to sigmoid(logits) > 0.5, but the threshold version is cheaper — no need to compute the sigmoid at all when we only want the final decision.

That is exactly what we do in `1 logit` experiment. We use `predictions = (logits > 0).long()` on raw logits instead of applying an activation.

**The logits > 0.5 Trap**

A common mistake is writing `logits > 0.5` when we actually have logits (not probabilities). Since logits live on the whole real line, `0.5` is not a meaningful threshold — it doesn't correspond to a `50% probability`. That would only be correct if the tensor were already passed through a sigmoid:

```python
# Wrong: 0.5 is not the decision boundary for logits
prediction = (logits > 0.5).long()

# Correct: threshold logits at 0
prediction = (logits > 0).long()

# Or: apply sigmoid first, then threshold at 0.5
probs = torch.sigmoid(logits)
prediction = (probs > 0.5).long()

Common Use Cases
Binary classification evaluation

logits = model(input_ids)              # shape [batch, 1] or [batch]
prediction = (logits > 0).long()       # shape [batch], values in {0, 1}
accuracy = (prediction == labels).float().mean()
```

This pattern only applies to binary classification. For multi-class, we'd instead take the argmax over the class dimension:

```python
# logits shape: [batch, num_classes]
prediction = torch.argmax(logits, dim=-1)   # shape [batch]
```

However, for general K-class classification problems, where K >= 2, it is consistent that we always generate K logits. For this reason, `2 logits` version is often recommended for its pedagogical consistency even though one-logit w/ or w/o sigmoid is arguably the more elegant formulation for a binary task. And importantly, neither should be thought of as fundamentally more capable for SST-2. They're two closely related parameterizations of binary classification.

That's why we do both in our progression:

```text
|                  | 2-logit classification | 1-logit binary classification |
| ---------------- | ---------------------- | ----------------------------- |
| Head             | `Linear(H/768, 2)`     | `Linear(H/768, 1)`            |
| Output           | 2 logits               | 1 logit                       |
| Probability      | softmax                | sigmoid                       |
| Loss             | CrossEntropyLoss       | BCEWithLogitsLoss             |
| Prediction       | `argmax`               | `logit > 0`                   |
| Target           | integer 0/1            | float 0/1                     |
| Trainable params | 2H+2/1,538             | H+1/769                       |
| Concept          | K-class classification | Binary classification         |

```
So the one-logit version even has half the classification-head parameters, although 769 vs. 1,538 is completely negligible relative to GPT-2's ~124M parameters.

Then, our next experiment `E2 (NER)` will make the distinction even clearer because we'll move from `[B, 768] → [B, K]` to `[B, T, 768] → [B, T, K]`, where every token gets its own classification decision.

**A Note on sigmoid + (BCELoss vs BCEWithLogitsLoss)**

If we're training with nn.BCELoss, the model output is already sigmoid-activated (probabilities), so we'd use probs > 0.5. But if we're using nn.BCEWithLogitsLoss (the numerically stable version), the output stays as logits, and we should use logits > 0. _Mixing these up is a frequent source of subtle bugs_.


## classification or regression?

So, when modelling binary sentiment classification as a classification problem, our two experiments work like this:

   1 logit output: negative < 0, positive otherwise

   2 logits output: index 0: negative probability, index 1: positive probability

But there's an issue with modelling it as a "regression" problem:

   a singular scalar output: closer to 1 for "positive" and closer to 0 for "negative"

Suppose our labels are:

```text
negative = 0
positive = 1
```

and the model produces: `ŷ = 0.73`

We could interpret `0.73` as:

"The model predicts something closer to positive."

But if we train it with ordinary regression: train against `positive=1, negative=0` with `loss = nn.MSELoss()(prediction, labels.float())`, we're asking the model to minimize: 

$$ (\hat y-y)^2 $$.

So for a positive example (y=1), predictions of:

```text
0.9 → small loss
0.7 → larger loss
0.2 → very large loss
```

This may seem to work on the surface. But how would we decide, given an output value, if it should map to a "positive" or "negative"? How about 0.5 as the threshold? The 0.5 threshold (or any numerical value) is fragile. Because our output is a regression prediction, essentially an arbitrary real number, without an inherent probability interpretation. The 0.5 cut is arbitrary in that it does not come from the training objective. With few examples, the model's output distribution might be centered at, say, 0.3 with a narrow spread—so every test point falls below 0.5. Once the model starts fitting real structure, the output distribution spreads out, and some points cross 0.5 in a meaningful way. With enough data, the spread stabilizes into two clusters, hopefully.

So, by modelling SST-2 as regression, our training objective only says:

   "Get the prediction numerically close to 0 for negative examples and close to 1 for positive examples."

It does not say:

"Put the decision boundary at exactly 0.5."

We've imposed that afterward.

Fundamental problem we see in this is that we train the model to minimize MSE instead of predicting the probability of "positive" or "negative" and minizing that prediction error, which has to be measured as a BCELoss or BCEWithLogitsLoss, forcing us back to a binary classification problem.

### There's another subtle problem with our regression experiment

Imagine two models:

```text
Model A:
negative example → 0.10
positive example → 0.90

Model B:
negative example → 0.40
positive example → 0.60
```

Both classify perfectly with our `0.5` threshold.

But their MSE is very different:

```text
Model A: very confident / far from target
Model B: barely separated
```

MSE encourages the former.

Now imagine:

```text
negative → -0.1
positive → 0.6
```

The classifier using our threshold still works:

```text
negative → negative
positive → positive
```

but regression says the negative prediction `-0.1` is actually better than `0.4`, because it is closer to `target 0`.

That illustrates the mismatch:

   MSE cares about numerical distance from arbitrary target values; 
   
   Classification cares about separating classes.

### There's actually a beautiful connection to logistic regression

This experiment is a good stepping stone to understanding why logistic regression exists.

Ordinary regression:

`x → Linear → arbitrary real number`

Binary classification:

`x → Linear → logit → sigmoid → probability`

The sigmoid constrains the output to:

$$ 0 < p < 1 $$

and, more importantly, binary cross-entropy directly optimizes the probability of the correct class.

So instead of saying:

   "Make positive examples numerically close to 1 and negative examples numerically close to 0."

we say:

   "Assign high probability to the correct class."

That's a much more natural objective for SST-2.

### There is an even more interesting point: regression is actually more natural for the original SST dataset

This is where SST becomes particularly educational.

The original SST dataset contains a continuous sentiment score from 0.0 to 1.0. The Hugging Face sst dataset describes the label as the degree of positive sentiment. SST-2 is created by turning that into a binary classification problem.

Conceptually, the original SST:

```text
"movie was okay"
        ↓
sentiment score = 0.62
```

versus SST-2:

```text
0.62
 ↓
binary class
 ↓
1 = positive
```

So there really are two related tasks:

```text
SST sentiment scoring
        ↓
regression
        ↓
continuous sentiment ∈ [0,1]
```

and:

```text
SST-2 sentiment classification
        ↓
classification
        ↓
negative / positive
```

That's actually a great example of why the task definition determines the appropriate objective.

Here's the link to Stanford NLP SST dataset on Hugging Face:

https://huggingface.co/datasets/stanfordnlp/sst


## Loss Function

```python
torch.nn.BCELoss # w/ 1-logit classifier probabilities
torch.nn.BCEWithLogitsLoss # w/ 1-logit classifier raw logits, preferred
torch.nn.CrossEntropyLoss # for 2-logit classifier
torch.nn.MSELoss # for regression
```

Refer to discussion from above.

## Evaluation Metrics

Accuracy

## Understand the Code

In the following, our discussion is based on `2 logits` experiment.

**1. We are using GPT2Model, not GPT2LMHeadModel**

Previously:

`GPT2LMHeadModel` gave us:

```text
GPT-2
  ↓
LM head
  ↓
[B, T, 50257]
```

Now:

GPT2Model stops at:

```text
GPT-2
  ↓
[B, T, 768]
```

We then add our own:

`nn.Linear(768, 2)`

So we're explicitly seeing the distinction between the Transformer backbone and the task head.

**2. `[B,T,768] → [B,768]` is the key operation**

Suppose:

```text
B = 4
T = 20
H = 768
```

GPT-2 produces:

```text
hidden_states
    [4, 20, 768]
```

But sentiment classification needs one representation per sentence.

We therefore select the final real token:

```text
[4, 20, 768]
       ↓
[4, 768]
```

Then:

```text
[4, 768]
    ↓ Linear
[4, 2]
```

The two logits correspond to:

```text
logit[:, 0] → negative
logit[:, 1] → positive
```

This last-token strategy is particularly natural for GPT-2's causal architecture, and is also the approach used by the Hugging Face GPT-2 sequence-classification implementation.

**3. The loss is completely different from our SFT loss**

Our previous response-only SFT was roughly:

```text
GPT-2
 ↓
[B,T,V]
 ↓
CrossEntropyLoss over target tokens
```

Now:

```text
GPT-2
 ↓
[B,T,768]
 ↓
last token
 ↓
[B,768]
 ↓
Linear
 ↓
[B,2]
 ↓
CrossEntropyLoss
```

For one example, if:

```text
logits = tensor([1.2, 2.8])
label = 1
```

then CrossEntropyLoss simply asks:

   How much probability did the model assign to class 1?

There is no next-token prediction whatsoever.

**4. Why freeze GPT-2 first?**

This is a particularly useful experiment for understanding transfer learning.

Initially, GPT-2 has already learned general language representations:

```text
sentence
   ↓
GPT-2
   ↓
768-dimensional representation
```

We're asking:

   Can a tiny randomly initialized linear layer learn to map those existing representations to sentiment?

The only trainable parameters are:

```python
self.classifier = nn.Linear(768, 2)
```

which gives:

```text
768 × 2 + 2
= 1,538 parameters
```

compared with roughly 124M GPT-2 parameters.

So this is an extremely clean feature extraction experiment.

Then E1 can naturally evolve into:

```text
E1.1  Frozen GPT-2 + classification head
       ↓
E1.2  Full GPT-2 fine-tuning + classification head (not included in this repo)
       ↓
E1.3  LoRA GPT-2 + classification head (not included in this repo)
       ↓
E1.4  QLoRA GPT-2 + classification head (not included in this repo)
```

That would mirror our C/D progression nicely.


## Things to Inspect

In _regression_ experiment, use the following training sizes: 10, 1k, 5k, 8k, 9k, 10k (and maybe use even more examples incrementally for training if we have access to enough compute power and are interested), then:

1) printout the average of the actual labels to know the factual distribution; 
   
2) print out the predicted labels to understand the prediction distribution of the trained model;

3) print out the raw regression outputs for the four test sentences used in all three experiments;

to understand why the trained model got them right or wrong, by using all the three numbers from above. Look for patterns or no patterns how these three numbers change with the size of training examples (Remember, we shuffle our examples for training in our code. So, we can craft a specific training set without shuffling to study specific point of interest).