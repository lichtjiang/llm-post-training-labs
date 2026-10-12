# SFT WNUT17 GPT-2 E2

E2 is a very nice next experiment in our progression because it introduces an important issue that E1 didn't have: the dataset labels are at the word/token level, while GPT-2's input is at the BPE subword-token level.

We'll build E2 manually, just as we did E1, rather than using `GPT2ForTokenClassification.`

E2 architecture

```text
GPT-2
   ↓
[B, T, 768]
   ↓
Linear(768 → K)
   ↓
[B, T, K]
```
For WNUT 2017, K = 13 entity labels. The dataset has 3,394 training examples, 1,009 validation examples, and 1,287 test examples, and is licensed CC BY 4.0.

So if:

```text
B = 4
T = 50
```

we get:

```text
hidden states: [4, 50, 768]

              ↓ Linear(768 → 13)

NER logits:    [4, 50, 13]
```

Each token gets its own 13-way classification.

## Learning Objectives

- Milti-way NER label classification at token level
- NER Label alignment
- Add classifier head


## Comparison to the previous experiment


In `E1`, there is one label for the entire sequence:

```text
sentence
   ↓
GPT-2
   ↓
[B,T,768]
   ↓
last token
   ↓
[B,768]
   ↓
Linear(768, 2)
   ↓
[B,2]
```

So, one classification decision per example.

In `E2`, one label for each token:

```text
sentence
   ↓
GPT-2
   ↓
[B,T,768]
   ↓
Linear(768, 13)
   ↓
[B,T,13]
```

One classification decision per token.

So the classification head itself is almost trivial:

self.classifier = nn.Linear(
    768,
    num_labels
)

The really interesting engineering difference is what happens around that head:

```text
E1:
[B,T,768]
     ↓
select ONE representation
     ↓
[B,768]
```

and 
```
E2:
[B,T,768]
     ↓
keep ALL token representations
     ↓
[B,T,768]
```

And then both ultimately use the same fundamental classification machinery:

```text
logits
   ↓
CrossEntropyLoss
   ↓
class index
```

with argmax for prediction.

One other thing to keep in mind: token-level accuracy isn't the standard headline metric for NER. Eventually we should calculate entity-level precision, recall, and F1, because getting `O` correct on lots of ordinary tokens can make raw token accuracy look deceptively good. For this first E2 experiment, however, token-level accuracy is useful because we're primarily learning the mechanics of [B,T,H] → [B,T,K] and label alignment.

## Training Data

### Dataset

flaitenberger/wnut_17 version of WNUT 2017 on Hugging Face

### Input and Label

```text
Input: A sentence as a list of tokens
Label: A list of NER IDs of the input sentence
```

Example:

```text
tokens: [ "MetroPCS", "has", "the", "worst", "customer", "service", "I", "'ve", "experienced", "in", "months", "(", ...]

ner_tags: [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
```


## Architecture

```text
GPT-2
   ↓
[B, T, 768]
   ↓
Linear(768 → K)
   ↓
[B, T, K]
```
For WNUT 2017, K = 13 entity labels.

## Loss Function

$$ [B,T,768] \rightarrow \text{classification head} \rightarrow [B,T,K] $$

with token-level cross entropy (torch.nn.CrossEntropyLoss), where K is the number of NER labels/ids.


## Evaluation Metrics

Accuracy

## Understand the Code

### Label alignment

This is the part we should pay the most attention to.

Suppose WNUT gives us:

```text
words:
["John", "works", "at", "Microsoft"]

NER:
[9, 0, 0, 1]
```

The exact IDs depend on the dataset's label mapping, but conceptually that's:

```text
John        B-person
works       O
at          O
Microsoft   B-corporation
```

But GPT-2 doesn't necessarily see exactly four tokens.

It might tokenize something like:

```python
John       → ["John"]
works      → ["works"]
at         → ["at"]
Microsoft  → ["Microsoft", ...]
```

and a word can potentially become multiple BPE tokens.

Therefore we have:

```text
WNUT annotation level:

word1     word2     word3
  ↓         ↓         ↓
 label1    label2    label3
```

```text
GPT-2 representation:

subtoken1 subtoken2 subtoken3 subtoken4 ...
    ↓         ↓         ↓         ↓
 hidden1   hidden2   hidden3   hidden4
```

We need to establish the mapping:

```text
word
 ↓
BPE subtokens
 ↓
NER labels
```

That's why we use:

`word_ids = tokenized.word_ids()`

and construct:

```python
labels = [
    actual_label,
    -100,
    actual_label,
    actual_label,
    -100,
    ...
]
```

`-100` has a special meaning to PyTorch's CrossEntropyLoss:

```nn.CrossEntropyLoss(ignore_index=-100)```

Those positions simply don't contribute to the loss, which is further explained as follows.

When a word that gets split into multiple GPT-2 BPE tokens, only the first subtoken receives the word's NER label. The remaining subtokens receive `-100` and therefore don't contribute to the loss. This keeps our first E2 experiment mechanically simple. Later we can experiment with label propagation/alignment strategies.
