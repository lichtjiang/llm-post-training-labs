# LLM Post Training Labs

This repo is a progressive lab series for learning how to fine-tune and post-train LLMs. To make these experiments more approachable and able to be trained locally on a computer with a CPU and a 16GB RAM, a 124M-parameter GPT-2 model is used as a base model. In the current version, we will learn the following topics through hands-on: full-sequence supervised fine-tuning (SFT) (continued training), response-only SFT, parameter efficient fine-tuning (PEFT) with LoRA and QLoRA, adding task-specific head for binary sentiment classification and Named Entity Recognition (NER).

## Base Model

To make this series of SFT experiments more approachable to a wider learners, the following base model is picked to post-train/fine-tune for three different tasks, namely, child dialoue, binary sentiment classification, and named entity recognition (see ["Dataset and Tasks"](https://github.com/lichtjiang/llm-post-training-labs#dataset-and-tasks)):

The Hugging Face openai-community/gpt2

https://huggingface.co/openai-community/gpt2


### GPT2 Architecture

The Hugging Face openai-community/gpt2 model is already a causal language model with a language-modeling head. It was pretrained specifically to predict the next token.

Conceptually,
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

Here's a detailed layer by layer topology of the model:

```text
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()
          (c_proj): Conv1D()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
```

### Why 124M GPT2?

This base model is tiny by modern LLM standards: it only has 124M parameters.

Even the raw FP32 parameters are only roughly: 

$$ 124M \times 4 \approx 496MB $$

Of course, training requires additional memory for gradients, optimizer state, activations, etc., so the actual footprint is considerably larger. But it is still a very manageable model for a general use personal computer typically configured with a 16 GB RAM, especially if we keep:

- batch size small
- sequence length modest
- gradient accumulation controlled
- dataset subset initially small

As a heads-up, our ultimate goal is to understand the mechanics of SFT and post-training. So, we could even deliberately start with a small subset, something like 5,000–10,000 examples. That's actually preferable for our experiments. No one wants to spend hours training only to discover that his/her data formatting was wrong.


## Dataset and Tasks

### 1. CHILDES Eng-UK Conversational Pairs Dataset on Hugging Face
   
It is derived from the English-UK portion of CHILDES and contains naturalistic adult–child conversational pairs.

https://huggingface.co/datasets/nshah-fbcs/childes-engUK-conversational-pairs

Some userful characteristics:

| Property          | Dataset                           |
| ----------------- | --------------------------------- |
| Language          | English                           |
| Source            | Natural child/adult conversations |
| Training examples | ~250,757 pairs                    |
| Validation        | ~13,197 pairs                     |
| Test              | ~14,504 pairs                     |
| Total             | ~278K pairs                       |
| Format            | JSON                              |
| Task              | Text generation                   |
| License           | CC BY-NC 4.0                      |
| Size              | Small enough for experimentation  |

#### Why CHILDES Dataset?

The first goal in this repo is to demonstrate different post-training approaches through building a `Child Dialoue LM`. But we focus on learning the mechanics of these methods instead of building a commercial grade chatbot for kids. Therefore, we want to build a simple model that can respond to an adult like a child in one turn.

Each record in this dataset is a single adjacent adult-child or child-adult utterance pair formatted in JSON. Each JSONL record contains:

speaker_a (string): "child" or "adult"

utterance_a (string): the first utterance, lowercased and cleaned

speaker_b (string): "child" or "adult" (the other role)

utterance_b (string): the second utterance, lowercased and cleaned

The examples look like this:

```text
child: "what's that?"
adult: "i don't know."
```

or

```text
adult: "what are you?"
child: "a big boy."
```

The dataset creator extracted adjacent child/adult utterances from CHILDES transcripts and explicitly describes it as intended for training small language models on naturalistic child-directed speech.

So this isn't merely a generic conversation dataset that happens to contain the word "child". It is genuinely child-centered conversational data.

We could construct examples such as:

```text
<|user|> What's that?
<|assistant|> I don't know.

<|user|> Look at this!
<|assistant|> Wow, that's a big truck!

<|user|> Do you wanna play?
<|assistant|> Yeah! Let's play dinosaurs.
```

where the assistant response is specifically the child's utterance.

That gives us a very clean SFT problem:

**_Given the preceding conversational context, generate a child-like response._**

Note though, we aren't trying to build a genuinely useful children's chatbot yet here. We're using the problem as an experimental vehicle for learning:

dataset → preprocessing → tokenization → train/validation split → GPT-2 → causal LM loss → backprop → optimizer → generation → evaluation

That's almost exactly the SFT learning exercise we'd recommend a new comer to generative AI to get hands on. Because at this stage our objective should be formed as: I want to understand how to SFT an existing Transformer model.

Then the kids-dialogue task is unusually nice because we can observe the effect of fine-tuning directly.

Start with pretrained GPT-2:

User: Do you like dinosaurs?

GPT-2:

```text
"Do you like dinosaurs? The company announced..."
```

We should see that vanilla GPT-2 can already generate English, but it doesn't specifically behave like a child.

Then fine-tune it on child-oriented dialogue:

```text
User: Do you like dinosaurs?
```

Fine-tuned GPT-2:

```text
"Yeah! I like the big ones!"
```

The model isn't becoming a fundamentally new model. We're modifying the distribution of continuations it prefers.

That makes this a very nice concrete demonstration of what fine-tuning actually does.

As a result, we desgined our first eight experiments on this dataset, fine-tuning the GPT-2 model to train a `Child Dialogue LM` model that can respond to an adult like a child (see ["Overall Learning Path"](https://github.com/lichtjiang/llm-post-training-labs#building-a-child-dialogue-lm-using-the-childes-eng-uk-dataset) for the complete list of these experiments).

#### Credits

CHILDES is maintained by Brian MacWhinney at Carnegie Mellon University. Original corpora cited individually are listed at https://childes.talkbank.org/.

The curation, preprocessing scripts, and split definitions are released under CC-BY-NC-4.0. The underlying CHILDES content is the property of its respective contributors and is governed by the TalkBank Code of Ethics.

MacWhinney, B. (2000). *The CHILDES project: Tools for analyzing talk* (3rd ed.). Lawrence Erlbaum Associates

### 2. Sentiment Analysis

SST-2 (Stanford Sentiment Treebank) is a binary sentiment dataset created at Stanford NLP.

The dataset is listed on Hugging Face as `stanfordnlp/sst2`:

https://huggingface.co/datasets/stanfordnlp/sst2


- English
- binary sentiment: negative / positive
- 67,349 training examples
- 872 validation examples
- 1,821 test examples
- single-sentence inputs (Source: Rotten Tomatoes)
- human sentiment annotations

Each example is an English sentence with a class label (`0` for negative and `1` for positive):

```text
"contains no wit , only labored gags", 0
"the greatest musicians", 1
```

We would use this dataset for adapting GPT-2's representations to a binary sentiment classification by adding a new task-specific head.

#### Credits

As mentioned above, this dataset is sourced from Hugging Face.

Credit goes to:

Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C. (2013).
    Recursive deep models for semantic compositionality over a sentiment treebank.
    In *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*
    (pp. 1631–1642). Association for Computational Linguistics.
    https://www.aclweb.org/anthology/D13-1170

### 3. Named Entity Recognition (NER)

Sourced from Hugging Face, flaitenberger/wnut_17 version of WNUT 2017 is explicitly CC BY 4.0, English, and contains:

- emerging/rare entities from noisy social-media text
- 3,394 training examples
- 1,009 validation examples
- 1,287 test examples
- 13 token-level NER labels

```
O
B-corporation / I-corporation
B-creative-work / I-creative-work
B-group / I-group
B-location / I-location
B-person / I-person
B-product / I-product
```

Each example in the dataset appears as a pair of two lists, `tokens` list and `ner_tags` list, as illustrated by the following:

```python
tokens: [ "MetroPCS", "has", "the", "worst", "customer", "service", "I", "'ve", "experienced", "in", "months", "(", ...]

ner_tags: [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
```

We are going to use this dataset to fine-tune a base model to recognize named entities at token-level.

#### Credits

This dataset is a Parquet conversion of the original WNT-17 dataset, originally created by Leon Derczynski.


## Overview

### "Build SFT from First Principles"

**Phase 1 — Establish the Pretrained GPT-2 Baseline**

Load openai-community/gpt2 and experiment with generation to understand:

```text
prompt
  ↓
tokenizer
  ↓
input_ids
  ↓
GPT-2
  ↓
logits [batch, sequence, vocab]
  ↓
next-token sampling
```

We should see that vanilla GPT-2 can already generate English, but it doesn't specifically behave like a child.

**Phase 2 — Prepare the CHILDES Data**

In `Experiment A`, we learn both ways: `adult → child and child → adult` without checking the order of utterance of an adult and a child. 

But then in `Experiment B`, we parse the dataset and make sure adult utterance always comes first. So, if a child's utterance is seen before an adult's in a pair, we  transform the pair as follows (This is fine for studying post training machineary. But later, we will correct this transformation in the experiment C3.):

```text
child:  what's that ?
adult:  i don't know .
```

into something like:

```text
Adult: what's that?
Child: i don't know.
```

and potentially:

```text
Adult: what's that?
Child: i don't know.
<|endoftext|>
```

This is where we'll learn an important practical aspect of SFT: _the dataset format is part of the task definition_.

**Phase 3 — SFT GPT-2** 

("Experiement A", see ["Overall learning path"](https://github.com/lichtjiang/llm-post-training-labs#overall-learning-path) below)

For a causal language model, our training example can essentially become:

```text
Adult: What are you doing?
Child: I'm playing.
```

and the training objective remains ordinary next-token prediction:

```math
\mathcal{L} = -\sum_t \log P(x_t\mid x_{<t})
```

We aren't introducing a new loss function.

We're taking the pretrained GPT-2 and continuing its language-model training on a different data distribution.

That's the core idea behind this experiment.

**Phase 4 — Make it Actual Conversational SFT**

 ("Experiement B", , see ["Overall learning path"](https://github.com/lichtjiang/llm-post-training-labs#overall-learning-path) below)

Then introduce a simple format:

```text
User:
Do you want to play?

Assistant:
Yeah! Let's play!
```
and train GPT-2 to generate the assistant portion.

This introduces an important SFT concept that will be useful later when we work with modern instruction-tuned models: **prompt/completion masking.**

We initially train on the entire sequence (`Experiment A`), then implement the more proper approach where the loss is calculated only on the assistant tokens. That gives us `experiment B`:

```text
loss on prompt + response
          vs.
loss only on response
```

*_Important Note_*

We don't immediately jump to Hugging Face `Trainer` and let it hide everything from the very begining. This means that for the first two experiments `Experiment A` and `Experiment B`, we do something like the following in `PyTorch`:

```python
for batch in dataloader:

    outputs = model(
        input_ids=batch["input_ids"],
        labels=batch["labels"],
    )

    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Then inspect

```python
loss
model.parameters()
gradients
learning_rate
```
and periodically:

```python
model.generate(...)
```

Once we've made that work, then we start using `Trainer` from `Experiment C1` (see ["Overall learning path"](https://github.com/lichtjiang/llm-post-training-labs#overall-learning-path) below). (BTW, we don't use `SFTTrainer` throughout the project.)

Thus, initially, our progression would look like the following:

```text

GPT-2 pretrained
       │
       ▼
CHILDES preprocessing
       │
       ▼
basic continued LM training
i.e. fine-tune the existing GPT-2 as-is
 ("Experiment A")
       │
       ▼
prompt → child response
       │
       ▼
assistant-only loss masking
       │
       ▼
conversation/reponse-only SFT 
("Experiment B")
       │
       ▼
evaluation
       │
       ▼
Hugging Face Trainer ("Experiment C1")
```

In summary, we start with a simple project idea as follows.

Base model: GPT-2 124M

Dataset: CHILDES English child/adult dialogue

Goal/task: fine-tune GPT-2 to generate child-like responses.

It's small enough to run locally, the data is readily available, the task is intuitively understandable, and—most importantly—it allows us see the entire SFT pipeline rather than hiding it behind a large framework at the begining.

**Phase 5 - PEFT with LoRA**

("Experiment C1, C2 and C3", see ["Overall learning path"](https://github.com/lichtjiang/llm-post-training-labs#overall-learning-path) below)

With the rapid development in generative AI, more and more organizations have adopted or plan to adopt the technology but with one big concern. That is, they have to send their business data to gen AI providers. When factors, such as requirement of laws and regulations, privacy, and the potential loss of business values by exposing business data to an ousider, are taken into consideration, many organizations tend to refrain from this path. 

As a result, training and running their own models are a valuable alternative. Due to prohibitive cost of training a large model, the solution is to train from a large model, often open source or open weights such as Llama, Gemma, Nemotron, Qwen, Kimi, Deepseek, and MiniMax, among many others, efficiently and economically, to obtain a small enough model that is comparable to a large model in terms of performance. This approach is promising because businesses only need a special model that targets their specific use cases, rather than a general model that can serve a wide range of demands from a remote AI service provider.

Low Rank Adaptation of LLM, or LoRA in short, is such a parameter efficient finetuning (PEFT) technology. We experiment with different LoRA setups in this phase.

**Phase 6 - PEFT with QLoRA**

("Experiment D1 and D2", see ["Overall learning path"](https://github.com/lichtjiang/llm-post-training-labs#overall-learning-path) below)

Quantization is another PEFT technique that helps futher cut down budget. We spend two experiments on this topic.

**Phase 7 - SFT with Classification Head for Binary Sentiment Classification**

 ("Experiment E1")

Up to this phase, we have spent quite a few experiments with generative tasks. Starting from this phase, however, we shift our interests to non-generative tasks that needs extra task-specific head on top of GPT-2.

First, three experiments are designed to study a specific task, that is, a binary sentiment classification ("positive" vs "negative"). Each experiment models the task differently. Although the first two modeling the task as a binary classification problem, the experiment `E1-1_1logit` works slightly differently than `E1-1_2logits` in how many logits are generated in the output: a single logit in the former vs two logits output in the latter. We will show that these two are essentially equivalent but the two logits implementation is a recommended way by some practioners to approach binary classification, as it extends nuturally to multiple classes by keeping the number of logits and the number of the classes the same; while some others are in favor of the single logit implementation because it is considered simple and classical.

The third experiment is to demonstrate a mistake that some beginners may tend to make. That is, modelling the binary classification as a regression problem of predicting a numerical value that represents the probability of being one of the two classes, e.g., "positive" in the case of binary sentiment classification. We will show that a model trained in such a way can be easily biased by the distribution of training data, and that it lacks a vigorous support in math.

**Phase 8 - SFT with Classification Head for Named Entity Recognition (NER)**

("Experiment E2")

This is the final phase in the current version of the repo. It is a 13-way token level classification for NER.


### Overall learning path:

Based on the above design principles, we arrive at a complete list of experiments in this repo.

#### Building a Child-Dialogue LM Using the CHILDES Eng-UK Dataset

see ["Dataset and Tasks"](https://github.com/lichtjiang/llm-post-training-labs#dataset-and-tasks) section above for the dataset info

  1. "Experiment A", full-sequence fine-tuning (continued training), [README](https://github.com/lichtjiang/llm-post-training-labs/blob/main/child-dialogue-lm/continued-training/sft_childes_gpt2_experiment_a.md)
  
  2. "Experiment B", response-only supervised fine-tuning, [README](https://github.com/lichtjiang/llm-post-training-labs/blob/main/child-dialogue-lm/response-only-sft/sft_childes_gpt2_experiment_b.md)
  
  3. "Experiment C1", LoRA on mlp.c_fc + mlp.c_proj in block 11 only, [README](https://github.com/lichtjiang/llm-post-training-labs/blob/main/child-dialogue-lm/lora/sft_childes_gpt2_lora_c1.md)
  
  4. "Experiment C1-inspect", LoRA inspectation, [README](https://github.com/lichtjiang/llm-post-training-labs/blob/main/child-dialogue-lm/lora/sft_childes_gpt2_lora_c1.md#things-to-inspect)
  
  5. "Experiment C2", LoRA on mlp.c_fc + mlp.c_proj in all 12 blocks, [README](https://github.com/lichtjiang/llm-post-training-labs/blob/main/child-dialogue-lm/lora/sft_childes_gpt2_lora_c2.md)
  
  6. "Experiment C3", add GPT-2 attention layers to LoRA targets, [README](https://github.com/lichtjiang/llm-post-training-labs/blob/main/child-dialogue-lm/lora/sft_childes_gpt2_lora_c3.md)
  
  7. "Experiment D1", 4-bit NF4 QLoRA targeting all MLP projections, [README](https://github.com/lichtjiang/llm-post-training-labs/blob/main/child-dialogue-lm/qlora/sft_childes_gpt2_qlora_4bitNF4base_d1.md)
  
  8. "Experiment D2", 4-bit NF4 QloRA targeting all MLP projections and attn layers, [README](https://github.com/lichtjiang/llm-post-training-labs/blob/main/child-dialogue-lm/qlora/sft_childes_gpt2_qlora_4bitNF4base_d2.md)

#### Building a Binary Sentiment Classification Model Using SST-2 Dataset (Stanford Sentiment Treebank

see ["Dataset and Tasks"](https://github.com/lichtjiang/llm-post-training-labs#dataset=and-tasks) section above

  9. "Experiment E1-1", 1-logit output classification, [README](https://github.com/lichtjiang/llm-post-training-labs/blob/main/binary-sentiment-classification/sft_sst2_gpt2_E1-1.md)
   
  10.  "Experiment E1-1", 2-logits output classification, [README](https://github.com/lichtjiang/llm-post-training-labs/blob/main/binary-sentiment-classification/sft_sst2_gpt2_E1-1.md)
  
  11.  "Experiment E1-1", regression, [README](https://github.com/lichtjiang/llm-post-training-labs/blob/main/binary-sentiment-classification/sft_sst2_gpt2_E1-1.md)
    
#### Building a NER Classficiation Model using WNUT 2017 Dataset

See ["Dataset and Tasks"](https://github.com/lichtjiang/llm-post-training-labs#dataset-and-tasks) section above

  12.  "Experiment E2", NER classification, [README](https://github.com/lichtjiang/llm-post-training-labs/blob/main/named-entity-recognition/sft_wnut17_gpt2_E2.md)

## Learning Objectives

  - Preprocessing and preparing datasets
  - Applying causal language modeling loss
  - Masking prompt tokens for response-only SFT
  - Freezing base-model parameters
  - Configuring LoRA adapters
  - Understanding LoRA and model inspectation
  - Understanding QLoRA
  - Adding task-specific heads
  - Fine-tuning GPT-2 for binary classification and token-level multi-NER-label classification

Note: it is worth re-stressing that our major goal is not to train hard for a frontier model; instead, we want to learn and study SFT machinery through these hands-on labs.

## Repository Structure

```text
.
├── README.md
├── child-dialogue-lm
│   ├── continued-training
│   │   ├── sft_childes_gpt2_experiment_a.html
│   │   ├── sft_childes_gpt2_experiment_a.ipynb
│   │   ├── sft_childes_gpt2_experiment_a.md
│   │   └── sft_childes_gpt2_experiment_a.py
│   ├── response-only-sft
│   │   ├── sft_childes_gpt2_experiment_b.html
│   │   ├── sft_childes_gpt2_experiment_b.ipynb
│   │   ├── sft_childes_gpt2_experiment_b.md
│   │   └── sft_childes_gpt2_experiment_b.py
│   ├── lora
│   │   ├── sft_childes_gpt2_lora_c1.html
│   │   ├── sft_childes_gpt2_lora_c1.ipynb
│   │   ├── sft_childes_gpt2_lora_c1.md
│   │   ├── sft_childes_gpt2_lora_c1.py
│   │   ├── sft_childes_gpt2_lora_c1_inspect.ipynb
│   │   ├── sft_childes_gpt2_lora_c1_inspect.py
│   │   ├── sft_childes_gpt2_lora_c2.html
│   │   ├── sft_childes_gpt2_lora_c2.ipynb
│   │   ├── sft_childes_gpt2_lora_c2.md
│   │   ├── sft_childes_gpt2_lora_c2.py
│   │   ├── sft_childes_gpt2_lora_c3.html
│   │   ├── sft_childes_gpt2_lora_c3.ipynb
│   │   ├── sft_childes_gpt2_lora_c3.md
│   │   └── sft_childes_gpt2_lora_c3.py
│   └── qlora
│       ├── sft_childes_gpt2_qlora_4bitNF4base_d1.html
│       ├── sft_childes_gpt2_qlora_4bitNF4base_d1.ipynb
│       ├── sft_childes_gpt2_qlora_4bitNF4base_d1.md
│       ├── sft_childes_gpt2_qlora_4bitNF4base_d1.py
│       ├── sft_childes_gpt2_qlora_4bitNF4base_d2.html
│       ├── sft_childes_gpt2_qlora_4bitNF4base_d2.ipynb
│       ├── sft_childes_gpt2_qlora_4bitNF4base_d2.md
│       └── sft_childes_gpt2_qlora_4bitNF4base_d2.py
├── binary-sentiment-classification
│   ├── sft_sst2_gpt2_E1-1.md
│   ├── sft_sst2_gpt2_E1-1_1logit.html
│   ├── sft_sst2_gpt2_E1-1_1logit.ipynb
│   ├── sft_sst2_gpt2_E1-1_1logit.py
│   ├── sft_sst2_gpt2_E1-1_2logits.html
│   ├── sft_sst2_gpt2_E1-1_2logits.ipynb
│   ├── sft_sst2_gpt2_E1-1_2logits.py
│   ├── sft_sst2_gpt2_E1-1_3in1.py
│   ├── sft_sst2_gpt2_E1-1_regression.html
│   ├── sft_sst2_gpt2_E1-1_regression.ipynb
│   └── sft_sst2_gpt2_E1-1_regression.py
└── named-entity-recognition
    ├── sft_wnut17_gpt2_E2.html
    ├── sft_wnut17_gpt2_E2.ipynb
    ├── sft_wnut17_gpt2_E2.md
    └── sft_wnut17_gpt2_E2.py
```

## Environment

Although all experiments can be conducted on a 16GB computer, if we really want to train hard with more examples longer (meaning more Epochs), we will spend long time painfully waiting for training to complete.

That is why notebook files (.ipynb) are included for all 12 experiments. So, feel free to load it into Google CoLab and run experiments with a T4 GPU.

Most packages, such as modern Python, PyTorch, and HuggingFace's Transformers have already been installed by default in CoLab. So, most time we will be good.

But there are times some of us may encounter missing packages, import errors or other erros like TypeErros. If this is the case, simply go ahead and pip install packages that are reported not found, or upgrade relevant packages if we see their names appear in an error message, problems should resolve instantly. (For who are curious, here is a problem I once ran into in CoLab. `bitsandbytes` was missing and I fixed it with a `!pip install bitsandbytes>=0.46.1` exactly as prompted to me in the error message.)

For these types of reasons, I didn't include a `requires.txt` or a project file like `pyproject.toml`. 

Another fair point behind this is that the exact versions of dependencies depend on where we run these experiments. For example, if we run on an old Mac, we won't be able to install the latest versions of some of the packages needed for this repo. But that doesn't exclude us from doing any of these experiments, except that we have to do our due diligence to find out what versions that work on our Mac. Exactly due to the same reason, I use certain versions of some of the dependencies on one of my computer and some other versions on another one, also different than those installed in CoLab.

So, we are free to use tools of our choice to create an environment to run these experiments, uv, conda, venv, and pip. In most cases, just install the latest version of a package we find in "import" section of a script.

## Installation and How to Run

For most experiments, four files are provided: a standalone Python script (`.py`), a Notebook (`.ipynb`), an HTML file that includes both code and output, exported from Google Colab (`.html`), and a README markdown file (`.md`). 

As mentioned from above, we don't include a dependency list in a `requires.txt` or a `pyproject.toml` file.

We can take a `.ipynb` and upload it to Google Colab to run its corresponding experiment interactively. This is the simplest way to run experiments without a need to install anything beforehand (in most cases all packages are already pre-installed in Colab).

If we want to run an experiment as a standalone program, simply take a look at `import` section in a `.py` file for the experiment and install pacakges when needed. Then, run `.py` file using commands such as `python` or `uv run` commands, depending on our environment.



## Experiment Results

### Child Dialogue LM

We are interested in whether the style of text generated by our Child Dialogue LM experiments resembles that of a child — that is, short, often grammatically incomplete, and no longer than a single dialogue turn. Another thing to keep in mind is that the generated examples from our experiments listed below are the results from models that are trained from 10K samples for 1 epoch. Again, we are learning and studying mechanics of post-training techniques instead of training our models hard to arrive at a frontier chatbot.

First of all, to have something for our fine-tuned model to compare to, here's an example generated by the base model `GPT-2`.

```text
Adult: waht did you do today?
Child: hmm no. I don't know if he got out of the car, the driver was in the car, but I'm sure they'd never seen him.
S: Yeah, but that's just him.
Child: i am sorry
```
As we can see, GPT-2 is ready to generate English pretty well. But the generated text after `Child:` does not seem like a response from a child and it seems that GPT-2 tries to generate multiple turns of a chat.

Next, here's a list of generated child responses by `Experiment A`:

```text
----------------------------------------------------------------------
Prompt:
adult: what's that ?
child:

Generation:
adult: what's that ?
child: that's another one .
child: that's another one .
child: ++ that's another one .
child: that's another one .
child: ++ that's another one .
adult: where's the other one ?
adult

----------------------------------------------------------------------
Prompt:
adult: where are you going ?
child:

Generation:
adult: where are you going ?
child: i going to sit here .
child: i'm getting dressed .
child: i'm getting dressed .
adult: are you going to sit here .
child: i want you to sit here .
adult: is that what you're
```

From these examples, it's clear that the new model has already picked up some pattern from our CHILDES dataset. Each generated sequence of text after `Child:` becomes short. 

But the model doesn't know to just generate a single reponse yet. This is expected, because we didn't actually post-train our first model to generate a single `response`.

But this changes in `Experiment B`, where we do a response-only SFT. Here's the result:

----------------------------------------------------------------------
Adult: what are you doing ?
Child: i'm doing the chair .<|endoftext|>

----------------------------------------------------------------------
Adult: do you want to play ?
Child: i want to play .<|endoftext|>

----------------------------------------------------------------------
Adult: what is that ?
Child: that's a spider .<|endoftext|>

----------------------------------------------------------------------
Adult: where are you going ?
Child: i gotta have that one .<|endoftext|>

----------------------------------------------------------------------
Adult: what did you do today ?
Child: i'm going to .<|endoftext|>

Evidently from these examples, our new model is now trained to form a new behavior, i.e., generate a single short response given a prompt.

Similarly to `B`, series `C` & `D` experiments demonstrate a child-like response style in its generated output as follows:

```text
Experiment C1

Adult: what did you do today ?
Child: like a baby .<|endoftext|>
------------------------------------------------------------
Adult: what are you playing with ?
Child: no it's just going to be a normal two man game .<|endoftext|>
------------------------------------------------------------
Adult: where is mommy ?
Child: where mommy .<|endoftext|>
------------------------------------------------------------
Adult: what did you eat ?
Child: what did you eat ?<|endoftext|>
------------------------------------------------------------
Adult: do you want to go outside ?
Child: you want to go outside ?<|endoftext|>
------------------------------------------------------------

Experiment C2

Adult: what did you do today ?
Child: i did a .<|endoftext|>
------------------------------------------------------------
Adult: what are you playing with ?
Child: i don't know .<|endoftext|>
------------------------------------------------------------
Adult: where is mommy ?
Child: in the kitchen .<|endoftext|>
------------------------------------------------------------
Adult: what did you eat ?
Child: i ate raw meat .<|endoftext|>
------------------------------------------------------------
Adult: do you want to go outside ?
Child: don't want to go outside .<|endoftext|>
------------------------------------------------------------


Experiment C3

"What are you doing?"
i'm doing the dog's tail dance .<|endoftext|>

Experiment D1

"What are you doing?"
i'm getting some .<|endoftext|>

Experiment D2

"What are you doing?"
getting some of the chips .<|endoftext|>
```

### Binary Sentiment Classification

It is interesting to see that `1 logit` model gets one test example wrong while `regression` model successfully gives correct answers to all four test examples. Among all three, `2 logits` model stands out as the strongest one in that it shows confidence in classifying each test example. Note: in `regression` printout, "Probabilities" rows actually show the predicted scalar values in the final output.

```text
Experiment E1-1_1logit

Text: This movie was absolutely fantastic.
Prediction: positive
Probabilities: 0.9023447036743164

Text: The movie was boring and disappointing.
Prediction: negative
Probabilities: 0.4444046914577484

Text: I really enjoyed this film.
Prediction: positive
Probabilities: 0.8982937335968018

Text: I wouldn't recommend this movie.
Prediction: positive
Probabilities: 0.5675510168075562


Experiment E1-1_2logits

Text: This movie was absolutely fantastic.
Prediction: positive
Probabilities: [0.12439426779747009, 0.8756057620048523]

Text: The movie was boring and disappointing.
Prediction: negative
Probabilities: [0.8331819176673889, 0.16681812703609467]

Text: I really enjoyed this film.
Prediction: positive
Probabilities: [0.1540893316268921, 0.8459107279777527]

Text: I wouldn't recommend this movie.
Prediction: negative
Probabilities: [0.7852526903152466, 0.21474727988243103]


Experiment E1-1_regression

Text: This movie was absolutely fantastic.
Prediction: positive
Probabilities: 1.0128304958343506

Text: The movie was boring and disappointing.
Prediction: negative
Probabilities: 0.10487626492977142

Text: I really enjoyed this film.
Prediction: positive
Probabilities: 0.9946993589401245

Text: I wouldn't recommend this movie.
Prediction: negative
Probabilities: 0.31262919306755066
```

### NER

The following shows `Experiment E2` performance on three test examples. The model does a fair job by getting all NER labels correct whilst only missing out one named entity.

```text

Input: I live in San Francisco .
I               O
live            O
in              O
San             B-location
Francisco       I-location
.               O

Input: Apple released a new iPhone .
Apple           O
released        O
a               O
new             O
iPhone          B-product
.               O

Input: Michael visited London last week .
Michael         O
visited         O
London          B-location
last            O
week            O
.               O
```

## Limitations and Notes

This repo is for educational purposes. Therefore, considering its experimental nature, we should be aware of the limitations and differences from production-scale post-training. 

Here are some notable ones:

First, our base model is itself a small model by modern standard. To train a product-grade small model for local deployment and serving specific business tasks, start with a proven open source model or an open weight model, such as Llama, Gemma, Nemotron, Qwen, Kimi, Deepseek, MiniMax, GPT OSS, Mistral, and GLM, among others.

Second, training data is where `knowledge`, `skill` or `intelligence` come from. In our case, CHILDES dataset has an inherent limitation: only contains one turn chat pairs between an adult and a child, which keeps it from being a reasonable dataset if our goal is to train a commercial-grade chatbot for kids. Plus, we may still need millions to tens of milions of examples in fine-tuning or distilling from a large model in order to build an LM for real world uses cases.

Third, given much larger base model and larger dataset, to make use of multiple GPUs, reduce memory footprint, accelerate math operations, etc., we need distributed training and fine-tuning approaches and platforms, such as distributed data-parallel or model-parallel, distributed optimizer, checkpointing & reproducibility, sync vs async SGD, etc.

Finally, building our own models through post-training is only half the journey. Real MLOps is the other half that poses even more challenges to engineering: model serving for high throughput and low latency, GPU provisioning, autoscaling, monitoring, and timely response and problem solving when something goes wrong.

