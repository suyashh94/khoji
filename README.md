# khoji

**Make retrieval models actually work on your data**

[Installation](#installation) | [Quick Start](#quick-start) | [Retrieval Modes](#retrieval-modes) | [Training Concepts](#training-concepts) | [Extensibility](#extensibility) | [Architecture](#architecture)

---

Pretrained retrieval models (BERT, CLIP, BLIP-2) are trained on generic web data. They work reasonably well out of the box, but struggle on domain-specific queries — legal documents, medical images, satellite imagery, fashion products, internal knowledge bases. The standard fix is fine-tuning, but wiring together the data pipeline, negative mining, LoRA, training loop, and evaluation for retrieval is a lot of boilerplate.

khoji handles all of that. You point it at your data and a base model, and it fine-tunes a retrieval adapter using LoRA — with hard negative mining, standard IR evaluation, and support for text search, image search, and composed image retrieval (e.g., "find this dress but in red"). It works as a single YAML config for quick experiments, or as composable Python components when you need full control.

### Three retrieval modes

| Mode | Query | Target | Models | Use case |
|------|-------|--------|--------|----------|
| **Text → Text** | text | text | BERT, BGE, sentence-transformers | Document search, FAQ matching, semantic search |
| **Text → Image** | text | image | CLIP, SigLIP | Image search from text descriptions |
| **(Image + Text) → Image** | reference image + modification caption | image | BLIP-2 | "Find me this dress but in red" |

### Two levels of abstraction

| Level | What you write | Best for |
|-------|---------------|----------|
| **Config-driven** | A YAML file → `run()` / `run_multimodal()` / `run_composed()` | Reproducible experiments, quick iteration |
| **Python API** | Compose individual components (model, trainer, evaluator, data) | Custom workflows, non-standard data sources, research |

### What you can plug in

| Component | Built-in | Custom |
|-----------|----------|--------|
| **Models** | Any HuggingFace model (auto-detected) | Any `nn.Module` + encode function |
| **Datasets** | BEIR (20+), Flickr30k, RSICD, FashionIQ | JSONL/TSV files, or raw Python dicts |
| **Loss functions** | Triplet Margin, InfoNCE, Contrastive | Any `(query, pos, neg) -> scalar` callable |
| **Metrics** | nDCG@k, MRR@k, Recall@k | Any `(ranked_ids, qrel, k) -> float` callable |
| **Negative mining** | Random, hard, mixed | Build your own `Triplet` / `ComposedTriplet` objects |

### Other highlights

- Parameter-efficient fine-tuning via **LoRA** (or full fine-tuning with `lora: null`)
- Auto-detection of model pooling strategy and LoRA target modules
- Iterative hard negative mining (re-mine with the fine-tuned model)
- Mixed precision training (fp16/bf16)
- All metrics implemented from scratch — no external IR evaluation frameworks
- Hardware support: CUDA, Apple Silicon (MPS), CPU

---

## Installation

**Requirements:** Python >= 3.10

```bash
# From PyPI
pip install khoji

# From source
git clone https://github.com/suyashh94/khoji.git
cd khoji
uv sync              # or: pip install -e .
uv sync --group dev  # adds pytest, ruff
```

---

## Quick Start

### Text → Text

```bash
khoji init                   # generate example configs
khoji fiqa_quick.yaml        # train + evaluate
```

```python
from khoji import ForgeConfig, run

config = ForgeConfig.from_yaml("fiqa_quick.yaml")
result = run(config)
print(result.finetuned.metrics)   # {"ndcg@1": 0.45, "mrr@5": 0.52, ...}
print(result.adapter_dir)         # path to saved LoRA adapter
```

### Text → Image

```bash
khoji multimodal flickr30k_quick.yaml
```

```python
from khoji import MultimodalForgeConfig, run_multimodal

config = MultimodalForgeConfig.from_yaml("flickr30k_quick.yaml")
result = run_multimodal(config)
```

### (Image + Text) → Image (Composed Retrieval)

```python
from khoji import ComposedForgeConfig, run_composed

config = ComposedForgeConfig.from_yaml("composed_config.yaml")
result = run_composed(config)
```

---

## Retrieval Modes

### 1. Text → Text

Fine-tune text embedding models (BERT, BGE, sentence-transformers) for domain-specific document retrieval.

#### Config-driven

```yaml
model:
  name: BAAI/bge-base-en-v1.5
  # adapter_path: null          # warm-start from existing adapter
  # dtype: null                 # "fp16", "bf16", or null (fp32)

data:
  dataset: fiqa                  # BEIR dataset name or path to local directory
  split: train
  negatives: mixed               # "random", "hard", or "mixed"
  n_random: 2                    # random negatives per pair (mixed mode)
  n_hard: 1                      # hard negatives per pair (mixed mode)
  # n_negatives: 1              # negatives per pair (random/hard modes)
  # n_queries: null             # subset of queries (null = all)
  # corpus_size: null           # corpus limit for mining (null = all)
  # top_k: 50                   # top-k for hard negative mining
  # skip_top: 0                 # skip top N non-relevant (avoids false negatives)
  # mining_rounds: 1            # iterative mining rounds (re-mine with fine-tuned model)

lora:
  r: 8
  alpha: 16
  dropout: 0.1
  # target_modules: null        # auto-detected per architecture
# lora: null                    # uncomment for full fine-tuning

train:
  epochs: 3
  batch_size: 8
  grad_accum_steps: 4            # effective batch = batch_size * grad_accum_steps
  lr: 2e-5
  weight_decay: 0.01
  warmup_steps: 100              # linear warmup then linear decay
  max_grad_norm: 1.0
  max_length: 512
  loss: infonce                  # "triplet", "infonce", or "contrastive"
  margin: 0.2                    # for triplet loss
  temperature: 0.05              # for infonce loss
  # mixed_precision: null        # "fp16", "bf16", or null
  # overfit_batches: null        # set to 1 for debugging
  sanity_check_samples: 10

eval:
  # dataset: null               # null = use data.dataset
  k_values: [1, 5, 10]
  split: test
  run_before: true
  run_after: true

seed: 42
output_dir: ./forge-output
```

#### Python API (component-by-component)

```python
from khoji import (
    EmbeddingModel, Evaluator, Trainer, TrainingConfig,
    TripletDataset, LoRASettings,
    load_beir, build_mixed_negatives,
)

# 1. Load data
dataset = load_beir("fiqa", split="train")

# 2. Build training triplets
model = EmbeddingModel("BAAI/bge-base-en-v1.5")
triplets = build_mixed_negatives(dataset, model, n_random=2, n_hard=1, top_k=50)

# 3. Train
config = TrainingConfig(
    epochs=3, batch_size=8, lr=2e-5,
    lora=LoRASettings(r=8, alpha=16),
    save_dir="./my-adapter",
)
trainer = Trainer("BAAI/bge-base-en-v1.5", config)
history = trainer.train(TripletDataset(triplets))

# 4. Evaluate
evaluator = Evaluator("BAAI/bge-base-en-v1.5", adapter_path="./my-adapter")
result = evaluator.evaluate("fiqa", split="test", k_values=[1, 5, 10])
result.print()

# 5. Inference
model = EmbeddingModel("BAAI/bge-base-en-v1.5", adapter_path="./my-adapter")
embeddings = model.encode(["What is compound interest?", "How do bonds work?"])
```

#### Custom datasets

Every dataset in khoji is just three things: **queries**, a **corpus**, and **relevance judgments** (qrels) mapping which corpus items are relevant to which queries. You can provide these as local files or as Python dicts.

**Option A: Local files** — create a directory with three files:

```
my_dataset/
  queries.jsonl   # {"_id": "q1", "text": "What is compound interest?"}
  corpus.jsonl    # {"_id": "d1", "text": "Compound interest is ...", "title": "Optional Title"}
  qrels.tsv       # q1\td1\t1  (tab-separated: query_id, doc_id, relevance_score. No header.)
```

```yaml
data:
  dataset: ./my_dataset    # point to the directory
```

**Option B: Python dicts** — build from any source (database, CSV, API, etc.):

```python
from khoji import RetrievalDataset

dataset = RetrievalDataset(
    queries={"q1": "What is compound interest?"},
    corpus={"d1": "Compound interest is interest on interest.", "d2": "Unrelated doc."},
    qrels={"q1": {"d1": 1}},   # query q1 → doc d1 is relevant (score 1)
)
# Pass this to build_random_negatives(), Evaluator.evaluate(dataset=...), etc.
```

Training and evaluation datasets are independent — you can train on one and evaluate on another:

```yaml
data:
  dataset: ./my_train_data
eval:
  dataset: ./my_eval_data     # null = same as data.dataset
```

#### Supported models

Any HuggingFace model compatible with `AutoModel` / `AutoTokenizer`. Pooling is auto-detected from the model's sentence-transformers config.

| Model | Pooling | Architecture |
|-------|---------|-------------|
| `BAAI/bge-base-en-v1.5` | CLS | BERT |
| `sentence-transformers/all-MiniLM-L6-v2` | Mean | BERT |

Auto-detected LoRA targets per architecture:

| Architecture | Target Modules |
|-------------|---------------|
| BERT, RoBERTa, XLM-RoBERTa | `query`, `key`, `value` |
| DistilBERT | `q_lin`, `k_lin`, `v_lin` |
| DeBERTa (v1/v2) | `query_proj`, `key_proj`, `value_proj` |
| Mistral, LLaMA | `q_proj`, `k_proj`, `v_proj` |

---

### 2. Text → Image

Fine-tune cross-modal models (CLIP, SigLIP) where queries are text and documents are images.

#### Config-driven

```yaml
model:
  name: openai/clip-vit-base-patch32
  # adapter_path: null
  # dtype: null
  lora_target: both              # "vision", "text", or "both"

data:
  dataset: nlphuji/flickr30k     # or "arampacha/rsicd" or local path
  split: train
  negatives: random
  n_negatives: 1
  cache_dir: null                 # cache downloaded images locally
  # ... all other data params same as text-to-text

lora:
  r: 8
  alpha: 16
  dropout: 0.1

train:
  epochs: 3
  batch_size: 8
  lr: 2e-5
  max_length: 77                  # CLIP default
  loss: infonce
  temperature: 0.05
  # ... all other train params same as text-to-text

# Optional: override image preprocessing
# preprocess:
#   image_size: 224
#   mean: [0.48145466, 0.4578275, 0.40821073]
#   std: [0.26862954, 0.26130258, 0.27577711]

eval:
  k_values: [1, 5, 10]
  run_before: true
  run_after: true

output_dir: ./forge-output/flickr30k
```

```bash
khoji multimodal flickr30k_quick.yaml
```

#### Python API

```python
from khoji import (
    MultimodalEmbeddingModel, MultimodalEvaluator,
    MultimodalTrainer, MultimodalTrainingConfig,
    MultimodalTripletDataset, LoRASettings,
    load_custom_multimodal, build_random_negatives_multimodal,
)

# 1. Load data
dataset = load_custom_multimodal("./my_image_dataset")

# 2. Build triplets
triplets = build_random_negatives_multimodal(dataset, n_negatives=1)

# 3. Train
config = MultimodalTrainingConfig(
    epochs=3, batch_size=8, lr=2e-5,
    lora=LoRASettings(r=8, alpha=16),
    lora_target="both",
    save_dir="./my-clip-adapter",
    base_dir="./my_image_dataset",
)
trainer = MultimodalTrainer("openai/clip-vit-base-patch32", config)
history = trainer.train(MultimodalTripletDataset(triplets))

# 4. Evaluate
evaluator = MultimodalEvaluator("openai/clip-vit-base-patch32", adapter_path="./my-clip-adapter")
result = evaluator.evaluate(dataset=dataset, k_values=[1, 5, 10])
result.print()

# 5. Inference
model = MultimodalEmbeddingModel("openai/clip-vit-base-patch32", adapter_path="./my-clip-adapter")
text_emb = model.encode_text(["a photo of a sunset"])
img_emb = model.encode_image_sources(["sunset.jpg", "cat.jpg"], base_dir="./photos")

import torch
scores = torch.mm(text_emb, img_emb.t()).squeeze(0)
```

#### LoRA targeting

Control which encoder(s) to fine-tune:

| `lora_target` | What's trained | When to use |
|---------------|---------------|-------------|
| `both` | Text + vision encoders | Default. General domain adaptation. |
| `vision` | Vision encoder only | Text understanding is fine, images are domain-specific (satellite, medical). |
| `text` | Text encoder only | Images are generic, queries are domain-specific. |

#### Custom image datasets

Same three-file structure as text-to-text, but `corpus.jsonl` uses an `image` field instead of `text`. Image paths are relative to the dataset directory; HTTP(S) URLs also work.

```
my_image_dataset/
  queries.jsonl   # {"_id": "q1", "text": "a dog playing fetch"}
  corpus.jsonl    # {"_id": "d1", "image": "images/dog.jpg"}   (relative path or URL)
  qrels.tsv       # q1\td1\t1
  images/         # local image files
```

Or build in Python:

```python
from khoji import MultimodalRetrievalDataset

dataset = MultimodalRetrievalDataset(
    queries={"q1": "a dog playing fetch"},
    corpus={"d1": "images/dog.jpg", "d2": "images/cat.jpg"},
    qrels={"q1": {"d1": 1}},
    base_dir="./my_image_dataset",   # resolve relative paths from here
)
```

#### Built-in datasets

| Dataset | Config name | Description |
|---------|------------|-------------|
| Flickr30k | `nlphuji/flickr30k` | ~30k images, 5 captions each. General purpose. |
| RSICD | `arampacha/rsicd` | ~10k satellite/aerial images. Domain where CLIP wasn't trained. |

#### Supported models

| Model | Type | Embedding Dim |
|-------|------|--------------|
| `openai/clip-vit-base-patch32` | CLIP | 512 |
| `openai/clip-vit-large-patch14` | CLIP | 768 |
| `google/siglip-base-patch16-224` | SigLIP | 768 |

Any CLIP or SigLIP variant on HuggingFace should work.

---

### 3. (Image + Text) → Image (Composed Retrieval)

Fine-tune joint encoder models (BLIP-2) for composed image retrieval: given a reference image and a modification caption ("make it red"), retrieve the correct target image from a gallery.

To try this out on FashionIQ (dress/shirt/toptee categories), download the annotations first:

```bash
python scripts/fashioniq/download_data.py
python scripts/train_composed_retrieval_api.py --category dress
```

#### Config-driven

```yaml
model:
  name: Salesforce/blip2-itm-vit-g
  # adapter_path: null
  # dtype: null

data:
  dataset: ./data/my_composed_dataset    # local directory
  split: train
  negatives: mixed
  n_random: 2
  n_hard: 1
  top_k: 50
  skip_top: 5
  mining_rounds: 1
  cache_dir: null

lora:
  r: 8
  alpha: 16
  dropout: 0.1

train:
  epochs: 5
  batch_size: 8
  lr: 2e-5
  warmup_steps: 50
  loss: infonce
  temperature: 0.05

eval:
  k_values: [1, 5, 10, 50]
  run_before: true
  run_after: true

output_dir: ./output/composed-retrieval
```

```python
from khoji import ComposedForgeConfig, run_composed

config = ComposedForgeConfig.from_yaml("composed_config.yaml")
result = run_composed(config)
```

#### Python API

```python
from khoji import (
    JointEmbeddingModel, ComposedEvaluator,
    ComposedTrainer, ComposedTrainingConfig,
    ComposedTripletDataset, LoRASettings,
    load_custom_composed, build_random_negatives_composed,
)

# 1. Load data
dataset = load_custom_composed("./my_composed_dataset")

# 2. Build triplets
triplets = build_random_negatives_composed(dataset, n_negatives=3)

# 3. Train
config = ComposedTrainingConfig(
    epochs=5, batch_size=8, lr=2e-5,
    lora=LoRASettings(r=8, alpha=16),
    save_dir="./my-composed-adapter",
    cache_dir="./image_cache",
)
trainer = ComposedTrainer("Salesforce/blip2-itm-vit-g", config)
history = trainer.train(ComposedTripletDataset(triplets))

# 4. Evaluate
evaluator = ComposedEvaluator(
    "Salesforce/blip2-itm-vit-g", adapter_path="./my-composed-adapter"
)
result = evaluator.evaluate(dataset=dataset, k_values=[1, 5, 10, 50])
result.print()

# 5. Inference
model = JointEmbeddingModel(
    "Salesforce/blip2-itm-vit-g", adapter_path="./my-composed-adapter"
)
from khoji import load_image
ref_img = load_image("reference.jpg")
query_emb = model.encode(images=[ref_img], texts=["make it red"])
gallery_emb = model.encode(images=[img1, img2, img3])

import torch
scores = torch.mm(query_emb, gallery_emb.t()).squeeze(0)
best_match = scores.argmax().item()
```

#### Custom composed datasets

Composed datasets differ from the other two modes in one key way: each **query is an (image, text) pair**, not just text. The query says "here's a reference image, and here's what I want changed about it." The corpus (gallery) is still just images.

**Local files:**

```
my_composed_dataset/
  queries.jsonl   # {"_id": "q1", "image": "imgs/ref.jpg", "text": "make it red"}
  corpus.jsonl    # {"_id": "d1", "image": "imgs/target.jpg"}
  qrels.tsv       # q1\td1\t1
```

Note that `queries.jsonl` has **both** an `image` and a `text` field. This is the key difference from the other modes.

**Python dicts:**

```python
from khoji import ComposedRetrievalDataset

dataset = ComposedRetrievalDataset(
    queries={
        "q1": ("imgs/ref_dress.jpg", "make it red"),       # (reference_image, modification_text)
        "q2": ("imgs/ref_shirt.jpg", "shorter sleeves"),
    },
    corpus={
        "d1": "imgs/red_dress.jpg",
        "d2": "imgs/short_sleeve_shirt.jpg",
        "d3": "imgs/other.jpg",
    },
    qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
    base_dir="./my_dataset",    # resolve relative image paths from here
)
```

#### Supported models

| Model | Type | Description |
|-------|------|-------------|
| `Salesforce/blip2-itm-vit-g` | BLIP-2 | Joint image-text encoder with Q-Former. 256-dim shared space. |

Any BLIP-2 variant on HuggingFace should work. Custom joint encoders are also supported (see [Extensibility](#extensibility)).

---

## Training Concepts

These concepts apply across all three retrieval modes.

### Loss functions

| Loss | Config value | Formula | Key param | When to use |
|------|-------------|---------|-----------|-------------|
| **Triplet Margin** | `triplet` | `relu(d(q,p) - d(q,n) + margin)` | `margin: 0.2` | Good default. Works with small batches and random negatives. |
| **InfoNCE** | `infonce` | Cross-entropy with in-batch negatives | `temperature: 0.05` | Best with larger batches and hard negatives. Typically strongest. |
| **Contrastive** | `contrastive` | `-cos(q,p) + cos(q,n)` | (none) | Simple baseline. No hyperparameters beyond LR. |

Custom loss functions are supported via the Python API — any `(query_emb, pos_emb, neg_emb) -> scalar` callable works.

### Negative mining strategies

Retrieval fine-tuning requires triplets: (query, relevant item, non-relevant item). The non-relevant item is the "negative." How you choose negatives has a big impact on what the model learns.

#### Random negatives (`negatives: random`)

Randomly sample non-relevant items from the corpus. Fast (no model encoding needed), and sufficient for initial training where the model needs to learn basic relevance signals.

```yaml
data:
  negatives: random
  n_negatives: 3       # 3 random negatives per (query, positive) pair
```

#### Hard negatives (`negatives: hard`)

Encode the entire corpus and all queries with the current model, then for each query pick the **most similar non-relevant items** as negatives. These are items the model currently thinks are relevant but aren't — forcing the model to learn finer distinctions.

How it works:
1. Encode all corpus items and queries into embeddings
2. For each query, rank corpus items by cosine similarity
3. From the top-`top_k` results, filter out actually-relevant items
4. Optionally skip the top N (`skip_top`) — see below
5. Pick `n_negatives` from the remaining as hard negatives

```yaml
data:
  negatives: hard
  n_negatives: 3       # 3 hard negatives per (query, positive) pair
  top_k: 50            # consider top-50 most similar corpus items
  skip_top: 0          # how many to skip (see below)
```

#### Mixed negatives (`negatives: mixed`)

Combines random and hard negatives in the same training set. Random negatives teach basic "this is clearly irrelevant" discrimination. Hard negatives push fine-grained ranking — "these two items look similar but only one is correct." This usually gives the best training signal.

```yaml
data:
  negatives: mixed
  n_random: 2          # 2 random negatives per pair
  n_hard: 1            # 1 hard negative per pair
  top_k: 50
```

Note: `n_negatives` is used by `random` and `hard` modes. `n_random` and `n_hard` are used by `mixed` mode. They are separate parameters because mixed mode needs counts for each type.

#### `top_k` — mining search window

When mining hard negatives, `top_k` controls how many top-ranked corpus items to consider. A larger `top_k` searches deeper but takes longer. Typical value: 50.

If `top_k` is too small, you may not find enough non-relevant items (especially for queries where many top results are relevant). If it's too large, the "hard" negatives become easy (they're far down the ranking).

#### `skip_top` — avoiding false negatives

Most retrieval datasets have **incomplete relevance judgments** (qrels). A document might be perfectly relevant to a query but isn't labeled as such, simply because a human annotator didn't see it. These unlabeled positives tend to cluster at the very top of the model's ranking — they look relevant because they *are* relevant.

If you mine these as "hard negatives," you're training the model to push away items that are actually good matches. This hurts performance.

`skip_top` skips the top N non-relevant results before picking hard negatives:

```yaml
data:
  skip_top: 5          # skip the 5 most similar non-relevant items
  top_k: 50            # then pick from ranks 6-50
```

**When to use it:**
- Datasets with sparse qrels (few labeled positives per query): `skip_top: 5-10`
- Datasets with comprehensive qrels: `skip_top: 0` is fine
- When in doubt, `skip_top: 5` is a safe default for hard/mixed negatives

#### `mining_rounds` — iterative re-mining

A single round of hard negative mining uses the **pretrained model** to find hard negatives. But after training, the model has improved — what was "hard" before may now be easy. Iterative mining repeats the mine-train cycle:

```
Round 1: pretrained model → mine negatives → train → adapter_r1
Round 2: fine-tuned model (adapter_r1) → re-mine harder negatives → train → adapter_r2 (final)
```

Each round halves the learning rate to avoid overshooting as negatives get harder.

```yaml
data:
  negatives: mixed      # only meaningful for hard/mixed (random doesn't use mining)
  mining_rounds: 2      # 2 rounds of mine → train
```

**When to use it:**
- 1 round is usually sufficient for most tasks
- 2 rounds helps when the pretrained model is already reasonable on your domain and you need to push further
- 3+ rounds has diminishing returns and risk of overfitting to hard negatives

#### Choosing a strategy

| Situation | Recommended |
|-----------|------------|
| First experiment / quick iteration | `random` with `n_negatives: 1-3` |
| Production training | `mixed` with `n_random: 2, n_hard: 1` |
| Squeezing last bits of performance | `mixed` with `mining_rounds: 2, skip_top: 5` |
| Very large corpus (>1M items) | `random` first, then `hard` on a `corpus_size` subset |

### LoRA vs full fine-tuning

**LoRA (default)**: Only adapter weights are trained and saved (~few MB). Base model weights are frozen.

```yaml
lora:
  r: 8        # rank (4, 8, 16, 32 — higher = more capacity)
  alpha: 16   # scaling factor (convention: 2 * r)
  dropout: 0.1
```

**Full fine-tuning**: All parameters are trained and saved (hundreds of MB). Use a lower learning rate.

```yaml
lora: null
train:
  lr: 1e-5   # lower LR to avoid catastrophic forgetting
```

### Model precision

Two independent controls:

| Setting | What it does | Values |
|---------|-------------|--------|
| `model.dtype` | Precision of base model weights in memory | `null` (fp32), `"fp16"`, `"bf16"` |
| `train.mixed_precision` | AMP during forward/backward pass | `null` (fp32), `"fp16"`, `"bf16"` |

Use both together for maximum memory savings:

```yaml
model:
  dtype: bf16
train:
  mixed_precision: bf16
```

### Evaluation metrics

All implemented from scratch (no external IR evaluation libraries).

| Metric | Description |
|--------|-------------|
| **nDCG@k** | Normalized Discounted Cumulative Gain. Measures ranking quality with graded relevance. |
| **MRR@k** | Mean Reciprocal Rank. 1 / position of the first relevant result. |
| **Recall@k** | Fraction of all relevant documents found in top-k. |

### Output structure

```
output_dir/
  config.yaml              # saved config for reproducibility
  train_history.json       # per-step loss, LR, grad norms, per-epoch loss
  adapter/                 # final LoRA adapter weights
    adapter_model.safetensors
    adapter_config.json
  adapter_r1/              # round 1 adapter (only when mining_rounds > 1)
  baseline.json            # baseline eval metrics (if run_before: true)
  finetuned.json           # fine-tuned eval metrics (if run_after: true)
```

### Result objects

**`RunResult`** (returned by `run()`, `run_multimodal()`, `run_composed()`):

| Field | Type | Description |
|-------|------|-------------|
| `history` | `TrainHistory` | `step_loss`, `step_lr`, `step_grad_norm`, `epoch_loss` |
| `baseline` | `EvalResult \| None` | Baseline metrics (None if `run_before: false`) |
| `finetuned` | `EvalResult \| None` | Fine-tuned metrics (None if `run_after: false`) |
| `adapter_dir` | `str \| None` | Path to saved LoRA adapter |

**`EvalResult`**:

| Field | Type |
|-------|------|
| `metrics` | `dict[str, float]` — e.g. `{"ndcg@5": 0.42, "mrr@5": 0.51}` |
| `model_name` | `str` |
| `dataset_name` | `str` |
| `num_queries` | `int` |
| `num_corpus` | `int` |

Methods: `print()`, `save(path)`, `to_dict()`.

---

## Extensibility

### Custom models (non-HuggingFace)

Every mode supports custom PyTorch models. The pattern is the same across all three: you provide an `nn.Module` (which holds the trainable parameters) and one or more **encode functions** (which define how inputs become embeddings). khoji calls your encode functions during training with gradients enabled, handles L2 normalization, and applies LoRA/optimizer/scheduler around your module.

The key difference between modes is **what your encode functions receive and return**:

| Mode | Encode functions | Input | Output |
|------|-----------------|-------|--------|
| Text → Text | Wired automatically from model + tokenizer + pooling mode | — | — |
| Text → Image | `encode_text_fn` and `encode_image_fn` | `list[str]` (texts) and `list[str]` (image file paths/URLs) | `Tensor (batch, dim)` each |
| Composed | `encode_query_fn` and `encode_image_fn` | `(list[PIL.Image], list[str])` and `list[PIL.Image]` | `Tensor (batch, dim)` each |

Note the difference: Text → Image `encode_image_fn` receives **file paths** (the trainer handles loading), while Composed `encode_image_fn` receives **PIL images** (the trainer loads images before calling your function).

#### Text → Text

Your model must follow the HuggingFace convention: `forward(input_ids, attention_mask, ...)` returns an object with a `.last_hidden_state` attribute of shape `(batch, seq_len, hidden_dim)`. khoji applies pooling (CLS, mean, max, etc.) on top.

Your tokenizer must support `tokenizer(texts, padding=True, truncation=True, max_length=N, return_tensors="pt")`.

```python
from khoji import EmbeddingModel, Trainer, TrainingConfig

# For inference / evaluation
embedding_model = EmbeddingModel(
    model=my_encoder,           # nn.Module
    tokenizer=my_tokenizer,     # HuggingFace-compatible tokenizer
    pooling="mean",             # "cls", "mean", "max", "weightedmean", "lasttoken"
)
embeddings = embedding_model.encode(["hello world"])

# For training
trainer = Trainer(
    model=my_encoder,
    tokenizer=my_tokenizer,
    pooling="mean",
    config=TrainingConfig(
        epochs=3,
        lora=None,              # full fine-tuning (LoRA also works if your model has attention layers)
    ),
)
```

#### Text → Image

You provide two encode functions. Both receive strings — `encode_text_fn` gets query texts, `encode_image_fn` gets image source paths/URLs (the trainer calls `load_image()` for you within `encode_image_fn` if needed, or you handle loading yourself).

The `model` parameter should be the `nn.Module` that holds all trainable parameters. Both encode functions should operate on `self.model` (or capture it in a closure) so that gradients flow through to the optimizer.

```python
from khoji import MultimodalTrainer, MultimodalTrainingConfig

trainer = MultimodalTrainer(
    model=my_clip_model,          # nn.Module holding all parameters
    encode_text_fn=my_text_fn,    # (list[str]) -> Tensor (batch, dim)
    encode_image_fn=my_image_fn,  # (list[str]) -> Tensor (batch, dim)  ← receives file paths
    config=MultimodalTrainingConfig(
        epochs=3,
        lora=None,
        base_dir="./my_images",   # base directory for resolving relative image paths
    ),
)
```

#### (Image + Text) → Image

You provide two encode functions. Both receive **PIL images** (not file paths — the trainer loads images before calling your functions).

- `encode_query_fn(images, texts)`: Encode (reference_image + caption) pairs jointly
- `encode_image_fn(images)`: Encode target/gallery images

```python
from khoji import ComposedTrainer, ComposedTrainingConfig, JointEmbeddingModel

# For training
trainer = ComposedTrainer(
    model=my_model,                # nn.Module holding all parameters
    encode_query_fn=my_joint_fn,   # (list[PIL.Image], list[str]) -> Tensor (batch, dim)
    encode_image_fn=my_image_fn,   # (list[PIL.Image]) -> Tensor (batch, dim)
    config=ComposedTrainingConfig(
        epochs=3,
        lora=None,
        base_dir="./my_images",
    ),
)

# For inference / evaluation
model = JointEmbeddingModel(
    encoder=my_encoder_fn,  # (images: list[PIL]|None, texts: list[str]|None, device) -> Tensor
)
# The encoder must handle three calling patterns:
#   encoder(images=[...], texts=None, device)      → image-only embeddings
#   encoder(images=None, texts=[...], device)       → text-only embeddings
#   encoder(images=[...], texts=[...], device)      → joint (image+text) embeddings
```

#### LoRA with custom models

LoRA works with custom models as long as your `nn.Module` contains standard attention layers (Linear modules named `query`, `key`, `value`, `q_proj`, etc.). If your module uses non-standard names, specify them explicitly:

```python
config = TrainingConfig(
    lora=LoRASettings(r=8, alpha=16, target_modules=["my_attn_q", "my_attn_k", "my_attn_v"]),
)
```

If your model doesn't have attention layers suitable for LoRA, use `lora=None` for full fine-tuning.

### Custom loss functions

Pass any callable to `TrainingConfig.loss_fn` (Python API only):

```python
import torch

def circle_loss(query_emb, positive_emb, negative_emb, margin=0.25, gamma=64):
    pos_sim = torch.nn.functional.cosine_similarity(query_emb, positive_emb)
    neg_sim = torch.nn.functional.cosine_similarity(query_emb, negative_emb)
    alpha_p = torch.clamp(1 + margin - pos_sim, min=0)
    alpha_n = torch.clamp(neg_sim + margin, min=0)
    logit_p = -gamma * alpha_p * (pos_sim - (1 - margin))
    logit_n = gamma * alpha_n * (neg_sim - margin)
    return torch.nn.functional.softplus(logit_n - logit_p).mean()

config = TrainingConfig(loss_fn=circle_loss, ...)
```

### Custom metrics

Pass `extra_metrics` to any `Evaluator.evaluate()`:

```python
def precision_at_k(ranked_doc_ids, qrel, k):
    relevant = {d for d, s in qrel.items() if s > 0}
    return sum(1 for d in ranked_doc_ids[:k] if d in relevant) / k

result = evaluator.evaluate(
    dataset=my_dataset,
    k_values=[1, 5, 10],
    extra_metrics={"precision": precision_at_k},
)
# result.metrics includes both built-in and custom metrics
```

The built-in metric functions are also exported for standalone use:

```python
from khoji import ndcg_at_k, mrr_at_k, recall_at_k

ranked = ["d3", "d1", "d5", "d2"]
qrel = {"d1": 2, "d5": 1}
print(recall_at_k(ranked, qrel, k=3))  # 1.0
```

### Custom image preprocessing (Text → Image only)

Three tiers, from most to least automatic:

1. **Auto (default)**: Loads `AutoProcessor` from HuggingFace.
2. **YAML overrides**: Override specific values (`image_size`, `mean`, `std`).
3. **Custom callable** (Python API): Full control over augmentations and transforms.

```python
import torch, torchvision.transforms as T
from PIL import Image

transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])

def my_preprocessor(images: list[Image.Image]) -> torch.Tensor:
    return torch.stack([transform(img) for img in images])

trainer = MultimodalTrainer(
    "openai/clip-vit-base-patch32",
    preprocess_overrides={"custom_fn": my_preprocessor},
    config=MultimodalTrainingConfig(...),
)
```

---

## Architecture

```
src/khoji/
  # ── Text → Text ──────────────────────────────
  config.py                 ForgeConfig (YAML)
  run.py                    run() orchestrator
  dataset.py                load_beir, load_custom, RetrievalDataset
  data.py                   Triplet, TripletDataset, negative mining
  model.py                  EmbeddingModel (pooling auto-detection)
  trainer.py                Trainer, TrainingConfig, TrainHistory
  evaluator.py              Evaluator, EvalResult

  # ── Text → Image ─────────────────────────────
  multimodal_config.py      MultimodalForgeConfig
  multimodal_run.py         run_multimodal()
  multimodal_dataset.py     load_flickr30k, load_rsicd, load_custom_multimodal
  multimodal_data.py        MultimodalTriplet, negative mining
  multimodal_model.py       MultimodalEmbeddingModel, JointEmbeddingModel
  multimodal_trainer.py     MultimodalTrainer
  multimodal_evaluator.py   MultimodalEvaluator

  # ── (Image + Text) → Image ───────────────────
  composed_config.py        ComposedForgeConfig
  composed_run.py           run_composed()
  composed_dataset.py       load_custom_composed, ComposedRetrievalDataset
  composed_data.py          ComposedTriplet, negative mining
  composed_trainer.py       ComposedTrainer
  composed_evaluator.py     ComposedEvaluator

  # ── Shared ────────────────────────────────────
  loss.py                   triplet_margin_loss, infonce_loss, contrastive_loss
  metrics.py                ndcg_at_k, mrr_at_k, recall_at_k
  lora.py                   LoRASettings, apply_lora
  image_utils.py            load_image, load_images_batch, build_image_processor
  device.py                 get_device (CUDA > MPS > CPU)
```

### Data flow (all modes follow the same pattern)

```
Config (YAML or Python)
  │
  ├─ Dataset loading ──> queries + corpus + qrels
  │
  ├─ Baseline evaluation (optional)
  │
  └─ Mining round loop:
       │
       ├─ Build triplets (random / hard / mixed)
       │    (round 2+ uses fine-tuned model for mining)
       │
       ├─ Trainer.train() ──> TrainHistory + adapter
       │
       └─ adapter feeds next round
  │
  ├─ Fine-tuned evaluation (optional)
  │
  └─ RunResult (history + baseline + finetuned + adapter_dir)
```

---

## Example Scripts

| Script | Mode | Description |
|--------|------|-------------|
| `scripts/train_text_retrieval.py` | Text → Text | Config-driven + manual API on FiQA |
| `scripts/train_multimodal_retrieval.py` | Text → Image | Config-driven + manual API on RSICD |
| `scripts/train_composed_retrieval.py` | Composed | Standalone FashionIQ training (low-level) |
| `scripts/train_composed_retrieval_api.py` | Composed | Config-driven + manual API on FashionIQ |
| `scripts/fashioniq/download_data.py` | Data setup | Download FashionIQ annotations (required for composed scripts) |

The composed retrieval scripts require FashionIQ annotations. Download them first:

```bash
python scripts/fashioniq/download_data.py          # default: ./data/fashioniq
python scripts/fashioniq/download_data.py ./my_dir  # custom output directory
```

This downloads captions, image splits, and ASIN-to-URL mappings (~3 categories: dress, shirt, toptee). Images are fetched on-the-fly from URLs during training and cached locally. Then run either composed script:

```bash
python scripts/train_composed_retrieval.py                          # standalone low-level script
python scripts/train_composed_retrieval.py --category shirt         # different category

python scripts/train_composed_retrieval_api.py                      # uses khoji API
python scripts/train_composed_retrieval_api.py --approach api       # manual API only
python scripts/train_composed_retrieval_api.py --approach config    # config-driven only
```

---

## Example Configs

Generated by `khoji init`:

| Config | Mode | Description |
|--------|------|-------------|
| `fiqa_quick.yaml` | Text → Text | 50 queries, random negatives, 2 epochs. Quick iteration. |
| `fiqa_full.yaml` | Text → Text | Full dataset, hard negatives, 5 epochs, InfoNCE. |
| `fiqa_mixed.yaml` | Text → Text | Mixed negatives (random + hard), InfoNCE. |
| `fiqa_overfit.yaml` | Text → Text | 1 batch, 50 epochs. Pipeline debugging. |
| `flickr30k_quick.yaml` | Text → Image | 50 queries, random negatives, 2 epochs. |
| `flickr30k_full.yaml` | Text → Image | Full dataset, hard negatives, image caching. |
| `flickr30k_overfit.yaml` | Text → Image | Overfit mode for debugging. |

---

## Hardware

Auto-detected: CUDA (1st) > MPS (2nd) > CPU (3rd).

**MPS tip**: If you hit OOM, reduce `batch_size` and increase `grad_accum_steps` to maintain the same effective batch size.

---

## Development

### Running tests

```bash
uv run pytest tests/ -v    # 132 tests
```

### Test coverage

| Module | Tests |
|--------|-------|
| `metrics.py` | nDCG, MRR, Recall — edge cases, graded relevance, k cutoffs |
| `model.py` | All pooling modes, auto-detection, L2 normalization |
| `data.py` | Random/hard/mixed negatives, determinism, correctness |
| `loss.py` | All 3 losses — shapes, values, gradient flow |
| `config.py` | YAML roundtrip, type coercion, defaults |
| `lora.py` | apply_lora, auto-detection, custom targets |
| `evaluator.py` | Custom datasets, extra metrics, serialization |
| `trainer.py` | Training loop, history tracking |
| `dataset.py` | load_custom, missing files, RetrievalDataset |
| `multimodal` | CLIP encoding, config, datasets, training, LoRA targeting, evaluation |
| `composed` | Dataset format, triplets, config YAML, custom model training, evaluation |
| `integration` | BEIR loading, retrieval sanity checks |

### Linting

```bash
uv run ruff check src/ tests/
```

---

## Roadmap

- [x] Text → Text retrieval (BERT, BGE, sentence-transformers)
- [x] Text → Image retrieval (CLIP, SigLIP)
- [x] (Image + Text) → Image composed retrieval (BLIP-2)
- [x] Full fine-tuning (`lora: null`)
- [x] Custom models, loss functions, metrics
- [ ] Validation loss tracking during training
- [ ] Early stopping
- [ ] Distributed training (multi-GPU via DDP)
- [ ] Checkpoint resumption
- [ ] Adapter merging (LoRA → base model)
- [ ] Logging integration (W&B, TensorBoard)

---

## License

MIT
