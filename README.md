# khoji

**Fine-tune embedding models for domain-specific retrieval using LoRA**

[Installation](#installation) | [Quick Start](#quick-start) | [Configuration](#configuration-reference) | [Python API](#python-api) | [Architecture](#architecture) | [Contributing](#development)

---

**khoji** is a lightweight, modular Python library for fine-tuning transformer-based embedding models on domain-specific retrieval tasks. It uses [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) for parameter-efficient training, supports multiple loss functions, and provides end-to-end evaluation with standard IR metrics — all written from scratch with no dependency on high-level evaluation frameworks.

**Key features:**

- Parameter-efficient fine-tuning via LoRA (only adapter weights are trained and saved)
- Multiple loss functions: Triplet Margin, InfoNCE, Contrastive
- Negative mining strategies: random, hard, or mixed (random + hard combined)
- Custom retrieval metrics: nDCG@k, MRR@k, Recall@k — implemented from scratch
- Auto-detection of model pooling strategy and LoRA target modules
- YAML-driven configuration for reproducible experiments
- Full Python API — inspect training history, plot loss curves, programmatic access to everything
- Hardware support: CUDA, Apple Silicon (MPS), and CPU

---

## Installation

**Requirements:** Python >= 3.10

### From PyPI

```bash
pip install khoji
```

### From source (recommended during development)

```bash
# Clone the repository
git clone https://github.com/suyashh94/khoji.git
cd khoji

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Dev dependencies

```bash
uv sync --group dev   # installs pytest, ruff
```

---

## Quick Start

### CLI

```bash
# Generate example config files in the current directory
khoji init

# Or in a specific directory
khoji init configs/

# Run a quick training experiment on FiQA
khoji fiqa_quick.yaml

# Or via Python module
python -m khoji.run fiqa_quick.yaml
```

### Python API

```python
from khoji import ForgeConfig, run

# Load config and run full pipeline
config = ForgeConfig.from_yaml("configs/fiqa_quick.yaml")
result = run(config)

# Inspect training history
print(result.history.epoch_loss)        # [0.182, 0.091, ...]
print(result.history.step_loss[:5])     # per optimizer step
print(result.history.step_lr[:5])       # learning rate schedule
print(result.history.step_grad_norm[:5])# gradient norms

# Plot loss curve
import matplotlib.pyplot as plt
plt.plot(result.history.step_loss)
plt.xlabel("Optimizer Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("loss_curve.png")

# Compare baseline vs fine-tuned
if result.baseline and result.finetuned:
    for metric in result.baseline.metrics:
        b = result.baseline.metrics[metric]
        f = result.finetuned.metrics[metric]
        print(f"{metric}: {b:.4f} -> {f:.4f} ({f-b:+.4f})")

# Use the trained model for inference
from khoji import EmbeddingModel
model = EmbeddingModel("BAAI/bge-base-en-v1.5", adapter_path=result.adapter_dir, dtype="bf16")
embeddings = model.encode(["What is compound interest?", "How do bonds work?"])
```

---

## Configuration Reference

All configuration is driven by a single YAML file. Below is every parameter with its type, default, and description.

### Full annotated config

```yaml
# ── Model ─────────────────────────────────────────────────────────
model:
  name: BAAI/bge-base-en-v1.5        # HuggingFace model identifier
  # adapter_path: null               # Path to existing LoRA adapter (for continued training)
  # dtype: null                      # Load base model in "fp16", "bf16", or null (fp32)
                                      #   Reduces memory. LoRA weights always stay in fp32.

# ── Data ──────────────────────────────────────────────────────────
data:
  dataset: fiqa                       # BEIR dataset name (e.g., fiqa, scifact, nfcorpus, etc.)
  split: train                        # Dataset split: "train", "validation", or "test"
  negatives: random                   # Negative strategy: "random", "hard", or "mixed"
  n_negatives: 1                      # Negatives per pair (used by "random" and "hard" modes)
  # n_random: 1                      # Random negatives per pair (only for "mixed" mode)
  # n_hard: 1                        # Hard negatives per pair (only for "mixed" mode)
  n_queries: null                     # Number of queries to use (null = all)
  corpus_size: null                   # Corpus size limit (null = full). Only relevant for hard negatives.
  # top_k: 50                        # Top-k corpus docs to consider for hard negative mining
  # mining_rounds: 1                 # Iterative mining rounds (hard/mixed only)
                                      #   Round 2+ re-mines using the fine-tuned model

# ── LoRA ──────────────────────────────────────────────────────────
# Set to null for full fine-tuning (all parameters trained):
#   lora: null
lora:
  r: 8                                # LoRA rank (higher = more parameters, more capacity)
  alpha: 16                           # LoRA scaling factor (typically 2x rank)
  dropout: 0.1                        # Dropout on LoRA layers (0.0 for overfitting experiments)
  # target_modules: null              # Layers to apply LoRA to (null = auto-detect by architecture)

# ── Training ──────────────────────────────────────────────────────
train:
  epochs: 3                           # Number of training epochs
  batch_size: 8                       # Micro-batch size (per forward pass)
  grad_accum_steps: 4                 # Gradient accumulation steps
                                      #   effective batch size = batch_size * grad_accum_steps
  lr: 2e-5                            # Learning rate (AdamW optimizer)
  weight_decay: 0.01                  # AdamW weight decay
  warmup_steps: 100                   # Linear warmup steps, then linear decay
  max_grad_norm: 1.0                  # Gradient clipping (max L2 norm)
  max_length: 512                     # Max token length for tokenization
  loss: triplet                       # Loss function: "triplet", "infonce", or "contrastive"
  margin: 0.2                         # Margin for triplet loss (ignored by other losses)
  temperature: 0.05                   # Temperature for infonce loss (ignored by other losses)
  # mixed_precision: null             # "fp16", "bf16", or null (disabled)
  sanity_check_samples: 10            # Sample N training triplets and report cosine similarity
                                      #   before/after training as a quick sanity check
  # overfit_batches: null             # Set to 1 (or N) to overfit on N batches for debugging
  # save_every_n_steps: null          # Save checkpoint every N optimizer steps
  # keep_all_checkpoints: false       # true = keep all, false = keep only latest

# seed: null                          # Global seed for reproducibility

# ── Evaluation ────────────────────────────────────────────────────
eval:
  # dataset: null                     # Eval dataset — BEIR name or local path
                                      #   null = use data.dataset. Set to evaluate on a
                                      #   different dataset than you train on.
  k_values: [1, 5, 10]               # K values for nDCG@k, MRR@k, Recall@k
  split: test                         # Evaluation dataset split
  n_queries: null                     # Number of eval queries (null = all)
  corpus_size: null                   # Eval corpus size (null = full)
  run_before: true                    # Evaluate baseline model before training
  run_after: true                     # Evaluate fine-tuned model after training

# ── Output ────────────────────────────────────────────────────────
output_dir: ./forge-output            # Directory for adapter weights, configs, eval results
```

### Parameter details

#### `model`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `BAAI/bge-base-en-v1.5` | Any HuggingFace transformer model. Sentence-transformer models are fully supported with auto-detected pooling. |
| `adapter_path` | `str \| null` | `null` | Path to a previously saved LoRA adapter to continue training from. |
| `dtype` | `str \| null` | `null` | Load base model weights in reduced precision: `"fp16"`, `"bf16"`, or `null` (fp32). Reduces memory — LoRA weights are always kept in fp32. See [Model Precision](#model-precision). |

**Tested models:** `BAAI/bge-base-en-v1.5`, `sentence-transformers/all-MiniLM-L6-v2`. Any model on HuggingFace that works with `AutoModel` should work.

#### `data`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `str` | `fiqa` | BEIR dataset name **or** path to a local dataset directory. See [Custom Datasets](#custom-datasets). |
| `split` | `str` | `train` | Which split to build training triplets from. |
| `negatives` | `str` | `random` | `"random"`: fast, no model needed. `"hard"`: mines negatives using model embeddings (slower, better signal). `"mixed"`: both random and hard negatives combined — balanced training signal. |
| `n_negatives` | `int` | `1` | Negatives per (query, positive) pair. Used by `random` and `hard` modes. Ignored by `mixed`. |
| `n_random` | `int` | `1` | Random negatives per pair. Only used when `negatives: mixed`. |
| `n_hard` | `int` | `1` | Hard negatives per pair. Only used when `negatives: mixed`. |
| `n_queries` | `int \| null` | `null` | Subset of queries to use. `null` = all queries in the split. Useful for quick experiments. |
| `corpus_size` | `int \| null` | `null` | Corpus size limit for hard negative mining. Relevant docs are always included. `null` = full corpus. |
| `top_k` | `int` | `50` | Top similar docs to consider when mining hard negatives (for `hard` and `mixed` modes). |
| `mining_rounds` | `int` | `1` | Iterative mining rounds (for `hard` and `mixed` modes). Round 2+ re-mines negatives using the fine-tuned model from the previous round. LR is halved each round. |

#### `lora`

Set the entire `lora` section to `null` to disable LoRA and train **all model parameters** (full fine-tuning):

```yaml
lora: null    # full fine-tuning — all parameters are trained and saved
```

When `lora` is set (default), only the LoRA adapter weights are trained and saved:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | `int` | `8` | LoRA rank. Controls the bottleneck dimension. Higher rank = more trainable parameters, more model capacity. Typical values: 4, 8, 16, 32. |
| `alpha` | `int` | `16` | LoRA scaling factor. The effective scaling is `alpha / r`. Convention is `alpha = 2 * r`. |
| `dropout` | `float` | `0.1` | Dropout applied to LoRA layers. Set to `0.0` when overfitting for debugging. |
| `target_modules` | `list[str] \| null` | `null` | Which attention layers to apply LoRA to. `null` = auto-detect based on model architecture. |

**Auto-detected target modules by architecture:**

| Architecture | Target Modules |
|-------------|---------------|
| BERT | `query`, `key`, `value` |
| RoBERTa | `query`, `key`, `value` |
| DistilBERT | `q_lin`, `k_lin`, `v_lin` |
| XLM-RoBERTa | `query`, `key`, `value` |
| DeBERTa (v1/v2) | `query_proj`, `key_proj`, `value_proj` |
| Mistral | `q_proj`, `k_proj`, `v_proj` |
| LLaMA | `q_proj`, `k_proj`, `v_proj` |

If your model architecture is not listed, set `target_modules` explicitly in the config.

#### `train`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | `int` | `3` | Number of passes over the training data. |
| `batch_size` | `int` | `8` | Micro-batch size. Reduce this if you hit OOM errors (especially on MPS). |
| `grad_accum_steps` | `int` | `4` | Accumulate gradients over N micro-batches before updating. Effective batch size = `batch_size * grad_accum_steps`. |
| `lr` | `float` | `2e-5` | Learning rate for AdamW. Typical range: `1e-5` to `1e-3`. Use higher values (`1e-3`) for overfit debugging. |
| `weight_decay` | `float` | `0.01` | AdamW weight decay. Set to `0.0` to disable. |
| `warmup_steps` | `int` | `100` | Number of linear warmup steps. LR ramps from 0 to `lr`, then linearly decays to 0. |
| `max_grad_norm` | `float` | `1.0` | Gradient clipping threshold (max L2 norm). Prevents exploding gradients. |
| `max_length` | `int` | `512` | Maximum token length. Texts are truncated beyond this. Used consistently for both training and evaluation. |
| `loss` | `str` | `triplet` | Loss function — see [Loss Functions](#loss-functions) below. |
| `margin` | `float` | `0.2` | Margin for triplet loss. Only used when `loss: triplet`. |
| `temperature` | `float` | `0.05` | Temperature for InfoNCE loss. Lower = sharper distribution. Only used when `loss: infonce`. |
| `mixed_precision` | `str \| null` | `null` | Automatic mixed precision: `"fp16"`, `"bf16"`, or `null` (disabled). Reduces memory and speeds up training on CUDA. See [Mixed Precision](#mixed-precision). |
| `sanity_check_samples` | `int` | `10` | Before and after training, sample N triplets and report cosine similarities and margins. Set to `0` to disable. |
| `overfit_batches` | `int \| null` | `null` | Debug mode: train on only N batches for many epochs. Useful to verify the training pipeline works end to end. |
| `save_every_n_steps` | `int \| null` | `null` | Save a checkpoint every N optimizer steps during training. `null` = disabled. |
| `keep_all_checkpoints` | `bool` | `false` | `true` = keep all checkpoints (`checkpoint-100`, `checkpoint-200`, ...). `false` = keep only `checkpoint-latest` (overwritten each time). |

#### `eval`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `str \| null` | `null` | Evaluation dataset — BEIR name or local path. `null` = use `data.dataset`. Set this to evaluate on a **different** dataset than you train on. |
| `k_values` | `list[int]` | `[1, 5, 10]` | K values for computing nDCG@k, MRR@k, and Recall@k. |
| `split` | `str` | `test` | Dataset split to evaluate on (only used for BEIR datasets). |
| `n_queries` | `int \| null` | `null` | Number of queries for evaluation. `null` = all. |
| `corpus_size` | `int \| null` | `null` | Corpus size for evaluation. Relevant docs are always included + random filler. `null` = full. |
| `run_before` | `bool` | `true` | Evaluate baseline model before fine-tuning. |
| `run_after` | `bool` | `true` | Evaluate fine-tuned model after training. |

#### `seed`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int \| null` | `null` | Global random seed for reproducibility. Sets `torch`, `random`, and `numpy` seeds. `null` = non-deterministic. |

#### `output_dir`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` | `./forge-output` | Where to save adapter weights, config snapshot, training history, and evaluation results. |

**Output directory structure:**
```
forge-output/
  config.yaml            # Saved config for reproducibility
  train_history.json     # Training curves (step_loss, step_lr, step_grad_norm, epoch_loss)
  adapter/               # Final LoRA adapter weights (from last mining round)
    adapter_model.safetensors
    adapter_config.json
    checkpoint-latest/   # Intermediate checkpoint (if save_every_n_steps is set)
    checkpoint-100/      # Named checkpoints (if keep_all_checkpoints: true)
    checkpoint-200/
  adapter_r1/            # Round 1 adapter (only when mining_rounds > 1)
  adapter_r2/            # Round 2 adapter (only when mining_rounds > 2)
  baseline.json          # Baseline eval metrics (if run_before: true)
  finetuned.json         # Fine-tuned eval metrics (if run_after: true)
```

---

## Loss Functions

### Triplet Margin Loss (`loss: triplet`)

```
L = mean(relu(d(query, positive) - d(query, negative) + margin))
```

Where `d` is cosine distance (`1 - cosine_similarity`). Pushes positive pairs closer and negative pairs further apart by at least `margin`.

- **When to use:** Good default. Works well with random negatives and small batch sizes.
- **Key parameter:** `margin` (default: `0.2`)

### InfoNCE Loss (`loss: infonce`)

```
L = -log(exp(sim(q, p+) / t) / sum(exp(sim(q, pi) / t)))
```

Cross-entropy over in-batch positives with all other positives and explicit hard negatives as distractors. Leverages in-batch negatives for richer training signal.

- **When to use:** Best with larger batch sizes and hard negatives. Typically strongest performance.
- **Key parameter:** `temperature` (default: `0.05`). Lower = sharper, more discriminative.

### Contrastive Loss (`loss: contrastive`)

```
L = mean(-cos_sim(query, positive) + cos_sim(query, negative))
```

Direct optimization of cosine similarity without margin or temperature.

- **When to use:** Simple baseline. No hyperparameters to tune beyond learning rate.

---

## Custom Datasets

Training and evaluation datasets are **fully independent**. You can mix and match BEIR datasets, local datasets, and programmatic datasets in any combination.

### Dataset format

Create a directory with three files:

```
my_dataset/
  queries.jsonl      # One JSON object per line
  corpus.jsonl       # One JSON object per line
  qrels.tsv          # Tab-separated, no header
```

**queries.jsonl** — one query per line:
```json
{"_id": "q1", "text": "What is compound interest?"}
{"_id": "q2", "text": "How do index funds work?"}
```

**corpus.jsonl** — one document per line (`title` is optional, gets prepended to `text`):
```json
{"_id": "d1", "text": "Compound interest is interest on interest.", "title": "Compound Interest"}
{"_id": "d2", "text": "An index fund tracks a market index like the S&P 500."}
```

**qrels.tsv** — relevance judgments (tab-separated: `query_id`, `doc_id`, `score`). No header row:
```
q1	d1	1
q2	d2	1
```

### Using separate train and eval datasets (YAML)

The `data.dataset` and `eval.dataset` fields are independent. You can train on one dataset and evaluate on another:

```yaml
# Train on your domain data, evaluate on a BEIR benchmark
data:
  dataset: ./my_domain_data       # local custom dataset for training
  negatives: random
  n_negatives: 1

eval:
  dataset: fiqa                    # BEIR dataset for evaluation (different from training!)
  split: test
  run_before: true
  run_after: true
```

```yaml
# Train on BEIR, evaluate on your own held-out test set
data:
  dataset: fiqa
  split: train

eval:
  dataset: ./my_test_set           # local custom dataset for evaluation
  run_before: true
  run_after: true
```

```yaml
# Train and evaluate on different local datasets
data:
  dataset: ./my_train_data

eval:
  dataset: ./my_eval_data
```

If `eval.dataset` is omitted, it defaults to `data.dataset` (same dataset for both).

### Programmatic: bring your own data in any format

The library doesn't force you to use JSONL files or BEIR downloads. Every component accepts a `RetrievalDataset`, which is just three dicts. If your data lives in a database, CSV, Parquet, API, or anywhere else, just build the dicts yourself:

```python
from khoji import RetrievalDataset

# RetrievalDataset is just three dicts — build it however you want
dataset = RetrievalDataset(
    queries={"q1": "What is compound interest?", "q2": "How do index funds work?"},
    corpus={"d1": "Compound interest is ...", "d2": "An index fund ...", "d3": "Unrelated doc"},
    qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
)
```

See [Writing a custom training script](#writing-a-custom-training-script) for a complete example of using this with the full pipeline.

---

## Python API

The library is designed for programmatic use. All key classes and functions are importable from the top-level package. Every component works independently — you can use the full `run()` pipeline, or pick individual pieces and compose your own workflow.

### Full pipeline

```python
from khoji import ForgeConfig, run

config = ForgeConfig.from_yaml("configs/fiqa_quick.yaml")
result = run(config)

# result.history     -> TrainHistory (step_loss, step_lr, step_grad_norm, epoch_loss)
# result.baseline    -> EvalResult or None
# result.finetuned   -> EvalResult or None
# result.adapter_dir -> str (path to saved adapter)
# result.config      -> ForgeConfig
```

### Component-by-component pipeline

Each component can be used independently. Here's the standard flow broken down:

```python
from khoji import (
    EmbeddingModel,
    Evaluator,
    LoRASettings,
    Trainer,
    TrainingConfig,
    TripletDataset,
    build_mixed_negatives,
    build_random_negatives,
    load_beir,
    mine_hard_negatives,
)

# 1. Load dataset
dataset = load_beir("fiqa", split="train")
print(f"Queries: {len(dataset.queries)}, Corpus: {len(dataset.corpus)}")

# 2. Build training triplets — three strategies:

# Option A: Random negatives (fast, no model needed)
triplets = build_random_negatives(dataset, n_negatives=3, n_queries=100)

# Option B: Hard negatives (requires encoding the corpus)
# model = EmbeddingModel("BAAI/bge-base-en-v1.5")
# triplets = mine_hard_negatives(dataset, model, n_negatives=3, top_k=50)

# Option C: Mixed — random + hard combined (recommended)
# model = EmbeddingModel("BAAI/bge-base-en-v1.5")
# triplets = build_mixed_negatives(dataset, model, n_random=2, n_hard=1, top_k=50)

torch_ds = TripletDataset(triplets)

# 3. Configure and train
config = TrainingConfig(
    epochs=3,
    batch_size=4,
    lr=2e-5,
    mixed_precision="bf16",            # AMP for faster training
    dtype="bf16",                      # load base model in bf16 to save memory
    lora=LoRASettings(r=8, alpha=16),
    save_dir="./my-adapter",
    save_every_n_steps=100,            # checkpoint every 100 optimizer steps
    keep_all_checkpoints=False,        # only keep latest checkpoint
)
trainer = Trainer("BAAI/bge-base-en-v1.5", config)
history = trainer.train(torch_ds)

# 4. Evaluate
evaluator = Evaluator("BAAI/bge-base-en-v1.5", adapter_path="./my-adapter", dtype="bf16")
result = evaluator.evaluate("fiqa", split="test", k_values=[1, 5, 10])
result.print()
result.save("eval_results.json")

# 5. Use for inference
model = EmbeddingModel("BAAI/bge-base-en-v1.5", adapter_path="./my-adapter", dtype="bf16")
query_emb = model.encode(["What is compound interest?"])
doc_emb = model.encode(["Compound interest is interest calculated on the initial principal..."])

# Compute similarity
import torch
similarity = torch.mm(query_emb, doc_emb.t())  # cosine sim (already L2-normalized)
print(f"Similarity: {similarity.item():.4f}")
```

### Writing a custom training script

If your data doesn't come from BEIR or JSONL files, you can skip the data loading entirely and wire things up yourself. The key insight is that every component is independent:

- **`RetrievalDataset`** is just three dicts — build it from any source
- **`build_random_negatives` / `mine_hard_negatives` / `build_mixed_negatives`** take a `RetrievalDataset` and return `list[Triplet]`
- **`TripletDataset`** wraps triplets for PyTorch — or you can construct `Triplet` objects directly
- **`Trainer`** takes a `TripletDataset` and returns `TrainHistory`
- **`Evaluator.evaluate()`** accepts a `RetrievalDataset` directly via the `dataset=` parameter

Here's an example loading data from a pandas DataFrame and a CSV:

```python
import pandas as pd
from khoji import (
    RetrievalDataset,
    Evaluator,
    LoRASettings,
    Trainer,
    TrainingConfig,
    TripletDataset,
    build_mixed_negatives,
    build_random_negatives,
    mine_hard_negatives,
    EmbeddingModel,
)
from khoji.data import Triplet

# ──────────────────────────────────────────────────────────
# 1. Load YOUR data from whatever source you have
# ──────────────────────────────────────────────────────────

# Example: customer support tickets from a CSV
tickets = pd.read_csv("support_tickets.csv")  # columns: ticket_id, question, resolution
kb = pd.read_csv("knowledge_base.csv")        # columns: article_id, content
labels = pd.read_csv("labels.csv")            # columns: ticket_id, article_id, relevance

# Build RetrievalDataset from your dataframes
train_dataset = RetrievalDataset(
    queries={str(r.ticket_id): r.question for r in tickets.itertuples()},
    corpus={str(r.article_id): r.content for r in kb.itertuples()},
    qrels={
        str(tid): {str(aid): int(rel) for _, aid, rel in group.itertuples()}
        for tid, group in labels.groupby("ticket_id")
    },
)

# ──────────────────────────────────────────────────────────
# 2. Build triplets — choose your strategy
# ──────────────────────────────────────────────────────────

# Option A: Random negatives (fast, no model needed)
triplets = build_random_negatives(train_dataset, n_negatives=3)

# Option B: Hard negatives (better quality, needs model encoding)
# base_model = EmbeddingModel("BAAI/bge-base-en-v1.5")
# triplets = mine_hard_negatives(train_dataset, base_model, n_negatives=3, top_k=50)

# Option C: Mixed — random + hard combined (recommended for balanced training)
# base_model = EmbeddingModel("BAAI/bge-base-en-v1.5")
# triplets = build_mixed_negatives(train_dataset, base_model, n_random=2, n_hard=1, top_k=50)

# Option D: Build triplets yourself if you have your own mining logic
# triplets = [
#     Triplet(query="...", positive="...", negative="...")
#     for query, pos, neg in your_custom_mining_function()
# ]

# ──────────────────────────────────────────────────────────
# 3. Train
# ──────────────────────────────────────────────────────────

config = TrainingConfig(
    epochs=5,
    batch_size=4,
    grad_accum_steps=4,        # effective batch size = 16
    lr=2e-5,
    weight_decay=0.01,         # AdamW weight decay
    warmup_steps=50,
    max_grad_norm=1.0,         # gradient clipping
    mixed_precision="bf16",    # AMP for faster training
    dtype="bf16",              # load base model in bf16 to save memory
    lora=LoRASettings(r=16, alpha=32),
    save_dir="./support-adapter",
    sanity_check_samples=10,   # verify training is working
    save_every_n_steps=200,    # checkpoint every 200 optimizer steps
    keep_all_checkpoints=False,# only keep latest
)

trainer = Trainer("BAAI/bge-base-en-v1.5", config)
history = trainer.train(TripletDataset(triplets))

# Plot training curves
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history.step_loss)
axes[0].set_title("Loss per step")
axes[1].plot(history.step_lr)
axes[1].set_title("Learning rate")
axes[2].plot(history.step_grad_norm)
axes[2].set_title("Gradient norm")
plt.tight_layout()
plt.savefig("training_curves.png")

# ──────────────────────────────────────────────────────────
# 4. Evaluate — on your own test set OR a standard benchmark
# ──────────────────────────────────────────────────────────

evaluator = Evaluator("BAAI/bge-base-en-v1.5", adapter_path="./support-adapter", dtype="bf16")

# Evaluate on your own held-out data (pass dataset directly)
test_dataset = RetrievalDataset(
    queries={"t1": "how to reset password", "t2": "refund policy"},
    corpus={...},   # your test corpus
    qrels={...},    # your test relevance judgments
)
custom_result = evaluator.evaluate(dataset=test_dataset, k_values=[1, 5, 10])
custom_result.print()

# Also evaluate on a standard BEIR benchmark for comparison
beir_result = evaluator.evaluate(dataset_name="fiqa", split="test", k_values=[1, 5, 10])
beir_result.print()

# ──────────────────────────────────────────────────────────
# 5. Use the fine-tuned model for inference
# ──────────────────────────────────────────────────────────

model = EmbeddingModel("BAAI/bge-base-en-v1.5", adapter_path="./support-adapter", dtype="bf16")

# Encode a user query and your knowledge base
query_emb = model.encode(["How do I reset my password?"])
kb_embs = model.encode(["To reset your password, go to ...", "Refund policy: ...", ...])

# Find most relevant articles
import torch
scores = torch.mm(query_emb, kb_embs.t()).squeeze(0)
top_indices = torch.topk(scores, k=5).indices.tolist()
# top_indices now contains the indices of the 5 most relevant KB articles
```

### `RunResult`

| Field | Type | Description |
|-------|------|-------------|
| `history` | `TrainHistory` | Training metrics per step and per epoch |
| `baseline` | `EvalResult \| None` | Baseline evaluation (None if `run_before: false`) |
| `finetuned` | `EvalResult \| None` | Fine-tuned evaluation (None if `run_after: false`) |
| `adapter_dir` | `str \| None` | Path to saved LoRA adapter |
| `config` | `ForgeConfig \| None` | The config used for this run |

### `TrainHistory`

| Field | Type | Description |
|-------|------|-------------|
| `step_loss` | `list[float]` | Loss per optimizer step |
| `step_lr` | `list[float]` | Learning rate per optimizer step |
| `step_grad_norm` | `list[float]` | Gradient L2 norm per optimizer step |
| `epoch_loss` | `list[float]` | Average loss per epoch |

### `EvalResult`

| Field | Type | Description |
|-------|------|-------------|
| `metrics` | `dict[str, float]` | Metric name to value (e.g., `{"ndcg@1": 0.45, "mrr@5": 0.52}`) |
| `model_name` | `str` | Model used |
| `dataset_name` | `str` | Dataset evaluated on |
| `split` | `str` | Split evaluated on |
| `num_queries` | `int` | Number of queries evaluated |
| `num_corpus` | `int` | Corpus size |
| `k_values` | `list[int]` | K values used |
| `timestamp` | `str` | ISO timestamp |

| Method | Description |
|--------|-------------|
| `print()` | Pretty-print results as a formatted table |
| `save(path)` | Save to JSON file |
| `to_dict()` | Convert to dictionary with all metadata |

---

## Supported Models

Any HuggingFace model compatible with `AutoModel` / `AutoTokenizer` is supported. The library auto-detects:

- **Pooling strategy** from the model's `1_Pooling/config.json` (sentence-transformers convention). Supported modes: CLS, Mean, Max, WeightedMean, LastToken, Mean-sqrt-len. Falls back to CLS if config is not found.
- **LoRA target modules** from model architecture (see [LoRA config](#lora) table above).

**Tested models:**

| Model | Pooling | Architecture |
|-------|---------|-------------|
| `BAAI/bge-base-en-v1.5` | CLS | BERT |
| `sentence-transformers/all-MiniLM-L6-v2` | Mean | BERT |

---

## Extensibility

Every major component is pluggable. You can bring your own model, loss function, or metrics without forking the library.

### Full fine-tuning (no LoRA)

By default, khoji uses LoRA for parameter-efficient training. To train **all model parameters** instead, set `lora: null` in your YAML config:

```yaml
lora: null    # train all parameters, not just LoRA adapters

train:
  epochs: 3
  lr: 1e-5    # use a lower learning rate for full fine-tuning
```

Or via the Python API:

```python
from khoji import Trainer, TrainingConfig

config = TrainingConfig(
    epochs=3,
    lr=1e-5,
    lora=None,    # full fine-tuning
    save_dir="./my-full-model",
)
trainer = Trainer("BAAI/bge-base-en-v1.5", config)
```

**Notes:**
- Full fine-tuning trains and saves **all** model weights (hundreds of MB), whereas LoRA only saves adapter weights (~few MB).
- Use a lower learning rate (`1e-5` to `5e-6`) compared to LoRA fine-tuning to avoid catastrophic forgetting.
- Requires more GPU memory since all parameters need gradients.

### Custom models (non-HuggingFace)

If you have a custom PyTorch encoder (not from HuggingFace), pass it directly to `EmbeddingModel`, `Trainer`, and `Evaluator`. The only requirement is:

- **Model**: a `torch.nn.Module` that returns an object with a `.last_hidden_state` attribute (shape `batch, seq_len, hidden_dim`) when called with tokenizer outputs.
- **Tokenizer**: anything that supports `tokenizer(texts, padding=True, truncation=True, max_length=N, return_tensors="pt")` and returns a dict with `"attention_mask"`.

```python
import torch
import torch.nn as nn
from khoji import EmbeddingModel, Trainer, TrainingConfig, Evaluator

# Example: a simple custom encoder
class MyEncoder(nn.Module):
    def __init__(self, vocab_size=30000, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=4,
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        # Return an object with .last_hidden_state (like HuggingFace models)
        return type("Output", (), {"last_hidden_state": x})()

my_model = MyEncoder()
my_tokenizer = ...  # your tokenizer

# Use for inference
embedding_model = EmbeddingModel(
    model=my_model,
    tokenizer=my_tokenizer,
    pooling="mean",       # "cls", "mean", "max", "weightedmean", "lasttoken", "mean_sqrt_len"
    max_length=512,       # max token length
)
embeddings = embedding_model.encode(["hello world"])

# Use for training (set lora=None to train the full model, or keep LoRA)
trainer = Trainer(
    model=my_model,
    tokenizer=my_tokenizer,
    pooling="mean",
    config=TrainingConfig(
        epochs=3,
        lora=None,              # no LoRA for custom models (trains full model)
        mixed_precision="bf16", # AMP still works with custom models
    ),
)

# Use for evaluation
evaluator = Evaluator(embedding_model=embedding_model)
result = evaluator.evaluate(dataset_name="fiqa", split="test")
```

### Custom loss functions

The `TrainingConfig.loss_fn` accepts any callable with the signature:

```python
def my_loss(
    query_emb: torch.Tensor,     # (batch, dim), L2-normalized
    positive_emb: torch.Tensor,  # (batch, dim), L2-normalized
    negative_emb: torch.Tensor,  # (batch, dim), L2-normalized
) -> torch.Tensor:               # scalar loss
```

```python
import torch
from khoji import Trainer, TrainingConfig, LoRASettings

# Example: custom circle loss
def circle_loss(query_emb, positive_emb, negative_emb, margin=0.25, gamma=64):
    pos_sim = torch.nn.functional.cosine_similarity(query_emb, positive_emb)
    neg_sim = torch.nn.functional.cosine_similarity(query_emb, negative_emb)

    alpha_p = torch.clamp(1 + margin - pos_sim, min=0)
    alpha_n = torch.clamp(neg_sim + margin, min=0)

    logit_p = -gamma * alpha_p * (pos_sim - (1 - margin))
    logit_n = gamma * alpha_n * (neg_sim - margin)

    loss = torch.nn.functional.softplus(logit_n - logit_p)
    return loss.mean()

# Use it
config = TrainingConfig(
    loss_fn=circle_loss,  # pass your function directly
    lora=LoRASettings(r=8, alpha=16),
)
trainer = Trainer("BAAI/bge-base-en-v1.5", config)
```

Through YAML, only the built-in losses (`triplet`, `infonce`, `contrastive`) are available. Custom losses require the Python API.

### Custom metrics

The `Evaluator.evaluate()` accepts `extra_metrics` — a dict of metric functions that are computed alongside the built-in nDCG, MRR, and Recall. Each function must have the signature:

```python
def my_metric(
    ranked_doc_ids: list[str],     # doc IDs ranked by similarity (most similar first)
    qrel: dict[str, int],         # ground truth: {doc_id: relevance_score}
    k: int,                        # cutoff
) -> float:                        # per-query score (averaged across all queries)
```

```python
from khoji import Evaluator

# Example metrics
def precision_at_k(ranked_doc_ids, qrel, k):
    """Fraction of top-k results that are relevant."""
    relevant = {d for d, s in qrel.items() if s > 0}
    found = sum(1 for d in ranked_doc_ids[:k] if d in relevant)
    return found / k

def hit_rate_at_k(ranked_doc_ids, qrel, k):
    """1 if any relevant doc is in top-k, else 0."""
    relevant = {d for d, s in qrel.items() if s > 0}
    return 1.0 if any(d in relevant for d in ranked_doc_ids[:k]) else 0.0

# Use them
evaluator = Evaluator("BAAI/bge-base-en-v1.5", adapter_path="./my-adapter")
result = evaluator.evaluate(
    dataset_name="fiqa",
    split="test",
    k_values=[1, 5, 10],
    extra_metrics={
        "precision": precision_at_k,
        "hit_rate": hit_rate_at_k,
    },
)

# Results include both built-in and custom metrics
# {"ndcg@5": 0.42, "mrr@5": 0.51, "recall@5": 0.65, "precision@5": 0.12, "hit_rate@5": 0.78, ...}
print(result.metrics)
```

The built-in metric functions (`ndcg_at_k`, `mrr_at_k`, `recall_at_k`) are also exported from the package, so you can reuse them in custom evaluation scripts:

```python
from khoji import ndcg_at_k, mrr_at_k, recall_at_k

ranked = ["d3", "d1", "d5", "d2"]
qrel = {"d1": 2, "d5": 1}

print(ndcg_at_k(ranked, qrel, k=3))   # 0.7654
print(mrr_at_k(ranked, qrel, k=3))    # 0.5
print(recall_at_k(ranked, qrel, k=3)) # 1.0
```

---

## Example Configs

Four configs are included for different use cases:

### `configs/fiqa_quick.yaml` — Quick iteration
- 50 training queries, random negatives
- 2 epochs, batch_size=4, grad_accum_steps=4
- Evaluation disabled for speed
- Good for testing config changes and verifying the pipeline

### `configs/fiqa_full.yaml` — Full training
- All queries, hard negative mining (top_k=50, n_negatives=3)
- 5 epochs, batch_size=32, InfoNCE loss
- LoRA rank=16, alpha=32
- Full baseline + fine-tuned evaluation on all test queries

### `configs/fiqa_mixed.yaml` — Mixed negatives
- All queries, 2 random + 1 hard negative per pair
- 2 epochs, batch_size=32, InfoNCE loss
- LoRA rank=16, alpha=32, target_modules: [query, key, value, dense]
- Random negatives provide easy training signal, hard negatives push fine-grained ranking

### `configs/fiqa_overfit.yaml` — Debug pipeline
- 1 batch, 50 epochs, lr=1e-3, no dropout
- Verifies training can drive loss to zero and margins improve
- Reports per-sample cosine similarity before and after training
- No evaluation (pure training loop verification)

---

## Evaluation Metrics

All metrics are implemented from scratch (no external IR evaluation libraries).

| Metric | Description |
|--------|-------------|
| **nDCG@k** | Normalized Discounted Cumulative Gain. Measures ranking quality with graded relevance. Accounts for position — relevant docs ranked higher contribute more. 1.0 = perfect ranking. |
| **MRR@k** | Mean Reciprocal Rank. 1 / position of the first relevant result. Focuses on where the first relevant result appears. 1.0 = relevant doc is the first result. |
| **Recall@k** | Fraction of all relevant documents found in the top-k results. Measures coverage. 1.0 = all relevant docs retrieved within top-k. |

---

## Hardware

The library auto-detects the best available device:

| Device | Priority | Notes |
|--------|----------|-------|
| CUDA | 1st | NVIDIA GPUs. Best performance. |
| MPS | 2nd | Apple Silicon (M1/M2/M3). Uses shared GPU/CPU memory. |
| CPU | 3rd | Fallback. Slower but always works. |

**MPS tips:**
- If you see `MPS backend out of memory`, reduce `batch_size` (e.g., to 4) and increase `grad_accum_steps` to maintain effective batch size.
- The effective batch size (`batch_size * grad_accum_steps`) is what matters for training dynamics. Smaller micro-batches with more accumulation steps gives equivalent results with less peak memory.

### Model Precision

There are **two independent precision controls** — one for how model weights are stored in memory, and one for how computations are done during training:

#### `model.dtype` — Base model weight precision

Controls the precision of the **base model weights** when loaded from HuggingFace. LoRA adapter weights are always kept in fp32 for training stability.

```yaml
model:
  name: BAAI/bge-base-en-v1.5
  dtype: bf16    # load base model in bfloat16 (saves ~50% memory)
```

| Value | Memory | Description |
|-------|--------|-------------|
| `null` | 100% | Default. Full fp32 precision. |
| `fp16` | ~50% | Half precision. Good for inference and memory-constrained setups. |
| `bf16` | ~50% | BFloat16. Same range as fp32, recommended for training. |

This setting applies everywhere the base model is loaded: training, evaluation, hard negative mining, and inference.

#### `train.mixed_precision` — Training computation precision

Controls automatic mixed precision (AMP) during the **forward/backward pass**. This is separate from `model.dtype`:

```yaml
train:
  mixed_precision: bf16   # or "fp16"
```

| Mode | When to use |
|------|-------------|
| `bf16` | Recommended for modern GPUs (Ampere+). Same dynamic range as fp32, no grad scaling needed. |
| `fp16` | Wider GPU support. Uses gradient scaling automatically to prevent underflow. |
| `null` | Default. Full fp32 precision. |

#### Combining both

You can use both together for maximum memory savings:

```yaml
model:
  dtype: bf16              # load base weights in bf16
train:
  mixed_precision: bf16    # run forward/backward in bf16
```

**Notes:**
- On **CUDA**, both `fp16` and `bf16` work. `bf16` is preferred on Ampere+ GPUs (A100, RTX 3090+).
- On **MPS** (Apple Silicon), `bf16` has limited support. Use with caution or stick with `null`.
- On **CPU**, mixed precision has minimal benefit.
- Gradient scaling is handled automatically for `fp16` (via `torch.amp.GradScaler`). `bf16` does not require scaling.

---

## Architecture

```
khoji/
  __init__.py          # Public API exports
  config.py            # YAML config dataclasses (ForgeConfig)
  run.py               # Orchestration: data prep -> train -> eval (RunResult)
  dataset.py           # BEIR dataset loading (load_beir, RetrievalDataset)
  data.py              # Triplet construction, hard negative mining
  model.py             # Embedding model with pooling auto-detection
  lora.py              # LoRA configuration and application
  trainer.py           # Training loop with grad accumulation, clipping, scheduling
  loss.py              # Loss functions (triplet, infonce, contrastive)
  evaluator.py         # Retrieval evaluation (nDCG, MRR, Recall)
  metrics.py           # Individual metric implementations
  device.py            # Hardware auto-detection (CUDA > MPS > CPU)
```

### Data flow

```
YAML Config
    |
    v
ForgeConfig.from_yaml()
    |
    v
load_beir() ──> RetrievalDataset (queries, corpus, qrels)
    |
    v
┌─────────────────── mining round loop ───────────────────┐
│                                                          │
│  build_random_negatives()   ──> list[Triplet]            │
│    or mine_hard_negatives()                              │
│    or build_mixed_negatives()                            │
│    (round 2+ uses fine-tuned model for mining)           │
│       |                                                  │
│       v                                                  │
│  Trainer.train() ──> TrainHistory + adapter saved        │
│       |                                                  │
│       └──── adapter feeds next round's mining ───────────┘
    |
    v
Evaluator.evaluate() ──> EvalResult (nDCG, MRR, Recall @ k)
    |
    v
RunResult (history + baseline + finetuned + adapter_dir)
```

---

## Development

### Running tests

```bash
uv run pytest tests/ -v
```

### Test coverage

| Module | What's tested |
|--------|--------------|
| `metrics.py` | nDCG@k, MRR@k, Recall@k — edge cases, perfect/worst rankings, graded relevance, k cutoffs |
| `model.py` | All 6 pooling modes (with/without padding), pooling auto-detection, embedding shape and L2 normalization |
| `data.py` | TripletDataset, random negatives (determinism, relevance correctness), hard negatives (semantic similarity) |
| `device.py` | Device selection returns valid device, tensor creation works on selected device |
| `integration` | BEIR dataset loading, retrieval sanity checks (relevant docs rank higher than random) |

### Linting

```bash
uv run ruff check src/ tests/
```

---

## Roadmap

Planned features and known gaps:

- [ ] **Validation loss tracking** — monitor loss on a held-out set during training
- [ ] **Early stopping** — stop training when validation metric stops improving
- [ ] **Distributed training** — multi-GPU support via DDP
- [ ] **Checkpoint resumption** — resume training from a saved checkpoint
- [ ] **Adapter merging** — merge LoRA weights back into base model for faster inference
- [ ] **Benchmark suite** — automated evaluation across multiple BEIR datasets
- [ ] **Logging integration** — Weights & Biases, TensorBoard support
- [ ] **Multi-positive contrastive learning** — leverage multiple positives per query in a single forward pass

---

## License

MIT
