# Your Retrieval Model Doesn't Understand Your Data. Here's How to Fix It.

*A practitioner's guide to fine-tuning retrieval models — from understanding why pretrained models fail on domain-specific data, to building a pipeline that actually works.*

---

You've built a search system, a RAG pipeline, or a recommendation engine. You plugged in a pretrained embedding model — BGE, CLIP, or something from the OpenAI API. It works okay on generic queries.

Then someone searches for "duty of care in negligence claims" in your legal corpus, and the top result is about employment duties. Or your satellite monitoring system can't tell a river delta from farmland. Or your fashion app can't handle "find me this dress but in blue."

The model isn't broken. It was just never trained on your domain.

This post walks through everything involved in fixing that — from understanding *why* pretrained models fail, to choosing a training strategy, to actually running the experiments. We use [khoji](https://github.com/suyashh94/khoji) throughout because it handles the full pipeline, but the concepts apply regardless of tooling.

---

## The Journey from "My Search Doesn't Work" to a Working Retriever

Fine-tuning a retrieval model isn't like fine-tuning a classifier. There are no fixed labels. Instead, you have queries, a corpus, and a notion of relevance — and you need to teach a model to encode both into the same embedding space so that similar things land near each other.

That involves answering a chain of questions:

1. **What kind of retrieval do I need?** Text search? Image search? "Find this but different"?
2. **What model should I start from?** And how much of it should I change?
3. **How do I create training data?** What makes a good training example?
4. **What should the model optimize for?** Which loss function? What margin? What temperature?
5. **How do I know if it's working?** What metrics, evaluated on what data?
6. **How do I iterate?** First attempt rarely works perfectly — what do I adjust?

Let's work through each one.

---

## Step 1: What Kind of Retrieval Do You Need?

Before you touch any code, you need to know what type of query-to-result matching you're building. This determines everything downstream — the model architecture, the data format, and how embeddings work.

![Three retrieval modes](./figures/three_modes_architecture.png)

### Text → Text

The query is text. The corpus is text. You need the model to understand that a question like "What is compound interest?" should match a document explaining how compound interest works, not one about simple interest.

**When this comes up:** Document search, FAQ matching, semantic search over knowledge bases, RAG retrieval, legal discovery, code search.

**Models:** BERT, BGE, MiniLM, any sentence-transformer. khoji auto-detects the pooling strategy (CLS, mean, max) and LoRA target modules.

### Text → Image

The query is text. The corpus is images. The model needs to bridge the gap between language and vision — understanding that "a river flowing through farmland" should match an aerial photograph showing exactly that.

**When this comes up:** Image search, content discovery, catalog search, multimodal RAG.

**Models:** CLIP, SigLIP. These have separate text and vision encoders that project into a shared embedding space. khoji lets you fine-tune one or both encoders.

### (Image + Text) → Image

The query is a **pair**: a reference image and a modification caption. "Here's a red dress — find me one like this but in blue and longer." The model must understand both what to keep from the image and what to change from the text.

![Composed retrieval concept](./figures/07_composed_retrieval_concept.png)

**When this comes up:** Fashion search, interior design, creative tools, visual recommendation with modification intent.

**Models:** BLIP-2 (joint image-text encoder with Q-Former). khoji fine-tunes the Q-Former — the component that bridges vision and language.

khoji supports all three modes with the same workflow: load data → mine negatives → train → evaluate. The data format differs, but the pipeline is the same.

---

## Step 2: Choosing Your Base Model and What to Change

You don't train from scratch. You start from a pretrained model that already understands language (or vision) in general, and adapt it to understand your specific domain.

### How Much Should You Change?

![LoRA: Low-Rank Adaptation](./figures/lora_architecture.png)

**Full fine-tuning** updates every parameter. Powerful, but expensive — hundreds of MB of weights, risk of catastrophic forgetting, needs more data and lower learning rates.

**LoRA** (Low-Rank Adaptation) inserts small trainable matrices into the attention layers while keeping everything else frozen. Only ~0.1% of parameters are trained. The adapter is ~2MB. The base model retains its general capabilities and you can hot-swap adapters for different domains.

khoji defaults to LoRA. The key parameters:

| Parameter | What to set | Guidance |
|-----------|------------|----------|
| `r` (rank) | 8 for most tasks, 16 for harder domains | Higher = more capacity but slower |
| `alpha` | 2 * r (convention) | Scales the LoRA contribution |
| `dropout` | 0.1 (production), 0.0 (debug) | Regularization |
| `target_modules` | null (auto-detect) | Explicitly set for non-standard architectures |

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: null   # auto: query/key/value for BERT, q_proj/k_proj/v_proj for CLIP
```

For full fine-tuning, set `lora: null` and drop the learning rate to 1e-5.

### For Multimodal: Which Encoder to Adapt?

With CLIP/SigLIP, you choose which side to fine-tune:

| `lora_target` | When to use |
|---------------|-------------|
| `"both"` | Default. The domain is novel for both vision and text. |
| `"vision"` | Text queries are standard, but images are domain-specific (satellite, medical). |
| `"text"` | Images are generic, but queries use domain jargon. |

---

## Step 3: Creating Training Data — The Hardest Part

This is where most retrieval fine-tuning efforts succeed or fail. You need **triplets**: (query, relevant item, non-relevant item). The quality of your negatives determines the quality of your model.

### Your Data Is Just Three Dicts

Every dataset in khoji is the same structure:

| Component | Text | Multimodal | Composed |
|-----------|------|------------|----------|
| **queries** | `{id: text}` | `{id: text}` | `{id: (image_path, text)}` |
| **corpus** | `{id: text}` | `{id: image_path}` | `{id: image_path}` |
| **qrels** | `{query_id: {doc_id: score}}` | same | same |

You can build these from anything — JSONL files, CSV, a database, a dataframe:

```python
# From local files
dataset = load_custom("./my_dataset")          # text
dataset = load_custom_multimodal("./my_data")  # images
dataset = load_custom_composed("./my_data")    # composed

# From BEIR benchmarks
dataset = load_beir("fiqa", split="train")

# From Python dicts (any source)
dataset = RetrievalDataset(
    queries={"q1": "What is compound interest?"},
    corpus={"d1": "Compound interest is...", "d2": "Unrelated."},
    qrels={"q1": {"d1": 1}},
)
```

### The Negative Mining Problem

The query and positive item come from your relevance labels. But what about the negative? This is the critical decision.

![How negative mining strategies differ](./figures/negative_mining_visual.png)

**Random negatives** — sample from the corpus at random. Fast, no model needed. But these are easy: the model already knows a financial document isn't a cooking recipe. Good for getting started, but the model plateaus quickly.

```yaml
data:
  negatives: random
  n_negatives: 3
```

**Hard negatives** — use the model to encode everything, find the items it currently confuses with relevant ones, and train on those. Forces fine-grained distinctions. But there's a trap: the model's top-ranked "negatives" are often actually relevant items with missing labels. That's where `skip_top` comes in — skip the most suspicious candidates.

```yaml
data:
  negatives: hard
  n_negatives: 3
  top_k: 50
  skip_top: 5          # skip top 5 likely false negatives
```

**Mixed negatives** — the sweet spot. Random negatives prevent collapse; hard negatives push the boundary. This is what we recommend for production.

```yaml
data:
  negatives: mixed
  n_random: 2
  n_hard: 1
```

### Iterative Mining: Getting Harder Over Time

After one round of training, the model has improved. What was hard before is now easy. So you mine again — using the fine-tuned model — and train on the new, harder negatives. khoji automates this:

![Iterative mining rounds](./figures/mining_rounds_workflow.png)

```yaml
data:
  negatives: mixed
  mining_rounds: 2     # mine → train → re-mine with improved model → train again
```

Each round halves the learning rate automatically. Two rounds is usually the sweet spot.

---

## Step 4: What Should the Model Optimize?

The loss function defines what "better" means during training. All three take the same input — L2-normalized (query, positive, negative) embeddings — but create different learning signals.

**Triplet Margin Loss** — push positive and negative apart by at least `margin`:

```
L = relu(cos_distance(q, pos) - cos_distance(q, neg) + margin)
```

Simple, works with small batches. Good starting point. `margin: 0.2` is the default.

**InfoNCE Loss** — the strongest option. Treats every other item in the batch as an additional negative (in-batch negatives), giving much richer signal per training step. Essentially asks: "out of all these items, which one is the correct match?"

```
L = -log(exp(sim(q, pos)/t) / sum(exp(sim(q, all_items)/t)))
```

Works best with larger batches. `temperature: 0.05` is the default — lower means sharper.

**Contrastive Loss** — directly maximize positive similarity, minimize negative:

```
L = -cos_sim(q, pos) + cos_sim(q, neg)
```

No hyperparameters beyond learning rate. Good baseline.

```yaml
train:
  loss: infonce         # best overall
  temperature: 0.05
  batch_size: 16        # larger batches → more in-batch negatives
  grad_accum_steps: 4   # effective batch = 64
```

Need something custom? Pass any function via the Python API:

```python
def my_loss(query_emb, positive_emb, negative_emb):
    # Your custom computation
    return scalar_loss

config = TrainingConfig(loss_fn=my_loss)
```

---

## Step 5: How Do You Know It's Working?

### Metrics That Matter for Retrieval

Classification has accuracy. Retrieval has three metrics that each tell you something different:

- **nDCG@k** — Are relevant items ranked high? Accounts for both relevance grade and position. The most holistic metric.
- **MRR@k** — How quickly does the first relevant result appear? Good for "I need one good answer" use cases.
- **Recall@k** — How many of the relevant items are in the top k? Good for "don't miss anything" use cases.

khoji computes all three automatically. You set the k values:

```yaml
eval:
  k_values: [1, 5, 10]
  run_before: true      # baseline (no fine-tuning)
  run_after: true       # after fine-tuning
```

### Baseline-Then-Compare

One of the most important features: khoji evaluates the pretrained model *before* training (`run_before: true`) so you can see the exact delta. If the baseline is already good enough, you save yourself the training cost.

### Custom Metrics

Need precision@k, hit rate, or something domain-specific?

```python
def precision_at_k(ranked_doc_ids, qrel, k):
    relevant = {d for d, s in qrel.items() if s > 0}
    return sum(1 for d in ranked_doc_ids[:k] if d in relevant) / k

result = evaluator.evaluate(
    dataset=my_dataset,
    extra_metrics={"precision": precision_at_k},
)
```

---

## Step 6: Running the Experiments

This is where everything comes together. khoji gives you two paths, and you can mix them freely.

### Path A: Write a YAML, Run One Command

![Two ways to use khoji](./figures/two_abstraction_levels.png)

For the common case — HuggingFace model, standard dataset, standard training — a single YAML config and one function call does everything:

```python
from khoji import ForgeConfig, run
result = run(ForgeConfig.from_yaml("config.yaml"))
```

Or from the CLI:

```bash
khoji configs/minilm_scifact_full.yaml                  # text → text
khoji multimodal configs/clip_rsicd_full.yaml            # text → image
```

Three runner functions, one per mode:
- `run()` → text-to-text
- `run_multimodal()` → text-to-image
- `run_composed()` → composed retrieval

Each returns a `RunResult` with everything you need:

```python
result.history      # TrainHistory: step_loss, step_lr, epoch_loss, grad_norms
result.baseline     # EvalResult: pre-training metrics (or None)
result.finetuned    # EvalResult: post-training metrics (or None)
result.adapter_dir  # path to the saved LoRA adapter
```

### Path B: Compose the Pipeline Yourself

When you need full control — custom data sources, non-standard mining, hyperparameter sweeps, integration with existing infrastructure — use the components directly:

```python
from khoji import (
    EmbeddingModel, Evaluator, Trainer, TrainingConfig,
    TripletDataset, LoRASettings,
    load_beir, build_mixed_negatives,
)

# Each step is independent — swap, skip, or extend as needed
dataset = load_beir("fiqa", split="train")
model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
triplets = build_mixed_negatives(dataset, model, n_random=2, n_hard=1)

config = TrainingConfig(
    epochs=3, batch_size=16, lr=2e-5,
    lora=LoRASettings(r=16, alpha=32),
    save_dir="./my-adapter",
)
trainer = Trainer("sentence-transformers/all-MiniLM-L6-v2", config)
history = trainer.train(TripletDataset(triplets))

evaluator = Evaluator("sentence-transformers/all-MiniLM-L6-v2", adapter_path="./my-adapter")
result = evaluator.evaluate("fiqa", split="test", k_values=[1, 5, 10])
result.print()
```

Every mode has the same component set:

| Component | Text | Multimodal | Composed |
|-----------|------|------------|----------|
| **Dataset** | `RetrievalDataset` | `MultimodalRetrievalDataset` | `ComposedRetrievalDataset` |
| **Mining** | `build_mixed_negatives()` | `build_mixed_negatives_multimodal()` | `build_mixed_negatives_composed()` |
| **Trainer** | `Trainer` | `MultimodalTrainer` | `ComposedTrainer` |
| **Evaluator** | `Evaluator` | `MultimodalEvaluator` | `ComposedEvaluator` |
| **Model** | `EmbeddingModel` | `MultimodalEmbeddingModel` | `JointEmbeddingModel` |

### Bringing Your Own Model

You're not limited to HuggingFace models. Every trainer accepts custom PyTorch modules with encode functions:

```python
# Text → Text: model must return .last_hidden_state
trainer = Trainer(model=my_encoder, tokenizer=my_tok, pooling="mean", config=config)

# Text → Image: provide separate encode functions
trainer = MultimodalTrainer(
    model=my_clip,
    encode_text_fn=my_text_fn,     # list[str] -> Tensor
    encode_image_fn=my_image_fn,   # list[str] -> Tensor (receives file paths)
    config=config,
)

# Composed: encode functions receive PIL images directly
trainer = ComposedTrainer(
    model=my_model,
    encode_query_fn=my_joint_fn,   # (list[PIL], list[str]) -> Tensor
    encode_image_fn=my_img_fn,     # list[PIL] -> Tensor
    config=config,
)
```

---

## What the Experiments Actually Show

We ran three experiments to validate the approach across all three retrieval modes.

### Text → Text: FiQA Financial Q&A

**Setup:** Fine-tune MiniLM (22M params) on financial questions. Compare against BGE-base (110M) as the "large model" reference.

![Text retrieval results](./figures/01_text_retrieval_results.png)

| Model | nDCG@10 | Recall@10 | MRR@10 |
|-------|---------|-----------|--------|
| BGE-base (110M, no fine-tuning) | 0.3909 | 0.4572 | 0.4740 |
| MiniLM (22M, no fine-tuning) | 0.3610 | 0.4325 | 0.4369 |
| **MiniLM (22M, fine-tuned)** | **0.3861** | **0.4527** | **0.4624** |

84% of the nDCG gap closed. A 5x smaller model nearly matches the larger one — at a fraction of the inference cost.

### Text → Image: RSICD Satellite Imagery

**Setup:** Fine-tune CLIP-B/32 (151M) on satellite images. Compare against CLIP-L/14 (428M).

![Multimodal retrieval results](./figures/02_multimodal_retrieval_results.png)

| Model | nDCG@10 | Recall@10 |
|-------|---------|-----------|
| CLIP ViT-L/14 (428M) | 0.1522 | 0.2937 |
| CLIP ViT-B/32 (151M, baseline) | 0.1439 | 0.2715 |
| **CLIP ViT-B/32 (151M, fine-tuned)** | **0.2639** | **0.4776** |

The fine-tuned small model surpasses the large one by 73%. Neither had seen satellite imagery, but fine-tuning on a few thousand examples gives domain knowledge that size alone can't provide.

Here's what that looks like in practice — the same text query, before and after fine-tuning:

![Satellite retrieval example: airport query](./figures/multimodal_example_1.png)

The baseline retrieves vaguely aerial-looking images. The fine-tuned model retrieves actual airports with runways and terminals.

![Satellite retrieval example: residential area](./figures/multimodal_example_2.png)

![Satellite retrieval example: river through farmland](./figures/multimodal_example_3.png)

### Composed Retrieval: FashionIQ Dress

**Setup:** Fine-tune BLIP-2 on "find this dress but different" queries.

![Composed retrieval results](./figures/03_composed_retrieval_results.png)

| Model | Recall@1 | Recall@10 | Recall@50 |
|-------|---------|-----------|-----------|
| BLIP-2 (baseline) | 0.0000 | 0.1489 | 0.2872 |
| **BLIP-2 (fine-tuned)** | **0.0638** | **0.2979** | **0.4574** |

The pretrained model couldn't get a single Recall@1 hit. After fine-tuning, Recall@10 doubled. This is a case where fine-tuning doesn't improve an existing capability — it creates one from scratch.

Here are real examples — the reference image, the modification caption, and the top-5 gallery results before and after:

![Composed example: "is shiny and silver with shorter sleeves" — rank 3 → 1](./figures/composed_example_1.png)

The target dress (shiny, silver, short sleeves) went from rank #3 to rank #1. The fine-tuned model understands the modification intent.

![Composed example: "is grey with black design" — rank 2 → 1](./figures/composed_example_2.png)

### Training Curves

![Training loss curves](./figures/04_training_curves.png)

### Mining Strategy Impact

![Impact of mining strategy](./figures/06_mining_strategy_comparison.png)

The progression from random → hard → mixed → mixed with 2 rounds shows consistent, compounding gains. Each technique builds on the previous one.

---

## Debugging and Iteration

Training rarely works perfectly the first time. khoji provides tools to diagnose and iterate.

### Sanity Checks

Before and after training, khoji samples training triplets and reports cosine similarity:

```
[BEFORE training] Sanity check (10 samples):
    Avg cos_sim(query, pos):  0.4521
    Avg cos_sim(query, neg):  0.4198
    Avg margin (pos - neg):   0.0323
    Samples where pos > neg:  6/10

[AFTER training] Sanity check (10 samples):
    Avg cos_sim(query, pos):  0.7812
    Avg cos_sim(query, neg):  0.2134
    Avg margin (pos - neg):   0.5678
    Samples where pos > neg:  10/10
```

If the margin doesn't improve, something is wrong — bad data, learning rate too low, or the model is already good on these examples.

### Overfit Debugging

Before committing to a full run, verify the pipeline works end-to-end:

```yaml
train:
  overfit_batches: 1     # train on 1 batch only
  epochs: 50             # many epochs to drive loss to ~0
  lr: 1e-3               # high LR for fast convergence
```

If loss doesn't drop to near zero, there's a bug in the data or model setup. khoji includes `_overfit` configs for exactly this.

### Training History

Every run saves per-step metrics for diagnosis:

```python
history.step_loss       # loss per optimizer step
history.step_lr         # learning rate per step
history.step_grad_norm  # gradient norm per step
history.epoch_loss      # average loss per epoch
history.save("train_history.json")
```

---

## What Gets Saved and How to Deploy

### Output Structure

```
output_dir/
  config.yaml                  # saved config (reproducibility)
  train_history.json           # training curves
  adapter/                     # final LoRA adapter (~2-4MB)
    adapter_model.safetensors
    adapter_config.json
  adapter_r1/                  # round 1 adapter (if mining_rounds > 1)
  baseline.json                # pre-training metrics
  finetuned.json               # post-training metrics
```

### Loading for Inference

```python
# Text
model = EmbeddingModel("model-name", adapter_path="./adapter")
embeddings = model.encode(["query text"])

# Multimodal
model = MultimodalEmbeddingModel("clip-model", adapter_path="./adapter")
text_emb = model.encode_text(["search query"])
img_emb = model.encode_image_sources(["photo.jpg"], base_dir="./images")
similarity = torch.mm(text_emb, img_emb.t())

# Composed
model = JointEmbeddingModel("blip2-model", adapter_path="./adapter")
query_emb = model.encode(images=[ref_img], texts=["make it red"])
gallery_emb = model.encode(images=gallery_images)
scores = torch.mm(query_emb, gallery_emb.t())
```

The adapter is 2-4MB. The base model is shared. You can serve multiple domain-specific retrievers from one base model by swapping adapters at request time.

---

## The Complete Parameter Reference

For when you need to tune something specific. Every parameter khoji accepts, across all three modes.

<details>
<summary><strong>Click to expand full parameter reference</strong></summary>

### `model`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | `BAAI/bge-base-en-v1.5` | HuggingFace model ID |
| `adapter_path` | null | Existing adapter for continued training |
| `dtype` | null | `"fp16"`, `"bf16"`, or null (fp32) |
| `lora_target` | `"both"` | Multimodal: `"vision"`, `"text"`, or `"both"` |

### `data`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset` | `fiqa` | BEIR name, HF dataset, or local path |
| `negatives` | `random` | `"random"`, `"hard"`, or `"mixed"` |
| `n_negatives` | 1 | Per pair (random/hard) |
| `n_random` / `n_hard` | 1 / 1 | Per pair (mixed mode) |
| `top_k` | 50 | Hard negative search window |
| `skip_top` | 0 | Skip likely false negatives |
| `mining_rounds` | 1 | Iterative mining rounds |
| `n_queries` | null | Subset of queries (null = all) |
| `corpus_size` | null | Corpus limit for mining (null = all) |

### `lora`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` | 8 | Rank |
| `alpha` | 16 | Scaling (convention: 2*r) |
| `dropout` | 0.1 | LoRA dropout |
| `target_modules` | null | Auto-detected |

### `train`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 3 | Training epochs |
| `batch_size` | 8 | Micro-batch size |
| `grad_accum_steps` | 4 | Effective batch = batch_size * this |
| `lr` | 2e-5 | Learning rate |
| `warmup_steps` | 100 | Linear warmup then decay |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `loss` | `triplet` | `"triplet"`, `"infonce"`, `"contrastive"` |
| `margin` | 0.2 | For triplet loss |
| `temperature` | 0.05 | For InfoNCE |
| `mixed_precision` | null | `"fp16"`, `"bf16"`, or null |
| `overfit_batches` | null | Debug: train on N batches |
| `sanity_check_samples` | 10 | Pre/post training check |

### `eval`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset` | null | Eval dataset (null = training dataset) |
| `k_values` | [1, 5, 10] | Metrics cutoffs |
| `run_before` | true | Baseline evaluation |
| `run_after` | true | Post-training evaluation |

</details>

---

## Getting Started

```bash
pip install khoji

# Text → text
khoji configs/minilm_scifact_full.yaml

# Text → image
khoji multimodal configs/clip_rsicd_full.yaml

# Composed retrieval
python scripts/fashioniq/download_data.py
python scripts/train_composed_retrieval_api.py
```

Four configs are included: `minilm_scifact_full`, `minilm_scifact_overfit`, `clip_rsicd_full`, `clip_rsicd_overfit`. The `_full` configs run complete training + evaluation. The `_overfit` configs verify the pipeline works.

Full documentation, example scripts, and Jupyter notebooks at [github.com/suyashh94/khoji](https://github.com/suyashh94/khoji).
