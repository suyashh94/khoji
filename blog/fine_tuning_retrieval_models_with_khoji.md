# Fine-Tuning Retrieval Models: When It's the Right Call and How to Do It

*A practical guide to choosing your retrieval architecture in 2025 — and where fine-tuned bi-encoders still win.*

---

## The Real Question Isn't "Should I Fine-Tune?"

If you're building a search system, a RAG pipeline, or a recommendation engine, you have more options in 2025 than ever: call an embedding API, use an LLM to re-rank results, prompt GPT-4o to score query-document pairs directly, or fine-tune your own retrieval model. Each has a sweet spot.

The question isn't whether fine-tuning works — it does. The question is whether it's the right architecture for *your* situation. Here's the decision tree:

![Which retrieval architecture do you need?](./figures/08_decision_flow.png)

Fine-tuning a local bi-encoder is the right call in two situations:

1. **Your data can't leave your infrastructure.** Hospitals (HIPAA), law firms (client privilege), defense (classified), EU data (GDPR), edge devices (no internet). An embedding API is not an option. A fine-tuned local model is the only viable architecture.

2. **You need high throughput at low latency.** At 100K+ queries/day, calling an LLM re-ranker on every query is prohibitively expensive. A bi-encoder encodes the query in ~2ms, looks up an ANN index, and returns results — three orders of magnitude cheaper than an LLM call.

If neither applies — if your volume is low and your data isn't sensitive — a pretrained embedding model or an LLM re-ranker is probably simpler and gives better quality per dollar. Be honest about this before investing in fine-tuning.

This post focuses on the cases where fine-tuning *is* the right call. We ran three experiments and used [khoji](https://github.com/suyashh94/khoji) to handle the pipeline:

![Three experiments, one conclusion: fine-tuning works](./figures/00_hero_results_summary.png)

---

## The 2025 Retrieval Landscape

Before diving into experiments, here's an honest overview of the realistic options and where each fits.

**LLM Re-Ranking.** Use GPT-4o, Claude, or a local LLM to score each (query, document) pair. Quality ceiling is very high — the LLM understands nuance that embedding models miss. But at ~100-500ms and ~$0.01 per query, it's only viable as a second stage on a small candidate set, or for low-volume applications. You cannot run this on every query at scale.

**Embedding APIs with Fine-Tuning.** OpenAI (`text-embedding-3-small`), Cohere (`embed-v3`), and Google offer embedding endpoints with fine-tuning. Easy to start — upload your data, get a model back. But: your data leaves your infrastructure, you have no control over the model architecture, per-query costs add up at scale, and you're locked into a vendor. For prototyping and moderate volume with no privacy constraints, this is often the simplest path.

**Caption-Then-Retrieve (Multimodal).** Use a vision LLM to generate text descriptions of images at index time, then do standard text-to-text retrieval. Works well for generic images where captions capture the relevant content. Fails for specialized visual domains — a caption like "an aerial view of land" loses the spatial and textural features that distinguish a river delta from a farm field. Also requires running a VLM on every new image in your corpus.

**Two-Stage Pipeline.** The production standard. A fast bi-encoder retrieves the top 50-100 candidates in milliseconds, then a cross-encoder or LLM re-ranks the top 10-20 for quality. khoji produces the first-stage model. The better your first stage, the better candidates the re-ranker sees — garbage in, garbage out applies here.

**Fine-Tuned Bi-Encoders.** What this post covers. Best when you need low latency, high throughput, privacy/edge deployment, or your domain is so specialized that pretrained models are genuinely bad at it. Weakest when you have little labeled data, a rapidly shifting domain, or generic queries where pretrained models already work well.

---

## Where Fine-Tuning Definitively Helps

Fine-tuning produces the largest gains when a domain has specialized vocabulary, unusual visual patterns, or a notion of relevance that generic training data never captured.

| Domain | Why pretrained models fail | Volume / latency pressure | Privacy driver |
|--------|---------------------------|---------------------------|----------------|
| **Legal / compliance** | "Negligence" vs "gross negligence" — worlds apart legally, identical to a generic model | Firm-wide search, high | Client privilege, data residency |
| **Medical imaging** | X-rays, histopathology, fundus images — never in CLIP's training data | Moderate | HIPAA, on-device at point of care |
| **Satellite / geospatial** | Spectral signatures, land-use patterns — alien to web-trained models | Real-time monitoring | Defense, sovereignty |
| **E-commerce** | Relevance is behavioral (clicks, purchases), not semantic | Very high (millions of users) | Less critical |
| **Cybersecurity** | CVE descriptions, TTPs, IOCs — rapidly evolving, specialized vocabulary | Real-time SIEM | Air-gapped SOC |
| **Code search** | Internal APIs, custom patterns, project-specific naming | IDE-integrated, low latency | Internal IP |

> **The edge deployment angle is underappreciated.** A fine-tuned 22M-parameter model with a 2MB LoRA adapter runs on a laptop CPU, a hospital workstation, a phone, or a Jetson edge device. No API calls, no network dependency, no data exfiltration risk. For air-gapped environments — which describes a surprisingly large portion of healthcare, defense, legal, and manufacturing — this isn't a cost optimization. It's the *only* architecture that works.

---

## How khoji Works

![khoji pipeline architecture](./figures/05_pipeline_architecture.png)

The pipeline is the same across all three retrieval modes:

1. **Your data** (queries + corpus + relevance judgments) goes in
2. **Negative mining** selects non-relevant items as training signal — randomly, by model similarity (hard), or both (mixed)
3. **Triplets** (query, positive, negative) are constructed
4. **Training** applies LoRA to the base model, optimizes with your chosen loss function, and saves a ~2MB adapter
5. Optionally, **mining rounds** repeat steps 2-4 using the fine-tuned model to find progressively harder negatives

khoji produces a first-stage retrieval model. Re-ranking (stage 2) is outside its scope — use a cross-encoder or LLM on top if you need it.

---

## Experiment 1: Text → Text on Financial Q&A

### The Setup

We compare two models on [FiQA](https://sites.google.com/view/fiqa/) — a financial question-answering dataset with 648 test queries over 57K documents:

| Model | Parameters | Role |
|-------|-----------|------|
| `BAAI/bge-base-en-v1.5` | 110M | The "big" reference model (no fine-tuning) |
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | The small model we fine-tune |

### The Results

![Text retrieval results](./figures/01_text_retrieval_results.png)

| Model | nDCG@10 | Recall@10 | MRR@10 |
|-------|---------|-----------|--------|
| BGE-base (110M, no fine-tuning) | 0.3909 | 0.4572 | 0.4740 |
| MiniLM (22M, no fine-tuning) | 0.3610 | 0.4325 | 0.4369 |
| **MiniLM (22M, fine-tuned)** | **0.3861** | **0.4527** | **0.4624** |

The fine-tuned MiniLM closed **84% of the nDCG@10 gap** to the 5x larger BGE model.

**An honest note:** The fine-tuned model (0.386) still slightly trails BGE (0.391). On a different test split, the gap could flip. The point is not that fine-tuning always wins on raw numbers — it's that you get equivalent quality at 5x lower inference cost with full control over the model. For most production systems, that 0.005 difference doesn't matter. The 5x cost reduction does.

We used BGE-base as the baseline here. Larger models exist (E5-Mistral-7B, `text-embedding-3-large`) that would score higher. But they also cost proportionally more to serve — the cost-quality tradeoff is the same: fine-tune a small model for your domain rather than scaling up a generic one.

**Training details:** 6 epochs (2 mining rounds of 3 epochs each), mixed negatives, InfoNCE loss. ~3,990 optimizer steps on a single NVIDIA H100. The LoRA adapter is 2MB.

### The Code

```yaml
model:
  name: sentence-transformers/all-MiniLM-L6-v2

data:
  dataset: fiqa
  split: train
  negatives: mixed
  n_random: 2
  n_hard: 1
  mining_rounds: 2

lora:
  r: 16
  alpha: 32
  dropout: 0.1

train:
  epochs: 3
  batch_size: 16
  grad_accum_steps: 4
  lr: 2e-5
  warmup_steps: 50
  loss: infonce
  temperature: 0.05

eval:
  k_values: [1, 5, 10]
  run_before: true
  run_after: true

output_dir: ./output/text-retrieval
```

```python
from khoji import ForgeConfig, run

config = ForgeConfig.from_yaml("fiqa_config.yaml")
result = run(config)
```

At inference, load the 2MB adapter on top of the frozen base model:

```python
from khoji import EmbeddingModel

model = EmbeddingModel(
    "sentence-transformers/all-MiniLM-L6-v2",
    adapter_path="./output/text-retrieval/adapter"
)
embeddings = model.encode(["What is compound interest?"])
```

### Why Mixed Negatives and Mining Rounds Matter

![Mining strategy comparison](./figures/06_mining_strategy_comparison.png)

**Mixed negatives** combine two types of training signal:

- **Random negatives** are items sampled randomly from the corpus. Easy for the model — clearly irrelevant — but they teach basic discrimination. *"This financial document is not about cooking recipes."*
- **Hard negatives** are items the model currently ranks highly but that aren't actually relevant. Mined by encoding the entire corpus, finding the top-k most similar non-relevant items, and using those as negatives. They force fine-grained distinctions. *"This document about bond yields is not the same as this document about bond ratings."*

Using both together gives the most balanced signal. Random negatives prevent collapse; hard negatives push the ranking boundary.

**Mining rounds** take this further. After round 1, the model has improved — what was "hard" is now easy. So we re-mine negatives using the fine-tuned model and train again, halving the learning rate to avoid overshooting:

```
Round 1: pretrained model → mine negatives → train → adapter_r1
Round 2: adapter_r1 → re-mine harder negatives → train → final adapter
```

One subtlety: **`skip_top`**. Most datasets have incomplete relevance labels. The model's top-ranked "negatives" are often actually relevant items the annotator missed. `skip_top: 5` skips the 5 most similar non-relevant items before picking hard negatives, avoiding training on likely false negatives.

---

## Experiment 2: Text → Image on Satellite Imagery

### The Setup

CLIP models are trained on internet photos. They've never seen satellite imagery. We test on [RSICD](https://github.com/201528014227051/RSICD_optimal) — ~10K satellite images with text captions.

| Model | Parameters | Role |
|-------|-----------|------|
| `openai/clip-vit-large-patch14` | 428M | The "big" CLIP (no fine-tuning) |
| `openai/clip-vit-base-patch32` | 151M | The small CLIP we fine-tune |

**A note on baselines:** CLIP ViT-B/32 dates to 2021. Newer models like SigLIP 2, EVA-CLIP, and MetaCLIP close some of the domain gap without fine-tuning. We chose CLIP because it remains the most widely deployed baseline and khoji supports it natively. The principle holds regardless: if your visual domain wasn't in the pretraining data, fine-tuning helps — the gain may just be smaller against a stronger baseline.

### The Results

![Multimodal retrieval results](./figures/02_multimodal_retrieval_results.png)

| Model | nDCG@10 | Recall@10 |
|-------|---------|-----------|
| CLIP ViT-L/14 (428M, no fine-tuning) | 0.1522 | 0.2937 |
| CLIP ViT-B/32 (151M, no fine-tuning) | 0.1439 | 0.2715 |
| **CLIP ViT-B/32 (151M, fine-tuned)** | **0.2639** | **0.4776** |

The fine-tuned small CLIP **surpasses the large one by 73% on nDCG@10**. Recall@10 nearly doubles — from 0.27 to 0.48. Neither model was trained on satellite imagery, but fine-tuning gives the small model domain knowledge the large model simply doesn't have.

**Why not caption-then-retrieve?** An alternative is to caption each satellite image with an LLM, then do text-to-text retrieval. This works for generic images but fails here: captions like "an aerial view of land" lose the spatial and textural details that distinguish a river delta from an agricultural field. Fine-tuning the vision encoder directly preserves these features.

**Catastrophic forgetting?** Because we use LoRA (rank 16, ~0.1% of parameters), the base CLIP weights are frozen. The model retains its general capabilities. If you need both domain-specific and generic performance, you can hot-swap adapters at inference time.

**Training details:** 6 epochs (2 mining rounds), mixed negatives with `skip_top: 5`, InfoNCE loss. ~24,570 optimizer steps.

### The Code

```yaml
model:
  name: openai/clip-vit-base-patch32
  lora_target: both          # fine-tune both vision and text encoders

data:
  dataset: arampacha/rsicd
  split: train
  negatives: mixed
  n_random: 2
  n_hard: 1
  top_k: 50
  skip_top: 5
  mining_rounds: 2

lora:
  r: 16
  alpha: 32

train:
  epochs: 3
  batch_size: 16
  grad_accum_steps: 2
  lr: 2e-5
  loss: infonce
  temperature: 0.05

eval:
  k_values: [1, 5, 10]
  run_before: true
  run_after: true

output_dir: ./output/multimodal-retrieval
```

```python
from khoji import MultimodalForgeConfig, run_multimodal

config = MultimodalForgeConfig.from_yaml("rsicd_config.yaml")
result = run_multimodal(config)
```

For multimodal models, `lora_target` controls which encoder(s) to fine-tune: `"both"` (default), `"vision"` only, or `"text"` only. For satellite imagery, `"both"` makes sense — the model needs to learn both what satellite images look like and how text maps to those visual features.

---

## Experiment 3: Composed Image Retrieval on Fashion

### The Concept

![Composed retrieval concept](./figures/07_composed_retrieval_concept.png)

Composed image retrieval is the most complex mode. The query is a **pair**: a reference image and a modification caption. *"Here's a red dress — find me one that's similar but in blue and longer."* The model must understand both the visual reference and the textual modification, then retrieve the right target from a gallery.

**What's inside BLIP-2.** This isn't a simple bi-encoder. BLIP-2 has three components: a frozen vision encoder (ViT-G), a lightweight Querying Transformer (Q-Former), and a frozen large language model (OPT or FlanT5). The Q-Former bridges vision and language. When we apply LoRA, we're fine-tuning the Q-Former's attention layers — the component that decides how visual features map into the shared embedding space. The frozen LLM provides the text representation backbone but is not updated.

### The Results

![Composed retrieval results](./figures/03_composed_retrieval_results.png)

| Model | Recall@1 | Recall@10 | Recall@50 | MRR@10 |
|-------|---------|-----------|-----------|--------|
| BLIP-2 (no fine-tuning) | 0.0000 | 0.1489 | 0.2872 | 0.0491 |
| **BLIP-2 (fine-tuned)** | **0.0638** | **0.2979** | **0.4574** | **0.1325** |

The pretrained BLIP-2 can't get a single Recall@1 hit — it has no concept of "find this but different." After fine-tuning, **Recall@10 doubles** from 15% to 30%, and Recall@50 jumps from 29% to 46%.

This is a case where fine-tuning doesn't just improve performance — it **enables an entirely new capability**.

**Training details:** 5 epochs, random negatives, InfoNCE loss. ~13,415 optimizer steps.

### The Code

```python
from khoji import (
    ComposedTrainer, ComposedTrainingConfig,
    ComposedTripletDataset, LoRASettings,
    build_random_negatives_composed, infonce_loss,
)
from functools import partial

triplets = build_random_negatives_composed(dataset, n_negatives=3)

config = ComposedTrainingConfig(
    epochs=5, batch_size=8, lr=2e-5,
    loss_fn=partial(infonce_loss, temperature=0.05),
    lora=LoRASettings(r=8, alpha=16, dropout=0.1),
    save_dir="./output/composed/adapter",
)

trainer = ComposedTrainer("Salesforce/blip2-itm-vit-g", config)
trainer.train(ComposedTripletDataset(triplets))
```

### How Composed Encoding Works

BLIP-2 encodes images and text into a shared 256-dimensional space. For composed queries, the default fusion is addition:

```
composed_embedding = image_embedding + text_embedding
```

The image embedding captures "what this dress looks like" and the text embedding captures "what should change."

**This is a known weak baseline.** The composed retrieval literature (TIRG, ARTEMIS, CompoDiff) has moved well beyond simple addition, using learned transformations, attention-based fusion, and diffusion models. khoji defaults to addition for simplicity but supports custom fusion — you can plug in a learned FFN or attention layer by passing custom encode functions to `ComposedTrainer`.

**Competing approaches exist.** An alternative is to generate the target image via a diffusion model (e.g., InstructPix2Pix) given the reference + caption, then do image-to-image similarity search. This avoids composed embeddings entirely but requires a generative model at query time, which is slow and expensive. The bi-encoder approach is orders of magnitude faster for high-volume use cases.

---

## Training Curves

![Training curves for all three experiments](./figures/04_training_curves.png)

All three experiments show clean loss convergence. The text and multimodal experiments (6 epochs each across 2 mining rounds) show the characteristic restart when re-mining produces harder negatives — the loss jumps up as the model faces harder examples, then converges again.

---

## Production Architecture: Where khoji Fits

In production, retrieval is almost never a single model. It's a pipeline.

![Production retrieval architecture](./figures/09_production_architecture.png)

**Stage 1 — Bi-Encoder (~2ms).** Encode the query, look up an ANN index (FAISS, Milvus), return top-100 candidates. This runs on every query and must be fast. khoji produces this model.

**Stage 2 — Re-Ranker (~50-200ms).** A cross-encoder or LLM scores each (query, candidate) pair from the top-100, re-orders them, returns top-10. Expensive per query but only runs on a small candidate set.

The quality of Stage 1 directly caps the quality of Stage 2. If the relevant document isn't in the top-100, the re-ranker never sees it. This is why a domain-fine-tuned first stage matters even when you have a powerful re-ranker downstream.

**Edge and local deployment.** A 22M MiniLM with a 2MB LoRA adapter runs on a laptop CPU. No API calls, no network dependency, no data leaving the device. This is the deployment story that embedding APIs cannot match — and it covers a surprisingly large slice of real-world use: hospitals, law firms, defense, manufacturing floors, mobile apps, sovereign cloud, and any environment subject to GDPR or data residency laws.

**Multi-domain serving.** Because LoRA adapters are 2MB and the base model is shared, you can serve N domain-specific retrievers from one base model. Load the adapter at request time based on the user's domain. This is impractical with full fine-tuning or vendor APIs.

---

## Training Strategies

### Negative mining

| Situation | Strategy | Config |
|-----------|----------|--------|
| First experiment, quick iteration | Random | `negatives: random`, `n_negatives: 3` |
| Production training | Mixed | `negatives: mixed`, `n_random: 2`, `n_hard: 1` |
| Maximizing performance | Mixed + 2 rounds | Add `mining_rounds: 2`, `skip_top: 5` |
| Very large corpus (>1M) | Random first, then hard on subset | `corpus_size: 50000` for mining |

### Loss functions

| Loss | When to use | Key parameter |
|------|------------|---------------|
| **InfoNCE** | Best overall. In-batch negatives give richer signal. | `temperature: 0.05` |
| **Triplet Margin** | Small batch sizes, random negatives | `margin: 0.2` |
| **Contrastive** | Simple baseline, no hyperparams | — |

### LoRA rank

LoRA rank controls adapter capacity. Higher rank = more parameters = more expressive but slower to train. Our experiments used `r=16` for text/multimodal and `r=8` for composed retrieval. As a rule of thumb: `r=8` is a safe default, `r=16` for harder domains, `r=32+` only with abundant data.

---

## When NOT to Fine-Tune

| Situation | Why fine-tuning is wrong | What to do instead |
|-----------|------------------------|--------------------|
| **Low query volume** (<1K/day) | Cost savings are negligible; LLM re-ranker gives better quality per dollar | Pretrained model + LLM re-ranker, or embedding API |
| **Rapidly shifting domain** (news, trends) | Fine-tuned model goes stale; retraining cost accumulates | Pretrained model + LLM re-ranker with in-context examples |
| **Very small labeled set** (<500 pairs) | Insufficient signal; risk of overfitting | Embedding API fine-tuning (less data-hungry), or generate synthetic labels with an LLM first |
| **Generic queries on generic data** | Pretrained model is already good enough | Don't fine-tune. Evaluate first — you may not need it. |

**Fine-tuning is not a silver bullet.** It requires labeled relevance data — at minimum ~1K-2K query-document pairs for text, ~2K-5K image-caption pairs for multimodal. It produces a static model that must be retrained when the domain shifts. And on generic benchmarks, a fine-tuned small model will still lose to a large pretrained one — the advantage only appears on *your* specific domain.

If you're not sure whether your domain is specialized enough, run the pretrained baseline first. khoji makes this easy: set `run_before: true` in your config and check the numbers before investing in fine-tuning.

---

## Practical Guide

### How much data do you need?

As a rough guide: 1K-5K labeled query-document pairs is a good starting point for text retrieval. 2K-10K image-caption pairs for multimodal. More data helps, but diminishing returns set in quickly with LoRA — most of the gain comes in the first few epochs.

If you don't have labeled data, consider using an LLM to generate synthetic query-relevance pairs from your corpus. Prompt GPT-4o to generate 5 diverse queries per document, label them as relevant, and use those as training data. This synthetic-data-then-fine-tune pipeline is increasingly the practical workflow — khoji handles the fine-tuning step.

### Training time

| Experiment | Dataset size | Optimizer steps | Adapter size |
|-----------|-------------|-----------------|-------------|
| Text (MiniLM, FiQA) | 5.7K queries, 57K docs | 3,990 | 2MB |
| Multimodal (CLIP-B/32, RSICD) | ~10K images | 24,570 | 3MB |
| Composed (BLIP-2, FashionIQ) | ~18K triplets | 13,415 | 4MB |

All experiments ran on a single NVIDIA H100. MiniLM and CLIP-B/32 fine-tune comfortably on 8GB VRAM (T4, RTX 3070). BLIP-2 requires 16-24GB (A10, A100, RTX 4090). Apple Silicon (MPS) is supported for all modes.

### Getting started

```bash
pip install khoji
khoji init                                   # generate example configs
khoji fiqa_quick.yaml                        # text → text
khoji multimodal flickr30k_quick.yaml        # text → image
```

For composed retrieval:

```bash
python scripts/fashioniq/download_data.py
python scripts/train_composed_retrieval_api.py
```

Full documentation, scripts, and notebooks at [github.com/suyashh94/khoji](https://github.com/suyashh94/khoji).

---

## What You Should Do Next

1. **Evaluate your pretrained baseline first.** Run khoji with `run_before: true` and no training. If metrics are already above your threshold, stop.

2. **If there's a gap, fine-tune the smallest model that meets your latency budget.** Start with MiniLM (22M) for text, CLIP-B/32 (151M) for images. Use mixed negatives and 2 mining rounds.

3. **If your visual domain was never in pretraining data, fine-tuning is almost certainly worth it.** Satellite, medical, industrial, geospatial — expect 50-100% retrieval improvements.

4. **For production, pair the fine-tuned retriever with a re-ranker.** khoji handles the first stage. Add a cross-encoder or LLM re-ranker on top-20 for maximum quality.

5. **If you need on-prem or edge deployment, this is your architecture.** A 22M model with a 2MB adapter runs anywhere. No API dependency, no data exfiltration risk.

6. **If you have fewer than 500 labeled pairs, don't fine-tune yet.** Generate synthetic labels with an LLM, or use an embedding API's fine-tuning endpoint, and revisit when you have more data.

---

*Code and experiment results are available at [github.com/suyashh94/khoji](https://github.com/suyashh94/khoji).*
