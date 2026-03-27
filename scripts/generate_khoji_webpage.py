"""Generate the main khoji tutorial webpage from pre-collected data.

Loads retrieval examples from output/webpage/data/ (JSON files with actual
query/document text, images as base64, scores, and relevance labels) and
generates a single self-contained HTML file with 5 tabs:

  Tab 1: Embedding Fine-Tuning Concepts
  Tab 2: Text -> Text Retrieval (FiQA)
  Tab 3: Text -> Image Retrieval (RSICD)
  Tab 4: Composed Retrieval (img+txt -> img) — FashionIQ
  Tab 5: Mixed-Mode SKU Matching (img+txt -> img+txt)

Usage:
    python scripts/generate_khoji_webpage.py
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path("./output/webpage/data")
OUTPUT_DIR = Path("./output/webpage")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading helpers ────────────────────────────────


def load_json(name: str):
    """Load a JSON file from the data directory."""
    path = DATA_DIR / name
    with open(path) as f:
        return json.load(f)


def img_data_uri(b64: str, fmt: str = "jpeg") -> str:
    """Wrap raw base64 as a data URI."""
    if not b64:
        return ""
    if b64.startswith("data:"):
        return b64
    return f"data:image/{fmt};base64,{b64}"


def blog_figure_uri(figures: dict, name: str) -> str:
    """Get a data URI for a blog figure by filename."""
    b64 = figures.get(name, "")
    if not b64:
        return ""
    # Blog figures are PNGs (from matplotlib)
    return f"data:image/png;base64,{b64}"


# ── Metrics table builder ──────────────────────────────


def metrics_table_html(
    baseline: dict,
    finetuned: dict,
    keys: list[str] | None = None,
    show_percent: bool = False,
) -> str:
    """Build an HTML metrics comparison table."""
    if keys is None:
        keys = list(baseline.keys())
    rows = ""
    for m in keys:
        b = baseline.get(m, 0)
        f = finetuned.get(m, 0)
        d = f - b
        color = "#16a34a" if d > 0.001 else ("#dc2626" if d < -0.001 else "#94a3b8")
        sign = "+" if d >= 0 else ""
        if show_percent:
            rows += (
                f'<tr><td>{m}</td>'
                f'<td>{b:.2%}</td>'
                f'<td><b>{f:.2%}</b></td>'
                f'<td style="color:{color};font-weight:600">{sign}{d:.2%}</td></tr>\n'
            )
        else:
            rows += (
                f'<tr><td>{m}</td>'
                f'<td>{b:.4f}</td>'
                f'<td><b>{f:.4f}</b></td>'
                f'<td style="color:{color};font-weight:600">{sign}{d:.4f}</td></tr>\n'
            )
    return f'''<table class="metrics-tbl">
<thead><tr><th>Metric</th><th>Baseline</th><th>Fine-tuned</th><th>Delta</th></tr></thead>
<tbody>{rows}</tbody></table>'''


# ── Text retrieval example builder ─────────────────────


def text_example_html(qid: str, query_text: str, baseline_results: list, finetuned_results: list, top_k: int = 5) -> str:
    """Build a before/after comparison for a text retrieval query."""

    def result_row(results, label):
        items = ""
        for i, r in enumerate(results[:top_k]):
            border_color = "#16a34a" if r["relevant"] else "#dc2626"
            bg_color = "#f0fdf4" if r["relevant"] else "#fef2f2"
            icon = "&#10003;" if r["relevant"] else "&#10007;"
            icon_color = "#16a34a" if r["relevant"] else "#dc2626"
            text_snippet = r["text"][:180] + ("..." if len(r["text"]) > 180 else "")
            items += f'''<div class="text-result-card" style="border-left:3px solid {border_color};background:{bg_color}">
<div class="text-result-header">
<span class="relevance-icon" style="color:{icon_color}">{icon}</span>
<span class="result-rank">#{i+1}</span>
<span class="result-score">score: {r["score"]:.4f}</span>
</div>
<div class="text-result-body">{text_snippet}</div>
</div>'''
        return items

    baseline_hits = sum(1 for r in baseline_results[:top_k] if r["relevant"])
    finetuned_hits = sum(1 for r in finetuned_results[:top_k] if r["relevant"])

    return f'''<div class="example-block">
<div class="example-query">
<div class="query-label">Query</div>
<div class="query-text">{query_text}</div>
<div class="hit-summary">
<span class="hit-badge baseline-badge">Baseline: {baseline_hits}/{top_k} relevant</span>
<span class="hit-badge finetuned-badge">Fine-tuned: {finetuned_hits}/{top_k} relevant</span>
</div>
</div>
<div class="example-comparison">
<div class="comparison-col">
<div class="comparison-label before-label">Before (Baseline)</div>
{result_row(baseline_results, "baseline")}
</div>
<div class="comparison-col">
<div class="comparison-label after-label">After (Fine-tuned)</div>
{result_row(finetuned_results, "finetuned")}
</div>
</div>
</div>'''


# ── Image retrieval example builder ────────────────────


def image_example_html(qid: str, query_text: str, baseline_results: list, finetuned_results: list, top_k: int = 5) -> str:
    """Build a before/after comparison for a text->image retrieval query."""

    def img_row(results):
        items = ""
        for i, r in enumerate(results[:top_k]):
            border_color = "#16a34a" if r["relevant"] else "#dc2626"
            uri = img_data_uri(r.get("image_b64", ""))
            if not uri:
                continue
            items += f'''<div class="img-result-card">
<img src="{uri}" class="result-img" style="border:3px solid {border_color}">
<div class="img-result-meta">
<span class="result-rank">#{i+1}</span>
<span class="result-score">{r["score"]:.3f}</span>
</div>
</div>'''
        return items

    baseline_hits = sum(1 for r in baseline_results[:top_k] if r["relevant"])
    finetuned_hits = sum(1 for r in finetuned_results[:top_k] if r["relevant"])

    return f'''<div class="example-block">
<div class="example-query">
<div class="query-label">Query</div>
<div class="query-text">{query_text}</div>
<div class="hit-summary">
<span class="hit-badge baseline-badge">Baseline: {baseline_hits}/{top_k} relevant</span>
<span class="hit-badge finetuned-badge">Fine-tuned: {finetuned_hits}/{top_k} relevant</span>
</div>
</div>
<div class="example-comparison">
<div class="comparison-col">
<div class="comparison-label before-label">Before (Baseline)</div>
<div class="img-results-row">{img_row(baseline_results)}</div>
</div>
<div class="comparison-col">
<div class="comparison-label after-label">After (Fine-tuned)</div>
<div class="img-results-row">{img_row(finetuned_results)}</div>
</div>
</div>
</div>'''


# ── SKU matching example builder ───────────────────────


def sku_example_html(
    qid: str,
    query_text: str,
    query_image_b64: str,
    query_brand: str,
    query_family: str,
    query_variant: str,
    baseline_results: list,
    finetuned_results: list,
    top_k: int = 5,
) -> str:
    """Build a before/after comparison for a SKU matching query."""

    def sku_row(results):
        items = ""
        for i, r in enumerate(results[:top_k]):
            border_color = "#16a34a" if r["relevant"] else "#dc2626"
            uri = img_data_uri(r.get("image_b64", ""))
            brand = r.get("brand", "")
            variant = r.get("variant", "")
            if not uri:
                continue
            items += f'''<div class="sku-result-card">
<img src="{uri}" class="result-img" style="border:3px solid {border_color}">
<div class="sku-result-meta">
<div class="sku-brand">{brand}</div>
<div class="sku-variant">{variant}</div>
<span class="result-score">{r["score"]:.3f}</span>
</div>
</div>'''
        return items

    query_uri = img_data_uri(query_image_b64)
    baseline_hits = sum(1 for r in baseline_results[:top_k] if r["relevant"])
    finetuned_hits = sum(1 for r in finetuned_results[:top_k] if r["relevant"])

    short_text = query_text[:80] + ("..." if len(query_text) > 80 else "")

    return f'''<div class="example-block">
<div class="example-query sku-query">
<div class="sku-query-inner">
{f'<img src="{query_uri}" class="query-img">' if query_uri else ''}
<div class="sku-query-info">
<div class="query-label">Query — {query_brand}</div>
<div class="sku-query-family">{query_family} / {query_variant}</div>
<div class="query-text" style="font-size:12px">{short_text}</div>
</div>
</div>
<div class="hit-summary">
<span class="hit-badge baseline-badge">Baseline: {baseline_hits}/{top_k} relevant</span>
<span class="hit-badge finetuned-badge">Fine-tuned: {finetuned_hits}/{top_k} relevant</span>
</div>
</div>
<div class="example-comparison">
<div class="comparison-col">
<div class="comparison-label before-label">Before (Baseline)</div>
<div class="img-results-row">{sku_row(baseline_results)}</div>
</div>
<div class="comparison-col">
<div class="comparison-label after-label">After (Fine-tuned)</div>
<div class="img-results-row">{sku_row(finetuned_results)}</div>
</div>
</div>
</div>'''


# ════════════════════════════════════════════════════════
#  TAB 1: EMBEDDING FINE-TUNING CONCEPTS
# ════════════════════════════════════════════════════════


def tab1_concepts() -> str:
    return '''
<h2>Embedding Fine-Tuning Concepts</h2>
<p class="lead">A comprehensive guide to how khoji fine-tunes embedding models for domain-specific retrieval. Whether you are searching financial documents, satellite imagery, fashion products, or grocery catalogs, the core principles are the same: encode your data into vectors, define what "similar" means through training examples, and teach the model your domain.</p>

<!-- ── What is Embedding Retrieval? ────────────────────── -->
<div class="concept-card">
<h3>What is Embedding Retrieval?</h3>
<p>Embedding retrieval maps queries and documents into a shared high-dimensional vector space (typically 256-1024 dimensions). Items that are semantically similar end up close together, measured by cosine similarity.</p>

<div class="diagram-box">
<div class="diagram-title">The Embedding Space</div>
<div class="diagram-content">
<pre class="diagram-pre">
  Pretrained Model                     Fine-tuned Model

  "bond yield curve"   ──→  [0.23, 0.87, ...]       [0.23, 0.87, ...]
                                  │                        │
                                  │ cos_sim = 0.41         │ cos_sim = 0.89  ← CLOSER
                                  │                        │
  "treasury rate term  ──→  [0.45, 0.12, ...]       [0.28, 0.82, ...]
   structure"
                                  │                        │
                                  │ cos_sim = 0.38         │ cos_sim = 0.22  ← FARTHER
                                  │                        │
  "best pizza recipe"  ──→  [0.67, 0.33, ...]       [0.91, 0.04, ...]
</pre>
</div>
</div>

<p>A pretrained model (BERT, CLIP, BLIP-2) encodes general semantics learned from web data. It knows that "bond" and "treasury" are somewhat related, but it does not understand that in a <b>financial QA</b> context, "bond yield curve" and "treasury rate term structure" are asking about the same concept. Fine-tuning teaches the model: "in MY domain, these are equivalent."</p>

<div class="insight-box">
<b>Why pretrained models fail on domain data:</b> A model trained on Wikipedia and web crawls has never seen your internal knowledge base, your satellite imagery categories, or your product catalog structure. Domain-specific vocabulary, visual patterns, and relevance criteria are foreign to it. Fine-tuning bridges this gap without retraining from scratch.
</div>
</div>

<!-- ── Triplet Training ────────────────────────────────── -->
<div class="concept-card">
<h3>Triplet Training: The Core Learning Signal</h3>
<p>Fine-tuning requires <b>triplets</b>: (query, positive, negative). Each triplet gives the model one learning signal: "the query should be more similar to the positive than to the negative."</p>

<div class="diagram-box">
<div class="diagram-title">Triplet Structure</div>
<div class="diagram-content">
<pre class="diagram-pre">
                    ┌─────────────┐
                    │    Query    │  "What is a bond yield?"
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │                         │
     ┌────────▼────────┐     ┌─────────▼────────┐
     │   Positive (+)  │     │   Negative (-)   │
     │  "Bond yields   │     │  "The best pizza  │
     │   represent     │     │   dough recipe    │
     │   the return    │     │   uses 00 flour"  │
     │   on debt..."   │     │                   │
     └─────────────────┘     └──────────────────┘

  Goal: sim(query, positive) >> sim(query, negative)
</pre>
</div>
</div>

<p>The quality of your triplets determines the quality of your fine-tuned model. The query-positive pairs come from your relevance labels (e.g., QA pairs, image-caption pairs, product matches). The negatives are items that are NOT relevant to the query — and choosing these wisely is the key to effective training.</p>

<div class="triplet-data-box">
<h4>Data Format</h4>
<p>khoji uses a standard format based on BEIR (Benchmarking IR). Three files define your dataset:</p>
<div class="code-block">queries.jsonl — one query per line
{"_id": "q1", "text": "What is compound interest?"}
{"_id": "q2", "text": "How to diversify a portfolio?"}

corpus.jsonl — one document per line
{"_id": "d1", "text": "Compound interest is interest on interest..."}
{"_id": "d2", "text": "Portfolio diversification reduces risk by..."}

qrels.tsv — relevance judgments (query_id, corpus_id, score)
q1    d1    1
q2    d2    1</div>
<p>For multimodal datasets, documents have an <code>"image"</code> field instead of (or in addition to) <code>"text"</code>.</p>
</div>
</div>

<!-- ── Loss Functions ──────────────────────────────────── -->
<div class="concept-card">
<h3>Loss Functions</h3>
<p>The loss function defines precisely <i>how</i> the model learns from each triplet. Different loss functions create different learning dynamics and work best in different scenarios.</p>

<div class="loss-grid">
<div class="loss-item">
<div class="loss-header">Triplet Margin Loss</div>
<div class="formula-box">
<code class="formula">L = max(0, d(q, pos) - d(q, neg) + margin)</code>
</div>
<p class="loss-desc">Pushes the positive closer to the query than the negative by at least <code>margin</code> (typically 0.2-0.5). Once the margin is satisfied, the loss becomes zero and the model stops learning from that triplet.</p>
<div class="loss-when">
<b>When to use:</b> Small batches (4-16), random negatives, when you want stable training. The margin acts as a natural regularizer — the model does not overfit to already-learned distinctions.
</div>
</div>

<div class="loss-item">
<div class="loss-header highlight">InfoNCE Loss (Recommended)</div>
<div class="formula-box">
<code class="formula">L = -log( exp(sim(q, pos) / &tau;) / &Sigma;<sub>i</sub> exp(sim(q, neg<sub>i</sub>) / &tau;) )</code>
</div>
<p class="loss-desc">Softmax cross-entropy over similarities. The temperature &tau; (typically 0.05-0.1) controls sharpness: lower &tau; makes the model more discriminative. Each query treats ALL other positives in the batch as additional negatives (in-batch negatives), giving O(batch_size) negatives per query for free.</p>
<div class="loss-when">
<b>When to use:</b> Medium to large batches (16-64). The in-batch negative mechanism means a batch of 32 gives each query 31 extra negatives beyond the explicit one. This is khoji's default loss and works best in almost all cases.
</div>
</div>

<div class="loss-item">
<div class="loss-header">Contrastive Loss</div>
<div class="formula-box">
<code class="formula">L = -cos(q, pos) + max(0, cos(q, neg) - margin)</code>
</div>
<p class="loss-desc">Directly maximizes positive similarity and minimizes negative similarity. No temperature parameter. A simple hinge on the negative prevents the model from wasting capacity pushing already-distant negatives farther.</p>
<div class="loss-when">
<b>When to use:</b> As a simple baseline to verify your data pipeline is correct, or when you want minimal hyperparameters. Usually outperformed by InfoNCE for retrieval tasks.
</div>
</div>
</div>
</div>

<!-- ── Custom Loss Functions ───────────────────────────── -->
<div class="concept-card">
<h3>Custom Loss Functions</h3>
<p>khoji accepts any callable with the signature <code>(query_emb, pos_emb, neg_emb) -> scalar</code>. You can implement any metric learning loss and pass it directly to the training config.</p>
<div class="code-block"># khoji accepts any callable: (query_emb, pos_emb, neg_emb) -> scalar

# ── Circle Loss ──────────────────────────────────────
def circle_loss(query_emb, positive_emb, negative_emb, margin=0.25, gamma=64):
    pos_sim = torch.nn.functional.cosine_similarity(query_emb, positive_emb)
    neg_sim = torch.nn.functional.cosine_similarity(query_emb, negative_emb)
    alpha_p = torch.clamp(1 + margin - pos_sim, min=0)
    alpha_n = torch.clamp(neg_sim + margin, min=0)
    logit_p = -gamma * alpha_p * (pos_sim - (1 - margin))
    logit_n = gamma * alpha_n * (neg_sim - margin)
    return torch.nn.functional.softplus(logit_n - logit_p).mean()

# ── Angular Loss ─────────────────────────────────────
import math
def angular_loss(query_emb, positive_emb, negative_emb, alpha=45):
    angle = alpha * math.pi / 180
    sq_tan = math.tan(angle) ** 2
    # Center of (query, positive)
    center = (query_emb + positive_emb) / 2
    neg_dist = 1 - torch.nn.functional.cosine_similarity(center, negative_emb)
    pos_dist = 1 - torch.nn.functional.cosine_similarity(query_emb, positive_emb)
    return torch.relu(pos_dist - sq_tan * neg_dist).mean()

# ── Use with any trainer ─────────────────────────────
from functools import partial

config = khoji.TrainingConfig(
    loss_fn=circle_loss,                              # direct
    # loss_fn=partial(circle_loss, margin=0.3),       # with custom params
    # loss_fn=partial(khoji.infonce_loss, temperature=0.07),  # built-in with custom temp
)</div>
</div>

<!-- ── Negative Mining Strategies ──────────────────────── -->
<div class="concept-card">
<h3>Negative Mining Strategies</h3>
<p>The choice of negatives has a <b>massive impact</b> on what the model learns — often more than the choice of loss function, learning rate, or model architecture. A random negative tells the model "pizza is not finance." A hard negative tells the model "this specific bond analysis document is NOT about yield curves despite using similar terminology."</p>

<div class="mining-grid">
<div class="mining-item">
<div class="mining-header random">Random Negatives</div>
<div class="mining-diagram">
<pre class="diagram-pre-sm">
  Query: "bond yield curve"
  Positive: "treasury rate term structure"
  Negative: "best hiking trails in Colorado"  ← easy to distinguish
</pre>
</div>
<p>Randomly sample non-relevant items from the corpus. <b>Pros:</b> Fast (no model inference needed), teaches basic "this is clearly not relevant" discrimination, good for initial training rounds. <b>Cons:</b> Most random negatives are too easy — the model learns nothing from them after the first few epochs.</p>
</div>

<div class="mining-item">
<div class="mining-header hard">Hard Negatives</div>
<div class="mining-diagram">
<pre class="diagram-pre-sm">
  Query: "bond yield curve"
  Positive: "treasury rate term structure"
  Negative: "bond rating downgrades in Q3"  ← confusingly similar!

  Process:
  1. Encode entire corpus with current model
  2. For each query, rank all docs by similarity
  3. Pick top-K non-relevant docs as hard negatives
</pre>
</div>
<p>Encode the full corpus, find the most similar non-relevant items — things the model <i>currently</i> confuses with relevant results. <b>Pros:</b> Forces fine-grained distinctions, dramatically improves precision. <b>Cons:</b> Requires full corpus encoding (slow), risk of selecting false negatives (see skip_top below).</p>
</div>

<div class="mining-item">
<div class="mining-header mixed">Mixed Negatives (Recommended)</div>
<div class="mining-diagram">
<pre class="diagram-pre-sm">
  Query: "bond yield curve"
  Positive: "treasury rate term structure"
  Negatives:
    1. "best hiking trails in Colorado"      ← random
    2. "pasta carbonara authentic recipe"     ← random
    3. "corporate bond market analysis"       ← hard (mined)
    4. "ski resort weather forecast"          ← random
    5. "bond rating methodology changes"      ← hard (mined)
</pre>
</div>
<p>Combine random and hard negatives for the best of both worlds. Random negatives maintain basic relevance discrimination while hard negatives push precision on confusing cases. Typically 2-4 random + 1-2 hard negatives per positive.</p>
</div>
</div>

<div class="insight-box">
<b>Rule of thumb:</b> Start with random negatives to verify your pipeline, then switch to mixed negatives for production training. Pure hard negatives can be unstable and may amplify false negative noise.
</div>
</div>

<!-- ── skip_top ────────────────────────────────────────── -->
<div class="concept-card">
<h3><code>skip_top</code> — Avoiding False Negatives</h3>
<p>Most real-world datasets have <b>incomplete relevance labels</b>. In FiQA, each query has 1-5 labeled relevant documents out of 57,000. Many unlabeled documents are actually relevant but were never annotated. These unlabeled positives tend to cluster at the very top of the model's ranking — they are the most similar non-labeled items.</p>

<div class="diagram-box">
<div class="diagram-title">The False Negative Problem</div>
<div class="diagram-content">
<pre class="diagram-pre">
  Model's ranking for "bond yield curve":

  Rank 1:  "Treasury yield curve analysis"      ← labeled relevant ✓
  Rank 2:  "Government bond yield movements"    ← NOT labeled, but actually relevant!
  Rank 3:  "Corporate yield spread trends"      ← NOT labeled, but actually relevant!
  Rank 4:  "Bond market quarterly review"       ← genuinely not relevant
  Rank 5:  "Fixed income duration analysis"     ← NOT labeled, but partially relevant!
  ...
  Rank 15: "Bond credit rating methodology"     ← genuinely not relevant ← SAFE to mine

  skip_top=0:  mines rank 2,3,5 as negatives → TEACHING MODEL TO AVOID GOOD RESULTS!
  skip_top=10: mines from rank 11+ → much safer, avoids likely false negatives
</pre>
</div>
</div>

<p>Setting <code>skip_top=N</code> skips the top N non-relevant results before selecting hard negatives. This avoids the most likely false negatives. For sparsely labeled datasets (1-5 labels per query), use <code>skip_top=50-100</code>. For well-labeled datasets, <code>skip_top=5-10</code> suffices.</p>
</div>

<!-- ── Mining Rounds ───────────────────────────────────── -->
<div class="concept-card">
<h3>Mining Rounds — Iterative Re-mining</h3>
<p>After training on the first set of hard negatives, the model has improved — what was "hard" before may now be easy. The negatives are stale. Iterative mining repeats the mine-train cycle to keep pushing the model:</p>

<div class="diagram-box">
<div class="diagram-title">Mining Round Cycle</div>
<div class="diagram-content">
<pre class="diagram-pre">
  ┌─────────────────────────────────────────────────────────────────┐
  │  Round 1                                                        │
  │  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐ │
  │  │ Pretrained    │───→│ Mine negatives│───→│ Train (lr=2e-5)  │ │
  │  │ model         │    │ from corpus   │    │ 3 epochs         │ │
  │  └──────────────┘    └───────────────┘    └────────┬─────────┘ │
  │                                                     │           │
  │                                            adapter_round_1      │
  └─────────────────────────────────────────────────────┼───────────┘
                                                        │
  ┌─────────────────────────────────────────────────────┼───────────┐
  │  Round 2                                            ▼           │
  │  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐ │
  │  │ Fine-tuned    │───→│ Re-mine HARDER│───→│ Train (lr=1e-5)  │ │
  │  │ model (r1)    │    │ negatives     │    │ 3 epochs         │ │
  │  └──────────────┘    └───────────────┘    └────────┬─────────┘ │
  │                                                     │           │
  │                                            adapter_final        │
  └─────────────────────────────────────────────────────────────────┘

  Key: each round HALVES the learning rate to avoid overshooting.
  2 rounds is usually sufficient; 3+ has diminishing returns.
</pre>
</div>
</div>

<p>In round 2, the fine-tuned model's hard negatives are genuinely harder — they are items that confused the improved model, not the original pretrained one. This forces increasingly fine-grained distinctions.</p>
</div>

<!-- ── LoRA ─────────────────────────────────────────────── -->
<div class="concept-card">
<h3>LoRA — Parameter-Efficient Fine-Tuning</h3>
<p>Instead of updating all model weights (110M for BERT-base, 400M+ for CLIP, 1B+ for BLIP-2), LoRA (Low-Rank Adaptation) injects small trainable matrices into attention layers. The original model weights are frozen.</p>

<div class="diagram-box">
<div class="diagram-title">LoRA Architecture</div>
<div class="diagram-content">
<pre class="diagram-pre">
  Original attention weight matrix W (d x d):

         ┌─────────────────┐
  input ─┤  W (frozen)     ├── output
         └─────────────────┘
                  +
         ┌────┐     ┌────┐
  input ─┤ A  ├─────┤ B  ├── delta_output    ← LoRA adapters (trainable)
         └────┘     └────┘
         (d x r)   (r x d)

  r = rank (e.g., 8 or 16)
  alpha = scaling factor (convention: 2 x r)
  Total trainable parameters: 2 x d x r per layer ≈ 0.1% of model
</pre>
</div>
</div>

<div class="code-block"># LoRA configuration in khoji
lora:
  r: 16        # rank — higher = more capacity, more parameters
  alpha: 32    # scaling factor (convention: 2 x r)
  dropout: 0.1 # regularization during training

# Resulting adapter size:
#   MiniLM (22M params):  ~0.3 MB adapter
#   CLIP (151M params):   ~1.5 MB adapter
#   BLIP-2 (1.2B params): ~8 MB adapter</div>

<p><b>Advantages of LoRA:</b></p>
<ul class="feature-list">
<li><b>Tiny adapter files</b> — a few MB vs. hundreds of MB for full model weights</li>
<li><b>Fast training</b> — only ~0.1% of parameters need gradients</li>
<li><b>No catastrophic forgetting</b> — base model knowledge is preserved</li>
<li><b>Multiple adapters, one base model</b> — swap adapters for different domains at inference time with zero extra memory for the base model</li>
</ul>
</div>

<!-- ── Four Retrieval Modes ────────────────────────────── -->
<div class="concept-card">
<h3>Four Retrieval Modes</h3>
<p>khoji supports four distinct retrieval modalities, each with different query and target types. The same core principles (triplets, mining, LoRA) apply to all four — only the model architecture and data format differ.</p>
<table class="modes-tbl">
<thead>
<tr><th>Mode</th><th>Query</th><th>Target</th><th>Model</th><th>Use Case</th></tr>
</thead>
<tbody>
<tr>
<td><span class="mode-badge mode-text">Text &rarr; Text</span></td>
<td>text</td><td>text</td>
<td>MiniLM, BERT, BGE</td>
<td>Document search, FAQ matching, semantic search</td>
</tr>
<tr>
<td><span class="mode-badge mode-multimodal">Text &rarr; Image</span></td>
<td>text</td><td>image</td>
<td>CLIP, SigLIP</td>
<td>Image search from descriptions, satellite imagery</td>
</tr>
<tr>
<td><span class="mode-badge mode-composed">Composed</span></td>
<td>image + text</td><td>image</td>
<td>BLIP-2</td>
<td>"Find this dress but in red," visual search with modification</td>
</tr>
<tr>
<td><span class="mode-badge mode-mixed">Mixed-Mode</span></td>
<td>image + text</td><td>image + text</td>
<td>BLIP-2</td>
<td>SKU matching, product search, multimodal catalog search</td>
</tr>
</tbody>
</table>

<div class="insight-box">
<b>Choosing a mode:</b> If your queries and documents are both text, use Text &rarr; Text. If users describe what they want and documents are images, use Text &rarr; Image. If users provide a reference image plus a modification ("like this but different"), use Composed. If both queries and documents have images AND text (e.g., product catalogs), use Mixed-Mode for best results.
</div>
</div>

<!-- ── Practical Tips ───────────────────────────────────── -->
<div class="concept-card">
<h3>Practical Tips</h3>

<h4>Resource Management</h4>
<div class="code-block"># Small GPU? Reduce batch sizes:
config = khoji.ComposedTrainingConfig(
    batch_size=4,              # smaller training batches
    grad_accum_steps=8,        # effective batch = 4 × 8 = 32
)

# Evaluation on large corpus? Control batch size:
result = evaluator.evaluate(
    dataset=my_dataset,
    batch_size=32,             # encoding batch size (default 64)
    corpus_size=1000,          # evaluate on a subset (default: full corpus)
    n_queries=50,              # limit test queries (default: all)
)

# Hard negative mining on large corpus?
triplets = khoji.build_mixed_negatives_composed(
    dataset, model,
    batch_size=32,             # mining encoding batch size
    corpus_size=5000,          # mine from a corpus subset
    n_queries=1000,            # mine for a query subset
)</div>

<h4>Debugging a New Pipeline</h4>
<div class="code-block"># Step 1: Verify data loads correctly
dataset = khoji.load_custom("./my_data")
print(f"Queries: {len(dataset.queries)}, Corpus: {len(dataset.corpus)}")

# Step 2: Overfit on 1 batch to verify training works
config = khoji.TrainingConfig(
    overfit_batches=1,         # train on just 1 batch repeatedly
    epochs=50,                 # many epochs on that 1 batch
    sanity_check_samples=5,    # show before/after cosine similarities
    batch_size=4,
    lora=khoji.LoRASettings(r=4, alpha=8),
)

# Step 3: If loss decreases, scale up to full training</div>

<h4>Mixed Precision Training</h4>
<div class="code-block"># Half the memory, ~2x faster on modern GPUs
config = khoji.ComposedTrainingConfig(
    mixed_precision="bf16",    # or "fp16" for older GPUs
    dtype="bf16",              # also load base model in bf16
)</div>

<h4>Checkpoint Saving</h4>
<div class="code-block">config = khoji.TrainingConfig(
    save_every_n_steps=500,         # save every 500 optimizer steps
    keep_all_checkpoints=True,      # keep all (default: only latest)
    save_dir="./my-adapter",
)</div>
</div>

<!-- ── Custom Evaluation Metrics ──────────────────────── -->
<div class="concept-card">
<h3>Custom Evaluation Metrics</h3>
<p>khoji computes nDCG@k, MRR@k, and Recall@k by default. Add your own metrics with <code>extra_metrics</code> — each metric receives the ranked document IDs, the relevance judgments for that query, and k.</p>
<div class="code-block"># khoji computes nDCG@k, MRR@k, Recall@k by default.
# Add your own with extra_metrics:

def precision_at_k(ranked_doc_ids: list[str], qrel: dict[str, int], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    relevant = {{d for d, s in qrel.items() if s > 0}}
    found = sum(1 for d in ranked_doc_ids[:k] if d in relevant)
    return found / k

def hit_rate_at_k(ranked_doc_ids: list[str], qrel: dict[str, int], k: int) -> float:
    """1 if any relevant doc in top-k, else 0."""
    relevant = {{d for d, s in qrel.items() if s > 0}}
    return 1.0 if any(d in relevant for d in ranked_doc_ids[:k]) else 0.0

def average_precision(ranked_doc_ids: list[str], qrel: dict[str, int], k: int) -> float:
    """Average precision at k (for MAP computation)."""
    relevant = {{d for d, s in qrel.items() if s > 0}}
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / min(len(relevant), k) if relevant else 0.0

# Pass to any evaluator:
result = evaluator.evaluate(
    dataset=test_ds,
    k_values=[1, 5, 10],
    extra_metrics={{
        "precision": precision_at_k,
        "hit_rate": hit_rate_at_k,
        "avg_precision": average_precision,
    }},
)
# result.metrics now includes: ndcg@k, mrr@k, recall@k, precision@k, hit_rate@k, avg_precision@k</div>
</div>
'''


# ════════════════════════════════════════════════════════
#  TAB 2: TEXT -> TEXT RETRIEVAL
# ════════════════════════════════════════════════════════


def tab2_text() -> str:
    summary = load_json("text_summary.json")
    baseline_data = load_json("text_baseline.json")
    finetuned_data = load_json("text_finetuned.json")

    selected_qids = summary["selected_qids"]
    baseline_m = summary["baseline_metrics"]
    finetuned_m = summary["finetuned_metrics"]
    training = summary["training"]

    # Build retrieval examples
    examples_html = ""
    for qid in selected_qids:
        if qid not in baseline_data or qid not in finetuned_data:
            continue
        bq = baseline_data[qid]
        fq = finetuned_data[qid]
        examples_html += text_example_html(
            qid, bq["query_text"], bq["retrieved"], fq["retrieved"], top_k=5
        )

    return f'''
<h2>Text &rarr; Text Retrieval</h2>
<p class="lead">Fine-tune a small text embedding model (MiniLM, 22M parameters) for financial domain question answering. The pretrained model knows general English but struggles with finance-specific terminology and domain relevance. After fine-tuning on FiQA, the tiny model rivals models 5x its size.</p>

<!-- Problem Statement -->
<div class="concept-card">
<h3>The Problem</h3>
<p>FiQA (Financial Question Answering) contains 5,500+ real financial questions from StackExchange and Reddit, with expert-annotated relevant answers from a corpus of 57,000 documents. Questions cover bonds, taxes, investing, insurance, mortgages, and more.</p>
<p>A pretrained MiniLM model understands general English semantics, but it does not know that "bond yield curve inversion" and "treasury spread tightening" refer to related financial concepts. It retrieves topically related but not actually relevant documents.</p>
</div>

<!-- Dataset Format -->
<div class="concept-card">
<h3>Dataset Format</h3>
<p>FiQA uses the standard BEIR format. khoji downloads it automatically:</p>
<div class="code-block"># queries.jsonl — real financial questions
{{"_id": "8002", "text": "How to calculate compound interest?"}}
{{"_id": "622",  "text": "What is the difference between a stock and a bond?"}}
{{"_id": "2460", "text": "Is it better to pay off debt or invest?"}}

# corpus.jsonl — 57,638 answer documents
{{"_id": "185274", "text": "Compound interest is calculated using the formula A = P(1 + r/n)^(nt)..."}}
{{"_id": "309171", "text": "Stocks represent ownership in a company, while bonds are debt..."}}

# qrels.tsv — relevance judgments (sparse: 1-5 per query)
8002    185274    1
622     309171    1</div>
</div>

<!-- Setup -->
<div class="setup-box">
<h3>Setup</h3>
<div class="setup-grid">
<div class="si"><span class="lb">Model</span><span class="vl">{summary["model"]} (22M params)</span></div>
<div class="si"><span class="lb">Dataset</span><span class="vl">{summary["dataset"]} — {summary["num_queries"]} test queries, {summary["corpus_size"]:,} corpus docs</span></div>
<div class="si"><span class="lb">Negatives</span><span class="vl">{training["negatives"]}</span></div>
<div class="si"><span class="lb">Training</span><span class="vl">{training["mining_rounds"]} mining rounds, {training["epochs_per_round"]} epochs/round, LR={training["lr"]}</span></div>
<div class="si"><span class="lb">LoRA</span><span class="vl">{training["lora"]}, dropout=0.1</span></div>
<div class="si"><span class="lb">Loss</span><span class="vl">InfoNCE (&tau;=0.05) with in-batch negatives</span></div>
</div>
<div class="code-block">git clone https://github.com/suyashh94/khoji.git && cd khoji
pip install -e .
python scripts/train_text_retrieval.py</div>
</div>

<!-- Complete Training Code -->
<h3>Complete Training Code</h3>
<p>The config-driven approach handles everything in one call:</p>
<div class="code-block">import khoji
from khoji.config import DataConfig, EvalConfig, LoRAConfig, ModelConfig, TrainConfig

config = khoji.ForgeConfig(
    model=ModelConfig(name="sentence-transformers/all-MiniLM-L6-v2"),
    data=DataConfig(
        dataset="fiqa",
        split="train",
        negatives="mixed",      # random + hard negatives
        n_random=2,             # 2 random negatives per positive
        n_hard=1,               # 1 hard negative per positive
        top_k=50,               # search top-50 for hard negatives
        skip_top=0,             # skip top-N to avoid false negatives
        mining_rounds=2,        # mine -> train -> re-mine -> train
    ),
    lora=LoRAConfig(r=16, alpha=32, dropout=0.1),
    train=TrainConfig(
        epochs=3,               # per mining round
        batch_size=16,
        lr=2e-5,
        warmup_steps=50,
        max_length=512,
        loss="infonce",
        temperature=0.05,
    ),
    eval=EvalConfig(
        k_values=[1, 5, 10],
        split="test",
        run_before=True,        # evaluate baseline first
        run_after=True,         # evaluate fine-tuned model
    ),
    seed=42,
    output_dir="./output/text-retrieval",
)

# One call — handles mining, training, re-mining, re-training, evaluation
result = khoji.run(config)</div>

<p>Or the manual API for full control over each step:</p>
<div class="code-block">from functools import partial
import khoji
from khoji.loss import infonce_loss

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
loss_fn = partial(infonce_loss, temperature=0.05)
lora = khoji.LoRASettings(r=16, alpha=32, dropout=0.1)

# ── Load data ──────────────────────────────────────────
train_ds = khoji.load_beir("fiqa", split="train")
test_ds  = khoji.load_beir("fiqa", split="test")

# ── Evaluate baseline ──────────────────────────────────
base_eval = khoji.Evaluator(MODEL)
base_result = base_eval.evaluate(
    dataset_name="fiqa",
    k_values=[1, 5, 10],
    batch_size=64,              # encoding batch size for eval
    dataset=test_ds,
)
base_result.print()

# ── Round 1: mine with base model, then train ─────────
mining_model = khoji.EmbeddingModel(MODEL)
triplets = khoji.build_mixed_negatives(
    train_ds, mining_model,
    n_random=2, n_hard=1, top_k=50,
    batch_size=64,              # encoding batch size for mining
)
del mining_model

config_r1 = khoji.TrainingConfig(
    epochs=3, batch_size=16, lr=2e-5, warmup_steps=50,
    max_length=512, loss_fn=loss_fn, lora=lora,
    save_dir="./output/adapter_r1",
)
trainer = khoji.Trainer(MODEL, config_r1)
trainer.train(khoji.TripletDataset(triplets))

# ── Round 2: re-mine with fine-tuned model, train again
ft_model = khoji.EmbeddingModel(MODEL, adapter_path="./output/adapter_r1")
triplets_r2 = khoji.build_mixed_negatives(
    train_ds, ft_model,
    n_random=2, n_hard=1, top_k=50,
    batch_size=64,              # encoding batch size for mining
)
del ft_model

config_r2 = khoji.TrainingConfig(
    epochs=3, batch_size=16, lr=1e-5,  # halved LR for round 2
    warmup_steps=30, max_length=512,
    loss_fn=loss_fn, lora=lora,
    save_dir="./output/adapter_final",
)
trainer_r2 = khoji.Trainer(MODEL, config_r2, adapter_path="./output/adapter_r1")
trainer_r2.train(khoji.TripletDataset(triplets_r2))

# ── Evaluate fine-tuned model ──────────────────────────
ft_eval = khoji.Evaluator(MODEL, adapter_path="./output/adapter_final")
ft_result = ft_eval.evaluate(
    dataset_name="fiqa",
    k_values=[1, 5, 10],
    batch_size=64,              # encoding batch size for eval
    dataset=test_ds,
)
ft_result.print()</div>

<!-- Using Your Own Data -->
<div class="concept-card">
<h3>Using Your Own Data</h3>
<p>You can use any text retrieval dataset — from local files or Python objects. The only requirement is queries, corpus, and relevance judgments (qrels).</p>
<div class="code-block"># Option 1: From local files
# Create these files:
#   my_data/queries.jsonl  — {{"_id": "q1", "text": "your query text"}}
#   my_data/corpus.jsonl   — {{"_id": "d1", "text": "your document text", "title": "optional title"}}
#   my_data/qrels.tsv      — q1\td1\t1  (tab-separated, no header)

dataset = khoji.load_custom("./my_data")

# Option 2: From Python dicts (any source: database, CSV, API)
dataset = khoji.RetrievalDataset(
    queries={{"q1": "What is compound interest?"}},
    corpus={{"d1": "Compound interest is interest on interest.", "d2": "Stocks are..."}},
    qrels={{"q1": {{"d1": 1}}}},
)</div>
</div>

<!-- Custom Model Support -->
<div class="concept-card">
<h3>Custom Model Support</h3>
<p>Use any HuggingFace text encoder, or bring your own <code>nn.Module</code>. khoji auto-detects pooling and architecture for HuggingFace models. For fully custom models, you control every detail.</p>
<div class="code-block"># ── Using any HuggingFace model ──────────────────────
# Just change the model name — khoji auto-detects pooling and architecture
trainer = khoji.Trainer("BAAI/bge-base-en-v1.5", config)
trainer = khoji.Trainer("sentence-transformers/all-mpnet-base-v2", config)
trainer = khoji.Trainer("intfloat/e5-large-v2", config)

# ── Custom nn.Module ─────────────────────────────────
# Your model must: forward(input_ids, attention_mask) → object with .last_hidden_state
import torch
import torch.nn as nn

class MyEncoder(nn.Module):
    def __init__(self, vocab_size=30522, hidden=256, layers=4):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embeddings(input_ids)
        x = self.transformer(x)
        # khoji expects .last_hidden_state
        return type('Output', (), {{'last_hidden_state': x}})()

from transformers import AutoTokenizer
model = MyEncoder()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

trainer = khoji.Trainer(
    model=model,
    tokenizer=tokenizer,
    pooling="mean",            # "cls", "mean", "max", "weightedmean", "lasttoken"
    config=config,
)
history = trainer.train(khoji.TripletDataset(triplets))</div>
</div>

<!-- All Training Parameters -->
<div class="concept-card">
<h3>All Training Parameters</h3>
<p>Complete reference for <code>TrainingConfig</code>. All parameters have sensible defaults — you only need to set what you want to change.</p>
<table class="param-tbl">
<thead><tr><th>Parameter</th><th>Default</th><th>Description</th></tr></thead>
<tbody>
<tr><td>epochs</td><td>3</td><td>Number of training epochs per mining round</td></tr>
<tr><td>batch_size</td><td>16</td><td>Training batch size (per GPU)</td></tr>
<tr><td>grad_accum_steps</td><td>1</td><td>Gradient accumulation steps; effective batch = batch_size &times; grad_accum_steps</td></tr>
<tr><td>lr</td><td>2e-5</td><td>Peak learning rate (with linear warmup)</td></tr>
<tr><td>weight_decay</td><td>0.01</td><td>AdamW weight decay for regularization</td></tr>
<tr><td>warmup_steps</td><td>50</td><td>Linear warmup steps before reaching peak LR</td></tr>
<tr><td>max_grad_norm</td><td>1.0</td><td>Gradient clipping norm to prevent exploding gradients</td></tr>
<tr><td>max_length</td><td>512</td><td>Maximum token length for text inputs</td></tr>
<tr><td>mixed_precision</td><td>None</td><td>Automatic mixed precision: None (off), "fp16", or "bf16"</td></tr>
<tr><td>loss_fn</td><td>infonce</td><td>Loss function: "infonce", "triplet", or "contrastive"</td></tr>
<tr><td>lora</td><td>None</td><td>LoRASettings(r, alpha, dropout) for parameter-efficient fine-tuning</td></tr>
<tr><td>save_dir</td><td>"./output"</td><td>Directory to save adapter weights and checkpoints</td></tr>
<tr><td>overfit_batches</td><td>None</td><td>If set, train on only N batches (for debugging)</td></tr>
<tr><td>sanity_check_samples</td><td>10</td><td>Number of samples to run before training to verify the pipeline</td></tr>
<tr><td>save_every_n_steps</td><td>None</td><td>Save a checkpoint every N steps (None = end of epoch only)</td></tr>
<tr><td>keep_all_checkpoints</td><td>False</td><td>If True, keep all checkpoints; otherwise keep only the latest</td></tr>
</tbody>
</table>
</div>

<!-- Metrics -->
<h3>Results</h3>
{metrics_table_html(baseline_m, finetuned_m)}

<div class="insight-box">
<b>Key insight:</b> MiniLM (22M params) with LoRA fine-tuning achieves nDCG@10 = {finetuned_m["ndcg@10"]:.4f}, up from {baseline_m["ndcg@10"]:.4f} baseline. This approaches BGE-base's 0.3909 (a 110M-parameter model, 5x larger) at a fraction of the inference cost. Fine-tuning a small, fast model for your domain can match or exceed larger general-purpose models.
</div>

<!-- Retrieval Examples -->
<h3>Retrieval Examples — Before vs. After</h3>
<p>Below are {len(selected_qids)} queries where fine-tuning made a visible difference. Green = relevant, red = not relevant. Notice how the fine-tuned model pushes relevant documents to the top and pushes irrelevant ones down.</p>
{examples_html}
'''


# ════════════════════════════════════════════════════════
#  TAB 3: TEXT -> IMAGE RETRIEVAL
# ════════════════════════════════════════════════════════


def tab3_multimodal(blog_figures: dict) -> str:
    summary = load_json("multimodal_summary.json")
    baseline_data = load_json("multimodal_baseline.json")
    finetuned_data = load_json("multimodal_finetuned.json")

    selected_qids = summary["selected_qids"]
    baseline_m = summary["baseline_metrics"]
    finetuned_m = summary["finetuned_metrics"]
    training = summary["training"]

    # Build retrieval examples
    examples_html = ""
    for qid in selected_qids:
        if qid not in baseline_data or qid not in finetuned_data:
            continue
        bq = baseline_data[qid]
        fq = finetuned_data[qid]
        examples_html += image_example_html(
            qid, bq["query_text"], bq["retrieved"], fq["retrieved"], top_k=5
        )

    # Blog figures
    blog_fig_html = ""
    for fig_name in ["multimodal_example_1.png", "multimodal_example_2.png", "multimodal_example_3.png"]:
        uri = blog_figure_uri(blog_figures, fig_name)
        if uri:
            blog_fig_html += f'''<div class="blog-figure">
<img src="{uri}" class="blog-figure-img" alt="{fig_name}">
</div>'''

    return f'''
<h2>Text &rarr; Image Retrieval</h2>
<p class="lead">Fine-tune CLIP (ViT-B/32) for satellite and aerial image retrieval on RSICD. CLIP was trained on billions of web images and their alt-text — it has never seen satellite imagery, so it struggles to match descriptions like "dense residential area with roads" to overhead photos. Fine-tuning with LoRA on both vision and text encoders teaches it the visual language of remote sensing.</p>

<!-- Problem Statement -->
<div class="concept-card">
<h3>The Problem</h3>
<p>RSICD (Remote Sensing Image Caption Dataset) contains satellite/aerial images with human-written text captions describing what is visible from above: buildings, roads, agricultural fields, airports, harbors, parking lots, and more.</p>
<p>CLIP's pretrained knowledge of "airport" comes from web photos of airport terminals, not bird's-eye views of runways. Similarly, "residential area" in CLIP means street-level neighborhood photos, not the geometric grid patterns visible from 1,000 meters up. The visual vocabulary of remote sensing is fundamentally different from web imagery.</p>
</div>

<!-- Dataset -->
<div class="concept-card">
<h3>Dataset: RSICD</h3>
<p>RSICD contains ~10,000 satellite images across 30+ categories (airport, bridge, commercial, desert, farmland, forest, harbor, industrial, meadow, parking, railway, residential, river, stadium, storage, etc.) with ~44,000 text captions. Each image has 3-5 captions from different annotators.</p>
<p>khoji auto-downloads the dataset from HuggingFace:</p>
<div class="code-block">from khoji.multimodal_dataset import load_rsicd

train_ds = load_rsicd(split="train")  # auto-downloads from HuggingFace
test_ds  = load_rsicd(split="test")

print(f"Train: {{len(train_ds.queries)}} captions, {{len(train_ds.corpus)}} images")
print(f"Test:  {{len(test_ds.queries)}} captions, {{len(test_ds.corpus)}} images")</div>
</div>

<!-- Setup -->
<div class="setup-box">
<h3>Setup</h3>
<div class="setup-grid">
<div class="si"><span class="lb">Model</span><span class="vl">{summary["model"]} (151M params)</span></div>
<div class="si"><span class="lb">Dataset</span><span class="vl">{summary["dataset"]} — {summary["num_queries"]:,} test captions, {summary["corpus_size"]:,} images</span></div>
<div class="si"><span class="lb">Negatives</span><span class="vl">{training["negatives"]}</span></div>
<div class="si"><span class="lb">Training</span><span class="vl">{training["mining_rounds"]} mining rounds, {training["epochs_per_round"]} epochs/round, LR={training["lr"]}</span></div>
<div class="si"><span class="lb">LoRA</span><span class="vl">{training["lora"]}, dropout=0.1</span></div>
<div class="si"><span class="lb">Loss</span><span class="vl">InfoNCE (&tau;=0.05) with in-batch negatives</span></div>
</div>
<div class="code-block">git clone https://github.com/suyashh94/khoji.git && cd khoji
pip install -e .
python scripts/train_multimodal_retrieval.py</div>
</div>

<!-- Complete Training Code -->
<h3>Complete Training Code</h3>
<p>YAML config approach — define everything in a config, one call does the rest:</p>
<div class="code-block"># rsicd_config.yaml
model:
  name: openai/clip-vit-base-patch32
  lora_target: both              # fine-tune BOTH vision + text encoders
data:
  dataset: arampacha/rsicd       # auto-downloads from HuggingFace
  split: train
  negatives: mixed               # random + hard
  n_random: 2
  n_hard: 1
  top_k: 50
  skip_top: 5
  mining_rounds: 2
lora:
  r: 16
  alpha: 32
  dropout: 0.1
train:
  epochs: 3
  batch_size: 16
  grad_accum_steps: 2            # effective batch = 32
  lr: 2e-5
  warmup_steps: 50
  max_length: 77                 # CLIP's text token limit
  loss: infonce
  temperature: 0.05
eval:
  k_values: [1, 5, 10]
  split: test
  run_before: true
  run_after: true
output_dir: ./output/multimodal-retrieval</div>

<div class="code-block"># Run with the config
from khoji.multimodal_config import MultimodalForgeConfig
from khoji.multimodal_run import run_multimodal

config = MultimodalForgeConfig.from_yaml("rsicd_config.yaml")
result = run_multimodal(config)</div>

<p>Or the manual API:</p>
<div class="code-block">from functools import partial
import khoji
from khoji.loss import infonce_loss
from khoji.multimodal_data import build_mixed_negatives_multimodal
from khoji.multimodal_dataset import load_rsicd

MODEL = "openai/clip-vit-base-patch32"
loss_fn = partial(infonce_loss, temperature=0.05)
lora = khoji.LoRASettings(r=16, alpha=32, dropout=0.1)

# ── Load data ──────────────────────────────────────────
train_ds = load_rsicd(split="train")
test_ds  = load_rsicd(split="test")

# ── Baseline evaluation ────────────────────────────────
base_eval = khoji.MultimodalEvaluator(MODEL)
base_result = base_eval.evaluate(
    dataset_name="rsicd",
    k_values=[1, 5, 10],
    batch_size=64,              # encoding batch size for eval
    dataset=test_ds,
)
base_result.print()
del base_eval

# ── Round 1: mine + train ─────────────────────────────
mining_model = khoji.MultimodalEmbeddingModel(MODEL)
triplets = build_mixed_negatives_multimodal(
    train_ds, mining_model,
    n_random=2, n_hard=1, top_k=50, skip_top=5,
    batch_size=64,              # encoding batch size for mining
)
del mining_model

config_r1 = khoji.MultimodalTrainingConfig(
    epochs=3, batch_size=16, grad_accum_steps=2,
    lr=2e-5, warmup_steps=50, max_length=77,
    loss_fn=loss_fn, lora=lora, lora_target="both",
    save_dir="./output/clip-rsicd/adapter_r1",
    base_dir=train_ds.base_dir,
)
trainer = khoji.MultimodalTrainer(MODEL, config_r1)
trainer.train(khoji.MultimodalTripletDataset(triplets))

# ── Round 2: re-mine with fine-tuned CLIP ─────────────
ft_mining = khoji.MultimodalEmbeddingModel(
    MODEL, adapter_path="./output/clip-rsicd/adapter_r1",
)
triplets_r2 = build_mixed_negatives_multimodal(
    train_ds, ft_mining,
    n_random=2, n_hard=1, top_k=50, skip_top=5,
    batch_size=64,              # encoding batch size for mining
)
del ft_mining

config_r2 = khoji.MultimodalTrainingConfig(
    epochs=3, batch_size=16, grad_accum_steps=2,
    lr=1e-5, warmup_steps=30, max_length=77,   # halved LR
    loss_fn=loss_fn, lora=lora, lora_target="both",
    save_dir="./output/clip-rsicd/adapter_final",
    base_dir=train_ds.base_dir,
)
trainer_r2 = khoji.MultimodalTrainer(
    MODEL, config_r2, adapter_path="./output/clip-rsicd/adapter_r1",
)
trainer_r2.train(khoji.MultimodalTripletDataset(triplets_r2))

# ── Evaluate ───────────────────────────────────────────
ft_eval = khoji.MultimodalEvaluator(
    MODEL, adapter_path="./output/clip-rsicd/adapter_final",
)
ft_result = ft_eval.evaluate(
    dataset_name="rsicd",
    k_values=[1, 5, 10],
    batch_size=64,              # encoding batch size for eval
    dataset=test_ds,
)
ft_result.print()</div>

<!-- Using Your Own Data -->
<div class="concept-card">
<h3>Using Your Own Data</h3>
<p>Provide text queries and image corpus items. Images can be local file paths or URLs.</p>
<div class="code-block"># Local files:
#   my_images/queries.jsonl  — {{"_id": "q1", "text": "a dog playing fetch"}}
#   my_images/corpus.jsonl   — {{"_id": "d1", "image": "images/dog.jpg"}}  (relative paths or URLs)
#   my_images/qrels.tsv      — q1\td1\t1

dataset = khoji.load_custom_multimodal("./my_images")

# Or from Python:
dataset = khoji.MultimodalRetrievalDataset(
    queries={{"q1": "a dog playing fetch"}},
    corpus={{"d1": "images/dog.jpg", "d2": "images/cat.jpg"}},
    qrels={{"q1": {{"d1": 1}}}},
    base_dir="./my_images",
)</div>
</div>

<!-- Custom Model Support -->
<div class="concept-card">
<h3>Custom Model Support</h3>
<p>Use any CLIP or SigLIP variant from HuggingFace, or provide fully custom text and image encoders. For custom encoders, you supply the model (holding trainable parameters) and two encoding functions.</p>
<div class="code-block"># ── Any CLIP/SigLIP variant ──────────────────────────
trainer = khoji.MultimodalTrainer("openai/clip-vit-large-patch14", config)
trainer = khoji.MultimodalTrainer("google/siglip-base-patch16-224", config)

# ── Fully custom encoders ────────────────────────────
import torch.nn as nn

class MyVisionTextModel(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.text_proj = nn.Linear(768, embed_dim)
        self.image_proj = nn.Linear(512, embed_dim)

    def forward(self, x):
        return x

# Your encode functions receive raw inputs and return embeddings
def my_text_encoder(texts: list[str]) -> torch.Tensor:
    # texts = ["a dog playing fetch", "sunset over ocean"]
    # Return: Tensor of shape (batch, embed_dim)
    tokenized = ...  # your tokenization
    features = my_model.text_proj(tokenized)
    return features

def my_image_encoder(image_paths: list[str]) -> torch.Tensor:
    # image_paths = ["images/dog.jpg", "images/sunset.jpg"]
    # NOTE: receives file paths, not PIL images
    # Return: Tensor of shape (batch, embed_dim)
    images = [load_and_preprocess(p) for p in image_paths]
    features = my_model.image_proj(torch.stack(images))
    return features

trainer = khoji.MultimodalTrainer(
    model=my_model,                 # nn.Module (holds trainable params)
    encode_text_fn=my_text_encoder, # (list[str]) -> Tensor
    encode_image_fn=my_image_encoder, # (list[str]) -> Tensor (file paths!)
    config=config,
)

# ── LoRA targeting ───────────────────────────────────
# Control which encoder gets fine-tuned:
config = khoji.MultimodalTrainingConfig(
    lora_target="both",     # fine-tune vision + text encoders
    # lora_target="vision", # only vision (for domain-specific images)
    # lora_target="text",   # only text (for domain-specific queries)
)</div>
</div>

<!-- All Training Parameters -->
<div class="concept-card">
<h3>All Training Parameters</h3>
<p>Complete reference for <code>MultimodalTrainingConfig</code>. Inherits all text parameters plus multimodal-specific options.</p>
<table class="param-tbl">
<thead><tr><th>Parameter</th><th>Default</th><th>Description</th></tr></thead>
<tbody>
<tr><td>epochs</td><td>3</td><td>Number of training epochs per mining round</td></tr>
<tr><td>batch_size</td><td>16</td><td>Training batch size (per GPU)</td></tr>
<tr><td>grad_accum_steps</td><td>1</td><td>Gradient accumulation steps; effective batch = batch_size &times; grad_accum_steps</td></tr>
<tr><td>lr</td><td>2e-5</td><td>Peak learning rate (with linear warmup)</td></tr>
<tr><td>weight_decay</td><td>0.01</td><td>AdamW weight decay for regularization</td></tr>
<tr><td>warmup_steps</td><td>50</td><td>Linear warmup steps before reaching peak LR</td></tr>
<tr><td>max_grad_norm</td><td>1.0</td><td>Gradient clipping norm to prevent exploding gradients</td></tr>
<tr><td>max_length</td><td>77</td><td>Maximum token length for text inputs (CLIP default: 77)</td></tr>
<tr><td>mixed_precision</td><td>None</td><td>Automatic mixed precision: None (off), "fp16", or "bf16"</td></tr>
<tr><td>loss_fn</td><td>infonce</td><td>Loss function: "infonce", "triplet", or "contrastive"</td></tr>
<tr><td>lora</td><td>None</td><td>LoRASettings(r, alpha, dropout) for parameter-efficient fine-tuning</td></tr>
<tr><td>lora_target</td><td>"both"</td><td>Which encoders get LoRA: "both", "vision", or "text"</td></tr>
<tr><td>save_dir</td><td>"./output"</td><td>Directory to save adapter weights and checkpoints</td></tr>
<tr><td>base_dir</td><td>""</td><td>Base directory for resolving relative image paths</td></tr>
<tr><td>overfit_batches</td><td>None</td><td>If set, train on only N batches (for debugging)</td></tr>
<tr><td>sanity_check_samples</td><td>10</td><td>Number of samples to run before training to verify the pipeline</td></tr>
<tr><td>save_every_n_steps</td><td>None</td><td>Save a checkpoint every N steps (None = end of epoch only)</td></tr>
<tr><td>keep_all_checkpoints</td><td>False</td><td>If True, keep all checkpoints; otherwise keep only the latest</td></tr>
</tbody>
</table>
</div>

<!-- Metrics -->
<h3>Results</h3>
{metrics_table_html(baseline_m, finetuned_m)}

<div class="insight-box">
<b>Key insight:</b> Recall@10 nearly doubles from {baseline_m["recall@10"]:.2%} to {finetuned_m["recall@10"]:.2%}. nDCG@10 jumps from {baseline_m["ndcg@10"]:.4f} to {finetuned_m["ndcg@10"]:.4f} (+{finetuned_m["ndcg@10"] - baseline_m["ndcg@10"]:.4f}). The pretrained CLIP model barely understands satellite imagery — after fine-tuning with LoRA on both the vision encoder (learns aerial visual patterns) and text encoder (learns remote sensing vocabulary), it learns the visual language of overhead imagery.
</div>

<!-- Blog Figures -->
{f"""<h3>Training Examples — Blog Figures</h3>
<p>These figures from the khoji blog show actual before/after retrieval results during development:</p>
<div class="blog-figures-container">{blog_fig_html}</div>""" if blog_fig_html else ""}

<!-- Retrieval Examples -->
<h3>Retrieval Examples — Before vs. After</h3>
<p>For each query caption, we show the top-5 retrieved satellite images. Green border = correct match, red border = wrong image. Notice how the baseline CLIP returns visually similar but semantically incorrect images, while the fine-tuned model finds the right category.</p>
{examples_html}
'''


# ════════════════════════════════════════════════════════
#  TAB 4: COMPOSED RETRIEVAL
# ════════════════════════════════════════════════════════


def tab4_composed(blog_figures: dict) -> str:
    # Hardcoded metrics from the longer FashionIQ training run
    baseline_metrics = {
        "recall@1": 0.0000,
        "recall@5": 0.1064,
        "recall@10": 0.1489,
        "recall@50": 0.2872,
    }
    finetuned_metrics = {
        "recall@1": 0.0638,
        "recall@5": 0.2234,
        "recall@10": 0.2979,
        "recall@50": 0.4574,
    }

    # Blog figures for composed retrieval examples
    composed_figs_html = ""
    for fig_name in ["composed_example_1.png", "composed_example_2.png"]:
        uri = blog_figure_uri(blog_figures, fig_name)
        if uri:
            composed_figs_html += f'''<div class="blog-figure">
<img src="{uri}" class="blog-figure-img" alt="{fig_name}">
</div>'''

    return f'''
<h2>Composed Retrieval (img+txt &rarr; img)</h2>
<p class="lead">Fine-tune BLIP-2 for composed image retrieval on FashionIQ: given a reference dress photo and a modification caption like "make it shorter and red," find the matching target dress from a gallery of 11,000 images. This is fundamentally different from text-only or image-only search — the model must <i>combine</i> visual and textual understanding.</p>

<!-- Problem Statement -->
<div class="concept-card">
<h3>The Problem</h3>
<p>Imagine browsing an online store. You see a blue floor-length dress and think: "I like this, but I want it shorter and in red." You cannot search for that with text alone (you would need to describe the entire dress). You cannot search with the image alone (you would get similar blue dresses). You need <b>composed retrieval</b>: reference image + modification text &rarr; target image.</p>

<div class="diagram-box">
<div class="diagram-title">Composed Query</div>
<div class="diagram-content">
<pre class="diagram-pre">
  ┌────────────────┐     ┌───────────────────────┐
  │ Reference Image │  +  │ "make it shorter      │
  │  (blue dress)   │     │  and in a red color"  │
  └────────┬───────┘     └───────────┬───────────┘
           │                         │
           └─────────┬───────────────┘
                     ▼
            ┌────────────────┐
            │ BLIP-2 Q-Former│  ← fuses image + text
            └────────┬───────┘
                     ▼
            ┌────────────────┐
            │ Query Embedding│
            └────────┬───────┘
                     │
                     ▼  cosine similarity search
            ┌────────────────────────────────────────┐
            │            Image Gallery (11K)          │
            │  [dress_1] [dress_2] ... [dress_11000]  │
            └────────────────────────────────────────┘
                     │
                     ▼
            Target: short red dress matching the style
</pre>
</div>
</div>
</div>

<!-- How BLIP-2 works -->
<div class="concept-card">
<h3>How BLIP-2 Fuses Image + Text</h3>
<p>BLIP-2's Q-Former is the key component. It uses a set of learned query tokens that attend to both the frozen image encoder output and the text input through cross-attention layers:</p>

<div class="diagram-box">
<div class="diagram-title">BLIP-2 Architecture for Composed Retrieval</div>
<div class="diagram-content">
<pre class="diagram-pre">
  Reference Image                    Modification Text
       │                                    │
       ▼                                    ▼
  ┌──────────────┐                  ┌──────────────┐
  │ ViT-G/14     │                  │ BERT Tokenizer│
  │ (frozen)     │                  │              │
  └──────┬───────┘                  └──────┬───────┘
         │ image features                  │ text tokens
         │                                 │
         └────────────┬────────────────────┘
                      ▼
              ┌───────────────┐
              │   Q-Former    │  ← 32 learned query tokens
              │               │    cross-attend to image + text
              │  LoRA adapts  │  ← we fine-tune HERE
              │  these layers │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Joint Embedding│  → cosine similarity with gallery
              └───────────────┘
</pre>
</div>
</div>

<p>LoRA adapters are inserted into the Q-Former attention layers. The massive ViT-G image encoder (1B+ params) stays frozen. Only the small Q-Former cross-attention layers are fine-tuned, making training feasible on a single GPU.</p>
</div>

<!-- Dataset -->
<div class="concept-card">
<h3>Dataset: FashionIQ</h3>
<p>FashionIQ contains ~12,000 triplets for the dress category: (reference image, modification caption, target image). Each query has two natural language captions describing the desired modification. The gallery contains ~11,000 dress images.</p>
<div class="code-block"># Example annotations
{{
  "candidate": "B006M3KRAQ",     // reference dress ASIN
  "target": "B0089RPGPO",        // target dress ASIN
  "captions": [
    "is more flowy and is white instead of yellow",
    "is longer with a white color and more flowing fabric"
  ]
}}</div>
</div>

<!-- Setup -->
<div class="setup-box">
<h3>Setup</h3>
<div class="setup-grid">
<div class="si"><span class="lb">Model</span><span class="vl">Salesforce/blip2-itm-vit-g (BLIP-2)</span></div>
<div class="si"><span class="lb">Dataset</span><span class="vl">FashionIQ — 12K queries (dress), 11K gallery images</span></div>
<div class="si"><span class="lb">Negatives</span><span class="vl">Mixed (4 random + 1 hard), skip_top=100</span></div>
<div class="si"><span class="lb">Training</span><span class="vl">5 epochs, InfoNCE (&tau;=0.05), LR=2e-5</span></div>
<div class="si"><span class="lb">LoRA</span><span class="vl">r=8, alpha=16, dropout=0.1 (Q-Former layers)</span></div>
<div class="si"><span class="lb">GPU Memory</span><span class="vl">~18 GB (batch_size=8 with grad accumulation)</span></div>
</div>
<div class="code-block">git clone https://github.com/suyashh94/khoji.git && cd khoji
pip install -e .
# Download FashionIQ annotations
python scripts/fashioniq/download_data.py
# Train
python scripts/train_composed_retrieval.py --category dress</div>
</div>

<!-- Complete Training Code -->
<h3>Complete Training Code</h3>
<p>The config-driven approach handles everything in one call:</p>
<div class="code-block">import khoji
from khoji.config import EvalConfig, LoRAConfig, TrainConfig

# First, build a ComposedRetrievalDataset from FashionIQ annotations
# (see scripts/train_composed_retrieval_api.py for full data loading code)
train_ds = build_composed_dataset("./data/fashioniq", "dress", "train", url_mapping)
val_ds   = build_composed_dataset("./data/fashioniq", "dress", "val", url_mapping)

config = khoji.ComposedForgeConfig(
    model=khoji.composed_config.ComposedModelConfig(
        name="Salesforce/blip2-itm-vit-g",
    ),
    data=khoji.composed_config.ComposedDataConfig(
        dataset="./data/fashioniq/train",   # path to saved dataset
        negatives="mixed",
        n_random=4,
        n_hard=1,
        top_k=50,
        skip_top=100,
        mining_rounds=1,
    ),
    lora=LoRAConfig(r=8, alpha=16, dropout=0.1),
    train=TrainConfig(
        epochs=5,
        batch_size=8,
        lr=2e-5,
        warmup_steps=50,
        loss="infonce",
        temperature=0.05,
        max_length=77,
    ),
    eval=EvalConfig(
        dataset="./data/fashioniq/val",
        k_values=[1, 5, 10, 50],
        n_queries=100,
        run_before=True,
        run_after=True,
    ),
    seed=42,
    output_dir="./output/composed-retrieval",
)

# One call — handles mining, training, evaluation
result = khoji.run_composed(config)</div>

<p>Or the manual API for full control over each step:</p>
<div class="code-block">import random
from functools import partial
from pathlib import Path

import torch
from tqdm import tqdm

import khoji
from khoji.image_utils import load_image
from khoji.loss import infonce_loss

MODEL = "Salesforce/blip2-itm-vit-g"
DATA_DIR = Path("./data/fashioniq")
CATEGORY = "dress"

# ── Load FashionIQ data ────────────────────────────────
import json

with open(DATA_DIR / f"captions/cap.{{CATEGORY}}.train.json") as f:
    annotations = json.load(f)

with open(DATA_DIR / f"image_splits/split.{{CATEGORY}}.train.json") as f:
    gallery_ids = json.load(f)

# Load ASIN -> image URL mapping (auto-downloads)
from scripts.train_composed_retrieval import load_url_mapping
url_mapping = load_url_mapping(DATA_DIR, CATEGORY)

print(f"Loaded {{len(annotations)}} annotations, {{len(gallery_ids)}} gallery images")

# ── Build composed dataset ─────────────────────────────
# Convert FashionIQ annotations to khoji's ComposedRetrievalDataset
# (see scripts/train_composed_retrieval_api.py for build_composed_dataset)
train_ds = build_composed_dataset(str(DATA_DIR), CATEGORY, "train", url_mapping)
val_ds   = build_composed_dataset(str(DATA_DIR), CATEGORY, "val", url_mapping)

# ── Baseline evaluation ────────────────────────────────
baseline_eval = khoji.ComposedEvaluator(MODEL)
baseline = baseline_eval.evaluate(
    dataset=val_ds,
    k_values=[1, 5, 10, 50],
    batch_size=64,              # encoding batch size for eval
    n_queries=100,              # limit eval queries (None = all)
    corpus_size=None,           # limit corpus size (None = all)
)
baseline.print()
del baseline_eval

# ── Build triplets (mixed: 4 random + 1 hard) ─────────
mining_model = khoji.JointEmbeddingModel(MODEL)
triplets = khoji.build_mixed_negatives_composed(
    train_ds, mining_model,
    n_random=4, n_hard=1,
    top_k=50, skip_top=100,
    batch_size=64,              # encoding batch size for mining
)
del mining_model

# ── Train ──────────────────────────────────────────────
training_config = khoji.ComposedTrainingConfig(
    epochs=5, batch_size=8,
    lr=2e-5, warmup_steps=50,
    loss_fn=partial(infonce_loss, temperature=0.05),
    lora=khoji.LoRASettings(r=8, alpha=16, dropout=0.1),
    save_dir="./output/composed-adapter",
)
trainer = khoji.ComposedTrainer(MODEL, training_config)
trainer.train(khoji.ComposedTripletDataset(triplets))

# ── Evaluate fine-tuned model ──────────────────────────
ft_eval = khoji.ComposedEvaluator(
    MODEL, adapter_path="./output/composed-adapter",
)
finetuned = ft_eval.evaluate(
    dataset=val_ds,
    k_values=[1, 5, 10, 50],
    batch_size=64,              # encoding batch size for eval
    n_queries=100,              # limit eval queries (None = all)
    corpus_size=None,           # limit corpus size (None = all)
)
finetuned.print()</div>

<!-- Using Your Own Data -->
<div class="concept-card">
<h3>Using Your Own Data</h3>
<p>Composed queries have both an image and a text modifier. Corpus items are images (with optional text). At least one of image or text is required in each query.</p>
<div class="code-block"># Local files:
#   my_data/queries.jsonl  — {{"_id": "q1", "image": "imgs/ref.jpg", "text": "make it red"}}
#   my_data/corpus.jsonl   — {{"_id": "d1", "image": "imgs/target.jpg"}}
#   my_data/qrels.tsv      — q1\td1\t1
# Note: both "image" and "text" fields are optional. At least one required.

dataset = khoji.load_custom_composed("./my_data")

# Or from Python:
dataset = khoji.ComposedRetrievalDataset(
    queries={{
        "q1": ("imgs/ref_dress.jpg", "make it red"),       # (image_path, text)
        "q2": ("", "red cocktail dress"),                   # text-only query
    }},
    corpus={{
        "d1": ("imgs/red_dress.jpg", ""),                   # image-only target
        "d2": ("imgs/blue_dress.jpg", "blue silk dress"),   # image+text target
    }},
    qrels={{"q1": {{"d1": 1}}}},
    base_dir="./my_data",
)</div>
</div>

<!-- Custom Model Support -->
<div class="concept-card">
<h3>Custom Model Support</h3>
<p>Use any BLIP-2 variant from HuggingFace, or build a fully custom fusion model. Your custom model must handle three encoding modes: image-only, text-only, and joint image+text.</p>
<div class="code-block"># ── Any BLIP-2 variant ───────────────────────────────
trainer = khoji.ComposedTrainer("Salesforce/blip2-itm-vit-g", config)

# ── Custom fusion model example ──────────────────────
# Build your own image+text fusion on top of BLIP-2:
import torch
import torch.nn as nn
from transformers import Blip2ForImageTextRetrieval, AutoProcessor

class FFNFusionModel(nn.Module):
    """Replace BLIP-2's default addition fusion with a learned FFN."""
    def __init__(self, model_name="Salesforce/blip2-itm-vit-g", embed_dim=256):
        super().__init__()
        self.blip2 = Blip2ForImageTextRetrieval.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        for p in self.blip2.parameters():
            p.requires_grad = False  # freeze backbone
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def encode(self, images=None, texts=None):
        # Must handle 3 modes: image-only, text-only, joint
        device = next(self.parameters()).device
        if images is not None and texts is not None:
            inputs = self.processor(images=images, text=texts,
                return_tensors="pt", padding=True, truncation=True).to(device)
            out = self.blip2(**inputs, use_image_text_matching_head=False)
            img_emb = out.image_embeds.max(dim=1).values
            txt_emb = out.text_embeds
            return self.fusion(torch.cat([img_emb, txt_emb], dim=1))
        elif images is not None:
            inputs = self.processor(images=images, text=[""] * len(images),
                return_tensors="pt", padding=True, truncation=True).to(device)
            out = self.blip2(**inputs, use_image_text_matching_head=False)
            return out.image_embeds.max(dim=1).values
        else:
            from PIL import Image as PILImage
            dummy = [PILImage.new("RGB", (224, 224), "black")] * len(texts)
            inputs = self.processor(images=dummy, text=texts,
                return_tensors="pt", padding=True, truncation=True).to(device)
            out = self.blip2(**inputs, use_image_text_matching_head=False)
            return out.text_embeds

model = FFNFusionModel().to("cuda")
trainer = khoji.ComposedTrainer(
    model=model,
    encode_fn=model.encode,     # single function handling all 3 modes
    config=config,
)</div>
</div>

<!-- All Training Parameters -->
<div class="concept-card">
<h3>All Training Parameters</h3>
<p>Complete reference for <code>ComposedTrainingConfig</code>. Extends the multimodal config with composed-retrieval-specific options.</p>
<table class="param-tbl">
<thead><tr><th>Parameter</th><th>Default</th><th>Description</th></tr></thead>
<tbody>
<tr><td>epochs</td><td>5</td><td>Number of training epochs per mining round</td></tr>
<tr><td>batch_size</td><td>8</td><td>Training batch size (smaller due to BLIP-2 memory)</td></tr>
<tr><td>grad_accum_steps</td><td>1</td><td>Gradient accumulation steps; effective batch = batch_size &times; grad_accum_steps</td></tr>
<tr><td>lr</td><td>2e-5</td><td>Peak learning rate (with linear warmup)</td></tr>
<tr><td>weight_decay</td><td>0.01</td><td>AdamW weight decay for regularization</td></tr>
<tr><td>warmup_steps</td><td>50</td><td>Linear warmup steps before reaching peak LR</td></tr>
<tr><td>max_grad_norm</td><td>1.0</td><td>Gradient clipping norm to prevent exploding gradients</td></tr>
<tr><td>max_length</td><td>77</td><td>Maximum token length for text inputs</td></tr>
<tr><td>mixed_precision</td><td>None</td><td>Automatic mixed precision: None (off), "fp16", or "bf16"</td></tr>
<tr><td>loss_fn</td><td>infonce</td><td>Loss function: "infonce", "triplet", or "contrastive"</td></tr>
<tr><td>lora</td><td>None</td><td>LoRASettings(r, alpha, dropout) — applies to Q-Former layers</td></tr>
<tr><td>save_dir</td><td>"./output"</td><td>Directory to save adapter weights and checkpoints</td></tr>
<tr><td>base_dir</td><td>""</td><td>Base directory for resolving relative image paths</td></tr>
<tr><td>eval_batch_size</td><td>64</td><td>Batch size for evaluation encoding (can be larger than training)</td></tr>
<tr><td>mining_batch_size</td><td>64</td><td>Batch size for hard negative mining encoding</td></tr>
<tr><td>overfit_batches</td><td>None</td><td>If set, train on only N batches (for debugging)</td></tr>
<tr><td>sanity_check_samples</td><td>10</td><td>Number of samples to run before training to verify the pipeline</td></tr>
<tr><td>save_every_n_steps</td><td>None</td><td>Save a checkpoint every N steps (None = end of epoch only)</td></tr>
<tr><td>keep_all_checkpoints</td><td>False</td><td>If True, keep all checkpoints; otherwise keep only the latest</td></tr>
</tbody>
</table>
</div>

<!-- Metrics -->
<h3>Results</h3>
{metrics_table_html(baseline_metrics, finetuned_metrics, show_percent=True)}

<div class="insight-box">
<b>Key insight:</b> Recall@10 doubles from 14.89% to 29.79%, and Recall@50 jumps from 28.72% to 45.74%. The pretrained BLIP-2 has no concept of "make it redder" or "shorter sleeves" — these are relational modifications that require understanding both the reference image AND the text instruction. Fine-tuning teaches the Q-Former to fuse these modalities for compositional visual search. FashionIQ is a challenging benchmark with subtle visual differences between dresses.
</div>

<!-- Before/After Blog Figures -->
<h3>Before vs. After — Real Retrieval Examples</h3>
<p>These figures show actual FashionIQ queries with the reference dress, modification text, and top-5 retrieved results before and after fine-tuning. Green borders indicate the correct target dress.</p>
{f'<div class="blog-figures-container">{composed_figs_html}</div>' if composed_figs_html else '<p class="note">Blog figure images not available. Run the data collection script to generate them.</p>'}
'''


# ════════════════════════════════════════════════════════
#  TAB 5: MIXED-MODE SKU MATCHING
# ════════════════════════════════════════════════════════


def tab5_sku_matching() -> str:
    summary = load_json("sku_summary.json")
    sku_brand_baseline = load_json("sku_brand_baseline.json")
    sku_brand_finetuned = load_json("sku_brand_finetuned.json")
    sku_product_baseline = load_json("sku_product_baseline.json")
    sku_product_finetuned = load_json("sku_product_finetuned.json")

    cfg = summary.get("config", {})
    brand_qids = summary.get("brand_qids", [])
    product_qids = summary.get("product_qids", [])

    # Build brand test examples (pick 8 diverse ones)
    brand_examples_html = ""
    shown_families = set()
    brand_shown = 0
    for qid in brand_qids:
        if qid not in sku_brand_baseline or qid not in sku_brand_finetuned:
            continue
        bq = sku_brand_baseline[qid]
        fq = sku_brand_finetuned[qid]
        family = bq.get("query_family", "")
        # Show diverse families
        if family in shown_families and brand_shown >= 4:
            continue
        shown_families.add(family)
        brand_examples_html += sku_example_html(
            qid,
            bq["query_text"],
            bq.get("query_image_b64", ""),
            bq.get("query_brand", ""),
            bq.get("query_family", ""),
            bq.get("query_variant", ""),
            bq["retrieved"],
            fq["retrieved"],
            top_k=5,
        )
        brand_shown += 1
        if brand_shown >= 8:
            break

    # Build product test examples (pick 5)
    product_examples_html = ""
    product_shown = 0
    for qid in product_qids:
        if qid not in sku_product_baseline or qid not in sku_product_finetuned:
            continue
        bq = sku_product_baseline[qid]
        fq = sku_product_finetuned[qid]
        product_examples_html += sku_example_html(
            qid,
            bq["query_text"],
            bq.get("query_image_b64", ""),
            bq.get("query_brand", ""),
            bq.get("query_family", ""),
            bq.get("query_variant", ""),
            bq["retrieved"],
            fq["retrieved"],
            top_k=5,
        )
        product_shown += 1
        if product_shown >= 5:
            break

    train_brands = cfg.get("train_brands", ["Sainsburys", "Tesco", "Aldi", "Lidl"])
    test_brand = cfg.get("test_brand", "Waitrose")
    holdout_families = cfg.get("holdout_families", ["Crisps", "Peanut Butter"])

    return f'''
<h2>Mixed-Mode: SKU Matching (img+txt &rarr; img+txt)</h2>
<p class="lead">Match products across supermarket brands using both product photos and text descriptions simultaneously. Given a Waitrose Organic Milk (photo + description), find the equivalent Tesco, Aldi, Lidl, and Sainsbury's products from their catalogs. This is the most complex retrieval mode — both queries and documents are multimodal.</p>

<!-- Problem Statement -->
<div class="concept-card">
<h3>Why Both Modalities Matter</h3>
<p>Cross-brand product matching is a real-world problem in grocery retail, price comparison, and supply chain. Neither images nor text alone are sufficient:</p>

<div class="why-grid">
<div class="why-item fail">
<h4>Image alone fails</h4>
<p>Same product, completely different packaging across brands. Tesco milk, Aldi milk, and Waitrose milk look nothing alike — different colors, logos, label designs. An image-only model sees three completely different products.</p>
</div>
<div class="why-item fail">
<h4>Text alone fails</h4>
<p>Different brand names and slightly different wording for identical products:</p>
<div class="text-examples">
<div class="text-ex">"Sainsbury's Whole Milk, 2L, &pound;1.45"</div>
<div class="text-ex">"Tesco Whole Milk, 2L, &pound;1.42"</div>
<div class="text-ex">"Aldi Whole Milk, 2L, &pound;1.09"</div>
</div>
<p style="margin-top:6px">A text model fixates on the brand name — Sainsbury's and Tesco look different, even though the product is identical.</p>
</div>
<div class="why-item success">
<h4>Image + Text together works</h4>
<p>The model learns: "these all look like milk cartons AND their descriptions say Whole Milk 2L — they are the same product regardless of brand packaging." Joint encoding captures both visual product cues AND semantic description overlap.</p>
</div>
</div>
</div>

<!-- Dataset -->
<div class="concept-card">
<h3>Dataset</h3>
<p>{summary["dataset"]}. Each product has an AI-generated photo and a structured text description with brand, product name, size, price, and attributes.</p>
<p><b>Brands:</b> {", ".join(train_brands + [test_brand])} | <b>Families:</b> Porridge Oats, Milk, Bread, Yoghurt, Pasta, Orange Juice, Crisps, Peanut Butter | <b>Variants:</b> 5 per family (Regular, Organic, Gluten-Free, etc.)</p>
</div>

<!-- Train/Test Split -->
<div class="concept-card">
<h3>Train/Test Split — Two Evaluation Dimensions</h3>
<p>One model is trained, then evaluated on <b>two orthogonal dimensions</b> of generalization:</p>

<div class="split-grid">
<div class="split-item">
<div class="split-header brand-split">Brand Generalization</div>
<p><b>{test_brand}</b> is held out entirely — the model has never seen any {test_brand} product image or description during training. {cfg.get("test_brand_queries", 30)} {test_brand} queries are searched against the {len(train_brands)}-brand corpus ({", ".join(train_brands)}).</p>
<p><b>Question:</b> Can the model match a brand it has never seen to known brands?</p>
</div>
<div class="split-item">
<div class="split-header product-split">Product Generalization</div>
<p><b>{", ".join(holdout_families)}</b> families are held out — the model never trained on these product categories. {cfg.get("test_product_queries", 10)} queries from Sainsbury's searching the other-brand corpus for held-out products.</p>
<p><b>Question:</b> Can the model match product categories it has never seen?</p>
</div>
</div>

<div class="diagram-box">
<div class="diagram-title">Train/Test Split Diagram</div>
<div class="diagram-content">
<pre class="diagram-pre">
                    Sainsburys  Tesco  Aldi  Lidl  │ Waitrose
  ─────────────────────────────────────────────────┼──────────
  Porridge Oats     TRAIN      TRAIN  TRAIN TRAIN │ TEST (brand)
  Milk              TRAIN      TRAIN  TRAIN TRAIN │ TEST (brand)
  Bread             TRAIN      TRAIN  TRAIN TRAIN │ TEST (brand)
  Yoghurt           TRAIN      TRAIN  TRAIN TRAIN │ TEST (brand)
  Pasta             TRAIN      TRAIN  TRAIN TRAIN │ TEST (brand)
  Orange Juice      TRAIN      TRAIN  TRAIN TRAIN │ TEST (brand)
  ─────────────────────────────────────────────────┼──────────
  Crisps            TEST(prod) CORPUS CORPUS CORPUS│ (excluded)
  Peanut Butter     TEST(prod) CORPUS CORPUS CORPUS│ (excluded)

  Note: held-out families exist in the CORPUS (as retrieval targets)
  but NOT in training triplets. This is "soft leakage" — the model
  has seen Crisps images as corpus items but never as query-positive pairs.
</pre>
</div>
</div>

<div class="insight-box">
<b>Soft leakage note:</b> The held-out product families ({", ".join(holdout_families)}) appear in the corpus as potential retrieval targets (the model's "catalog"). They do NOT appear in any training triplet. This mirrors a real scenario: your catalog already contains Crisps, but no customer has ever searched for cross-brand Crisps matches. The model must generalize from other families to match Crisps correctly.
</div>
</div>

<!-- Setup -->
<div class="setup-box">
<h3>Setup &amp; Training</h3>
<div class="setup-grid">
<div class="si"><span class="lb">Model</span><span class="vl">{summary["model"]} (BLIP-2) + LoRA r={cfg.get("lora_r", 8)}</span></div>
<div class="si"><span class="lb">Negatives</span><span class="vl">{cfg.get("negatives", "mixed")} negatives</span></div>
<div class="si"><span class="lb">Training</span><span class="vl">{cfg.get("mining_rounds", 2)} mining rounds, {cfg.get("epochs", 5)} epochs/round, LR={cfg.get("lr", 2e-5)}</span></div>
<div class="si"><span class="lb">Data</span><span class="vl">{cfg.get("train_queries", 120)} queries, {cfg.get("corpus_size", 160)} corpus, ~{cfg.get("triplets_per_round", 1800)} triplets/round</span></div>
<div class="si"><span class="lb">Loss</span><span class="vl">InfoNCE (&tau;=0.05)</span></div>
<div class="si"><span class="lb">Batch Size</span><span class="vl">{cfg.get("batch_size", 32)}</span></div>
</div>
<div class="code-block">git clone https://github.com/suyashh94/khoji.git && cd khoji
pip install -e .
python scripts/train_sku_matching.py --negatives mixed --mining-rounds 2 --epochs 5</div>
</div>

<!-- Complete Training Code -->
<h3>Complete Training Code</h3>
<div class="code-block">import json
import random
from functools import partial
from pathlib import Path

import torch
import khoji
from khoji.loss import infonce_loss

MODEL = "Salesforce/blip2-itm-vit-g"
DATA_DIR = Path("./data/sku-matching")
TRAIN_BRANDS = ["Sainsburys", "Tesco", "Aldi", "Lidl"]
TEST_BRAND = "Waitrose"
HOLDOUT_FAMILIES = ["Crisps", "Peanut Butter"]

# ── Load catalog ───────────────────────────────────────
with open(DATA_DIR / "metadata" / "catalog.json") as f:
    catalog = json.load(f)

# ── Build datasets ─────────────────────────────────────
# Partition: train items = train brands x non-holdout families
train_items = [it for it in catalog
               if it["brand"] in TRAIN_BRANDS and it["family"] not in HOLDOUT_FAMILIES]
test_brand_items = [it for it in catalog
                    if it["brand"] == TEST_BRAND and it["family"] not in HOLDOUT_FAMILIES]

# Corpus: all train-brand items (including holdout families)
corpus = {{}}
for item in [it for it in catalog if it["brand"] in TRAIN_BRANDS]:
    img_path = str(DATA_DIR / item["image_path"])
    corpus[item["sku_id"]] = (img_path, item["description"])

# Queries + relevance: same (family, variant) across brands
# ... (see scripts/train_sku_matching.py for full dataset building code)

# ── Baseline evaluation ────────────────────────────────
baseline_eval = khoji.ComposedEvaluator(MODEL)
baseline = baseline_eval.evaluate(
    dataset=test_ds,
    k_values=[1, 3, 5, 10],
    batch_size=64,              # encoding batch size for eval
    n_queries=None,             # limit eval queries (None = all)
    corpus_size=None,           # limit corpus size (None = all)
)
baseline.print()
del baseline_eval

# ── Build triplets (mixed negatives) ──────────────────
mining_model = khoji.JointEmbeddingModel(MODEL)
triplets = khoji.build_mixed_negatives_composed(
    train_ds, mining_model,
    n_random=4, n_hard=1,
    top_k=50, skip_top=10,
    batch_size=64,              # encoding batch size for mining
)
del mining_model

# ── Train ──────────────────────────────────────────────
loss_fn = partial(infonce_loss, temperature=0.05)
training_config = khoji.ComposedTrainingConfig(
    epochs=5, batch_size=8,
    lr=2e-5, warmup_steps=50,
    loss_fn=loss_fn,
    lora=khoji.LoRASettings(r=8, alpha=16, dropout=0.1),
    save_dir="./output/sku-matching/adapter",
)
trainer = khoji.ComposedTrainer(MODEL, training_config)
trainer.train(khoji.ComposedTripletDataset(triplets))

# ── Evaluate fine-tuned model ──────────────────────────
ft_eval = khoji.ComposedEvaluator(
    MODEL, adapter_path="./output/sku-matching/adapter",
)
finetuned = ft_eval.evaluate(
    dataset=test_ds,
    k_values=[1, 3, 5, 10],
    batch_size=64,              # encoding batch size for eval
    n_queries=None,             # limit eval queries (None = all)
    corpus_size=None,           # limit corpus size (None = all)
)
finetuned.print()</div>

<!-- Using Your Own Data -->
<div class="concept-card">
<h3>Using Your Own Data</h3>
<p>For mixed-mode retrieval, every item (query and corpus) has <b>both</b> an image and text. Use <code>""</code> for absent modalities. The same <code>ComposedRetrievalDataset</code> class works for all composed modes.</p>
<div class="code-block"># For SKU matching, every item has BOTH image and text
dataset = khoji.ComposedRetrievalDataset(
    queries={{
        # (product_photo, product_description)
        "q1": ("photos/brand_a/oats.jpg", "Brand A Organic Oats, 1kg, £2.50"),
    }},
    corpus={{
        # Same format — both sides are img+txt
        "d1": ("photos/brand_b/oats.jpg", "Brand B Organic Oats, 1kg, £2.30"),
        "d2": ("photos/brand_b/milk.jpg", "Brand B Whole Milk, 2L, £1.40"),
    }},
    qrels={{"q1": {{"d1": 1}}}},  # Brand A oats matches Brand B oats
)</div>
<div class="insight-box">
<b>Key points:</b>
<ul class="feature-list" style="margin-top:4px">
<li>Both queries AND corpus use <code>(image, text)</code> tuples</li>
<li>Use <code>""</code> for absent modalities (e.g., text-only query: <code>("", "red cocktail dress")</code>)</li>
<li>The same <code>ComposedRetrievalDataset</code> works for all composed modes — img+txt &rarr; img, img+txt &rarr; img+txt, etc.</li>
<li>Negatives are sampled automatically — you only need to annotate positives in qrels</li>
</ul>
</div>
</div>

<!-- Custom Model Support -->
<div class="concept-card">
<h3>Custom Model Support</h3>
<p>Use any BLIP-2 variant from HuggingFace, or build a fully custom fusion model. Your custom model must handle three encoding modes: image-only, text-only, and joint image+text.</p>
<div class="code-block"># ── Any BLIP-2 variant ───────────────────────────────
trainer = khoji.ComposedTrainer("Salesforce/blip2-itm-vit-g", config)

# ── Custom fusion model example ──────────────────────
# Build your own image+text fusion on top of BLIP-2:
import torch
import torch.nn as nn
from transformers import Blip2ForImageTextRetrieval, AutoProcessor

class FFNFusionModel(nn.Module):
    """Replace BLIP-2's default addition fusion with a learned FFN."""
    def __init__(self, model_name="Salesforce/blip2-itm-vit-g", embed_dim=256):
        super().__init__()
        self.blip2 = Blip2ForImageTextRetrieval.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        for p in self.blip2.parameters():
            p.requires_grad = False  # freeze backbone
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def encode(self, images=None, texts=None):
        # Must handle 3 modes: image-only, text-only, joint
        device = next(self.parameters()).device
        if images is not None and texts is not None:
            inputs = self.processor(images=images, text=texts,
                return_tensors="pt", padding=True, truncation=True).to(device)
            out = self.blip2(**inputs, use_image_text_matching_head=False)
            img_emb = out.image_embeds.max(dim=1).values
            txt_emb = out.text_embeds
            return self.fusion(torch.cat([img_emb, txt_emb], dim=1))
        elif images is not None:
            inputs = self.processor(images=images, text=[""] * len(images),
                return_tensors="pt", padding=True, truncation=True).to(device)
            out = self.blip2(**inputs, use_image_text_matching_head=False)
            return out.image_embeds.max(dim=1).values
        else:
            from PIL import Image as PILImage
            dummy = [PILImage.new("RGB", (224, 224), "black")] * len(texts)
            inputs = self.processor(images=dummy, text=texts,
                return_tensors="pt", padding=True, truncation=True).to(device)
            out = self.blip2(**inputs, use_image_text_matching_head=False)
            return out.text_embeds

model = FFNFusionModel().to("cuda")
trainer = khoji.ComposedTrainer(
    model=model,
    encode_fn=model.encode,     # single function handling all 3 modes
    config=config,
)</div>
</div>

<!-- All Training Parameters -->
<div class="concept-card">
<h3>All Training Parameters</h3>
<p>Complete reference for <code>ComposedTrainingConfig</code> used for both composed and mixed-mode retrieval.</p>
<table class="param-tbl">
<thead><tr><th>Parameter</th><th>Default</th><th>Description</th></tr></thead>
<tbody>
<tr><td>epochs</td><td>5</td><td>Number of training epochs per mining round</td></tr>
<tr><td>batch_size</td><td>8</td><td>Training batch size (smaller due to BLIP-2 memory)</td></tr>
<tr><td>grad_accum_steps</td><td>1</td><td>Gradient accumulation steps; effective batch = batch_size &times; grad_accum_steps</td></tr>
<tr><td>lr</td><td>2e-5</td><td>Peak learning rate (with linear warmup)</td></tr>
<tr><td>weight_decay</td><td>0.01</td><td>AdamW weight decay for regularization</td></tr>
<tr><td>warmup_steps</td><td>50</td><td>Linear warmup steps before reaching peak LR</td></tr>
<tr><td>max_grad_norm</td><td>1.0</td><td>Gradient clipping norm to prevent exploding gradients</td></tr>
<tr><td>max_length</td><td>77</td><td>Maximum token length for text inputs</td></tr>
<tr><td>mixed_precision</td><td>None</td><td>Automatic mixed precision: None (off), "fp16", or "bf16"</td></tr>
<tr><td>loss_fn</td><td>infonce</td><td>Loss function: "infonce", "triplet", or "contrastive"</td></tr>
<tr><td>lora</td><td>None</td><td>LoRASettings(r, alpha, dropout) — applies to Q-Former layers</td></tr>
<tr><td>save_dir</td><td>"./output"</td><td>Directory to save adapter weights and checkpoints</td></tr>
<tr><td>base_dir</td><td>""</td><td>Base directory for resolving relative image paths</td></tr>
<tr><td>eval_batch_size</td><td>64</td><td>Batch size for evaluation encoding (can be larger than training)</td></tr>
<tr><td>mining_batch_size</td><td>64</td><td>Batch size for hard negative mining encoding</td></tr>
<tr><td>overfit_batches</td><td>None</td><td>If set, train on only N batches (for debugging)</td></tr>
<tr><td>sanity_check_samples</td><td>10</td><td>Number of samples to run before training to verify the pipeline</td></tr>
<tr><td>save_every_n_steps</td><td>None</td><td>Save a checkpoint every N steps (None = end of epoch only)</td></tr>
<tr><td>keep_all_checkpoints</td><td>False</td><td>If True, keep all checkpoints; otherwise keep only the latest</td></tr>
</tbody>
</table>
</div>

<!-- Metrics: Brand Generalization -->
<h3>Results &mdash; Brand Generalization ({test_brand})</h3>
<p>Model has never seen any {test_brand} product. Can it match {test_brand} items to the {", ".join(train_brands)} catalog?</p>
{metrics_table_html(summary["baseline_brand"], summary["finetuned_brand"])}

<!-- Metrics: Product Generalization -->
<h3>Results &mdash; Product Generalization ({", ".join(holdout_families)})</h3>
<p>Model never trained on {" or ".join(holdout_families)}. Can it match these unseen categories?</p>
{metrics_table_html(summary["baseline_product"], summary["finetuned_product"])}

<div class="insight-box">
<b>Key insight:</b> The fine-tuned model achieves <b>perfect Recall@10 = 1.0</b> on brand generalization ({test_brand} products it has never seen) and <b>perfect nDCG = 1.0 across all k values</b> on product generalization ({", ".join(holdout_families)} categories it never trained on). The model learns a general "cross-brand product matching" capability that transfers to both unseen brands and unseen product categories. The baseline was already strong (BLIP-2 with joint encoding is powerful out of the box), but fine-tuning eliminates the remaining errors.
</div>

<!-- Brand Test Examples -->
<h3>Retrieval Examples &mdash; Brand Test ({test_brand})</h3>
<p>{test_brand} queries matched against the {", ".join(train_brands)} corpus. The query product image and description are on the left; top-5 retrieved products are shown with green (correct match = same family + variant) or red (wrong product) borders.</p>
{brand_examples_html}

<!-- Product Test Examples -->
<h3>Retrieval Examples &mdash; Product Test ({", ".join(holdout_families)})</h3>
<p>Queries for held-out product families ({", ".join(holdout_families)}) that the model never trained on. These products exist in the corpus but were never part of any training triplet.</p>
{product_examples_html}
'''


# ════════════════════════════════════════════════════════
#  CSS STYLES
# ════════════════════════════════════════════════════════


def get_css() -> str:
    return '''
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: #f8fafc;
    color: #1e293b;
    line-height: 1.6;
}

/* ── Header ─────────────────────────────────────────── */
.header {
    background: #0f172a;
    color: white;
    padding: 52px 24px 44px;
    text-align: center;
    border-bottom: 4px solid #2563eb;
}
.header h1 {
    font-size: 48px;
    margin-bottom: 12px;
    letter-spacing: -1.5px;
    font-weight: 800;
    color: #f8fafc;
}
.header h1 span { color: #60a5fa; }
.header .tagline {
    font-size: 18px;
    color: #94a3b8;
    max-width: 750px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ── Container ──────────────────────────────────────── */
.container {
    max-width: 1140px;
    margin: 0 auto;
    padding: 0 24px;
}

/* ── Navigation ──────────────────────────────────────── */
.page-layout {
    display: grid;
    grid-template-columns: 260px 1fr;
    gap: 0;
    margin-top: 0;
    min-height: calc(100vh - 200px);
}
.sidebar {
    background: #fff;
    border-right: 1px solid #e2e8f0;
    padding: 20px 0;
    position: sticky;
    top: 0;
    height: fit-content;
    max-height: 100vh;
    overflow-y: auto;
}
.sidebar-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #94a3b8;
    padding: 8px 20px 4px;
    font-weight: 600;
}
.nav-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 10px 20px;
    background: none;
    border: none;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    color: #475569;
    text-align: left;
    transition: all 0.15s;
    border-left: 3px solid transparent;
}
.nav-btn:hover { background: #f1f5f9; color: #1e293b; }
.nav-btn.active { background: #eff6ff; color: #1e40af; border-left-color: #2563eb; font-weight: 600; }
.nav-icon {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
}
.nav-icon.concepts { background: #f1f5f9; }
.nav-icon.text { background: #dbeafe; }
.nav-icon.image { background: #dcfce7; }
.nav-icon.composed { background: #fef3c7; }
.nav-icon.sku { background: #fce7f3; }
.main-content {
    padding: 32px 40px;
    max-width: 900px;
}
.tab-content { display: none; }
.tab-content.active { display: block; }

/* ── Typography ─────────────────────────────────────── */
h2 { font-size: 28px; color: #0f172a; margin-bottom: 14px; letter-spacing: -0.3px; }
h3 { font-size: 19px; color: #1e3a5f; margin: 24px 0 12px; }
h4 { font-size: 15px; color: #334155; margin-bottom: 6px; }
.lead { font-size: 16px; color: #475569; line-height: 1.7; margin-bottom: 20px; }
p { font-size: 14px; line-height: 1.7; color: #475569; margin-bottom: 8px; }
ul.feature-list { margin: 8px 0 8px 20px; }
ul.feature-list li { font-size: 14px; color: #475569; margin-bottom: 4px; line-height: 1.6; }

/* ── Setup box ──────────────────────────────────────── */
.setup-box {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    margin: 16px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.setup-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 14px;
}
.si {
    background: #f8fafc;
    padding: 8px 12px;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
}
.si .lb { font-size: 11px; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px; }
.si .vl { font-size: 13px; font-weight: 600; color: #1e293b; }

/* ── Metrics table ──────────────────────────────────── */
.metrics-tbl {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    background: white;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.metrics-tbl th {
    background: #f1f5f9;
    padding: 10px 16px;
    text-align: left;
    font-size: 12px;
    text-transform: uppercase;
    color: #64748b;
    letter-spacing: 0.5px;
}
.metrics-tbl td {
    padding: 10px 16px;
    border-bottom: 1px solid #f1f5f9;
    font-size: 14px;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
}
.metrics-tbl tr:last-child td { border-bottom: none; }

/* ── Code block ─────────────────────────────────────── */
.code-block {
    background: #0f172a;
    color: #e2e8f0;
    padding: 18px 20px;
    border-radius: 10px;
    font-family: 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace;
    font-size: 13px;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-x: auto;
    margin: 14px 0;
    line-height: 1.55;
    border: 1px solid #1e293b;
}

/* ── Concept cards ──────────────────────────────────── */
.concept-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 22px;
    margin: 16px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.concept-card h3 { margin-top: 0; }

/* ── Diagrams ───────────────────────────────────────── */
.diagram-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    margin: 14px 0;
    overflow: hidden;
}
.diagram-title {
    background: #f1f5f9;
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid #e2e8f0;
}
.diagram-content { padding: 16px; overflow-x: auto; }
.diagram-pre {
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-size: 12px;
    line-height: 1.5;
    color: #334155;
    margin: 0;
    white-space: pre;
}
.diagram-pre-sm {
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-size: 11px;
    line-height: 1.5;
    color: #334155;
    margin: 0;
    white-space: pre;
}

/* ── Loss/mining grids ──────────────────────────────── */
.loss-grid { display: grid; gap: 14px; margin-top: 12px; }
.loss-item {
    background: #f8fafc;
    padding: 16px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}
.loss-header {
    font-size: 15px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 8px;
}
.loss-header.highlight {
    color: #2563eb;
}
.formula-box {
    background: #0f172a;
    padding: 10px 14px;
    border-radius: 6px;
    margin: 8px 0;
    display: inline-block;
}
.formula {
    color: #93c5fd;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-size: 13px;
}
.loss-desc { font-size: 13px; color: #475569; margin: 8px 0; }
.loss-when {
    font-size: 12px;
    color: #64748b;
    background: #fff;
    padding: 8px 12px;
    border-radius: 6px;
    margin-top: 8px;
    border-left: 3px solid #2563eb;
}

.mining-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
    margin-top: 12px;
}
.mining-item {
    background: #f8fafc;
    padding: 14px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}
.mining-header {
    padding: 5px 12px;
    border-radius: 6px;
    color: white;
    font-weight: 700;
    font-size: 13px;
    display: inline-block;
    margin-bottom: 8px;
}
.mining-header.random { background: #3b82f6; }
.mining-header.hard { background: #ef4444; }
.mining-header.mixed { background: #16a34a; }
.mining-item p { font-size: 13px; color: #64748b; margin: 0; }
.mining-diagram {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px;
    margin: 8px 0;
    overflow-x: auto;
}

/* ── Triplet data box ───────────────────────────────── */
.triplet-data-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
    margin-top: 14px;
}
.triplet-data-box h4 { margin-bottom: 6px; }

/* ── Modes table ────────────────────────────────────── */
.modes-tbl {
    width: 100%;
    border-collapse: collapse;
    background: white;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin: 14px 0;
}
.modes-tbl thead th {
    background: #f1f5f9;
    padding: 10px 14px;
    text-align: left;
    font-size: 12px;
    text-transform: uppercase;
    color: #64748b;
}
.modes-tbl td {
    padding: 10px 14px;
    border-bottom: 1px solid #f1f5f9;
    font-size: 13px;
}
.modes-tbl tr:last-child td { border-bottom: none; }
.mode-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 12px;
    color: white;
}
.mode-text { background: #3b82f6; }
.mode-multimodal { background: #8b5cf6; }
.mode-composed { background: #f59e0b; color: #1e293b; }
.mode-mixed { background: #16a34a; }

/* ── Insight box ────────────────────────────────────── */
.insight-box {
    background: #eff6ff;
    border-left: 4px solid #2563eb;
    padding: 16px 20px;
    border-radius: 0 10px 10px 0;
    margin: 18px 0;
}
.insight-box b { color: #1e3a5f; }
.insight-box p { margin: 0; }

/* ── Example blocks ─────────────────────────────────── */
.example-block {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px;
    margin: 16px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.example-query {
    margin-bottom: 14px;
    padding-bottom: 14px;
    border-bottom: 1px solid #f1f5f9;
}
.query-label {
    font-size: 11px;
    text-transform: uppercase;
    color: #94a3b8;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
    font-weight: 600;
}
.query-text {
    font-size: 15px;
    color: #0f172a;
    font-weight: 600;
    line-height: 1.5;
}
.hit-summary {
    margin-top: 8px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.hit-badge {
    font-size: 12px;
    padding: 3px 10px;
    border-radius: 12px;
    font-weight: 600;
}
.baseline-badge { background: #fee2e2; color: #991b1b; }
.finetuned-badge { background: #dcfce7; color: #166534; }

/* ── Text results ───────────────────────────────────── */
.example-comparison {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}
.comparison-col { min-width: 0; }
.comparison-label {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 6px 12px;
    border-radius: 6px;
    margin-bottom: 10px;
    display: inline-block;
}
.before-label { background: #fee2e2; color: #991b1b; }
.after-label { background: #dcfce7; color: #166534; }

.text-result-card {
    padding: 10px 12px;
    border-radius: 8px;
    margin-bottom: 6px;
    transition: transform 0.1s;
}
.text-result-card:hover { transform: translateX(2px); }
.text-result-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}
.relevance-icon { font-size: 14px; font-weight: 700; }
.result-rank { font-size: 11px; color: #94a3b8; font-weight: 600; }
.result-score { font-size: 11px; color: #94a3b8; font-family: monospace; margin-left: auto; }
.text-result-body { font-size: 12px; color: #475569; line-height: 1.5; }

/* ── Image results ──────────────────────────────────── */
.img-results-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.img-result-card {
    text-align: center;
    width: 100px;
}
.result-img {
    width: 90px;
    height: 90px;
    border-radius: 8px;
    object-fit: cover;
    display: block;
    margin: 0 auto 4px;
}
.img-result-meta {
    display: flex;
    justify-content: center;
    gap: 6px;
    font-size: 10px;
    color: #94a3b8;
}

/* ── SKU results ────────────────────────────────────── */
.sku-query {
    display: flex;
    flex-direction: column;
}
.sku-query-inner {
    display: flex;
    gap: 14px;
    align-items: flex-start;
}
.query-img {
    width: 80px;
    height: 80px;
    border-radius: 8px;
    object-fit: cover;
    border: 2px solid #2563eb;
    flex-shrink: 0;
}
.sku-query-info { flex: 1; min-width: 0; }
.sku-query-family { font-size: 14px; font-weight: 700; color: #1e3a5f; }
.sku-result-card {
    text-align: center;
    width: 100px;
}
.sku-result-meta {
    font-size: 10px;
    line-height: 1.3;
}
.sku-brand { font-weight: 700; color: #334155; }
.sku-variant { color: #64748b; }

/* ── SKU matching specific ──────────────────────────── */
.why-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
    margin: 16px 0;
}
.why-item {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
}
.why-item.fail { border-left: 4px solid #ef4444; }
.why-item.success { border-left: 4px solid #16a34a; }
.why-item h4 { font-size: 14px; margin-bottom: 8px; }
.text-examples { font-size: 12px; }
.text-ex {
    background: #f8fafc;
    padding: 6px 10px;
    border-radius: 4px;
    margin: 4px 0;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-size: 11px;
    color: #475569;
}

.split-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin: 14px 0;
}
.split-item {
    background: #f8fafc;
    padding: 16px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}
.split-header {
    font-size: 13px;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 8px;
    color: white;
}
.split-header.brand-split { background: #8b5cf6; }
.split-header.product-split { background: #f59e0b; color: #1e293b; }

/* ── Blog figures ───────────────────────────────────── */
.blog-figures-container {
    display: flex;
    flex-direction: column;
    gap: 20px;
    margin: 16px 0;
}
.blog-figure {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.blog-figure-img {
    width: 100%;
    height: auto;
    display: block;
}

/* ── Parameter reference table ──────────────────────── */
.param-tbl {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    background: white;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    font-size: 13px;
}
.param-tbl thead th {
    background: #f1f5f9;
    padding: 10px 14px;
    text-align: left;
    font-size: 12px;
    text-transform: uppercase;
    color: #64748b;
    letter-spacing: 0.5px;
}
.param-tbl td {
    padding: 8px 14px;
    border-bottom: 1px solid #f1f5f9;
    vertical-align: top;
}
.param-tbl td:first-child {
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-weight: 600;
    color: #1e3a5f;
    white-space: nowrap;
}
.param-tbl td:nth-child(2) {
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    color: #64748b;
    white-space: nowrap;
}
.param-tbl tr:last-child td { border-bottom: none; }

/* ── Note ───────────────────────────────────────────── */
.note {
    font-style: italic;
    color: #94a3b8;
    font-size: 13px;
}

/* ── Footer ─────────────────────────────────────────── */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    padding: 28px 24px;
    margin-top: 32px;
    border-top: 1px solid #e2e8f0;
}
.footer a { color: #64748b; text-decoration: none; }
.footer a:hover { color: #2563eb; }

/* ── Responsive ─────────────────────────────────────── */
@media (max-width: 900px) {
    .example-comparison { grid-template-columns: 1fr; }
    .mining-grid { grid-template-columns: 1fr; }
    .why-grid { grid-template-columns: 1fr; }
}
@media (max-width: 768px) {
    .page-layout { grid-template-columns: 1fr; }
    .sidebar {
        position: static;
        display: flex;
        overflow-x: auto;
        border-right: none;
        border-bottom: 1px solid #e2e8f0;
        padding: 8px;
        gap: 4px;
    }
    .sidebar-label { display: none; }
    .nav-btn { padding: 8px 14px; white-space: nowrap; border-left: none; border-bottom: 3px solid transparent; font-size: 12px; }
    .nav-btn.active { border-left: none; border-bottom-color: #2563eb; }
    .nav-icon { width: 22px; height: 22px; font-size: 12px; }
    .main-content { padding: 20px 16px; }
    .setup-grid { grid-template-columns: 1fr; }
    .split-grid { grid-template-columns: 1fr; }
    .header h1 { font-size: 32px; }
    .header .tagline { font-size: 15px; }
    h2 { font-size: 22px; }
    .img-results-row { gap: 6px; }
    .img-result-card { width: 80px; }
    .result-img { width: 70px; height: 70px; }
    .sku-result-card { width: 80px; }
}
@media (max-width: 480px) {
    .code-block { font-size: 11px; padding: 12px; }
    .result-img { width: 60px; height: 60px; }
    .query-img { width: 60px; height: 60px; }
}
'''


# ════════════════════════════════════════════════════════
#  HTML ASSEMBLY
# ════════════════════════════════════════════════════════


def build_page() -> str:
    """Build the complete HTML page."""
    # Load blog figures
    blog_figures_path = DATA_DIR / "blog_figures.json"
    if blog_figures_path.exists():
        with open(blog_figures_path) as f:
            blog_figures = json.load(f)
    else:
        blog_figures = {}
        print("Warning: blog_figures.json not found, blog figures will be missing")

    # Build tab content
    tabs = [
        ("concepts", "Embedding Fine-Tuning Concepts", tab1_concepts()),
        ("text", "Text \u2192 Text Retrieval", tab2_text()),
        ("multimodal", "Text \u2192 Image Retrieval", tab3_multimodal(blog_figures)),
        ("composed", "Composed (img+txt \u2192 img)", tab4_composed(blog_figures)),
        ("sku", "Mixed-Mode (img+txt \u2192 img+txt)", tab5_sku_matching()),
    ]

    nav_icons = {
        "concepts": ("📚", "concepts"),
        "text": ("📝", "text"),
        "multimodal": ("🖼️", "image"),
        "composed": ("🔗", "composed"),
        "sku": ("🏷️", "sku"),
    }

    tab_buttons = '<div class="sidebar-label">Learn</div>\n'
    tab_contents = ""
    for i, (tid, label, content) in enumerate(tabs):
        active = " active" if i == 0 else ""
        icon_emoji, icon_class = nav_icons.get(tid, ("", "concepts"))
        tab_buttons += f'<button class="nav-btn{active}" data-tab="{tid}"><span class="nav-icon {icon_class}">{icon_emoji}</span>{label}</button>\n'
        if i == 1:
            tab_buttons += '<div class="sidebar-label" style="margin-top:12px">Tutorials</div>\n'
        active_class = " active" if i == 0 else ""
        tab_contents += f'<div id="tab-{tid}" class="tab-content{active_class}">{content}</div>\n'

    css = get_css()

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>khoji -- Fine-Tune Embedding Models for Domain-Specific Retrieval</title>
<style>
{css}
</style>
</head>
<body>

<div class="header">
<h1><span>khoji</span></h1>
<p class="tagline">Fine-tune embedding models for domain-specific retrieval. Text search, image search, composed retrieval, and cross-brand product matching &mdash; with complete tutorials, real code, and before/after results.</p>
</div>

<div class="page-layout">
<nav class="sidebar">
{tab_buttons}
</nav>
<div class="main-content">
{tab_contents}
</div>
</div>

<div class="footer">
khoji &mdash; MIT License &mdash; <a href="https://github.com/suyashh94/khoji">github.com/suyashh94/khoji</a>
</div>

<script>
(function() {{
    var buttons = document.querySelectorAll('.nav-btn');
    buttons.forEach(function(btn) {{
        btn.addEventListener('click', function() {{
            var tabId = this.getAttribute('data-tab');
            document.querySelectorAll('.tab-content').forEach(function(c) {{
                c.classList.remove('active');
            }});
            document.querySelectorAll('.nav-btn').forEach(function(b) {{
                b.classList.remove('active');
            }});
            document.getElementById('tab-' + tabId).classList.add('active');
            this.classList.add('active');
            document.querySelector('.main-content').scrollTo(0, 0);
        }});
    }});
}})();
</script>

</body>
</html>'''


# ── Main ────────────────────────────────────────────────


def main():
    print("Generating khoji webpage...")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Verify data files exist
    required_files = [
        "text_summary.json", "text_baseline.json", "text_finetuned.json",
        "multimodal_summary.json", "multimodal_baseline.json", "multimodal_finetuned.json",
        "sku_summary.json", "sku_brand_baseline.json", "sku_brand_finetuned.json",
        "sku_product_baseline.json", "sku_product_finetuned.json",
        "blog_figures.json",
    ]
    missing = [f for f in required_files if not (DATA_DIR / f).exists()]
    if missing:
        print(f"  Warning: missing data files: {missing}")
        print("  Run 'python scripts/collect_webpage_data.py' first to collect data.")

    html = build_page()
    out_path = OUTPUT_DIR / "index.html"
    out_path.write_text(html, encoding="utf-8")

    size_kb = len(html) / 1024
    size_mb = size_kb / 1024
    print(f"  Saved to: {out_path}")
    print(f"  Size: {size_kb:.0f} KB ({size_mb:.1f} MB)")
    print(f"  Lines: {html.count(chr(10)):,}")
    print("Done!")


if __name__ == "__main__":
    main()
