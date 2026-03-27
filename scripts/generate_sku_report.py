"""Generate HTML report for SKU matching experiment.

Shows before/after retrieval results with actual product images.

Usage:
    python scripts/generate_sku_report.py
"""
from __future__ import annotations

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

import khoji
from train_sku_matching import (
    DATA_DIR, MODEL, TRAIN_BRANDS, TEST_BRAND, HOLDOUT_FAMILIES,
    load_catalog, build_datasets,
)

OUTPUT_DIR = Path("./output/sku-matching")
TOP_K = 10


def img_to_base64(img_path: str, size: int = 140) -> str:
    img = Image.open(img_path).convert("RGB").resize((size, size))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def get_item_info(sku_id: str, catalog: list[dict]) -> dict | None:
    for item in catalog:
        if item["sku_id"] == sku_id:
            return item
    return None


def run_retrieval(model, dataset, catalog, top_k=10):
    """Run retrieval and return per-query results."""
    corpus_ids = list(dataset.corpus.keys())
    corpus_items = [dataset.corpus[cid] for cid in corpus_ids]

    imgs, txts = [], []
    for img_src, txt in corpus_items:
        img = khoji.load_image(img_src)
        if img is None:
            img = Image.new("RGB", (224, 224), "gray")
        imgs.append(img)
        txts.append(txt)

    corpus_embs = model.encode(images=imgs, texts=txts, batch_size=64)

    results = {}
    for qid in dataset.queries:
        img_src, txt = dataset.queries[qid]
        q_img = khoji.load_image(img_src)
        if q_img is None:
            continue
        q_emb = model.encode(images=[q_img], texts=[txt], show_progress=False)
        scores = torch.mm(q_emb, corpus_embs.t()).squeeze(0)
        topk_idx = torch.topk(scores, min(top_k, len(corpus_ids))).indices.tolist()

        relevant = set(dataset.qrels.get(qid, {}).keys())
        q_info = get_item_info(qid, catalog)

        retrieved = []
        for idx in topk_idx:
            cid = corpus_ids[idx]
            if cid == qid:
                continue  # skip self-match
            c_info = get_item_info(cid, catalog)
            retrieved.append({
                "sku_id": cid,
                "image": str(DATA_DIR / dataset.corpus[cid][0]) if not str(dataset.corpus[cid][0]).startswith("/") else dataset.corpus[cid][0],
                "text": dataset.corpus[cid][1],
                "score": scores[idx].item(),
                "relevant": cid in relevant,
                "brand": c_info["brand"] if c_info else "?",
                "family": c_info["family"] if c_info else "?",
                "variant": c_info["variant"] if c_info else "?",
            })

        results[qid] = {
            "query_image": img_src,
            "query_text": txt,
            "brand": q_info["brand"] if q_info else "?",
            "family": q_info["family"] if q_info else "?",
            "variant": q_info["variant"] if q_info else "?",
            "retrieved": retrieved,
            "num_relevant": len(relevant),
        }

    return results


def build_html(summary, baseline_brand, ft_brand, baseline_product, ft_product, catalog):
    """Build the full HTML report."""

    def metrics_table(baseline_m, ft_m):
        rows = ""
        for m in baseline_m:
            b, f = baseline_m[m], ft_m[m]
            d = f - b
            color = "#16a34a" if d > 0.001 else ("#dc2626" if d < -0.001 else "#888")
            sign = "+" if d >= 0 else ""
            rows += f'<tr><td>{m}</td><td>{b:.4f}</td><td><b>{f:.4f}</b></td><td style="color:{color}"><b>{sign}{d:.4f}</b></td></tr>'
        return rows

    def example_rows(baseline_results, ft_results, max_examples=15):
        html = ""
        qids = list(baseline_results.keys())[:max_examples]

        for qid in qids:
            br = baseline_results[qid]
            fr = ft_results[qid]

            q_b64 = img_to_base64(br["query_image"])

            base_hits = sum(1 for r in br["retrieved"][:5] if r["relevant"])
            ft_hits = sum(1 for r in fr["retrieved"][:5] if r["relevant"])
            diff = ft_hits - base_hits

            if diff > 0:
                badge = f'<span class="badge pos">+{diff}</span>'
            elif diff < 0:
                badge = f'<span class="badge neg">{diff}</span>'
            else:
                badge = '<span class="badge neu">=</span>'

            def result_cards(results, n=5):
                cards = ""
                for rank, r in enumerate(results[:n]):
                    border = "#16a34a" if r["relevant"] else "#dc2626"
                    tick = "&#10003;" if r["relevant"] else "&#10007;"
                    tc = "#16a34a" if r["relevant"] else "#dc2626"
                    r_b64 = img_to_base64(r["image"])
                    cards += f'''
                    <div class="card">
                        <span class="rk">#{rank+1}</span>
                        <img src="{r_b64}" style="border:3px solid {border}">
                        <span style="color:{tc};font-weight:700;font-size:14px">{tick}</span>
                        <span class="cn">{r["brand"]}</span>
                        <span class="cn">{r["variant"]}</span>
                        <span class="sc">{r["score"]:.3f}</span>
                    </div>'''
                return cards

            html += f'''
            <div class="ex">
                <div class="qsec">
                    <img src="{q_b64}" class="qi">
                    <div class="qinfo">
                        <div class="qb">{br["brand"]}</div>
                        <div class="qp">{br["family"]} — {br["variant"]}</div>
                        <div class="qt">{br["query_text"][:80]}</div>
                    </div>
                    {badge}
                </div>
                <div class="rsec">
                    <div class="rcol">
                        <h4>Baseline — {base_hits}/5 correct</h4>
                        <div class="rrow">{result_cards(br["retrieved"])}</div>
                    </div>
                    <div class="rcol">
                        <h4>Fine-tuned — {ft_hits}/5 correct</h4>
                        <div class="rrow">{result_cards(fr["retrieved"])}</div>
                    </div>
                </div>
            </div>'''
        return html

    cfg = summary["config"]

    brand_examples = example_rows(baseline_brand, ft_brand)
    product_examples = example_rows(baseline_product, ft_product)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SKU Matching — khoji Mixed-Mode Retrieval</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f8fafc;color:#1e293b;padding:20px}}
.c{{max-width:1300px;margin:0 auto}}
.hdr{{background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff;padding:32px;border-radius:16px;margin-bottom:20px}}
.hdr h1{{font-size:26px;margin-bottom:6px}}
.hdr .sub{{font-size:17px;opacity:.9;margin-bottom:10px}}
.hdr p{{opacity:.8;font-size:14px;line-height:1.5}}
.box{{background:#fff;border-radius:12px;padding:24px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.box h2{{font-size:19px;color:#1e3a5f;margin-bottom:14px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.si{{background:#f1f5f9;padding:10px 14px;border-radius:8px}}
.si .lb{{font-size:11px;text-transform:uppercase;color:#64748b;margin-bottom:2px}}
.si .vl{{font-size:14px;font-weight:600}}
table{{width:100%;border-collapse:collapse}}
th{{background:#f1f5f9;padding:8px 14px;text-align:left;font-size:12px;text-transform:uppercase;color:#64748b}}
td{{padding:8px 14px;border-bottom:1px solid #e2e8f0;font-size:14px}}
.ex{{border:1px solid #e2e8f0;border-radius:10px;padding:14px;margin-bottom:16px;background:#fafbfc}}
.qsec{{display:flex;align-items:center;gap:14px;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid #e2e8f0}}
.qi{{width:90px;height:90px;border-radius:8px;object-fit:cover;border:2px solid #2563eb}}
.qinfo{{flex:1}}
.qb{{font-weight:700;font-size:15px;color:#2563eb}}
.qp{{font-weight:600;font-size:14px}}
.qt{{color:#64748b;font-size:12px;margin-top:2px}}
.badge{{padding:3px 10px;border-radius:14px;font-size:13px;font-weight:700;white-space:nowrap}}
.pos{{background:#dcfce7;color:#16a34a}}
.neg{{background:#fee2e2;color:#dc2626}}
.neu{{background:#f1f5f9;color:#64748b}}
.rsec{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.rcol h4{{font-size:13px;color:#475569;margin-bottom:6px}}
.rrow{{display:flex;gap:6px;flex-wrap:wrap}}
.card{{text-align:center;width:88px}}
.card img{{width:76px;height:76px;border-radius:6px;object-fit:cover}}
.rk{{font-size:10px;color:#94a3b8;display:block}}
.cn{{font-size:9px;color:#475569;display:block;line-height:1.2}}
.sc{{font-size:9px;color:#94a3b8}}
.ft{{text-align:center;color:#94a3b8;font-size:12px;margin-top:16px}}
</style>
</head>
<body>
<div class="c">

<div class="hdr">
<h1>SKU Matching: Cross-Brand Product Retrieval</h1>
<div class="sub">Mixed-Mode (img+txt → img+txt) with khoji</div>
<p>Given a product from one brand (photo + description), find the equivalent product in another brand's catalog.
AI-generated product images across {len(cfg["train_brands"]) + 1} UK grocery brands. Test brand ({cfg["test_brand"]}) was never seen during training.</p>
</div>

<div class="box">
<h2>Experiment Setup</h2>
<div class="grid2">
<div class="si"><div class="lb">Model</div><div class="vl">BLIP-2 + LoRA r={cfg["lora_r"]}</div></div>
<div class="si"><div class="lb">Training</div><div class="vl">{cfg["negatives"]} negatives, {cfg["mining_rounds"]} mining rounds, {cfg["epochs"]} epochs/round</div></div>
<div class="si"><div class="lb">Train</div><div class="vl">{cfg["train_queries"]} queries from {', '.join(cfg["train_brands"])} ({cfg["corpus_size"]} corpus)</div></div>
<div class="si"><div class="lb">Test (Brand)</div><div class="vl">{cfg["test_brand_queries"]} {cfg["test_brand"]} queries (unseen brand)</div></div>
<div class="si"><div class="lb">Test (Product)</div><div class="vl">{cfg["test_product_queries"]} queries for held-out families: {', '.join(cfg["holdout_families"])}</div></div>
<div class="si"><div class="lb">Training Loss</div><div class="vl">{summary["training_loss_first"]:.3f} → {summary["training_loss_last"]:.3f}</div></div>
</div>
</div>

<div class="box">
<h2>Brand Generalization — {cfg["test_brand"]} (unseen brand)</h2>
<p style="color:#64748b;font-size:13px;margin-bottom:12px">Model has never seen any {cfg["test_brand"]} product images during training.</p>
<table>
<thead><tr><th>Metric</th><th>Baseline</th><th>Fine-tuned</th><th>Delta</th></tr></thead>
<tbody>{metrics_table(summary["baseline_brand"], summary["finetuned_brand"])}</tbody>
</table>
</div>

<div class="box">
<h2>Product Generalization — Held-out Families ({', '.join(cfg["holdout_families"])})</h2>
<p style="color:#64748b;font-size:13px;margin-bottom:12px">Model has never seen these product categories during training.</p>
<table>
<thead><tr><th>Metric</th><th>Baseline</th><th>Fine-tuned</th><th>Delta</th></tr></thead>
<tbody>{metrics_table(summary["baseline_product"], summary["finetuned_product"])}</tbody>
</table>
</div>

<div class="box">
<h2>Brand Generalization — Query Examples</h2>
<p style="color:#64748b;font-size:13px;margin-bottom:14px">{cfg["test_brand"]} products searching the train brand catalog. <span style="color:#16a34a;font-weight:600">Green = correct</span>, <span style="color:#dc2626;font-weight:600">Red = wrong</span>.</p>
{brand_examples}
</div>

<div class="box">
<h2>Product Generalization — Query Examples</h2>
<p style="color:#64748b;font-size:13px;margin-bottom:14px">Held-out product families ({', '.join(cfg["holdout_families"])}) matching across train brands. <span style="color:#16a34a;font-weight:600">Green = correct</span>, <span style="color:#dc2626;font-weight:600">Red = wrong</span>.</p>
{product_examples}
</div>

<div class="ft">Generated by khoji — Mixed-mode composed retrieval — AI-generated grocery dataset</div>

</div>
</body>
</html>'''


def main():
    catalog = load_catalog()
    _, test_brand_ds, test_product_ds = build_datasets(catalog)

    summary = json.loads((OUTPUT_DIR / "summary.json").read_text())

    print("Running baseline retrieval (brand)...")
    base_model = khoji.JointEmbeddingModel(MODEL)
    baseline_brand = run_retrieval(base_model, test_brand_ds, catalog, TOP_K)
    del base_model; torch.cuda.empty_cache()

    print("Running fine-tuned retrieval (brand)...")
    ft_model = khoji.JointEmbeddingModel(MODEL, adapter_path=str(OUTPUT_DIR / "adapter"))
    ft_brand = run_retrieval(ft_model, test_brand_ds, catalog, TOP_K)
    del ft_model; torch.cuda.empty_cache()

    print("Running baseline retrieval (product)...")
    base_model = khoji.JointEmbeddingModel(MODEL)
    baseline_product = run_retrieval(base_model, test_product_ds, catalog, TOP_K)
    del base_model; torch.cuda.empty_cache()

    print("Running fine-tuned retrieval (product)...")
    ft_model = khoji.JointEmbeddingModel(MODEL, adapter_path=str(OUTPUT_DIR / "adapter"))
    ft_product = run_retrieval(ft_model, test_product_ds, catalog, TOP_K)
    del ft_model; torch.cuda.empty_cache()

    print("Building HTML...")
    html = build_html(summary, baseline_brand, ft_brand, baseline_product, ft_product, catalog)
    report_path = OUTPUT_DIR / "report.html"
    report_path.write_text(html)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
