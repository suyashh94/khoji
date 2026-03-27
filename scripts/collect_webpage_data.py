"""Collect retrieval examples for the khoji webpage.

Runs baseline and fine-tuned inference for all 4 modes and saves
query-by-query results with actual text/images for the webpage.

Usage:
    python scripts/collect_webpage_data.py
"""
from __future__ import annotations

import base64
import json
import random
import sys
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

import khoji
from khoji.image_utils import load_image

OUTPUT = Path("./output/webpage/data")
OUTPUT.mkdir(parents=True, exist_ok=True)


def img_to_b64(path_or_img, size=140):
    try:
        if isinstance(path_or_img, str):
            img = Image.open(path_or_img).convert("RGB")
        else:
            img = path_or_img.convert("RGB")
        img = img.resize((size, size))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


# ── 1. TEXT RETRIEVAL ────────────────────────────────────


def collect_text_examples():
    print("\n" + "=" * 60)
    print("Collecting Text → Text examples (FiQA)")
    print("=" * 60)

    MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    ADAPTER = "output/text-retrieval/config-approach/adapter"

    test_ds = khoji.load_beir("fiqa", split="test")
    baseline_metrics = json.load(open("output/text-retrieval/config-approach/baseline.json"))["metrics"]
    finetuned_metrics = json.load(open("output/text-retrieval/config-approach/finetuned.json"))["metrics"]

    # Pick interesting queries with diverse topics
    query_ids = list(test_ds.queries.keys())
    random.seed(42)
    sample_qids = random.sample(query_ids, min(30, len(query_ids)))

    for label, adapter_path in [("baseline", None), ("finetuned", ADAPTER)]:
        print(f"  Running {label} retrieval...")
        model = khoji.EmbeddingModel(MODEL, adapter_path=adapter_path)

        corpus_ids = list(test_ds.corpus.keys())
        corpus_texts = [test_ds.corpus[cid] for cid in corpus_ids]
        corpus_embs = model.encode(corpus_texts, batch_size=64, show_progress=True)

        results = {}
        for qid in sample_qids:
            q_text = test_ds.queries[qid]
            q_emb = model.encode([q_text])
            scores = torch.mm(q_emb, corpus_embs.t()).squeeze(0)
            topk = torch.topk(scores, 10).indices.tolist()
            relevant = set(test_ds.qrels.get(qid, {}).keys())

            results[qid] = {
                "query_text": q_text,
                "retrieved": [
                    {
                        "doc_id": corpus_ids[idx],
                        "text": corpus_texts[idx][:200],
                        "score": scores[idx].item(),
                        "relevant": corpus_ids[idx] in relevant,
                    }
                    for idx in topk
                ],
            }

        del model
        torch.cuda.empty_cache()

        json.dump(results, open(OUTPUT / f"text_{label}.json", "w"), indent=2)

    # Pick 6 best examples (ones where finetuned improved)
    baseline_res = json.load(open(OUTPUT / "text_baseline.json"))
    finetuned_res = json.load(open(OUTPUT / "text_finetuned.json"))

    examples = []
    for qid in sample_qids:
        b_hits = sum(1 for r in baseline_res[qid]["retrieved"][:5] if r["relevant"])
        f_hits = sum(1 for r in finetuned_res[qid]["retrieved"][:5] if r["relevant"])
        examples.append((qid, b_hits, f_hits, f_hits - b_hits))

    examples.sort(key=lambda x: (-x[3], -x[2]))
    selected = [e[0] for e in examples[:8]]

    summary = {
        "model": MODEL,
        "dataset": "FiQA (Financial QA)",
        "corpus_size": len(corpus_ids),
        "num_queries": len(query_ids),
        "baseline_metrics": baseline_metrics,
        "finetuned_metrics": finetuned_metrics,
        "selected_qids": selected,
        "training": {
            "negatives": "mixed (2 random + 1 hard)",
            "mining_rounds": 2,
            "epochs_per_round": 3,
            "lora": "r=16, alpha=32",
            "lr": "2e-5",
            "batch_size": 16,
        },
    }
    json.dump(summary, open(OUTPUT / "text_summary.json", "w"), indent=2)
    print(f"  Saved {len(selected)} examples")


# ── 2. MULTIMODAL RETRIEVAL ──────────────────────────────


def collect_multimodal_examples():
    print("\n" + "=" * 60)
    print("Collecting Text → Image examples (RSICD)")
    print("=" * 60)

    MODEL = "openai/clip-vit-base-patch32"
    ADAPTER = "output/multimodal-retrieval/config-approach/adapter"

    from khoji.multimodal_dataset import load_rsicd
    test_ds = load_rsicd(split="test")
    baseline_metrics = json.load(open("output/multimodal-retrieval/config-approach/baseline.json"))["metrics"]
    finetuned_metrics = json.load(open("output/multimodal-retrieval/config-approach/finetuned.json"))["metrics"]

    query_ids = list(test_ds.queries.keys())
    random.seed(42)
    sample_qids = random.sample(query_ids, min(30, len(query_ids)))

    corpus_ids = list(test_ds.corpus.keys())
    corpus_sources = [test_ds.corpus[cid] for cid in corpus_ids]

    # Pre-encode images for corpus (load once)
    print("  Loading corpus images...")
    corpus_b64 = {}
    for cid in corpus_ids:
        src = test_ds.corpus[cid]
        b64 = img_to_b64(str(Path(test_ds.base_dir) / src) if test_ds.base_dir else src, 120)
        if b64:
            corpus_b64[cid] = b64

    for label, adapter_path in [("baseline", None), ("finetuned", ADAPTER)]:
        print(f"  Running {label} retrieval...")
        model = khoji.MultimodalEmbeddingModel(MODEL, adapter_path=adapter_path)

        # Encode corpus images
        img_list = []
        for src in corpus_sources:
            full_path = str(Path(test_ds.base_dir) / src) if test_ds.base_dir else src
            img = load_image(full_path)
            img_list.append(img if img else Image.new("RGB", (224, 224), "gray"))

        corpus_embs = model.encode_images(img_list, batch_size=64)

        results = {}
        for qid in sample_qids:
            q_text = test_ds.queries[qid]
            q_emb = model.encode_text([q_text])
            scores = torch.mm(q_emb, corpus_embs.t()).squeeze(0)
            topk = torch.topk(scores, 10).indices.tolist()
            relevant = set(test_ds.qrels.get(qid, {}).keys())

            results[qid] = {
                "query_text": q_text,
                "retrieved": [
                    {
                        "doc_id": corpus_ids[idx],
                        "image_b64": corpus_b64.get(corpus_ids[idx], ""),
                        "score": scores[idx].item(),
                        "relevant": corpus_ids[idx] in relevant,
                    }
                    for idx in topk
                ],
            }

        del model
        torch.cuda.empty_cache()

        json.dump(results, open(OUTPUT / f"multimodal_{label}.json", "w"), indent=2)

    # Pick best examples
    baseline_res = json.load(open(OUTPUT / "multimodal_baseline.json"))
    finetuned_res = json.load(open(OUTPUT / "multimodal_finetuned.json"))

    examples = []
    for qid in sample_qids:
        b_hits = sum(1 for r in baseline_res[qid]["retrieved"][:5] if r["relevant"])
        f_hits = sum(1 for r in finetuned_res[qid]["retrieved"][:5] if r["relevant"])
        examples.append((qid, b_hits, f_hits, f_hits - b_hits))

    examples.sort(key=lambda x: (-x[3], -x[2]))
    selected = [e[0] for e in examples[:8]]

    summary = {
        "model": MODEL,
        "dataset": "RSICD (Satellite Imagery)",
        "corpus_size": len(corpus_ids),
        "num_queries": len(query_ids),
        "baseline_metrics": baseline_metrics,
        "finetuned_metrics": finetuned_metrics,
        "selected_qids": selected,
        "training": {
            "negatives": "mixed (2 random + 1 hard)",
            "mining_rounds": 2,
            "epochs_per_round": 3,
            "lora": "r=16, alpha=32, target=both",
            "lr": "2e-5",
            "batch_size": 16,
        },
    }
    json.dump(summary, open(OUTPUT / "multimodal_summary.json", "w"), indent=2)
    print(f"  Saved {len(selected)} examples")


# ── 3. SKU MATCHING ──────────────────────────────────────


def collect_sku_examples():
    print("\n" + "=" * 60)
    print("Collecting SKU Matching examples")
    print("=" * 60)

    MODEL = "Salesforce/blip2-itm-vit-g"
    ADAPTER = "output/sku-matching/adapter"
    DATA_DIR = Path("data/sku-matching")

    from train_sku_matching import load_catalog, build_datasets

    catalog = load_catalog()
    _, test_brand_ds, test_product_ds = build_datasets(catalog)
    summary = json.load(open("output/sku-matching/summary.json"))

    def get_item_info(sku_id):
        return next((it for it in catalog if it["sku_id"] == sku_id), {})

    for test_name, test_ds in [("brand", test_brand_ds), ("product", test_product_ds)]:
        corpus_ids = list(test_ds.corpus.keys())

        # Pre-encode corpus images as b64
        corpus_b64 = {}
        for cid in corpus_ids:
            img_src = test_ds.corpus[cid][0]
            b64 = img_to_b64(img_src, 120)
            if b64:
                corpus_b64[cid] = b64

        for label, adapter_path in [("baseline", None), ("finetuned", ADAPTER)]:
            print(f"  Running {test_name} {label} retrieval...")
            model = khoji.JointEmbeddingModel(MODEL, adapter_path=adapter_path)

            imgs, txts = [], []
            for cid in corpus_ids:
                img_src, txt = test_ds.corpus[cid]
                img = load_image(img_src)
                imgs.append(img if img else Image.new("RGB", (224, 224), "gray"))
                txts.append(txt)

            corpus_embs = model.encode(images=imgs, texts=txts, batch_size=64)

            results = {}
            for qid in test_ds.queries:
                q_img_src, q_txt = test_ds.queries[qid]
                q_img = load_image(q_img_src)
                if q_img is None:
                    continue
                q_emb = model.encode(images=[q_img], texts=[q_txt], show_progress=False)
                scores = torch.mm(q_emb, corpus_embs.t()).squeeze(0)
                topk = torch.topk(scores, min(10, len(corpus_ids))).indices.tolist()
                relevant = set(test_ds.qrels.get(qid, {}).keys())

                q_info = get_item_info(qid)
                results[qid] = {
                    "query_text": q_txt,
                    "query_image_b64": img_to_b64(q_img_src, 120),
                    "query_brand": q_info.get("brand", ""),
                    "query_family": q_info.get("family", ""),
                    "query_variant": q_info.get("variant", ""),
                    "retrieved": [
                        {
                            "doc_id": corpus_ids[idx],
                            "image_b64": corpus_b64.get(corpus_ids[idx], ""),
                            "text": test_ds.corpus[corpus_ids[idx]][1][:100],
                            "score": scores[idx].item(),
                            "relevant": corpus_ids[idx] in relevant,
                            "brand": get_item_info(corpus_ids[idx]).get("brand", ""),
                            "family": get_item_info(corpus_ids[idx]).get("family", ""),
                            "variant": get_item_info(corpus_ids[idx]).get("variant", ""),
                        }
                        for idx in topk
                        if corpus_ids[idx] != qid  # skip self
                    ],
                }

            del model
            torch.cuda.empty_cache()

            json.dump(results, open(OUTPUT / f"sku_{test_name}_{label}.json", "w"), indent=2)

    # Save summary
    all_qids_brand = list(test_brand_ds.queries.keys())
    all_qids_product = list(test_product_ds.queries.keys())

    json.dump({
        "model": MODEL,
        "dataset": "AI-generated grocery (5 brands × 8 families × 5 variants)",
        "baseline_brand": summary["baseline_brand"],
        "finetuned_brand": summary["finetuned_brand"],
        "baseline_product": summary["baseline_product"],
        "finetuned_product": summary["finetuned_product"],
        "config": summary["config"],
        "brand_qids": all_qids_brand,
        "product_qids": all_qids_product,
    }, open(OUTPUT / "sku_summary.json", "w"), indent=2)
    print(f"  Saved brand={len(all_qids_brand)} product={len(all_qids_product)} examples")


# ── Main ─────────────────────────────────────────────────


def main():
    collect_text_examples()
    collect_multimodal_examples()
    collect_sku_examples()
    print("\n" + "=" * 60)
    print("All data collected!")
    print(f"Files in {OUTPUT}:")
    for f in sorted(OUTPUT.glob("*.json")):
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
