"""Compare evaluation: baseline (no adapter) vs fine-tuned (with adapter)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

import khoji
from khoji.image_utils import load_image

MODEL = "Salesforce/blip2-itm-vit-g"
CATEGORY = "dress"
DATA_DIR = "./data/fashioniq"
CACHE_DIR = "./data/fashioniq/image_cache"
ADAPTER_PATH = "./output/composed-retrieval/dress/adapter"
MAX_EVAL_QUERIES = 100

URL_MAP_BASE = (
    "https://raw.githubusercontent.com/"
    "hongwang600/fashion-iq-metadata/master/image_url"
)


def load_url_mapping(data_dir: Path, category: str) -> dict[str, str]:
    cache_file = data_dir / "image_url" / f"asin2url.{category}.txt"
    if not cache_file.exists():
        from urllib.request import urlretrieve
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(f"{URL_MAP_BASE}/asin2url.{category}.txt", cache_file)
    mapping = {}
    with open(cache_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[parts[0].strip()] = parts[1].strip()
    return mapping


def load_image_by_id(image_id, url_mapping, cache_dir=None):
    if cache_dir is not None:
        for ext in [".jpg", ".jpeg", ".png"]:
            local = Path(cache_dir) / f"{image_id}{ext}"
            if local.exists():
                try:
                    from PIL import Image
                    return Image.open(local).convert("RGB")
                except Exception:
                    pass
    url = url_mapping.get(image_id)
    if url is None:
        return None
    try:
        img = load_image(url, cache_dir=str(cache_dir) if cache_dir else None)
        return img.convert("RGB") if img else None
    except Exception:
        return None


def load_fashioniq(data_dir, category, split="val"):
    data_path = Path(data_dir)
    with open(data_path / "captions" / f"cap.{category}.{split}.json") as f:
        annotations = json.load(f)
    with open(data_path / "image_splits" / f"split.{category}.{split}.json") as f:
        gallery_ids = json.load(f)
    print(f"Loaded {category}/{split}: {len(annotations)} annotations, {len(gallery_ids)} images")
    return annotations, gallery_ids


@torch.no_grad()
def evaluate(model, annotations, gallery_ids, url_mapping, cache_dir, k_values=None, max_queries=None):
    if k_values is None:
        k_values = [1, 5, 10, 50]
    model._full_model.eval()

    print(f"Loading gallery ({len(gallery_ids)} images)...")
    gallery_images, valid_ids = [], []
    for gid in tqdm(gallery_ids, desc="Loading"):
        img = load_image_by_id(gid, url_mapping, cache_dir)
        if img is not None:
            gallery_images.append(img)
            valid_ids.append(gid)

    gallery_emb = model.encode(images=gallery_images)
    id_to_idx = {gid: i for i, gid in enumerate(valid_ids)}

    if max_queries:
        annotations = annotations[:max_queries]

    sums = {f"{m}@{k}": 0.0 for k in k_values for m in ["recall", "mrr"]}
    n = 0

    for ann in tqdm(annotations, desc="Evaluating"):
        if ann["target"] not in id_to_idx:
            continue
        c_img = load_image_by_id(ann["candidate"], url_mapping, cache_dir)
        if c_img is None:
            continue

        q_emb = model.encode(images=[c_img], texts=[ann["captions"][0]], show_progress=False)
        scores = torch.mm(q_emb, gallery_emb.t()).squeeze(0)
        ranked = [valid_ids[i] for i in scores.argsort(descending=True)]

        qrel = {ann["target"]: 1}
        for k in k_values:
            sums[f"recall@{k}"] += khoji.recall_at_k(ranked, qrel, k)
            sums[f"mrr@{k}"] += khoji.mrr_at_k(ranked, qrel, k)
        n += 1

    metrics = {k: round(v / max(n, 1), 4) for k, v in sums.items()}
    print(f"\nResults ({n} queries, {len(valid_ids)} gallery):")
    print(f"{'Metric':<12} {'Value':>8}")
    print("-" * 22)
    for k, v in metrics.items():
        print(f"{k:<12} {v:>8.4f}")
    return metrics


def main():
    data_path = Path(DATA_DIR)
    cache_path = Path(CACHE_DIR)

    val_anns, val_gallery = load_fashioniq(DATA_DIR, CATEGORY, "val")
    url_mapping = load_url_mapping(data_path, CATEGORY)
    print(f"URL mapping: {len(url_mapping)} entries")

    # --- Baseline (no adapter) ---
    print("\n" + "=" * 60)
    print("BASELINE — No adapter (pretrained BLIP-2 only)")
    print("=" * 60)
    model_base = khoji.JointEmbeddingModel(MODEL)
    baseline = evaluate(model_base, val_anns, val_gallery, url_mapping, cache_path, max_queries=MAX_EVAL_QUERIES)
    del model_base
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Fine-tuned (with adapter) ---
    print("\n" + "=" * 60)
    print(f"FINE-TUNED — With adapter from {ADAPTER_PATH}")
    print("=" * 60)
    model_ft = khoji.JointEmbeddingModel(MODEL, adapter_path=ADAPTER_PATH)
    finetuned = evaluate(model_ft, val_anns, val_gallery, url_mapping, cache_path, max_queries=MAX_EVAL_QUERIES)
    del model_ft
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Comparison ---
    print(f"\n{'=' * 55}")
    print(f"  COMPARISON — {CATEGORY.upper()} (max {MAX_EVAL_QUERIES} queries)")
    print(f"{'=' * 55}")
    print(f"  {'Metric':<12} {'Baseline':>10} {'Finetuned':>10} {'Delta':>10}")
    print(f"  {'-'*44}")
    for m in baseline:
        b, f = baseline[m], finetuned[m]
        sign = "+" if f - b >= 0 else ""
        print(f"  {m:<12} {b:>10.4f} {f:>10.4f} {sign}{f-b:>9.4f}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
