"""Composed image retrieval on FashionIQ using khoji.

Query = (reference image + modification caption) → retrieve target image.
Uses BLIP-2's Q-Former via khoji's MultimodalEmbeddingModel.encode() which
jointly encodes (image, text) into a single embedding.

Usage:
    # 1. Download annotations + URL mappings
    python scripts/fashioniq/download_data.py

    # 2. Run training (images auto-downloaded and cached)
    python scripts/fashioniq/train.py --category dress --epochs 5
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from tqdm import tqdm

import khoji
from khoji.image_utils import load_image
from khoji.loss import infonce_loss

CATEGORIES = ["dress", "shirt", "toptee"]

URL_MAP_BASE = (
    "https://raw.githubusercontent.com/"
    "hongwang600/fashion-iq-metadata/master/image_url"
)


# ──────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────


def load_url_mapping(data_dir: Path, category: str) -> dict[str, str]:
    """Load ASIN → image URL mapping for a category."""
    cache_file = data_dir / "image_url" / f"asin2url.{category}.txt"
    if not cache_file.exists():
        from urllib.request import urlretrieve

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        url = f"{URL_MAP_BASE}/asin2url.{category}.txt"
        print(f"  Downloading URL mapping: {url}")
        urlretrieve(url, cache_file)

    mapping = {}
    with open(cache_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[parts[0].strip()] = parts[1].strip()
    return mapping


def load_image_by_id(image_id, url_mapping, cache_dir=None):
    """Load an image by ASIN — from cache or URL."""
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
        img = load_image(
            url, cache_dir=str(cache_dir) if cache_dir else None
        )
        return img.convert("RGB") if img else None
    except Exception:
        return None


def load_fashioniq(data_dir, category, split="train"):
    """Load FashionIQ annotations and gallery IDs."""
    data_path = Path(data_dir)
    with open(data_path / "captions" / f"cap.{category}.{split}.json") as f:
        annotations = json.load(f)
    with open(data_path / "image_splits" / f"split.{category}.{split}.json") as f:
        gallery_ids = json.load(f)
    print(
        f"Loaded {category}/{split}: "
        f"{len(annotations)} annotations, {len(gallery_ids)} gallery images"
    )
    return annotations, gallery_ids


@dataclass
class ComposedTriplet:
    candidate_id: str
    caption: str
    target_id: str
    negative_id: str


def build_triplets(annotations, gallery_ids, n_negatives=1, seed=42):
    """Build composed retrieval triplets (2 captions per annotation)."""
    rng = random.Random(seed)
    triplets = []
    for ann in annotations:
        non_targets = [
            g for g in gallery_ids
            if g != ann["target"] and g != ann["candidate"]
        ]
        for caption in ann["captions"]:
            for neg_id in rng.sample(
                non_targets, min(n_negatives, len(non_targets))
            ):
                triplets.append(ComposedTriplet(
                    candidate_id=ann["candidate"],
                    caption=caption,
                    target_id=ann["target"],
                    negative_id=neg_id,
                ))
    rng.shuffle(triplets)
    print(f"Built {len(triplets)} composed triplets")
    return triplets


# ──────────────────────────────────────────────────────────
# Training — uses khoji's model.encode() for joint embedding
# ──────────────────────────────────────────────────────────


def train(
    model: khoji.MultimodalEmbeddingModel,
    triplets: list[ComposedTriplet],
    url_mapping: dict[str, str],
    cache_dir: Path | None = None,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 2e-5,
    warmup_steps: int = 50,
    temperature: float = 0.05,
) -> khoji.TrainHistory:
    """Train using khoji's unified encode() for composed retrieval."""
    history = khoji.TrainHistory()
    loss_fn = partial(infonce_loss, temperature=temperature)

    # LoRA params only
    trainable = [
        p for p in model._full_model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    total_steps = (len(triplets) // batch_size) * epochs
    warmup = min(warmup_steps, total_steps // 5)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        remaining = total_steps - step
        return max(0.0, remaining / max(total_steps - warmup, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model._full_model.train()
    print(
        f"\nTraining on {len(triplets)} triplets for {epochs} epochs "
        f"(batch_size={batch_size}, lr={lr})"
    )

    pbar = tqdm(total=total_steps, desc="Training", unit="batch")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        indices = list(range(len(triplets)))
        random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start: start + batch_size]
            if len(batch_idx) < 2:
                continue
            batch = [triplets[i] for i in batch_idx]

            # Load images
            cand_imgs, tgt_imgs, neg_imgs, captions = [], [], [], []
            skip = False
            for t in batch:
                c = load_image_by_id(t.candidate_id, url_mapping, cache_dir)
                tg = load_image_by_id(t.target_id, url_mapping, cache_dir)
                n = load_image_by_id(t.negative_id, url_mapping, cache_dir)
                if c is None or tg is None or n is None:
                    skip = True
                    break
                cand_imgs.append(c)
                tgt_imgs.append(tg)
                neg_imgs.append(n)
                captions.append(t.caption)
            if skip:
                continue

            # Key: use khoji's encode() for joint (image+text) embedding
            # This goes through Q-Former when model is BLIP-2
            query_emb = model.encode(
                images=cand_imgs, texts=captions,
                batch_size=len(cand_imgs), show_progress=False,
            ).to(model.device)

            target_emb = model.encode(
                images=tgt_imgs,
                batch_size=len(tgt_imgs), show_progress=False,
            ).to(model.device)

            neg_emb = model.encode(
                images=neg_imgs,
                batch_size=len(neg_imgs), show_progress=False,
            ).to(model.device)

            loss = loss_fn(query_emb, target_emb, neg_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            bl = loss.item()
            epoch_loss += bl
            epoch_batches += 1
            history.step_loss.append(bl)
            history.step_lr.append(scheduler.get_last_lr()[0])

            pbar.update(1)
            pbar.set_postfix(
                loss=f"{bl:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )

        avg = epoch_loss / max(epoch_batches, 1)
        history.epoch_loss.append(avg)
        tqdm.write(f"  Epoch {epoch+1}/{epochs} | Avg Loss: {avg:.4f}")

    pbar.close()
    return history


# ──────────────────────────────────────────────────────────
# Evaluation — uses khoji's encode() and metrics
# ──────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: khoji.MultimodalEmbeddingModel,
    annotations: list[dict],
    gallery_ids: list[str],
    url_mapping: dict[str, str],
    cache_dir: Path | None = None,
    k_values: list[int] | None = None,
    max_queries: int | None = None,
) -> dict[str, float]:
    """Evaluate composed retrieval using khoji's encode() + metrics."""
    if k_values is None:
        k_values = [1, 5, 10, 50]

    model._full_model.eval()

    # Pre-encode gallery images using khoji's encode()
    print(f"Loading {len(gallery_ids)} gallery images...")
    gallery_images, valid_ids = [], []
    for gid in tqdm(gallery_ids, desc="Loading gallery"):
        img = load_image_by_id(gid, url_mapping, cache_dir)
        if img is not None:
            gallery_images.append(img)
            valid_ids.append(gid)
    print(f"Loaded {len(valid_ids)}/{len(gallery_ids)} images")

    # Encode gallery with khoji
    gallery_emb = model.encode(images=gallery_images)
    id_to_idx = {gid: i for i, gid in enumerate(valid_ids)}

    if max_queries is not None:
        annotations = annotations[:max_queries]

    sums: dict[str, float] = {}
    for k in k_values:
        sums[f"recall@{k}"] = 0.0
        sums[f"mrr@{k}"] = 0.0
    n = 0

    for ann in tqdm(annotations, desc="Evaluating"):
        if ann["target"] not in id_to_idx:
            continue
        c_img = load_image_by_id(ann["candidate"], url_mapping, cache_dir)
        if c_img is None:
            continue

        # Composed query with khoji's encode()
        q_emb = model.encode(
            images=[c_img], texts=[ann["captions"][0]],
            show_progress=False,
        )
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


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────


def run_experiment(
    category: str,
    data_dir: str = "./data/fashioniq",
    cache_dir: str = "./data/fashioniq/image_cache",
    model_name: str = "Salesforce/blip2-itm-vit-g",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 2e-5,
    lora_r: int = 8,
    n_negatives: int = 1,
    max_eval_queries: int | None = 100,
    output_dir: str = "./output/fashioniq",
):
    """Run composed retrieval experiment using khoji."""
    data_path = Path(data_dir)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / category
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"# FashionIQ Composed Retrieval: {category.upper()}")
    print(f"# Model: {model_name} + LoRA (r={lora_r})")
    print(f"{'#' * 60}")

    # Load data
    train_anns, train_gallery = load_fashioniq(data_dir, category, "train")
    val_anns, val_gallery = load_fashioniq(data_dir, category, "val")

    print("\nLoading image URL mapping...")
    url_mapping = load_url_mapping(data_path, category)
    print(f"  {len(url_mapping)} URLs")

    # Build model using khoji's MultimodalEmbeddingModel
    # The .encode(images=..., texts=...) method jointly encodes via Q-Former
    lora = khoji.LoRASettings(r=lora_r, alpha=lora_r * 2, dropout=0.1)

    # We need to apply LoRA manually since MultimodalEmbeddingModel
    # loads in eval mode. We'll load base, apply LoRA, then train.
    model = khoji.MultimodalEmbeddingModel(model_name)
    model._full_model = khoji.lora.apply_lora(model._full_model, lora)

    # Baseline
    print("\n--- Baseline Evaluation ---")
    baseline = evaluate(
        model, val_anns, val_gallery, url_mapping,
        cache_dir=cache_path, max_queries=max_eval_queries,
    )

    # Train
    triplets = build_triplets(
        train_anns, train_gallery, n_negatives=n_negatives
    )
    history = train(
        model, triplets, url_mapping,
        cache_dir=cache_path, epochs=epochs,
        batch_size=batch_size, lr=lr,
    )
    history.save(str(out_path / "train_history.json"))

    # Fine-tuned eval
    print("\n--- Fine-tuned Evaluation ---")
    finetuned = evaluate(
        model, val_anns, val_gallery, url_mapping,
        cache_dir=cache_path, max_queries=max_eval_queries,
    )

    # Comparison
    print(f"\n{'=' * 50}")
    print(f"COMPARISON — {category.upper()}")
    print(f"{'=' * 50}")
    for m in baseline:
        b, f = baseline[m], finetuned[m]
        arrow = "+" if f - b >= 0 else ""
        print(f"  {m:<12} {b:.4f} → {f:.4f}  ({arrow}{f-b:.4f})")

    model._full_model.save_pretrained(str(out_path / "adapter"))
    print(f"\nAdapter saved to {out_path / 'adapter'}")


def main():
    parser = argparse.ArgumentParser(
        description="FashionIQ composed image retrieval using khoji"
    )
    parser.add_argument(
        "--category", default="dress",
        choices=["dress", "shirt", "toptee", "all"],
    )
    parser.add_argument("--data-dir", default="./data/fashioniq")
    parser.add_argument("--cache-dir", default="./data/fashioniq/image_cache")
    parser.add_argument(
        "--model", default="Salesforce/blip2-itm-vit-g",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--n-negatives", type=int, default=1)
    parser.add_argument("--max-eval-queries", type=int, default=100)
    parser.add_argument("--output-dir", default="./output/fashioniq")
    args = parser.parse_args()

    cats = CATEGORIES if args.category == "all" else [args.category]
    for cat in cats:
        run_experiment(
            category=cat, data_dir=args.data_dir,
            cache_dir=args.cache_dir, model_name=args.model,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, lora_r=args.lora_r,
            n_negatives=args.n_negatives,
            max_eval_queries=args.max_eval_queries,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
