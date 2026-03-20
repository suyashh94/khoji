"""Composed image retrieval: (image + text) → image on FashionIQ.

Demonstrates khoji's JointEmbeddingModel for composed retrieval:
- Query = reference image + modification caption ("make it red")
- Target = retrieve the correct target image from a gallery
- Uses BLIP-2 Q-Former for joint (image, text) encoding

Requires FashionIQ annotations:
    python scripts/fashioniq/download_data.py

Usage:
    python scripts/train_composed_retrieval.py
    python scripts/train_composed_retrieval.py --category shirt
    python scripts/train_composed_retrieval.py --negatives hard --skip-top 5
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
MODEL = "Salesforce/blip2-itm-vit-g"

URL_MAP_BASE = (
    "https://raw.githubusercontent.com/"
    "hongwang600/fashion-iq-metadata/master/image_url"
)


# ── Data loading ─────────────────────────────────────────


def load_url_mapping(data_dir: Path, category: str) -> dict[str, str]:
    """Load ASIN → image URL mapping."""
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
    """Load image by ASIN — from cache or URL."""
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


def load_fashioniq(data_dir, category, split="train"):
    """Load FashionIQ annotations and gallery IDs."""
    data_path = Path(data_dir)
    with open(data_path / "captions" / f"cap.{category}.{split}.json") as f:
        annotations = json.load(f)
    with open(data_path / "image_splits" / f"split.{category}.{split}.json") as f:
        gallery_ids = json.load(f)
    print(f"Loaded {category}/{split}: {len(annotations)} annotations, {len(gallery_ids)} images")
    return annotations, gallery_ids


@dataclass
class ComposedTriplet:
    candidate_id: str
    caption: str
    target_id: str
    negative_id: str


# ── Triplet building ─────────────────────────────────────


def build_random_triplets(annotations, gallery_ids, n_negatives=1, seed=42):
    """Random negatives for composed retrieval."""
    rng = random.Random(seed)
    triplets = []
    for ann in annotations:
        non_targets = [g for g in gallery_ids if g != ann["target"] and g != ann["candidate"]]
        for caption in ann["captions"]:
            for neg_id in rng.sample(non_targets, min(n_negatives, len(non_targets))):
                triplets.append(ComposedTriplet(ann["candidate"], caption, ann["target"], neg_id))
    rng.shuffle(triplets)
    print(f"Built {len(triplets)} random triplets")
    return triplets


@torch.no_grad()
def mine_hard_triplets(
    model, annotations, gallery_ids, url_mapping,
    cache_dir=None, n_negatives=1, top_k=50, skip_top=0, seed=42,
):
    """Mine hard negatives using composed query embeddings."""
    model._full_model.eval()
    rng = random.Random(seed)

    # Encode gallery
    print(f"Loading {len(gallery_ids)} gallery images...")
    gallery_images, valid_ids = [], []
    for gid in tqdm(gallery_ids, desc="Loading"):
        img = load_image_by_id(gid, url_mapping, cache_dir)
        if img is not None:
            gallery_images.append(img)
            valid_ids.append(gid)

    gallery_emb = model.encode(images=gallery_images)
    fetch_k = top_k + skip_top

    triplets = []
    for ann in tqdm(annotations, desc="Mining"):
        target_id = ann["target"]
        candidate_id = ann["candidate"]
        c_img = load_image_by_id(candidate_id, url_mapping, cache_dir)
        if c_img is None:
            continue
        target_set = {target_id, candidate_id}

        for caption in ann["captions"]:
            q_emb = model.encode(images=[c_img], texts=[caption], show_progress=False)
            scores = torch.mm(q_emb, gallery_emb.t()).squeeze(0)
            topk = torch.topk(scores, min(fetch_k, len(valid_ids))).indices.tolist()
            hard_negs = [valid_ids[i] for i in topk if valid_ids[i] not in target_set]
            hard_negs = hard_negs[skip_top:]

            if not hard_negs:
                fallback = [g for g in valid_ids if g not in target_set]
                hard_negs = rng.sample(fallback, min(n_negatives, len(fallback)))

            for neg_id in hard_negs[:n_negatives]:
                triplets.append(ComposedTriplet(candidate_id, caption, target_id, neg_id))

    rng.shuffle(triplets)
    print(f"Mined {len(triplets)} hard triplets")
    return triplets


def build_mixed_triplets(
    model, annotations, gallery_ids, url_mapping,
    cache_dir=None, n_random=2, n_hard=1, top_k=50, skip_top=0, seed=42,
):
    """Mixed random + hard negatives."""
    hard = mine_hard_triplets(
        model, annotations, gallery_ids, url_mapping,
        cache_dir=cache_dir, n_negatives=n_hard, top_k=top_k, skip_top=skip_top, seed=seed,
    )
    rand = build_random_triplets(annotations, gallery_ids, n_negatives=n_random, seed=seed)
    combined = hard + rand
    random.Random(seed).shuffle(combined)
    print(f"Mixed: {len(hard)} hard + {len(rand)} random = {len(combined)} total")
    return combined


# ── Training ─────────────────────────────────────────────


def train_composed(
    model, triplets, url_mapping, cache_dir=None,
    epochs=5, batch_size=8, lr=2e-5, warmup_steps=50, temperature=0.05,
):
    """Train using khoji's JointEmbeddingModel.encode()."""
    history = khoji.TrainHistory()
    loss_fn = partial(infonce_loss, temperature=temperature)

    trainable = [p for p in model._full_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    total_steps = (len(triplets) // batch_size) * epochs
    warmup = min(warmup_steps, total_steps // 5)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        return max(0.0, (total_steps - step) / max(total_steps - warmup, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    model._full_model.train()

    print(f"\nTraining on {len(triplets)} triplets for {epochs} epochs")
    pbar = tqdm(total=total_steps, desc="Training", unit="batch")

    for epoch in range(epochs):
        epoch_loss, epoch_batches = 0.0, 0
        indices = list(range(len(triplets)))
        random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start: start + batch_size]
            if len(batch_idx) < 2:
                continue
            batch = [triplets[i] for i in batch_idx]

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

            # Joint encoding: (candidate_image + caption) → embedding
            q = model.encode(images=cand_imgs, texts=captions, batch_size=len(cand_imgs), show_progress=False).to(model.device)
            p = model.encode(images=tgt_imgs, batch_size=len(tgt_imgs), show_progress=False).to(model.device)
            n = model.encode(images=neg_imgs, batch_size=len(neg_imgs), show_progress=False).to(model.device)

            loss = loss_fn(q, p, n)
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
            pbar.set_postfix(loss=f"{bl:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg = epoch_loss / max(epoch_batches, 1)
        history.epoch_loss.append(avg)
        tqdm.write(f"  Epoch {epoch+1}/{epochs} | Avg Loss: {avg:.4f}")

    pbar.close()
    return history


# ── Evaluation ───────────────────────────────────────────


@torch.no_grad()
def evaluate_composed(
    model, annotations, gallery_ids, url_mapping,
    cache_dir=None, k_values=None, max_queries=None,
):
    """Evaluate: Recall@k and MRR@k using khoji's metrics."""
    if k_values is None:
        k_values = [1, 5, 10, 50]
    model._full_model.eval()

    # Encode gallery
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


# ── Main ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="FashionIQ composed retrieval")
    parser.add_argument("--category", default="dress", choices=["dress", "shirt", "toptee"])
    parser.add_argument("--data-dir", default="./data/fashioniq")
    parser.add_argument("--cache-dir", default="./data/fashioniq/image_cache")
    parser.add_argument("--negatives", default="mixed", choices=["random", "hard", "mixed"])
    parser.add_argument("--n-random", type=int, default=2)
    parser.add_argument("--n-hard", type=int, default=1)
    parser.add_argument("--n-negatives", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--skip-top", type=int, default=0)
    parser.add_argument("--mining-rounds", type=int, default=1,
                        help="Iterative mining rounds (re-mine with fine-tuned model)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--max-eval-queries", type=int, default=100)
    parser.add_argument("--output-dir", default="./output/composed-retrieval")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    cache_path = Path(args.cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_dir) / args.category
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Composed Retrieval: FashionIQ {args.category.upper()}")
    print(f"Model: {MODEL} + LoRA (r={args.lora_r})")
    print(f"Negatives: {args.negatives}")
    print("=" * 60)

    # Load data
    train_anns, train_gallery = load_fashioniq(args.data_dir, args.category, "train")
    val_anns, val_gallery = load_fashioniq(args.data_dir, args.category, "val")

    print("\nLoading URL mapping...")
    url_mapping = load_url_mapping(data_path, args.category)
    print(f"  {len(url_mapping)} URLs")

    # Build model — JointEmbeddingModel for (image + text) → embedding
    lora = khoji.LoRASettings(r=args.lora_r, alpha=args.lora_r * 2, dropout=0.1)
    model = khoji.JointEmbeddingModel(MODEL)
    model._full_model = khoji.lora.apply_lora(model._full_model, lora)

    # Baseline
    print("\n--- Baseline ---")
    baseline = evaluate_composed(
        model, val_anns, val_gallery, url_mapping,
        cache_dir=cache_path, max_queries=args.max_eval_queries,
    )

    # Training with mining rounds
    for round_idx in range(args.mining_rounds):
        round_lr = args.lr / (2 ** round_idx)
        if args.mining_rounds > 1:
            print(f"\n--- Mining Round {round_idx + 1}/{args.mining_rounds} (lr={round_lr}) ---")

        # Build triplets
        if args.negatives == "hard":
            triplets = mine_hard_triplets(
                model, train_anns, train_gallery, url_mapping,
                cache_dir=cache_path, n_negatives=args.n_negatives,
                top_k=args.top_k, skip_top=args.skip_top,
            )
        elif args.negatives == "mixed":
            triplets = build_mixed_triplets(
                model, train_anns, train_gallery, url_mapping,
                cache_dir=cache_path, n_random=args.n_random, n_hard=args.n_hard,
                top_k=args.top_k, skip_top=args.skip_top,
            )
        else:
            triplets = build_random_triplets(train_anns, train_gallery, n_negatives=args.n_negatives)

        # Train
        history = train_composed(
            model, triplets, url_mapping,
            cache_dir=cache_path, epochs=args.epochs,
            batch_size=args.batch_size, lr=round_lr,
        )
        history.save(str(out_path / f"train_history_r{round_idx + 1}.json"))

    # Evaluate
    print("\n--- Fine-tuned ---")
    finetuned = evaluate_composed(
        model, val_anns, val_gallery, url_mapping,
        cache_dir=cache_path, max_queries=args.max_eval_queries,
    )

    # Comparison
    print(f"\n{'=' * 50}")
    print(f"COMPARISON — {args.category.upper()}")
    print(f"{'=' * 50}")
    for m in baseline:
        b, f = baseline[m], finetuned[m]
        arrow = "+" if f - b >= 0 else ""
        print(f"  {m:<12} {b:.4f} → {f:.4f}  ({arrow}{f-b:.4f})")

    model._full_model.save_pretrained(str(out_path / "adapter"))
    print(f"\nAdapter saved to {out_path / 'adapter'}")


if __name__ == "__main__":
    main()
