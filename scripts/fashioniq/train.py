"""Composed image retrieval on FashionIQ using BLIP-2 Q-Former.

Query = (reference image + modification caption) → retrieve target image.
Uses BLIP-2's Q-Former to jointly encode (image, text) into a shared space.

Images are loaded on the fly via URLs (cached locally on first download).

Usage:
    # 1. Download annotations + URL mappings
    python scripts/fashioniq/download_data.py

    # 2. Run training
    python scripts/fashioniq/train.py --category dress --epochs 5

    # Run all categories
    python scripts/fashioniq/train.py --category all --epochs 5
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

# khoji imports — reuse what we can
from khoji.device import get_device
from khoji.image_utils import load_image
from khoji.lora import LoRASettings, apply_lora
from khoji.loss import infonce_loss, triplet_margin_loss
from khoji.metrics import mrr_at_k, recall_at_k
from khoji.trainer import TrainHistory


# ──────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────

CATEGORIES = ["dress", "shirt", "toptee"]

URL_MAP_BASE = (
    "https://raw.githubusercontent.com/"
    "hongwang600/fashion-iq-metadata/master/image_url"
)


def load_url_mapping(data_dir: Path, category: str) -> dict[str, str]:
    """Load ASIN → image URL mapping for a category.

    Downloads from GitHub on first call, caches locally.
    """
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
                asin = parts[0].strip()
                url = parts[1].strip()
                mapping[asin] = url

    return mapping


def load_image_by_id(
    image_id: str,
    url_mapping: dict[str, str],
    cache_dir: Path | None = None,
) -> Image.Image | None:
    """Load an image by ASIN — from cache or URL."""
    # Try local cache first
    if cache_dir is not None:
        for ext in [".jpg", ".jpeg", ".png"]:
            local = cache_dir / f"{image_id}{ext}"
            if local.exists():
                try:
                    return Image.open(local).convert("RGB")
                except Exception:
                    pass

    # Load from URL
    url = url_mapping.get(image_id)
    if url is None:
        return None

    try:
        img = load_image(url, cache_dir=str(cache_dir) if cache_dir else None)
        return img.convert("RGB") if img else None
    except Exception:
        return None


@dataclass
class ComposedTriplet:
    """A composed retrieval triplet."""

    candidate_id: str  # reference image ID
    caption: str  # modification text
    target_id: str  # target image ID
    negative_id: str  # hard/random negative image ID


class ComposedTripletDataset(Dataset):
    """PyTorch Dataset for composed retrieval triplets."""

    def __init__(self, triplets: list[ComposedTriplet]):
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> ComposedTriplet:
        return self.triplets[idx]


def load_fashioniq(
    data_dir: str,
    category: str,
    split: str = "train",
) -> tuple[list[dict], list[str]]:
    """Load FashionIQ annotations and image IDs for a category.

    Returns:
        (annotations, gallery_image_ids)
    """
    data_path = Path(data_dir)

    cap_file = data_path / "captions" / f"cap.{category}.{split}.json"
    with open(cap_file) as f:
        annotations = json.load(f)

    split_file = data_path / "image_splits" / f"split.{category}.{split}.json"
    with open(split_file) as f:
        gallery_ids = json.load(f)

    print(
        f"Loaded {category}/{split}: "
        f"{len(annotations)} annotations, {len(gallery_ids)} gallery images"
    )
    return annotations, gallery_ids


def build_triplets(
    annotations: list[dict],
    gallery_ids: list[str],
    n_negatives: int = 1,
    seed: int = 42,
) -> list[ComposedTriplet]:
    """Build composed retrieval triplets with random negatives.

    Each annotation has 2 captions — we create a triplet for each caption.
    """
    rng = random.Random(seed)
    triplets = []

    for ann in annotations:
        candidate_id = ann["candidate"]
        target_id = ann["target"]
        non_targets = [
            gid for gid in gallery_ids
            if gid != target_id and gid != candidate_id
        ]

        for caption in ann["captions"]:
            neg_ids = rng.sample(
                non_targets, min(n_negatives, len(non_targets))
            )
            for neg_id in neg_ids:
                triplets.append(ComposedTriplet(
                    candidate_id=candidate_id,
                    caption=caption,
                    target_id=target_id,
                    negative_id=neg_id,
                ))

    rng.shuffle(triplets)
    print(f"Built {len(triplets)} composed triplets")
    return triplets


# ──────────────────────────────────────────────────────────
# Model: BLIP-2 Q-Former for composed retrieval
# ──────────────────────────────────────────────────────────


class ComposedRetrievalModel:
    """BLIP-2 Q-Former wrapper for composed image retrieval.

    encode_composed(image, text) → embedding (query)
    encode_images(images) → embeddings (targets)

    Both return vectors in the same Q-Former embedding space.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-itm-vit-g",
        device: torch.device | None = None,
        lora: LoRASettings | None = None,
    ):
        from transformers import AutoProcessor, Blip2Model

        self.device = device or get_device()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Blip2Model.from_pretrained(model_name).to(self.device)
        self.model_name = model_name

        if lora is not None:
            self.model = apply_lora(self.model, lora)

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(
            f"Loaded {model_name} | device: {self.device} | "
            f"trainable: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    def encode_composed(
        self,
        images: list[Image.Image],
        texts: list[str],
    ) -> torch.Tensor:
        """Encode (image, text) pairs → joint embeddings.

        Returns:
            (batch, embed_dim) tensor, L2-normalized.
        """
        inputs = self.processor(
            images=images, text=texts,
            return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)

        qformer_out = self.model.get_qformer_features(**inputs)
        embeddings = qformer_out.mean(dim=1)
        return F.normalize(embeddings, p=2, dim=1)

    @torch.no_grad()
    def encode_images(
        self,
        images: list[Image.Image],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Encode images → embeddings (no text).

        Returns:
            (n_images, embed_dim) tensor, L2-normalized.
        """
        all_embeddings = []
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding images", unit="batch")

        for start in iterator:
            batch_images = images[start: start + batch_size]
            inputs = self.processor(
                images=batch_images, return_tensors="pt",
            ).to(self.device)

            qformer_out = self.model.get_qformer_features(**inputs)
            embeddings = qformer_out.mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)


# ──────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────


def train_composed(
    model: ComposedRetrievalModel,
    triplets: list[ComposedTriplet],
    url_mapping: dict[str, str],
    cache_dir: Path | None = None,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 2e-5,
    warmup_steps: int = 50,
    loss_type: str = "infonce",
    temperature: float = 0.05,
) -> TrainHistory:
    """Train the composed retrieval model on triplets."""
    device = model.device
    history = TrainHistory()

    if loss_type == "infonce":
        loss_fn = partial(infonce_loss, temperature=temperature)
    else:
        loss_fn = partial(triplet_margin_loss, margin=0.2)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.model.parameters()),
        lr=lr, weight_decay=0.01,
    )

    total_steps = (len(triplets) // batch_size) * epochs
    warmup = min(warmup_steps, total_steps // 5)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        remaining = total_steps - step
        total_decay = total_steps - warmup
        return max(0.0, remaining / max(total_decay, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.model.train()
    dataset = ComposedTripletDataset(triplets)

    print(
        f"\nTraining on {len(triplets)} triplets for {epochs} epochs "
        f"(batch_size={batch_size}, lr={lr})"
    )

    pbar = tqdm(
        total=len(triplets) * epochs // batch_size,
        desc="Training", unit="batch",
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start: start + batch_size]
            if len(batch_indices) < 2:
                continue

            batch = [dataset[i] for i in batch_indices]

            # Load images via URL (with caching)
            candidate_imgs = []
            target_imgs = []
            negative_imgs = []
            captions = []
            valid = True

            for t in batch:
                c_img = load_image_by_id(t.candidate_id, url_mapping, cache_dir)
                t_img = load_image_by_id(t.target_id, url_mapping, cache_dir)
                n_img = load_image_by_id(t.negative_id, url_mapping, cache_dir)

                if c_img is None or t_img is None or n_img is None:
                    valid = False
                    break

                candidate_imgs.append(c_img)
                target_imgs.append(t_img)
                negative_imgs.append(n_img)
                captions.append(t.caption)

            if not valid:
                continue

            # Encode query: (candidate_image, caption) → embedding
            query_emb = model.encode_composed(candidate_imgs, captions)

            # Encode targets and negatives
            target_inputs = model.processor(
                images=target_imgs, return_tensors="pt",
            ).to(device)
            target_qf = model.model.get_qformer_features(**target_inputs)
            target_emb = F.normalize(target_qf.mean(dim=1), p=2, dim=1)

            neg_inputs = model.processor(
                images=negative_imgs, return_tensors="pt",
            ).to(device)
            neg_qf = model.model.get_qformer_features(**neg_inputs)
            neg_emb = F.normalize(neg_qf.mean(dim=1), p=2, dim=1)

            loss = loss_fn(query_emb, target_emb, neg_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.model.parameters(), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            epoch_batches += 1

            history.step_loss.append(batch_loss)
            history.step_lr.append(scheduler.get_last_lr()[0])

            pbar.update(1)
            pbar.set_postfix(
                loss=f"{batch_loss:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        avg_loss = epoch_loss / max(epoch_batches, 1)
        history.epoch_loss.append(avg_loss)
        tqdm.write(
            f"  Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.4f}"
        )

    pbar.close()
    return history


# ──────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_composed(
    model: ComposedRetrievalModel,
    annotations: list[dict],
    gallery_ids: list[str],
    url_mapping: dict[str, str],
    cache_dir: Path | None = None,
    k_values: list[int] | None = None,
    max_queries: int | None = None,
) -> dict[str, float]:
    """Evaluate composed retrieval: Recall@k and MRR@k over the gallery."""
    if k_values is None:
        k_values = [1, 5, 10, 50]

    model.model.eval()

    # Pre-encode all gallery images
    print(f"Loading and encoding {len(gallery_ids)} gallery images...")
    gallery_images = []
    valid_gallery_ids = []
    for gid in tqdm(gallery_ids, desc="Loading gallery"):
        img = load_image_by_id(gid, url_mapping, cache_dir)
        if img is not None:
            gallery_images.append(img)
            valid_gallery_ids.append(gid)

    print(
        f"Loaded {len(valid_gallery_ids)}/{len(gallery_ids)} gallery images"
    )
    gallery_embeddings = model.encode_images(gallery_images)
    gallery_id_to_idx = {
        gid: i for i, gid in enumerate(valid_gallery_ids)
    }

    if max_queries is not None:
        annotations = annotations[:max_queries]

    metric_sums: dict[str, float] = {}
    for k in k_values:
        metric_sums[f"recall@{k}"] = 0.0
        metric_sums[f"mrr@{k}"] = 0.0

    n_queries = 0

    print(f"Evaluating {len(annotations)} queries...")
    for ann in tqdm(annotations, desc="Evaluating"):
        candidate_id = ann["candidate"]
        target_id = ann["target"]

        if target_id not in gallery_id_to_idx:
            continue

        c_img = load_image_by_id(candidate_id, url_mapping, cache_dir)
        if c_img is None:
            continue

        caption = ann["captions"][0]
        query_emb = model.encode_composed([c_img], [caption])

        scores = torch.mm(query_emb, gallery_embeddings.t()).squeeze(0)
        ranked_indices = torch.argsort(scores, descending=True).tolist()
        ranked_ids = [valid_gallery_ids[i] for i in ranked_indices]

        qrel = {target_id: 1}
        for k in k_values:
            metric_sums[f"recall@{k}"] += recall_at_k(ranked_ids, qrel, k)
            metric_sums[f"mrr@{k}"] += mrr_at_k(ranked_ids, qrel, k)

        n_queries += 1

    metrics = {}
    for key, total in metric_sums.items():
        metrics[key] = round(total / max(n_queries, 1), 4)

    print(
        f"\nResults ({n_queries} queries, "
        f"{len(valid_gallery_ids)} gallery):"
    )
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
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 2e-5,
    lora_r: int = 8,
    n_negatives: int = 1,
    max_eval_queries: int | None = 100,
    output_dir: str = "./output/fashioniq",
):
    """Run a complete composed retrieval experiment."""
    data_path = Path(data_dir)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / category
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"# FashionIQ Composed Retrieval: {category.upper()}")
    print(f"# Model: BLIP-2 Q-Former + LoRA (r={lora_r})")
    print(f"# Images: loaded via URL, cached to {cache_dir}")
    print(f"{'#' * 60}")

    # Load annotations
    train_anns, train_gallery = load_fashioniq(data_dir, category, "train")
    val_anns, val_gallery = load_fashioniq(data_dir, category, "val")

    # Load URL mapping
    print("\nLoading image URL mapping...")
    url_mapping = load_url_mapping(data_path, category)
    print(f"  {len(url_mapping)} image URLs loaded")

    # Build model
    lora = LoRASettings(r=lora_r, alpha=lora_r * 2, dropout=0.1)
    model = ComposedRetrievalModel(lora=lora)

    # Baseline evaluation
    print("\n--- Baseline Evaluation ---")
    baseline = evaluate_composed(
        model, val_anns, val_gallery, url_mapping,
        cache_dir=cache_path, max_queries=max_eval_queries,
    )

    # Build triplets and train
    triplets = build_triplets(
        train_anns, train_gallery, n_negatives=n_negatives
    )
    history = train_composed(
        model, triplets, url_mapping,
        cache_dir=cache_path,
        epochs=epochs, batch_size=batch_size, lr=lr,
    )

    history.save(str(out_path / "train_history.json"))

    # Fine-tuned evaluation
    print("\n--- Fine-tuned Evaluation ---")
    finetuned = evaluate_composed(
        model, val_anns, val_gallery, url_mapping,
        cache_dir=cache_path, max_queries=max_eval_queries,
    )

    # Comparison
    print(f"\n{'=' * 50}")
    print(f"COMPARISON — {category.upper()}")
    print(f"{'=' * 50}")
    for metric in baseline:
        b = baseline[metric]
        f = finetuned[metric]
        delta = f - b
        arrow = "+" if delta >= 0 else ""
        print(f"  {metric:<12} {b:.4f} → {f:.4f}  ({arrow}{delta:.4f})")

    # Save model
    model.model.save_pretrained(str(out_path / "adapter"))
    print(f"\nAdapter saved to {out_path / 'adapter'}")

    return baseline, finetuned, history


def main():
    parser = argparse.ArgumentParser(
        description="FashionIQ composed image retrieval"
    )
    parser.add_argument(
        "--category", default="dress",
        choices=["dress", "shirt", "toptee", "all"],
    )
    parser.add_argument("--data-dir", default="./data/fashioniq")
    parser.add_argument(
        "--cache-dir", default="./data/fashioniq/image_cache",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--n-negatives", type=int, default=1)
    parser.add_argument("--max-eval-queries", type=int, default=100)
    parser.add_argument("--output-dir", default="./output/fashioniq")
    args = parser.parse_args()

    categories = (
        CATEGORIES if args.category == "all" else [args.category]
    )

    for cat in categories:
        run_experiment(
            category=cat,
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lora_r=args.lora_r,
            n_negatives=args.n_negatives,
            max_eval_queries=args.max_eval_queries,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
