"""Composed image retrieval using khoji's first-class API.

Demonstrates both config-driven and manual API approaches for
composed (image+text → image) retrieval on FashionIQ.

Requires FashionIQ annotations:
    python scripts/fashioniq/download_data.py

Usage:
    python scripts/train_composed_retrieval_api.py
    python scripts/train_composed_retrieval_api.py --category shirt
    python scripts/train_composed_retrieval_api.py --approach api
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import khoji
from khoji.config import EvalConfig, LoRAConfig, TrainConfig

MODEL = "Salesforce/blip2-itm-vit-g"
CATEGORIES = ["dress", "shirt", "toptee"]

URL_MAP_BASE = (
    "https://raw.githubusercontent.com/"
    "hongwang600/fashion-iq-metadata/master/image_url"
)


# ── Helpers ───────────────────────────────────────────────


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


def build_composed_dataset(
    data_dir: str,
    category: str,
    split: str,
    url_mapping: dict[str, str],
    cache_dir: str | None = None,
) -> khoji.ComposedRetrievalDataset:
    """Convert FashionIQ annotations to a ComposedRetrievalDataset.

    This maps FashionIQ's format into khoji's standard composed dataset:
    - queries: (reference_image_url, modification_caption)
    - corpus: gallery image URLs
    - qrels: query → target image
    """
    data_path = Path(data_dir)
    with open(data_path / "captions" / f"cap.{category}.{split}.json") as f:
        annotations = json.load(f)
    with open(data_path / "image_splits" / f"split.{category}.{split}.json") as f:
        gallery_ids = json.load(f)

    queries: dict[str, tuple[str, str]] = {}
    corpus: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    # Build corpus from gallery IDs
    for gid in gallery_ids:
        if gid in url_mapping:
            corpus[gid] = url_mapping[gid]

    # Build queries from annotations
    for i, ann in enumerate(annotations):
        candidate_id = ann["candidate"]
        target_id = ann["target"]

        if candidate_id not in url_mapping or target_id not in corpus:
            continue

        candidate_url = url_mapping[candidate_id]

        for cap_idx, caption in enumerate(ann["captions"]):
            qid = f"q_{i}_{cap_idx}"
            queries[qid] = (candidate_url, caption)
            qrels[qid] = {target_id: 1}

    print(
        f"Built ComposedRetrievalDataset ({category}/{split}): "
        f"{len(queries)} queries, {len(corpus)} gallery images"
    )

    return khoji.ComposedRetrievalDataset(
        queries=queries,
        corpus=corpus,
        qrels=qrels,
        base_dir=cache_dir,
    )


def save_as_custom_format(
    dataset: khoji.ComposedRetrievalDataset,
    output_path: str,
) -> None:
    """Save a ComposedRetrievalDataset in khoji's standard local format.

    Creates the directory with queries.jsonl, corpus.jsonl, qrels.tsv
    that can be loaded by khoji.load_custom_composed().
    """
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "queries.jsonl", "w") as f:
        for qid, (image, text) in dataset.queries.items():
            f.write(json.dumps({"_id": qid, "image": image, "text": text}) + "\n")

    with open(out / "corpus.jsonl", "w") as f:
        for did, image in dataset.corpus.items():
            f.write(json.dumps({"_id": did, "image": image}) + "\n")

    with open(out / "qrels.tsv", "w") as f:
        for qid, docs in dataset.qrels.items():
            for did, score in docs.items():
                f.write(f"{qid}\t{did}\t{score}\n")

    print(f"Saved dataset to {out}")


# ── Config-driven approach ────────────────────────────────


def run_config_approach(args):
    """Use ComposedForgeConfig + run_composed() for one-shot pipeline."""
    print("=" * 60)
    print("APPROACH 1: Config-driven (run_composed)")
    print("=" * 60)

    # Prepare data in khoji's standard format
    data_path = Path(args.data_dir)
    url_mapping = load_url_mapping(data_path, args.category)

    train_ds = build_composed_dataset(
        args.data_dir, args.category, "train", url_mapping, args.cache_dir
    )
    val_ds = build_composed_dataset(
        args.data_dir, args.category, "val", url_mapping, args.cache_dir
    )

    # Save to standard format so run_composed() can load it
    train_dir = str(Path(args.output_dir) / "data" / "train")
    val_dir = str(Path(args.output_dir) / "data" / "val")
    save_as_custom_format(train_ds, train_dir)
    save_as_custom_format(val_ds, val_dir)

    config = khoji.ComposedForgeConfig(
        model=khoji.composed_config.ComposedModelConfig(
            name=MODEL,
        ),
        data=khoji.composed_config.ComposedDataConfig(
            dataset=train_dir,
            negatives="mixed",
            n_random=2,
            n_hard=1,
            n_queries=args.max_queries,
            top_k=50,
            skip_top=5,
            mining_rounds=1,
            cache_dir=args.cache_dir,
        ),
        lora=LoRAConfig(r=args.lora_r, alpha=args.lora_r * 2, dropout=0.1),
        train=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=1,
            lr=2e-5,
            warmup_steps=50,
            loss="infonce",
            temperature=0.05,
            max_length=77,
        ),
        eval=EvalConfig(
            dataset=val_dir,
            k_values=[1, 5, 10, 50],
            n_queries=args.max_eval_queries,
            run_before=True,
            run_after=True,
        ),
        seed=42,
        output_dir=str(Path(args.output_dir) / "config-approach"),
    )

    result = khoji.run_composed(config)
    print(f"\nAdapter saved to: {result.adapter_dir}")
    return result


# ── Manual API approach ───────────────────────────────────


def run_api_approach(args):
    """Use khoji's Python API directly for full control."""
    print("=" * 60)
    print("APPROACH 2: Manual API")
    print("=" * 60)

    data_path = Path(args.data_dir)
    cache_path = Path(args.cache_dir) if args.cache_dir else None
    out_path = Path(args.output_dir) / "api-approach"
    out_path.mkdir(parents=True, exist_ok=True)

    url_mapping = load_url_mapping(data_path, args.category)

    train_ds = build_composed_dataset(
        args.data_dir, args.category, "train", url_mapping,
        str(cache_path) if cache_path else None,
    )
    val_ds = build_composed_dataset(
        args.data_dir, args.category, "val", url_mapping,
        str(cache_path) if cache_path else None,
    )

    # --- Baseline evaluation ---
    print("\n--- Baseline ---")
    baseline_eval = khoji.ComposedEvaluator(MODEL)
    baseline = baseline_eval.evaluate(
        dataset_name=f"fashioniq-{args.category}",
        dataset=val_ds,
        k_values=[1, 5, 10, 50],
        n_queries=args.max_eval_queries,
        cache_dir=str(cache_path) if cache_path else None,
    )
    baseline.print()
    baseline.save(str(out_path / "baseline.json"))

    # --- Build triplets (random negatives for simplicity) ---
    print("\n--- Building triplets ---")
    triplets = khoji.build_random_negatives_composed(
        train_ds,
        n_negatives=3,
        n_queries=args.max_queries,
        seed=42,
    )
    torch_ds = khoji.ComposedTripletDataset(triplets)

    # --- Train ---
    print("\n--- Training ---")
    from functools import partial
    training_config = khoji.ComposedTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=2e-5,
        warmup_steps=50,
        loss_fn=partial(khoji.infonce_loss, temperature=0.05),
        lora=khoji.LoRASettings(r=args.lora_r, alpha=args.lora_r * 2, dropout=0.1),
        save_dir=str(out_path / "adapter"),
        cache_dir=str(cache_path) if cache_path else None,
    )
    trainer = khoji.ComposedTrainer(MODEL, training_config)
    history = trainer.train(torch_ds)
    history.save(str(out_path / "train_history.json"))

    # --- Fine-tuned evaluation ---
    print("\n--- Fine-tuned ---")
    ft_eval = khoji.ComposedEvaluator(
        MODEL, adapter_path=str(out_path / "adapter")
    )
    finetuned = ft_eval.evaluate(
        dataset_name=f"fashioniq-{args.category}",
        dataset=val_ds,
        k_values=[1, 5, 10, 50],
        n_queries=args.max_eval_queries,
        cache_dir=str(cache_path) if cache_path else None,
    )
    finetuned.print()
    finetuned.save(str(out_path / "finetuned.json"))

    # --- Comparison ---
    print(f"\n{'=' * 55}")
    print(f"  COMPARISON — {args.category.upper()}")
    print(f"{'=' * 55}")
    for m in baseline.metrics:
        b, f = baseline.metrics[m], finetuned.metrics[m]
        sign = "+" if f - b >= 0 else ""
        print(f"  {m:<12} {b:.4f} → {f:.4f}  ({sign}{f-b:.4f})")

    return baseline, finetuned


# ── Main ──────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Composed retrieval using khoji API"
    )
    parser.add_argument("--category", default="dress", choices=CATEGORIES)
    parser.add_argument("--data-dir", default="./data/fashioniq")
    parser.add_argument("--cache-dir", default="./data/fashioniq/image_cache")
    parser.add_argument(
        "--approach", default="both", choices=["config", "api", "both"],
        help="Which approach to demo: config-driven, manual API, or both"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-eval-queries", type=int, default=100)
    parser.add_argument("--output-dir", default="./output/composed-retrieval-api")
    args = parser.parse_args()

    if args.approach in ("config", "both"):
        run_config_approach(args)

    if args.approach in ("api", "both"):
        run_api_approach(args)


if __name__ == "__main__":
    main()
