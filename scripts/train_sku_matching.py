"""SKU matching with mixed-mode retrieval on generated grocery dataset.

Cross-brand product matching: given a product (image + description) from
one brand, find the equivalent product from another brand's catalog.

Dataset: 200 AI-generated product images across 5 UK grocery brands,
8 product families, 5 variants each.

Train/Test split: Waitrose (brand E) is held out entirely for test.
Additionally, 2 product families are held out for product-level eval.

Usage:
    python scripts/train_sku_matching.py
    python scripts/train_sku_matching.py --epochs 5 --mining-rounds 2
"""
from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path

import torch

import khoji
from khoji.loss import infonce_loss, triplet_margin_loss


# ── Config ───────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sku-matching"
MODEL = "Salesforce/blip2-itm-vit-g"

TRAIN_BRANDS = ["Sainsburys", "Tesco", "Aldi", "Lidl"]
TEST_BRAND = "Waitrose"

# Held-out families for product-level eval (model never sees these during training)
HOLDOUT_FAMILIES = ["Crisps", "Peanut Butter"]


# ── Dataset building ─────────────────────────────────────


def load_catalog():
    """Load the generated catalog metadata."""
    with open(DATA_DIR / "metadata" / "catalog.json") as f:
        return json.load(f)


def build_datasets(catalog: list[dict], seed: int = 42):
    """Build train and test datasets with two evaluation dimensions.

    Returns:
        train_ds: Train brands × train families (for training)
        test_brand_ds: Test brand queries against train brand corpus (brand generalization)
        test_product_ds: Train brand queries for held-out families (product generalization)
    """
    rng = random.Random(seed)

    # Build product family mapping: (family, variant) -> list of catalog items
    family_variant_items: dict[tuple[str, str], list[dict]] = {}
    for item in catalog:
        key = (item["family"], item["variant"])
        family_variant_items.setdefault(key, []).append(item)

    # Partition items
    train_items = [it for it in catalog
                   if it["brand"] in TRAIN_BRANDS and it["family"] not in HOLDOUT_FAMILIES]
    test_brand_items = [it for it in catalog
                        if it["brand"] == TEST_BRAND and it["family"] not in HOLDOUT_FAMILIES]
    test_product_items = [it for it in catalog
                          if it["brand"] in TRAIN_BRANDS and it["family"] in HOLDOUT_FAMILIES]

    # Build corpus: all train brand items (train families + holdout families)
    # This is "our catalog" — what we match against
    all_train_brand_items = [it for it in catalog if it["brand"] in TRAIN_BRANDS]
    corpus = {}
    for item in all_train_brand_items:
        img_path = str(DATA_DIR / item["image_path"])
        corpus[item["sku_id"]] = (img_path, item["description"])

    # Helper: build qrels for a set of query items against a target corpus
    def _build_qrels(query_items, target_corpus=None):
        if target_corpus is None:
            target_corpus = corpus
        queries = {}
        qrels = {}
        for item in query_items:
            qid = item["sku_id"]
            img_path = str(DATA_DIR / item["image_path"])
            queries[qid] = (img_path, item["description"])

            # Positives: same (family, variant), different brand, in target corpus
            positives = {}
            for other in family_variant_items.get((item["family"], item["variant"]), []):
                if other["sku_id"] != qid and other["sku_id"] in target_corpus:
                    positives[other["sku_id"]] = 1
            if positives:
                qrels[qid] = positives
        # Filter to queries that have at least one positive
        queries = {q: queries[q] for q in qrels}
        return queries, qrels

    # Train: train brands × train families, matching across brands
    train_queries, train_qrels = _build_qrels(train_items)

    # Test (brand): Waitrose queries matching into train brand corpus
    test_brand_queries, test_brand_qrels = _build_qrels(test_brand_items)

    # Test (product): held-out families — use first train brand as query,
    # other train brands as targets. Build a separate corpus that excludes
    # the query brand's held-out family items to prevent self-matching.
    product_query_brand = TRAIN_BRANDS[0]
    test_product_query_items = [it for it in test_product_items if it["brand"] == product_query_brand]
    product_test_corpus = {
        sid: entry for sid, entry in corpus.items()
        if not any(
            item["sku_id"] == sid and item["brand"] == product_query_brand and item["family"] in HOLDOUT_FAMILIES
            for item in catalog
        )
    }
    test_product_queries, test_product_qrels = _build_qrels(test_product_query_items, product_test_corpus)

    train_ds = khoji.ComposedRetrievalDataset(
        queries=train_queries, corpus=corpus, qrels=train_qrels)
    test_brand_ds = khoji.ComposedRetrievalDataset(
        queries=test_brand_queries, corpus=corpus, qrels=test_brand_qrels)
    test_product_ds = khoji.ComposedRetrievalDataset(
        queries=test_product_queries, corpus=product_test_corpus, qrels=test_product_qrels)

    return train_ds, test_brand_ds, test_product_ds


def print_dataset_stats(train_ds, test_brand_ds, test_product_ds, catalog):
    """Print dataset statistics."""
    train_families = set()
    for qid in train_ds.queries:
        for item in catalog:
            if item["sku_id"] == qid:
                train_families.add(item["family"])

    print(f"\nDataset Statistics:")
    print(f"  Corpus: {len(train_ds.corpus)} items ({len(TRAIN_BRANDS)} brands)")
    print(f"  Train queries: {len(train_ds.queries)} ({len(TRAIN_BRANDS)} brands × {len(train_families)} families)")
    print(f"  Test (brand):  {len(test_brand_ds.queries)} ({TEST_BRAND} queries, unseen brand)")
    print(f"  Test (product): {len(test_product_ds.queries)} (held-out families: {HOLDOUT_FAMILIES})")

    # Show sample
    sample_qid = list(train_ds.queries.keys())[0]
    q_img, q_txt = train_ds.queries[sample_qid]
    pos_id = list(train_ds.qrels[sample_qid].keys())[0]
    p_img, p_txt = train_ds.corpus[pos_id]
    print(f"\n  Sample train pair:")
    print(f"    Query:    {Path(q_img).name} | {q_txt}")
    print(f"    Positive: {Path(p_img).name} | {p_txt}")


# ── Evaluation helper ────────────────────────────────────


def evaluate(model_name, adapter_path, dataset, name, k_values, batch_size):
    """Run evaluation and return results."""
    evaluator = khoji.ComposedEvaluator(model_name, adapter_path=adapter_path)
    result = evaluator.evaluate(
        dataset_name=name, dataset=dataset,
        k_values=k_values, batch_size=batch_size,
    )
    result.print()
    del evaluator
    torch.cuda.empty_cache()
    return result


# ── Main ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="SKU matching with mixed-mode retrieval")
    parser.add_argument("--negatives", default="mixed", choices=["random", "hard", "mixed"])
    parser.add_argument("--n-random", type=int, default=4)
    parser.add_argument("--n-hard", type=int, default=1)
    parser.add_argument("--n-negatives", type=int, default=5)
    parser.add_argument("--skip-top", type=int, default=10)
    parser.add_argument("--mining-rounds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--loss", default="infonce", choices=["infonce", "triplet"])
    parser.add_argument("--output-dir", default="./output/sku-matching")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("SKU MATCHING: img+txt → img+txt")
    print("Cross-brand product matching on AI-generated grocery dataset")
    print(f"Model: {MODEL} + LoRA (r={args.lora_r})")
    print(f"Train brands: {TRAIN_BRANDS}")
    print(f"Test brand: {TEST_BRAND} (unseen)")
    print(f"Held-out families: {HOLDOUT_FAMILIES}")
    print(f"Negatives: {args.negatives} | Rounds: {args.mining_rounds} | Epochs: {args.epochs}")
    print("=" * 65)

    # Load catalog and build datasets
    catalog = load_catalog()
    train_ds, test_brand_ds, test_product_ds = build_datasets(catalog)
    print_dataset_stats(train_ds, test_brand_ds, test_product_ds, catalog)

    k_values = [1, 3, 5, 10]

    # ── Baseline evaluation ──────────────────────────────
    print("\n" + "=" * 65)
    print("BASELINE EVALUATION")
    print("=" * 65)

    print("\n--- Brand generalization (Waitrose → train brands) ---")
    baseline_brand = evaluate(MODEL, None, test_brand_ds, "brand-test", k_values, args.eval_batch_size)
    baseline_brand.save(str(out_path / "baseline_brand.json"))

    print("\n--- Product generalization (held-out families) ---")
    baseline_product = evaluate(MODEL, None, test_product_ds, "product-test", k_values, args.eval_batch_size)
    baseline_product.save(str(out_path / "baseline_product.json"))

    # ── Training ─────────────────────────────────────────
    if args.loss == "infonce":
        loss_fn = partial(infonce_loss, temperature=0.05)
    else:
        loss_fn = partial(triplet_margin_loss, margin=0.2)

    lora = khoji.LoRASettings(r=args.lora_r, alpha=args.lora_r * 2, dropout=0.1)
    current_adapter = None
    all_step_loss = []

    for round_idx in range(args.mining_rounds):
        round_lr = args.lr / (2 ** round_idx)
        round_label = f"Round {round_idx + 1}/{args.mining_rounds}"

        print(f"\n{'=' * 65}")
        print(f"{round_label}: BUILDING TRIPLETS")
        print("=" * 65)

        if args.negatives == "mixed":
            mining_model = khoji.JointEmbeddingModel(MODEL, adapter_path=current_adapter)
            triplets = khoji.build_mixed_negatives_composed(
                train_ds, mining_model,
                n_random=args.n_random, n_hard=args.n_hard,
                top_k=50, skip_top=args.skip_top,
                batch_size=args.eval_batch_size,
            )
            del mining_model
            torch.cuda.empty_cache()
        elif args.negatives == "hard":
            mining_model = khoji.JointEmbeddingModel(MODEL, adapter_path=current_adapter)
            triplets = khoji.mine_hard_negatives_composed(
                train_ds, mining_model,
                n_negatives=args.n_negatives,
                top_k=50, skip_top=args.skip_top,
                batch_size=args.eval_batch_size,
            )
            del mining_model
            torch.cuda.empty_cache()
        else:
            triplets = khoji.build_random_negatives_composed(
                train_ds, n_negatives=args.n_negatives, seed=42 + round_idx)

        print(f"Triplets: {len(triplets)}")

        # Show sample
        t = triplets[0]
        print(f"\nSample triplet:")
        print(f"  Query:    {Path(t.query_image).name} | {t.query_text[:60]}...")
        print(f"  Positive: {Path(t.positive_image).name} | {t.positive_text[:60]}...")
        print(f"  Negative: {Path(t.negative_image).name} | {t.negative_text[:60]}...")

        # Train
        is_last = round_idx == args.mining_rounds - 1
        adapter_dir = str(out_path / "adapter") if is_last else str(out_path / f"adapter_r{round_idx + 1}")

        print(f"\n{'=' * 65}")
        print(f"{round_label}: TRAINING (lr={round_lr:.1e})")
        print("=" * 65)

        config = khoji.ComposedTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=1,
            lr=round_lr,
            warmup_steps=20,
            loss_fn=loss_fn,
            lora=lora,
            save_dir=adapter_dir,
            sanity_check_samples=5,
        )
        trainer = khoji.ComposedTrainer(
            MODEL, config,
            adapter_path=current_adapter if round_idx > 0 else None,
        )
        history = trainer.train(khoji.ComposedTripletDataset(triplets))
        history.save(str(out_path / f"history_r{round_idx + 1}.json"))
        all_step_loss.extend(history.step_loss)
        del trainer
        torch.cuda.empty_cache()

        current_adapter = adapter_dir

    # ── Fine-tuned evaluation ────────────────────────────
    print("\n" + "=" * 65)
    print("FINE-TUNED EVALUATION")
    print("=" * 65)

    print("\n--- Brand generalization (Waitrose → train brands) ---")
    ft_brand = evaluate(MODEL, current_adapter, test_brand_ds, "brand-test", k_values, args.eval_batch_size)
    ft_brand.save(str(out_path / "finetuned_brand.json"))

    print("\n--- Product generalization (held-out families) ---")
    ft_product = evaluate(MODEL, current_adapter, test_product_ds, "product-test", k_values, args.eval_batch_size)
    ft_product.save(str(out_path / "finetuned_product.json"))

    # ── Comparison ───────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("COMPARISON")
    print("=" * 65)

    print("\nBrand generalization (Waitrose, unseen brand):")
    for m in baseline_brand.metrics:
        b, f = baseline_brand.metrics[m], ft_brand.metrics[m]
        sign = "+" if f - b >= 0 else ""
        print(f"  {m:<12} {b:.4f} → {f:.4f}  ({sign}{f-b:.4f})")

    print(f"\nProduct generalization (held-out families: {HOLDOUT_FAMILIES}):")
    for m in baseline_product.metrics:
        b, f = baseline_product.metrics[m], ft_product.metrics[m]
        sign = "+" if f - b >= 0 else ""
        print(f"  {m:<12} {b:.4f} → {f:.4f}  ({sign}{f-b:.4f})")

    print(f"\nTraining loss: {all_step_loss[0]:.4f} → {all_step_loss[-1]:.4f}")
    print(f"Adapter saved to: {current_adapter}")

    # Save all results for report generation
    summary = {
        "baseline_brand": baseline_brand.metrics,
        "finetuned_brand": ft_brand.metrics,
        "baseline_product": baseline_product.metrics,
        "finetuned_product": ft_product.metrics,
        "training_loss_first": all_step_loss[0],
        "training_loss_last": all_step_loss[-1],
        "config": {
            "model": MODEL,
            "lora_r": args.lora_r,
            "negatives": args.negatives,
            "mining_rounds": args.mining_rounds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "train_brands": TRAIN_BRANDS,
            "test_brand": TEST_BRAND,
            "holdout_families": HOLDOUT_FAMILIES,
            "train_queries": len(train_ds.queries),
            "corpus_size": len(train_ds.corpus),
            "test_brand_queries": len(test_brand_ds.queries),
            "test_product_queries": len(test_product_ds.queries),
            "triplets_per_round": len(triplets),
        },
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
