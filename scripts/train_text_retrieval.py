"""Text-to-text retrieval: Fine-tune MiniLM on FiQA.

Demonstrates the full khoji Python API for text retrieval:
- Load a BEIR dataset
- Baseline evaluation (BGE-base as reference, MiniLM as target)
- Build training triplets (random, hard, mixed)
- Fine-tune with LoRA
- Evaluate improvement
- Inference with the fine-tuned model

Usage:
    python scripts/train_text_retrieval.py
"""

from __future__ import annotations

from functools import partial

import torch

import khoji
from khoji.loss import infonce_loss

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REFERENCE_MODEL = "BAAI/bge-base-en-v1.5"
DATASET = "fiqa"
OUTPUT_DIR = "./output/text-retrieval-fiqa"


def main():
    print("=" * 60)
    print("Text-to-Text Retrieval: MiniLM on FiQA")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────
    train_ds = khoji.load_beir(DATASET, split="train")
    test_ds = khoji.load_beir(DATASET, split="test")
    print(f"Train: {len(train_ds.queries)} queries, {len(train_ds.corpus)} docs")
    print(f"Test:  {len(test_ds.queries)} queries, {len(test_ds.corpus)} docs")

    # ── 2. Baseline evaluation ────────────────────────────
    print("\n--- Reference: BGE-base (110M) ---")
    ref_eval = khoji.Evaluator(REFERENCE_MODEL)
    ref_result = ref_eval.evaluate(
        dataset_name=DATASET, k_values=[1, 5, 10], dataset=test_ds,
    )
    ref_result.print()
    del ref_eval

    print("\n--- Baseline: MiniLM (22M) ---")
    base_eval = khoji.Evaluator(MODEL)
    base_result = base_eval.evaluate(
        dataset_name=DATASET, k_values=[1, 5, 10], dataset=test_ds,
    )
    base_result.print()
    del base_eval

    # ── 3. Build training triplets ────────────────────────
    # Mixed negatives: 2 random (easy) + 1 hard (challenging)
    mining_model = khoji.EmbeddingModel(MODEL)
    triplets = khoji.build_mixed_negatives(
        train_ds, mining_model,
        n_random=2, n_hard=1, top_k=50, skip_top=0,
    )
    del mining_model
    print(f"Total triplets: {len(triplets)}")

    # ── 4. Configure training ─────────────────────────────
    lora = khoji.LoRASettings(r=16, alpha=32, dropout=0.1)
    config = khoji.TrainingConfig(
        epochs=5,
        batch_size=16,
        grad_accum_steps=1,
        lr=2e-5,
        weight_decay=0.01,
        warmup_steps=50,
        max_grad_norm=1.0,
        max_length=512,
        loss_fn=partial(infonce_loss, temperature=0.05),
        lora=lora,
        save_dir=f"{OUTPUT_DIR}/adapter",
        sanity_check_samples=10,
    )

    # ── 5. Train ──────────────────────────────────────────
    trainer = khoji.Trainer(MODEL, config)
    history = trainer.train(khoji.TripletDataset(triplets))
    history.save(f"{OUTPUT_DIR}/train_history.json")

    # ── 6. Evaluate single-round model ──────────────────────
    print("\n--- Fine-tuned: MiniLM + LoRA (1 round) ---")
    ft_eval = khoji.Evaluator(MODEL, adapter_path=f"{OUTPUT_DIR}/adapter")
    ft_result = ft_eval.evaluate(
        dataset_name=DATASET, k_values=[1, 5, 10], dataset=test_ds,
    )
    ft_result.print()
    ft_result.save(f"{OUTPUT_DIR}/finetuned.json")
    del ft_eval

    # ── 7. Iterative mining rounds ────────────────────────
    # Round 2: re-mine negatives using the fine-tuned model,
    # then train again with halved LR for sharper negatives
    print("\n--- Round 2: re-mine with fine-tuned model ---")
    ft_mining_model = khoji.EmbeddingModel(
        MODEL, adapter_path=f"{OUTPUT_DIR}/adapter",
    )
    triplets_r2 = khoji.build_mixed_negatives(
        train_ds, ft_mining_model,
        n_random=2, n_hard=1, top_k=50, skip_top=0,
    )
    del ft_mining_model
    print(f"Round 2 triplets: {len(triplets_r2)}")

    config_r2 = khoji.TrainingConfig(
        epochs=3,
        batch_size=16,
        lr=1e-5,  # halved LR for round 2
        warmup_steps=30,
        max_length=512,
        loss_fn=partial(infonce_loss, temperature=0.05),
        lora=lora,
        save_dir=f"{OUTPUT_DIR}/adapter_r2",
        sanity_check_samples=10,
    )

    # Warm-start from round 1's adapter
    trainer_r2 = khoji.Trainer(
        MODEL, config_r2, adapter_path=f"{OUTPUT_DIR}/adapter",
    )
    history_r2 = trainer_r2.train(khoji.TripletDataset(triplets_r2))

    print("\n--- Fine-tuned: MiniLM + LoRA (2 rounds) ---")
    ft2_eval = khoji.Evaluator(
        MODEL, adapter_path=f"{OUTPUT_DIR}/adapter_r2",
    )
    ft2_result = ft2_eval.evaluate(
        dataset_name=DATASET, k_values=[1, 5, 10], dataset=test_ds,
    )
    ft2_result.print()
    del ft2_eval

    # ── 8. Comparison ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Model':<25} {'nDCG@10':>10} {'MRR@10':>10} {'R@10':>10}")
    print("-" * 57)
    for name, r in [
        ("BGE-base (110M)", ref_result),
        ("MiniLM baseline", base_result),
        ("MiniLM + LoRA (1 round)", ft_result),
        ("MiniLM + LoRA (2 rounds)", ft2_result),
    ]:
        m = r.metrics
        print(f"{name:<25} {m['ndcg@10']:>10.4f} {m['mrr@10']:>10.4f} {m['recall@10']:>10.4f}")

    gap_before = ref_result.metrics["ndcg@10"] - base_result.metrics["ndcg@10"]
    gap_after = ref_result.metrics["ndcg@10"] - ft2_result.metrics["ndcg@10"]
    if gap_before > 0:
        print(f"\nnDCG@10 gap vs BGE: {gap_before:.4f} → {gap_after:.4f}")
        print(f"Gap closed by {100 * (1 - gap_after / gap_before):.1f}%")

    # ── 8. Inference ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("INFERENCE DEMO")
    print(f"{'=' * 60}")

    model = khoji.EmbeddingModel(MODEL, adapter_path=f"{OUTPUT_DIR}/adapter_r2")

    query = "Vitamin D deficiency increases respiratory infection risk."
    docs = [
        "Low vitamin D levels correlate with acute respiratory infections.",
        "Calcium supplementation does not affect bone density in adults.",
        "Vitamin D reduces respiratory infection risk by 12% in trials.",
        "Iron deficiency anemia is the most common nutritional disorder.",
    ]

    q_emb = model.encode([query])
    d_emb = model.encode(docs)
    scores = torch.mm(q_emb, d_emb.t()).squeeze(0)
    ranked = torch.argsort(scores, descending=True).tolist()

    print(f"\nQuery: {query}\n")
    for rank, idx in enumerate(ranked):
        print(f"  {rank+1}. (score={scores[idx]:.4f}) {docs[idx][:70]}...")


if __name__ == "__main__":
    main()
