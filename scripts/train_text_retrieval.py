"""Text-to-text retrieval: Fine-tune MiniLM on FiQA.

Two approaches shown:
  1. Config approach — construct ForgeConfig, call run() once
  2. API approach — manual pipeline with full control per round

Usage:
    python scripts/train_text_retrieval.py
"""

from __future__ import annotations

from functools import partial

import torch

import khoji
from khoji.config import DataConfig, EvalConfig, LoRAConfig, ModelConfig, TrainConfig
from khoji.loss import infonce_loss

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REFERENCE_MODEL = "BAAI/bge-base-en-v1.5"
DATASET = "fiqa"
OUTPUT = "./output/text-retrieval"


def approach_config():
    """Approach 1: ForgeConfig with mining_rounds — one call does everything."""
    print("\n" + "=" * 60)
    print("APPROACH 1: Config-driven pipeline")
    print("=" * 60)

    config = khoji.ForgeConfig(
        model=ModelConfig(name=MODEL),
        data=DataConfig(
            dataset=DATASET,
            split="train",
            negatives="mixed",
            n_random=2,
            n_hard=1,
            top_k=50,
            skip_top=0,
            mining_rounds=2,  # mine → train → re-mine → train
        ),
        lora=LoRAConfig(r=16, alpha=32, dropout=0.1),
        train=TrainConfig(
            epochs=3,
            batch_size=16,
            lr=2e-5,
            warmup_steps=50,
            max_length=512,
            loss="infonce",
            temperature=0.05,
            sanity_check_samples=10,
        ),
        eval=EvalConfig(
            k_values=[1, 5, 10],
            split="test",
            run_before=True,
            run_after=True,
        ),
        seed=42,
        output_dir=f"{OUTPUT}/config-approach",
    )

    # One call — handles mining, training, re-mining, re-training, eval
    result = khoji.run(config)

    print(f"\nEpoch losses across {len(result.history.epoch_loss)} epochs:")
    print([f"{l:.4f}" for l in result.history.epoch_loss])
    return result


def approach_api():
    """Approach 2: Manual API — full control over each step."""
    print("\n" + "=" * 60)
    print("APPROACH 2: Manual Python API")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────
    train_ds = khoji.load_beir(DATASET, split="train")
    test_ds = khoji.load_beir(DATASET, split="test")
    print(f"Train: {len(train_ds.queries)} queries")
    print(f"Test:  {len(test_ds.queries)} queries")

    # ── Baselines ─────────────────────────────────────────
    print("\n--- BGE-base (110M) baseline ---")
    ref_eval = khoji.Evaluator(REFERENCE_MODEL)
    ref_result = ref_eval.evaluate(
        dataset_name=DATASET, k_values=[1, 5, 10], dataset=test_ds,
    )
    ref_result.print()
    del ref_eval

    print("\n--- MiniLM (22M) baseline ---")
    base_eval = khoji.Evaluator(MODEL)
    base_result = base_eval.evaluate(
        dataset_name=DATASET, k_values=[1, 5, 10], dataset=test_ds,
    )
    base_result.print()
    del base_eval

    # ── Round 1: mine + train ─────────────────────────────
    lora = khoji.LoRASettings(r=16, alpha=32, dropout=0.1)
    loss_fn = partial(infonce_loss, temperature=0.05)

    print("\n--- Round 1: mine with base model ---")
    mining_model = khoji.EmbeddingModel(MODEL)
    triplets = khoji.build_mixed_negatives(
        train_ds, mining_model,
        n_random=2, n_hard=1, top_k=50,
    )
    del mining_model

    config_r1 = khoji.TrainingConfig(
        epochs=3, batch_size=16, lr=2e-5, warmup_steps=50,
        max_length=512, loss_fn=loss_fn, lora=lora,
        save_dir=f"{OUTPUT}/api-approach/adapter_r1",
        sanity_check_samples=10,
    )
    trainer = khoji.Trainer(MODEL, config_r1)
    trainer.train(khoji.TripletDataset(triplets))

    # ── Round 2: re-mine with fine-tuned model + train ────
    print("\n--- Round 2: re-mine with fine-tuned model ---")
    ft_model = khoji.EmbeddingModel(
        MODEL, adapter_path=f"{OUTPUT}/api-approach/adapter_r1",
    )
    triplets_r2 = khoji.build_mixed_negatives(
        train_ds, ft_model,
        n_random=2, n_hard=1, top_k=50,
    )
    del ft_model

    config_r2 = khoji.TrainingConfig(
        epochs=3, batch_size=16, lr=1e-5,  # halved LR
        warmup_steps=30, max_length=512,
        loss_fn=loss_fn, lora=lora,
        save_dir=f"{OUTPUT}/api-approach/adapter_final",
        sanity_check_samples=10,
    )
    trainer_r2 = khoji.Trainer(
        MODEL, config_r2,
        adapter_path=f"{OUTPUT}/api-approach/adapter_r1",
    )
    trainer_r2.train(khoji.TripletDataset(triplets_r2))

    # ── Evaluate ──────────────────────────────────────────
    print("\n--- Fine-tuned (2 rounds) ---")
    ft_eval = khoji.Evaluator(
        MODEL, adapter_path=f"{OUTPUT}/api-approach/adapter_final",
    )
    ft_result = ft_eval.evaluate(
        dataset_name=DATASET, k_values=[1, 5, 10], dataset=test_ds,
    )
    ft_result.print()
    del ft_eval

    # ── Comparison ────────────────────────────────────────
    print(f"\n{'Model':<30} {'nDCG@10':>10} {'MRR@10':>10} {'R@10':>10}")
    print("-" * 62)
    for name, r in [
        ("BGE-base (110M)", ref_result),
        ("MiniLM baseline", base_result),
        ("MiniLM + LoRA (2 rounds)", ft_result),
    ]:
        m = r.metrics
        print(f"{name:<30} {m['ndcg@10']:>10.4f} {m['mrr@10']:>10.4f} {m['recall@10']:>10.4f}")

    # ── Inference ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("INFERENCE")
    print(f"{'=' * 60}")

    model = khoji.EmbeddingModel(
        MODEL, adapter_path=f"{OUTPUT}/api-approach/adapter_final",
    )

    query = "What is compound interest and how does it work?"
    docs = [
        "Compound interest is calculated on the initial principal and accumulated interest.",
        "Stock dividends are payments from company profits to shareholders.",
        "Compound interest grows exponentially over time, unlike simple interest.",
        "A 401k is a tax-advantaged retirement savings plan.",
    ]

    q_emb = model.encode([query])
    d_emb = model.encode(docs)
    scores = torch.mm(q_emb, d_emb.t()).squeeze(0)
    ranked = torch.argsort(scores, descending=True).tolist()

    print(f"\nQuery: {query}\n")
    for rank, idx in enumerate(ranked):
        print(f"  {rank+1}. (score={scores[idx]:.4f}) {docs[idx][:70]}...")


if __name__ == "__main__":
    # Run both approaches
    approach_config()
    approach_api()
