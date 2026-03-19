"""Demonstrate that khoji fine-tuning improves retrieval performance.

Two experiments:
  1. Weak model (MiniLM-L6) on FiQA   — smaller model with room to improve
  2. Strong model (BGE-base) on SciFact — scientific domain, 809 train queries

Usage:
    python scripts/demo_finetuning.py              # run both experiments
    python scripts/demo_finetuning.py minilm        # run only MiniLM on FiQA
    python scripts/demo_finetuning.py scifact       # run only BGE on SciFact
"""

from __future__ import annotations

import sys
from functools import partial

import khoji
from khoji.loss import infonce_loss


def run_experiment(
    name: str,
    model_name: str,
    dataset_name: str,
    train_split: str,
    eval_split: str,
    n_negatives: int,
    negatives: str,
    epochs: int,
    lr: float,
    lora_r: int,
    lora_alpha: int,
    batch_size: int,
    output_dir: str,
) -> None:
    """Run a single fine-tuning experiment and print comparison."""

    print(f"\n{'#' * 70}")
    print(f"# {name}")
    print(f"# Model: {model_name}")
    print(f"# Dataset: {dataset_name} (train={train_split}, eval={eval_split})")
    print(f"# Negatives: {negatives}, n={n_negatives}")
    print(f"# LoRA: r={lora_r}, alpha={lora_alpha}")
    print(f"# Training: {epochs} epochs, lr={lr}, batch_size={batch_size}")
    print(f"{'#' * 70}\n")

    # --- 1. Load dataset ---
    print("Loading dataset...")
    train_dataset = khoji.load_beir(dataset_name, split=train_split)
    eval_dataset = khoji.load_beir(dataset_name, split=eval_split)
    print(f"  Train: {len(train_dataset.queries)} queries, {len(train_dataset.corpus)} docs")
    print(f"  Eval:  {len(eval_dataset.queries)} queries, {len(eval_dataset.corpus)} docs")

    # --- 2. Baseline evaluation ---
    print("\n--- Baseline Evaluation ---")
    baseline_evaluator = khoji.Evaluator(model_name)
    baseline = baseline_evaluator.evaluate(
        dataset_name=dataset_name,
        k_values=[1, 5, 10],
        dataset=eval_dataset,
    )
    baseline.print()
    del baseline_evaluator

    # --- 3. Build training data ---
    print("\n--- Building Training Data ---")
    if negatives == "hard":
        mining_model = khoji.EmbeddingModel(model_name)
        triplets = khoji.mine_hard_negatives(
            train_dataset, mining_model,
            n_negatives=n_negatives, top_k=50,
        )
        del mining_model
    elif negatives == "mixed":
        mining_model = khoji.EmbeddingModel(model_name)
        triplets = khoji.build_mixed_negatives(
            train_dataset, mining_model,
            n_random=n_negatives, n_hard=1, top_k=50,
        )
        del mining_model
    else:
        triplets = khoji.build_random_negatives(
            train_dataset, n_negatives=n_negatives,
        )
    print(f"  Total triplets: {len(triplets)}")

    # --- 4. Train ---
    print("\n--- Training ---")
    lora = khoji.LoRASettings(r=lora_r, alpha=lora_alpha, dropout=0.1)
    config = khoji.TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=1,
        lr=lr,
        weight_decay=0.01,
        warmup_steps=50,
        max_grad_norm=1.0,
        max_length=512,
        loss_fn=partial(infonce_loss, temperature=0.05),
        lora=lora,
        save_dir=f"{output_dir}/adapter",
        sanity_check_samples=10,
    )
    trainer = khoji.Trainer(model_name, config)
    history = trainer.train(khoji.TripletDataset(triplets))

    # --- 5. Fine-tuned evaluation ---
    print("\n--- Fine-tuned Evaluation ---")
    ft_evaluator = khoji.Evaluator(
        model_name, adapter_path=f"{output_dir}/adapter",
    )
    finetuned = ft_evaluator.evaluate(
        dataset_name=dataset_name,
        k_values=[1, 5, 10],
        dataset=eval_dataset,
    )
    finetuned.print()
    del ft_evaluator

    # --- 6. Comparison ---
    print(f"\n{'=' * 60}")
    print(f"COMPARISON — {name}")
    print(f"{'=' * 60}")
    for metric in baseline.metrics:
        b = baseline.metrics[metric]
        f = finetuned.metrics[metric]
        delta = f - b
        arrow = "+" if delta >= 0 else ""
        marker = " ***" if delta > 0.005 else ""
        print(f"  {metric:<12} {b:.4f} → {f:.4f}  ({arrow}{delta:.4f}){marker}")

    # Save results
    baseline.save(f"{output_dir}/baseline.json")
    finetuned.save(f"{output_dir}/finetuned.json")
    history.save(f"{output_dir}/train_history.json")
    print(f"\nResults saved to {output_dir}/")


def experiment_minilm_fiqa():
    """MiniLM-L6 on FiQA — a weaker model with clear room to improve."""
    run_experiment(
        name="MiniLM-L6 on FiQA (weak model, financial QA)",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dataset_name="fiqa",
        train_split="train",
        eval_split="test",
        n_negatives=3,
        negatives="mixed",
        epochs=3,
        lr=2e-5,
        lora_r=16,
        lora_alpha=32,
        batch_size=32,
        output_dir="./forge-output/demo-minilm-fiqa",
    )


def experiment_bge_scifact():
    """BGE on SciFact — scientific claim verification, a domain BGE wasn't optimized for."""
    run_experiment(
        name="BGE-base on SciFact (scientific claim verification)",
        model_name="BAAI/bge-base-en-v1.5",
        dataset_name="scifact",
        train_split="train",
        eval_split="test",
        n_negatives=3,
        negatives="mixed",
        epochs=5,
        lr=2e-5,
        lora_r=16,
        lora_alpha=32,
        batch_size=16,
        output_dir="./forge-output/demo-bge-scifact",
    )


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"

    if which in ("minilm", "both"):
        experiment_minilm_fiqa()
    if which in ("scifact", "both"):
        experiment_bge_scifact()

    if which == "both":
        print(f"\n{'#' * 70}")
        print("# Both experiments complete. Check forge-output/ for results.")
        print(f"{'#' * 70}")
