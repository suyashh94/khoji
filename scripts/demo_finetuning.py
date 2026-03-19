"""Demonstrate that khoji fine-tuning improves retrieval performance.

Two experiments:
  1. Weak model (MiniLM-L6) on FiQA   — smaller model with room to improve
  2. Strong model (BGE-base) on SciFact — scientific domain, 809 train queries

Usage:
    python scripts/demo_finetuning.py                          # run both with defaults
    python scripts/demo_finetuning.py minilm                   # MiniLM on FiQA
    python scripts/demo_finetuning.py scifact                  # BGE on SciFact
    python scripts/demo_finetuning.py minilm --n-random 3 --n-hard 2   # custom mix
    python scripts/demo_finetuning.py scifact --n-random 0 --n-hard 3  # hard only
    python scripts/demo_finetuning.py scifact --n-random 3 --n-hard 0  # random only
"""

from __future__ import annotations

import argparse
from functools import partial

import khoji
from khoji.loss import infonce_loss


def run_experiment(
    name: str,
    model_name: str,
    dataset_name: str,
    train_split: str,
    eval_split: str,
    n_random: int,
    n_hard: int,
    epochs: int,
    lr: float,
    lora_r: int,
    lora_alpha: int,
    batch_size: int,
    output_dir: str,
) -> None:
    """Run a single fine-tuning experiment and print comparison."""

    neg_desc = f"{n_random} random + {n_hard} hard"
    if n_hard == 0:
        neg_desc = f"{n_random} random"
    elif n_random == 0:
        neg_desc = f"{n_hard} hard"

    print(f"\n{'#' * 70}")
    print(f"# {name}")
    print(f"# Model: {model_name}")
    print(f"# Dataset: {dataset_name} (train={train_split}, eval={eval_split})")
    print(f"# Negatives: {neg_desc}")
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
    if n_random > 0 and n_hard > 0:
        # Mixed: both random and hard negatives
        mining_model = khoji.EmbeddingModel(model_name)
        triplets = khoji.build_mixed_negatives(
            train_dataset, mining_model,
            n_random=n_random, n_hard=n_hard, top_k=50,
        )
        del mining_model
    elif n_hard > 0:
        # Hard only
        mining_model = khoji.EmbeddingModel(model_name)
        triplets = khoji.mine_hard_negatives(
            train_dataset, mining_model,
            n_negatives=n_hard, top_k=50,
        )
        del mining_model
    else:
        # Random only
        triplets = khoji.build_random_negatives(
            train_dataset, n_negatives=n_random,
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


EXPERIMENTS = {
    "minilm": {
        "name": "MiniLM-L6 on FiQA (weak model, financial QA)",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dataset_name": "fiqa",
        "train_split": "train",
        "eval_split": "test",
        "epochs": 3,
        "lr": 2e-5,
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 32,
        "output_dir": "./forge-output/demo-minilm-fiqa",
    },
    "scifact": {
        "name": "BGE-base on SciFact (scientific claim verification)",
        "model_name": "BAAI/bge-base-en-v1.5",
        "dataset_name": "scifact",
        "train_split": "train",
        "eval_split": "test",
        "epochs": 5,
        "lr": 2e-5,
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 16,
        "output_dir": "./forge-output/demo-bge-scifact",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Demo: khoji fine-tuning for retrieval")
    parser.add_argument(
        "experiment", nargs="?", default="both",
        choices=["minilm", "scifact", "both"],
        help="Which experiment to run (default: both)",
    )
    parser.add_argument(
        "--n-random", type=int, default=2,
        help="Random negatives per pair (default: 2)",
    )
    parser.add_argument(
        "--n-hard", type=int, default=1,
        help="Hard negatives per pair (default: 1)",
    )
    args = parser.parse_args()

    if args.n_random == 0 and args.n_hard == 0:
        parser.error("Need at least one of --n-random or --n-hard > 0")

    if args.experiment == "both":
        experiments_to_run = list(EXPERIMENTS.keys())
    else:
        experiments_to_run = [args.experiment]

    for exp_name in experiments_to_run:
        exp = EXPERIMENTS[exp_name]
        run_experiment(
            n_random=args.n_random,
            n_hard=args.n_hard,
            **exp,
        )

    if len(experiments_to_run) > 1:
        print(f"\n{'#' * 70}")
        print("# Both experiments complete. Check forge-output/ for results.")
        print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
