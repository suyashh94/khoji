"""Training script: load config, prepare data, train, evaluate."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

from khoji.config import ForgeConfig
from khoji.data import (
    TripletDataset,
    build_mixed_negatives,
    build_random_negatives,
    mine_hard_negatives,
)
from khoji.dataset import RetrievalDataset, load_beir, load_custom
from khoji.evaluator import EvalResult, Evaluator
from khoji.lora import LoRASettings
from khoji.loss import contrastive_loss, infonce_loss, triplet_margin_loss
from khoji.model import EmbeddingModel
from khoji.trainer import Trainer, TrainHistory, TrainingConfig


def _resolve_loss(config: ForgeConfig):
    """Map loss name string to the actual loss function."""
    if config.train.loss == "triplet":
        return partial(triplet_margin_loss, margin=config.train.margin)
    elif config.train.loss == "infonce":
        return partial(infonce_loss, temperature=config.train.temperature)
    elif config.train.loss == "contrastive":
        return contrastive_loss
    else:
        raise ValueError(f"Unknown loss: {config.train.loss}. Use 'triplet', 'infonce', or 'contrastive'.")


def _build_triplets(
    config: ForgeConfig,
    dataset: "RetrievalDataset",
    adapter_path: str | None,
) -> list:
    """Build training triplets using the configured negative strategy.

    For hard/mixed modes, loads the mining model with the adapter (if any)
    to mine negatives, then frees GPU memory.
    """
    if config.data.negatives in ("hard", "mixed"):
        # Load mining model — handle LoRA vs full fine-tuning
        if adapter_path is not None and config.lora is None:
            # Full fine-tuning: saved model IS the model
            mining_model = EmbeddingModel(
                adapter_path,
                max_length=config.train.max_length,
                dtype=config.model.dtype,
            )
        else:
            # LoRA or no adapter: base model + optional adapter
            mining_model = EmbeddingModel(
                config.model.name,
                adapter_path=adapter_path,
                max_length=config.train.max_length,
                dtype=config.model.dtype,
            )

    if config.data.negatives == "hard":
        triplets = mine_hard_negatives(
            dataset,
            mining_model,
            n_negatives=config.data.n_negatives,
            top_k=config.data.top_k,
            n_queries=config.data.n_queries,
            corpus_size=config.data.corpus_size,
        )
    elif config.data.negatives == "mixed":
        triplets = build_mixed_negatives(
            dataset,
            mining_model,
            n_random=config.data.n_random,
            n_hard=config.data.n_hard,
            top_k=config.data.top_k,
            n_queries=config.data.n_queries,
            corpus_size=config.data.corpus_size,
        )
    else:
        return build_random_negatives(
            dataset,
            n_negatives=config.data.n_negatives,
            n_queries=config.data.n_queries,
        )

    # Free mining model GPU memory
    del mining_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return triplets


def _set_seed(seed: int) -> None:
    """Set global random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


@dataclass
class RunResult:
    """Everything returned from a training run.

    Attributes:
        history: Training metrics (step_loss, step_lr, step_grad_norm, epoch_loss).
        baseline: Baseline eval result (None if eval was disabled).
        finetuned: Fine-tuned eval result (None if eval was disabled).
        adapter_dir: Path to saved LoRA adapter.
        config: The config used for this run.
    """

    history: TrainHistory
    baseline: EvalResult | None = None
    finetuned: EvalResult | None = None
    adapter_dir: str | None = None
    config: ForgeConfig | None = None


def run(config: ForgeConfig) -> RunResult:
    """Execute a full training run from config.

    Args:
        config: ForgeConfig with all settings.

    Returns:
        RunResult with training history, eval results, and adapter path.
    """
    # Set seed early, before any data loading
    if config.seed is not None:
        _set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    config.to_yaml(str(output_dir / "config.yaml"))

    result = RunResult(history=TrainHistory(), config=config)

    # --- Dataset loading helpers ---
    def _load(source: str, split: str) -> RetrievalDataset:
        if Path(source).is_dir():
            return load_custom(source)
        return load_beir(source, split=split)

    # Eval dataset can differ from training dataset
    eval_source = config.eval.dataset or config.data.dataset

    # --- Baseline evaluation ---
    if config.eval.run_before:
        print("\n" + "=" * 60)
        print("BASELINE EVALUATION")
        print("=" * 60)
        eval_dataset = _load(eval_source, config.eval.split)
        evaluator = Evaluator(config.model.name, max_length=config.train.max_length, dtype=config.model.dtype)
        baseline = evaluator.evaluate(
            dataset_name=eval_source,
            k_values=config.eval.k_values,
            n_queries=config.eval.n_queries,
            corpus_size=config.eval.corpus_size,
            dataset=eval_dataset,
        )
        baseline.print()
        baseline.save(str(output_dir / "baseline.json"))
        result.baseline = baseline

    # --- Data preparation ---
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    dataset = _load(config.data.dataset, config.data.split)

    if config.data.negatives not in ("random", "hard", "mixed"):
        raise ValueError(
            f"Unknown negatives mode: {config.data.negatives!r}. "
            "Use 'random', 'hard', or 'mixed'."
        )

    if config.data.negatives == "mixed":
        if config.data.n_random == 0 and config.data.n_hard == 0:
            raise ValueError("negatives: mixed requires n_random > 0 or n_hard > 0.")
    elif config.data.negatives in ("random", "hard"):
        if config.data.n_random != 1 or config.data.n_hard != 1:
            print(
                f"Note: n_random/n_hard are ignored when negatives "
                f"is '{config.data.negatives}'. "
                f"Using n_negatives={config.data.n_negatives} instead."
            )

    lora_settings = None
    if config.lora is not None:
        lora_settings = LoRASettings(
            r=config.lora.r,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
        )

    # Mining rounds: only for hard/mixed. Random negatives don't benefit from re-mining.
    uses_mining = config.data.negatives in ("hard", "mixed")
    rounds = config.data.mining_rounds if uses_mining else 1
    if config.data.mining_rounds > 1 and not uses_mining:
        print(
            "Note: mining_rounds > 1 has no effect with random negatives. "
            "Using 1 round."
        )

    current_adapter: str | None = config.model.adapter_path
    final_adapter_dir = str(output_dir / "adapter")

    for round_idx in range(rounds):
        round_label = f"Round {round_idx + 1}/{rounds}" if rounds > 1 else ""

        # --- Mine / build negatives ---
        if rounds > 1:
            print(f"\n{'=' * 60}")
            print(f"NEGATIVE MINING  {round_label}")
            print("=" * 60)

        triplets = _build_triplets(config, dataset, current_adapter)
        torch_ds = TripletDataset(triplets)

        # --- Training ---
        print("\n" + "=" * 60)
        print(f"TRAINING  {round_label}".rstrip())
        print("=" * 60)

        # Intermediate rounds save to adapter_r1/, etc. Final round saves to adapter/
        is_last_round = round_idx == rounds - 1
        if rounds > 1 and not is_last_round:
            adapter_dir = str(output_dir / f"adapter_r{round_idx + 1}")
        else:
            adapter_dir = final_adapter_dir

        # Halve LR for each subsequent round to avoid overshooting
        round_lr = config.train.lr / (2 ** round_idx)
        if rounds > 1 and round_idx > 0:
            print(f"LR decay: {config.train.lr} → {round_lr} (round {round_idx + 1})")

        training_config = TrainingConfig(
            epochs=config.train.epochs,
            batch_size=config.train.batch_size,
            grad_accum_steps=config.train.grad_accum_steps,
            lr=round_lr,
            weight_decay=config.train.weight_decay,
            warmup_steps=config.train.warmup_steps,
            max_grad_norm=config.train.max_grad_norm,
            max_length=config.train.max_length,
            mixed_precision=config.train.mixed_precision,
            loss_fn=_resolve_loss(config),
            lora=lora_settings,
            save_dir=adapter_dir,
            overfit_batches=config.train.overfit_batches,
            sanity_check_samples=config.train.sanity_check_samples,
            save_every_n_steps=config.train.save_every_n_steps,
            keep_all_checkpoints=config.train.keep_all_checkpoints,
            dtype=config.model.dtype,
        )

        # For round 2+ full fine-tuning, load from previous round's saved model
        if round_idx > 0 and config.lora is None:
            model_name = current_adapter  # type: ignore[assignment]
        else:
            model_name = config.model.name

        # For round 2+ LoRA, warm-start from previous adapter
        warm_start = (
            current_adapter
            if (round_idx > 0 and config.lora is not None)
            else None
        )
        trainer = Trainer(model_name, training_config, adapter_path=warm_start)
        round_history = trainer.train(torch_ds)

        # Accumulate history across rounds
        result.history.step_loss.extend(round_history.step_loss)
        result.history.step_lr.extend(round_history.step_lr)
        result.history.step_grad_norm.extend(round_history.step_grad_norm)
        result.history.epoch_loss.extend(round_history.epoch_loss)

        current_adapter = adapter_dir

    result.adapter_dir = final_adapter_dir

    # Save training history
    result.history.save(str(output_dir / "train_history.json"))

    # --- Post-training evaluation ---
    if config.eval.run_after:
        print("\n" + "=" * 60)
        print("FINE-TUNED EVALUATION")
        print("=" * 60)
        eval_dataset = _load(eval_source, config.eval.split)
        if config.lora is not None:
            # LoRA: load base model + adapter
            finetuned_evaluator = Evaluator(
                config.model.name,
                adapter_path=adapter_dir,
                max_length=config.train.max_length,
                dtype=config.model.dtype,
            )
        else:
            # Full fine-tuning: load the saved model directly
            finetuned_evaluator = Evaluator(
                adapter_dir,
                max_length=config.train.max_length,
                dtype=config.model.dtype,
            )
        finetuned = finetuned_evaluator.evaluate(
            dataset_name=eval_source,
            k_values=config.eval.k_values,
            n_queries=config.eval.n_queries,
            corpus_size=config.eval.corpus_size,
            dataset=eval_dataset,
        )
        finetuned.print()
        finetuned.save(str(output_dir / "finetuned.json"))
        result.finetuned = finetuned

    # --- Summary ---
    if result.baseline and result.finetuned:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        baseline_m = result.baseline.metrics
        finetuned_m = result.finetuned.metrics
        for metric in baseline_m:
            b = baseline_m[metric]
            f = finetuned_m[metric]
            delta = f - b
            arrow = "+" if delta >= 0 else ""
            print(f"  {metric:<12} {b:.4f} → {f:.4f}  ({arrow}{delta:.4f})")

    print(f"\nResults saved to {output_dir}")
    return result


def _init_configs(target_dir: str = ".") -> None:
    """Generate example config files in the target directory."""
    from khoji.example_configs import CONFIGS

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    for name, content in CONFIGS.items():
        path = target / name
        path.write_text(content)
        print(f"  Created {path}")

    print(f"\nGenerated {len(CONFIGS)} example configs in {target}/")
    print("Run:  khoji fiqa_quick.yaml")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  khoji <config.yaml>        Run a training pipeline")
        print("  khoji init [directory]      Generate example config files")
        sys.exit(1)

    if sys.argv[1] == "init":
        target = sys.argv[2] if len(sys.argv) > 2 else "."
        _init_configs(target)
        return

    config = ForgeConfig.from_yaml(sys.argv[1])
    run(config)


if __name__ == "__main__":
    main()
