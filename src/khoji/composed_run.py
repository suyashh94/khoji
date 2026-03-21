"""Orchestration for composed (image+text → image) training runs."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from khoji.composed_config import ComposedForgeConfig
from khoji.composed_data import (
    ComposedTripletDataset,
    build_mixed_negatives_composed,
    build_random_negatives_composed,
    mine_hard_negatives_composed,
)
from khoji.composed_dataset import ComposedRetrievalDataset, load_custom_composed
from khoji.composed_evaluator import ComposedEvaluator
from khoji.composed_trainer import ComposedTrainer, ComposedTrainingConfig
from khoji.lora import LoRASettings
from khoji.loss import contrastive_loss, infonce_loss, triplet_margin_loss
from khoji.multimodal_model import JointEmbeddingModel
from khoji.run import RunResult, _set_seed
from khoji.trainer import TrainHistory


def _resolve_loss(config: ComposedForgeConfig):
    """Map loss name string to the actual loss function."""
    if config.train.loss == "triplet":
        return partial(triplet_margin_loss, margin=config.train.margin)
    elif config.train.loss == "infonce":
        return partial(infonce_loss, temperature=config.train.temperature)
    elif config.train.loss == "contrastive":
        return contrastive_loss
    else:
        raise ValueError(
            f"Unknown loss: {config.train.loss}. "
            "Use 'triplet', 'infonce', or 'contrastive'."
        )


def _build_triplets_composed(
    config: ComposedForgeConfig,
    dataset: ComposedRetrievalDataset,
    adapter_path: str | None,
) -> list:
    """Build composed triplets using the configured negative strategy."""
    if config.data.negatives in ("hard", "mixed"):
        mining_model = JointEmbeddingModel(
            config.model.name,
            adapter_path=adapter_path,
            max_length=config.train.max_length,
            dtype=config.model.dtype,
        )

    if config.data.negatives == "hard":
        triplets = mine_hard_negatives_composed(
            dataset,
            mining_model,
            n_negatives=config.data.n_negatives,
            top_k=config.data.top_k,
            skip_top=config.data.skip_top,
            n_queries=config.data.n_queries,
            corpus_size=config.data.corpus_size,
            cache_dir=config.data.cache_dir,
        )
    elif config.data.negatives == "mixed":
        triplets = build_mixed_negatives_composed(
            dataset,
            mining_model,
            n_random=config.data.n_random,
            n_hard=config.data.n_hard,
            top_k=config.data.top_k,
            skip_top=config.data.skip_top,
            n_queries=config.data.n_queries,
            corpus_size=config.data.corpus_size,
            cache_dir=config.data.cache_dir,
        )
    else:
        return build_random_negatives_composed(
            dataset,
            n_negatives=config.data.n_negatives,
            n_queries=config.data.n_queries,
        )

    del mining_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return triplets


def run_composed(config: ComposedForgeConfig) -> RunResult:
    """Execute a full composed retrieval training run from config.

    Args:
        config: ComposedForgeConfig with all settings.

    Returns:
        RunResult with training history, eval results, and adapter path.
    """
    if config.seed is not None:
        _set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(str(output_dir / "config.yaml"))

    result = RunResult(history=TrainHistory(), config=None)

    # --- Dataset loading ---
    def _load(source: str) -> ComposedRetrievalDataset:
        return load_custom_composed(source)

    eval_source = config.eval.dataset or config.data.dataset

    # --- Baseline evaluation ---
    if config.eval.run_before:
        print("\n" + "=" * 60)
        print("BASELINE EVALUATION")
        print("=" * 60)
        eval_dataset = _load(eval_source)
        evaluator = ComposedEvaluator(
            config.model.name,
            max_length=config.train.max_length,
            dtype=config.model.dtype,
        )
        baseline = evaluator.evaluate(
            dataset_name=eval_source,
            k_values=config.eval.k_values,
            n_queries=config.eval.n_queries,
            corpus_size=config.eval.corpus_size,
            dataset=eval_dataset,
            cache_dir=config.data.cache_dir,
        )
        baseline.print()
        baseline.save(str(output_dir / "baseline.json"))
        result.baseline = baseline

    # --- Data preparation ---
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    dataset = _load(config.data.dataset)

    lora_settings = None
    if config.lora is not None:
        lora_settings = LoRASettings(
            r=config.lora.r,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
        )

    # Mining rounds
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

        if rounds > 1:
            print(f"\n{'=' * 60}")
            print(f"NEGATIVE MINING  {round_label}")
            print("=" * 60)

        triplets = _build_triplets_composed(config, dataset, current_adapter)
        torch_ds = ComposedTripletDataset(triplets)

        # --- Training ---
        print("\n" + "=" * 60)
        print(f"TRAINING  {round_label}".rstrip())
        print("=" * 60)

        is_last_round = round_idx == rounds - 1
        if rounds > 1 and not is_last_round:
            adapter_dir = str(output_dir / f"adapter_r{round_idx + 1}")
        else:
            adapter_dir = final_adapter_dir

        # Halve LR for each subsequent round
        round_lr = config.train.lr / (2 ** round_idx)
        if rounds > 1 and round_idx > 0:
            print(f"LR decay: {config.train.lr} → {round_lr} (round {round_idx + 1})")

        training_config = ComposedTrainingConfig(
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
            cache_dir=config.data.cache_dir,
            base_dir=dataset.base_dir,
        )

        # For round 2+ LoRA, warm-start from previous adapter
        warm_start = (
            current_adapter
            if (round_idx > 0 and config.lora is not None)
            else None
        )
        trainer = ComposedTrainer(
            config.model.name,
            training_config,
            adapter_path=warm_start,
        )
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
        eval_dataset = _load(eval_source)
        finetuned_evaluator = ComposedEvaluator(
            config.model.name,
            adapter_path=final_adapter_dir,
            max_length=config.train.max_length,
            dtype=config.model.dtype,
        )
        finetuned = finetuned_evaluator.evaluate(
            dataset_name=eval_source,
            k_values=config.eval.k_values,
            n_queries=config.eval.n_queries,
            corpus_size=config.eval.corpus_size,
            dataset=eval_dataset,
            cache_dir=config.data.cache_dir,
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
