"""Orchestration for multimodal (text-to-image) training runs."""

from __future__ import annotations

import random
from functools import partial
from pathlib import Path

import torch

from khoji.loss import contrastive_loss, infonce_loss, triplet_margin_loss
from khoji.multimodal_config import MultimodalForgeConfig
from khoji.multimodal_data import (
    MultimodalTripletDataset,
    build_random_negatives_multimodal,
    mine_hard_negatives_multimodal,
)
from khoji.multimodal_dataset import (
    MultimodalRetrievalDataset,
    load_custom_multimodal,
    load_flickr30k,
    load_rsicd,
)
from khoji.multimodal_evaluator import MultimodalEvaluator
from khoji.multimodal_model import MultimodalEmbeddingModel
from khoji.multimodal_trainer import MultimodalTrainer, MultimodalTrainingConfig
from khoji.lora import LoRASettings
from khoji.run import RunResult, _set_seed
from khoji.trainer import TrainHistory


def _resolve_loss(config: MultimodalForgeConfig):
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


def run_multimodal(config: MultimodalForgeConfig) -> RunResult:
    """Execute a full multimodal training run from config.

    Args:
        config: MultimodalForgeConfig with all settings.

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
    def _load(source: str, split: str) -> MultimodalRetrievalDataset:
        if Path(source).is_dir():
            return load_custom_multimodal(source)
        if "rsicd" in source.lower():
            return load_rsicd(split=split, cache_dir=config.data.cache_dir)
        if "flickr" in source.lower():
            return load_flickr30k(split=split, cache_dir=config.data.cache_dir)
        # Default: try as a generic HF dataset name
        return load_flickr30k(split=split, cache_dir=config.data.cache_dir)

    eval_source = config.eval.dataset or config.data.dataset

    # --- Baseline evaluation ---
    if config.eval.run_before:
        print("\n" + "=" * 60)
        print("BASELINE EVALUATION")
        print("=" * 60)
        eval_dataset = _load(eval_source, config.eval.split)
        evaluator = MultimodalEvaluator(
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
    dataset = _load(config.data.dataset, config.data.split)

    if config.data.negatives == "hard":
        mining_model = MultimodalEmbeddingModel(
            config.model.name,
            max_length=config.train.max_length,
            dtype=config.model.dtype,
        )
        triplets = mine_hard_negatives_multimodal(
            dataset,
            mining_model,
            n_negatives=config.data.n_negatives,
            top_k=config.data.top_k,
            n_queries=config.data.n_queries,
            corpus_size=config.data.corpus_size,
            cache_dir=config.data.cache_dir,
        )
    else:
        triplets = build_random_negatives_multimodal(
            dataset,
            n_negatives=config.data.n_negatives,
            n_queries=config.data.n_queries,
        )

    torch_ds = MultimodalTripletDataset(triplets)

    # --- Training ---
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    lora_settings = None
    if config.lora is not None:
        lora_settings = LoRASettings(
            r=config.lora.r,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
        )

    adapter_dir = str(output_dir / "adapter")

    # Build preprocess overrides dict
    preprocess_overrides = None
    if config.preprocess is not None:
        preprocess_overrides = {
            "image_size": config.preprocess.image_size,
            "mean": config.preprocess.mean,
            "std": config.preprocess.std,
        }

    training_config = MultimodalTrainingConfig(
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        grad_accum_steps=config.train.grad_accum_steps,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        warmup_steps=config.train.warmup_steps,
        max_grad_norm=config.train.max_grad_norm,
        max_length=config.train.max_length,
        mixed_precision=config.train.mixed_precision,
        loss_fn=_resolve_loss(config),
        lora=lora_settings,
        lora_target=config.model.lora_target,
        save_dir=adapter_dir,
        overfit_batches=config.train.overfit_batches,
        sanity_check_samples=config.train.sanity_check_samples,
        save_every_n_steps=config.train.save_every_n_steps,
        keep_all_checkpoints=config.train.keep_all_checkpoints,
        dtype=config.model.dtype,
        cache_dir=config.data.cache_dir,
        base_dir=dataset.base_dir,
    )

    trainer = MultimodalTrainer(
        config.model.name,
        training_config,
        preprocess_overrides=preprocess_overrides,
    )
    result.history = trainer.train(torch_ds)
    result.adapter_dir = adapter_dir

    # Save training history
    result.history.save(str(output_dir / "train_history.json"))

    # --- Post-training evaluation ---
    if config.eval.run_after:
        print("\n" + "=" * 60)
        print("FINE-TUNED EVALUATION")
        print("=" * 60)
        eval_dataset = _load(eval_source, config.eval.split)
        finetuned_evaluator = MultimodalEvaluator(
            config.model.name,
            adapter_path=adapter_dir,
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
