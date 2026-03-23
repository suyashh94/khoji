"""Text-to-image retrieval: Fine-tune CLIP on RSICD satellite imagery.

Two approaches shown:
  1. Config approach — construct MultimodalForgeConfig with mining_rounds
  2. API approach — manual pipeline with full control

Usage:
    python scripts/train_multimodal_retrieval.py
"""

from __future__ import annotations

from functools import partial

import torch

import khoji
from khoji.loss import infonce_loss
from khoji.multimodal_data import build_mixed_negatives_multimodal
from khoji.multimodal_dataset import load_rsicd

MODEL = "openai/clip-vit-base-patch32"
OUTPUT = "./output/multimodal-retrieval"


def approach_config():
    """Approach 1: MultimodalForgeConfig with mining_rounds."""
    print("\n" + "=" * 60)
    print("APPROACH 1: Config-driven pipeline")
    print("=" * 60)

    from khoji.multimodal_config import MultimodalForgeConfig
    from khoji.multimodal_run import run_multimodal

    config_yaml = f"""\
model:
  name: {MODEL}
  lora_target: both
data:
  dataset: arampacha/rsicd
  split: train
  negatives: mixed
  n_random: 2
  n_hard: 1
  top_k: 50
  skip_top: 5
  mining_rounds: 2
lora:
  r: 16
  alpha: 32
  dropout: 0.1
train:
  epochs: 3
  batch_size: 16
  grad_accum_steps: 2
  lr: 2e-5
  warmup_steps: 50
  max_length: 77
  loss: infonce
  temperature: 0.05
  sanity_check_samples: 5
seed: 42
eval:
  k_values: [1, 5, 10]
  split: test
  run_before: true
  run_after: true
output_dir: {OUTPUT}/config-approach
"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_path = f.name

    config = MultimodalForgeConfig.from_yaml(config_path)
    result = run_multimodal(config)

    print(f"\nEpoch losses across {len(result.history.epoch_loss)} epochs:")
    print([f"{l:.4f}" for l in result.history.epoch_loss])
    return result


def approach_api():
    """Approach 2: Manual API with full control."""
    print("\n" + "=" * 60)
    print("APPROACH 2: Manual Python API")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────
    train_ds = load_rsicd(split="train")
    test_ds = load_rsicd(split="test")
    print(f"Train: {len(train_ds.queries)} queries, {len(train_ds.corpus)} images")
    print(f"Test:  {len(test_ds.queries)} queries, {len(test_ds.corpus)} images")

    # ── Baseline ──────────────────────────────────────────
    print("\n--- CLIP baseline ---")
    base_eval = khoji.MultimodalEvaluator(MODEL)
    base_result = base_eval.evaluate(
        dataset_name="rsicd", k_values=[1, 5, 10], dataset=test_ds,
    )
    base_result.print()
    del base_eval

    # ── Round 1: mine + train ─────────────────────────────
    lora = khoji.LoRASettings(r=16, alpha=32, dropout=0.1)
    loss_fn = partial(infonce_loss, temperature=0.05)

    print("\n--- Round 1: mine with base CLIP ---")
    mining_model = khoji.MultimodalEmbeddingModel(MODEL)
    triplets = build_mixed_negatives_multimodal(
        train_ds, mining_model,
        n_random=2, n_hard=1, top_k=50, skip_top=5,
    )
    del mining_model

    config_r1 = khoji.MultimodalTrainingConfig(
        epochs=3, batch_size=16, grad_accum_steps=2,
        lr=2e-5, warmup_steps=50, max_length=77,
        loss_fn=loss_fn, lora=lora, lora_target="both",
        save_dir=f"{OUTPUT}/api-approach/adapter_r1",
        sanity_check_samples=5, base_dir=train_ds.base_dir,
    )
    trainer = khoji.MultimodalTrainer(MODEL, config_r1)
    trainer.train(khoji.MultimodalTripletDataset(triplets))

    # ── Round 2: re-mine with fine-tuned model ────────────
    print("\n--- Round 2: re-mine with fine-tuned CLIP ---")
    ft_mining = khoji.MultimodalEmbeddingModel(
        MODEL, adapter_path=f"{OUTPUT}/api-approach/adapter_r1",
    )
    triplets_r2 = build_mixed_negatives_multimodal(
        train_ds, ft_mining,
        n_random=2, n_hard=1, top_k=50, skip_top=5,
    )
    del ft_mining

    config_r2 = khoji.MultimodalTrainingConfig(
        epochs=3, batch_size=16, grad_accum_steps=2,
        lr=1e-5, warmup_steps=30, max_length=77,  # halved LR
        loss_fn=loss_fn, lora=lora, lora_target="both",
        save_dir=f"{OUTPUT}/api-approach/adapter_final",
        sanity_check_samples=5, base_dir=train_ds.base_dir,
    )
    trainer_r2 = khoji.MultimodalTrainer(
        MODEL, config_r2,
        adapter_path=f"{OUTPUT}/api-approach/adapter_r1",
    )
    trainer_r2.train(khoji.MultimodalTripletDataset(triplets_r2))

    # ── Evaluate ──────────────────────────────────────────
    print("\n--- Fine-tuned (2 rounds) ---")
    ft_eval = khoji.MultimodalEvaluator(
        MODEL, adapter_path=f"{OUTPUT}/api-approach/adapter_final",
    )
    ft_result = ft_eval.evaluate(
        dataset_name="rsicd", k_values=[1, 5, 10], dataset=test_ds,
    )
    ft_result.print()
    del ft_eval

    # ── Comparison ────────────────────────────────────────
    print(f"\n{'Model':<30} {'nDCG@10':>10} {'MRR@10':>10} {'R@10':>10}")
    print("-" * 62)
    for name, r in [
        ("CLIP baseline", base_result),
        ("CLIP + LoRA (2 rounds)", ft_result),
    ]:
        m = r.metrics
        print(
            f"{name:<30} {m['ndcg@10']:>10.4f} "
            f"{m['mrr@10']:>10.4f} {m['recall@10']:>10.4f}"
        )

    # ── Inference ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("INFERENCE: query satellite images with text")
    print(f"{'=' * 60}")

    model = khoji.MultimodalEmbeddingModel(
        MODEL, adapter_path=f"{OUTPUT}/api-approach/adapter_final",
    )

    queries = [
        "an airport with runways",
        "a river through forest",
        "residential houses",
    ]

    query_embs = model.encode_text(queries)
    corpus_sources = list(test_ds.corpus.values())[:200]
    image_embs = model.encode_image_sources(
        corpus_sources, base_dir=test_ds.base_dir,
    )

    scores = torch.mm(query_embs, image_embs.t())
    for i, query in enumerate(queries):
        top3 = torch.topk(scores[i], k=3).indices.tolist()
        print(f"\nQuery: '{query}'")
        for rank, idx in enumerate(top3):
            print(f"  {rank+1}. (score={scores[i][idx]:.4f}) {corpus_sources[idx]}")


if __name__ == "__main__":
    approach_config()
    approach_api()
