"""Text-to-image retrieval: Fine-tune CLIP on RSICD satellite imagery.

Demonstrates the full khoji Python API for multimodal retrieval:
- Load RSICD dataset (satellite/aerial images with captions)
- Baseline evaluation
- Build multimodal triplets (random, hard, mixed)
- Fine-tune CLIP with LoRA (both text + vision encoders)
- Evaluate improvement
- Inference: query satellite images with text

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
OUTPUT_DIR = "./output/multimodal-retrieval"


def main():
    print("=" * 60)
    print("Text-to-Image Retrieval: CLIP on RSICD")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────
    train_ds = load_rsicd(split="train")
    test_ds = load_rsicd(split="test")
    print(f"Train: {len(train_ds.queries)} queries, {len(train_ds.corpus)} images")
    print(f"Test:  {len(test_ds.queries)} queries, {len(test_ds.corpus)} images")

    # ── 2. Baseline evaluation ────────────────────────────
    print("\n--- Baseline: CLIP ViT-B/32 ---")
    base_eval = khoji.MultimodalEvaluator(MODEL)
    base_result = base_eval.evaluate(
        dataset_name="rsicd", k_values=[1, 5, 10], dataset=test_ds,
    )
    base_result.print()
    del base_eval

    # ── 3. Build training triplets ────────────────────────
    # Mixed: 4 random + 1 hard negative per pair
    # skip_top=5: skip top 5 non-relevant (likely mislabeled in RSICD)
    mining_model = khoji.MultimodalEmbeddingModel(MODEL)
    triplets = build_mixed_negatives_multimodal(
        train_ds, mining_model,
        n_random=4, n_hard=1, top_k=50, skip_top=5,
    )
    del mining_model
    print(f"Total triplets: {len(triplets)}")

    # ── 4. Configure training ─────────────────────────────
    lora = khoji.LoRASettings(r=16, alpha=32, dropout=0.1)
    config = khoji.MultimodalTrainingConfig(
        epochs=3,
        batch_size=16,
        grad_accum_steps=2,
        lr=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        max_grad_norm=1.0,
        max_length=77,
        loss_fn=partial(infonce_loss, temperature=0.05),
        lora=lora,
        lora_target="both",
        save_dir=f"{OUTPUT_DIR}/adapter",
        sanity_check_samples=10,
        base_dir=train_ds.base_dir,
    )

    # ── 5. Train ──────────────────────────────────────────
    trainer = khoji.MultimodalTrainer(MODEL, config)
    history = trainer.train(khoji.MultimodalTripletDataset(triplets))
    history.save(f"{OUTPUT_DIR}/train_history.json")

    # ── 6. Evaluate fine-tuned model ──────────────────────
    print("\n--- Fine-tuned: CLIP + LoRA ---")
    ft_eval = khoji.MultimodalEvaluator(
        MODEL, adapter_path=f"{OUTPUT_DIR}/adapter",
    )
    ft_result = ft_eval.evaluate(
        dataset_name="rsicd", k_values=[1, 5, 10], dataset=test_ds,
    )
    ft_result.print()
    ft_result.save(f"{OUTPUT_DIR}/finetuned.json")
    del ft_eval

    # ── 7. 2-round mining via YAML pipeline ─────────────────
    # Alternative: use the run() pipeline with mining_rounds
    # This does everything automatically (mine → train → re-mine → train)
    print("\n--- 2-round mining via run() pipeline ---")
    from khoji.multimodal_config import MultimodalForgeConfig
    from khoji.multimodal_run import run_multimodal

    config_yaml = f"""
model:
  name: {MODEL}
  lora_target: both
data:
  dataset: arampacha/rsicd
  split: train
  negatives: mixed
  n_random: 4
  n_hard: 1
  top_k: 50
  skip_top: 5
  mining_rounds: 2
lora:
  r: 16
  alpha: 32
  dropout: 0.1
train:
  epochs: 2
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
  run_before: false
  run_after: true
output_dir: {OUTPUT_DIR}/2rounds
"""
    with open("/tmp/rsicd_2rounds.yaml", "w") as f:
        f.write(config_yaml)

    cfg = MultimodalForgeConfig.from_yaml("/tmp/rsicd_2rounds.yaml")
    result_2rounds = run_multimodal(cfg)

    # ── 8. Comparison ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Model':<30} {'nDCG@10':>10} {'MRR@10':>10} {'R@10':>10}")
    print("-" * 62)
    for name, r in [
        ("CLIP baseline", base_result),
        ("CLIP + LoRA (1 round)", ft_result),
        ("CLIP + LoRA (2 rounds)", result_2rounds.finetuned),
    ]:
        m = r.metrics
        print(
            f"{name:<30} {m['ndcg@10']:>10.4f} "
            f"{m['mrr@10']:>10.4f} {m['recall@10']:>10.4f}"
        )

    # ── 9. Inference ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("INFERENCE DEMO")
    print(f"{'=' * 60}")

    model = khoji.MultimodalEmbeddingModel(
        MODEL, adapter_path=f"{OUTPUT_DIR}/adapter",
    )

    queries = [
        "an airport with multiple runways",
        "a river flowing through dense forest",
        "residential area with houses",
    ]

    # Encode queries
    query_embs = model.encode_text(queries)

    # Encode a subset of test gallery
    corpus_ids = list(test_ds.corpus.keys())[:200]
    corpus_sources = [test_ds.corpus[cid] for cid in corpus_ids]
    image_embs = model.encode_image_sources(
        corpus_sources, base_dir=test_ds.base_dir,
    )

    scores = torch.mm(query_embs, image_embs.t())

    for i, query in enumerate(queries):
        top3 = torch.topk(scores[i], k=3).indices.tolist()
        print(f"\nQuery: '{query}'")
        for rank, idx in enumerate(top3):
            print(
                f"  {rank+1}. (score={scores[i][idx]:.4f}) "
                f"{corpus_sources[idx]}"
            )


if __name__ == "__main__":
    main()
