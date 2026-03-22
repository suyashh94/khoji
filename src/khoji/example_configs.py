"""Bundled example configs for `khoji init`."""

CONFIGS = {
    "minilm_scifact_full.yaml": """\
# Text-to-text retrieval: MiniLM on SciFact
# A smaller model (22M params) fine-tuned on scientific claim verification
# Usage: khoji minilm_scifact_full.yaml
model:
  name: sentence-transformers/all-MiniLM-L6-v2
  adapter_path: null
  dtype: null

data:
  dataset: scifact
  split: train
  negatives: mixed
  n_negatives: 3
  n_random: 2
  n_hard: 1
  n_queries: null
  corpus_size: null
  top_k: 50
  skip_top: 0
  mining_rounds: 1

lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: null

train:
  epochs: 5
  batch_size: 16
  grad_accum_steps: 1
  lr: 2e-5
  weight_decay: 0.01
  warmup_steps: 50
  max_grad_norm: 1.0
  max_length: 512
  loss: infonce
  margin: 0.2
  temperature: 0.05
  mixed_precision: null
  overfit_batches: null
  sanity_check_samples: 10
  save_every_n_steps: null
  keep_all_checkpoints: false

seed: 42

eval:
  dataset: null
  k_values: [1, 5, 10]
  split: test
  n_queries: null
  corpus_size: null
  run_before: true
  run_after: true

output_dir: ./forge-output/minilm-scifact-full
""",
    "minilm_scifact_overfit.yaml": """\
# Text-to-text overfit debug: MiniLM on SciFact
# Verify the training pipeline works — loss should drop to ~0
# Usage: khoji minilm_scifact_overfit.yaml
model:
  name: sentence-transformers/all-MiniLM-L6-v2
  adapter_path: null
  dtype: null

data:
  dataset: scifact
  split: train
  negatives: random
  n_negatives: 1
  n_random: 1
  n_hard: 1
  n_queries: 5
  corpus_size: null
  top_k: 50
  skip_top: 0
  mining_rounds: 1

lora:
  r: 8
  alpha: 16
  dropout: 0.0
  target_modules: null

train:
  epochs: 50
  batch_size: 4
  grad_accum_steps: 1
  lr: 1e-3
  weight_decay: 0.0
  warmup_steps: 0
  max_grad_norm: 1.0
  max_length: 512
  loss: triplet
  margin: 0.2
  temperature: 0.05
  mixed_precision: null
  overfit_batches: 1
  sanity_check_samples: 10
  save_every_n_steps: null
  keep_all_checkpoints: false

seed: 42

eval:
  dataset: null
  k_values: [1, 5, 10]
  split: test
  n_queries: null
  corpus_size: null
  run_before: false
  run_after: false

output_dir: ./forge-output/minilm-scifact-overfit
""",
    # ── Multimodal (text-to-image) configs ──────────────────────
    "clip_rsicd_full.yaml": """\
# Text-to-image retrieval: CLIP ViT-B/32 on RSICD (satellite imagery)
# CLIP wasn't trained on satellite images — fine-tuning shows clear gains
# Usage: khoji multimodal clip_rsicd_full.yaml
model:
  name: openai/clip-vit-base-patch32
  adapter_path: null
  dtype: null
  lora_target: both

data:
  dataset: arampacha/rsicd
  split: train
  negatives: mixed
  n_negatives: 3
  n_random: 2
  n_hard: 1
  n_queries: null
  corpus_size: null
  top_k: 50
  skip_top: 0
  mining_rounds: 1
  cache_dir: null

lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: null

train:
  epochs: 3
  batch_size: 16
  grad_accum_steps: 2
  lr: 2e-5
  weight_decay: 0.01
  warmup_steps: 100
  max_grad_norm: 1.0
  max_length: 77
  loss: infonce
  margin: 0.2
  temperature: 0.05
  mixed_precision: null
  overfit_batches: null
  sanity_check_samples: 10
  save_every_n_steps: null
  keep_all_checkpoints: false

seed: 42

eval:
  dataset: null
  k_values: [1, 5, 10]
  split: test
  n_queries: null
  corpus_size: null
  run_before: true
  run_after: true

output_dir: ./forge-output/clip-rsicd-full
""",
    "clip_rsicd_overfit.yaml": """\
# Text-to-image overfit debug: CLIP ViT-B/32 on RSICD
# Verify the multimodal training pipeline works
# Usage: khoji multimodal clip_rsicd_overfit.yaml
model:
  name: openai/clip-vit-base-patch32
  adapter_path: null
  dtype: null
  lora_target: both

data:
  dataset: arampacha/rsicd
  split: train
  negatives: random
  n_negatives: 1
  n_random: 1
  n_hard: 1
  n_queries: 5
  corpus_size: null
  top_k: 50
  skip_top: 0
  mining_rounds: 1
  cache_dir: null

lora:
  r: 8
  alpha: 16
  dropout: 0.0
  target_modules: null

train:
  epochs: 50
  batch_size: 4
  grad_accum_steps: 1
  lr: 1e-3
  weight_decay: 0.0
  warmup_steps: 0
  max_grad_norm: 1.0
  max_length: 77
  loss: triplet
  margin: 0.2
  temperature: 0.05
  mixed_precision: null
  overfit_batches: 1
  sanity_check_samples: 10
  save_every_n_steps: null
  keep_all_checkpoints: false

seed: 42

eval:
  dataset: null
  k_values: [1, 5, 10]
  split: test
  n_queries: null
  corpus_size: null
  run_before: false
  run_after: false

output_dir: ./forge-output/clip-rsicd-overfit
""",
}
