"""Bundled example configs for `khoji init`."""

CONFIGS = {
    "fiqa_quick.yaml": """\
# Quick test config — small subset for fast iteration
model:
  name: BAAI/bge-base-en-v1.5
  adapter_path: null       # path to existing adapter (for continued training)
  dtype: null              # "fp16", "bf16", or null (fp32). Load base model in this precision

data:
  dataset: fiqa
  split: train
  negatives: random        # "random" or "hard"
  n_negatives: 1
  n_queries: 50            # small subset for testing. null = all queries
  corpus_size: null        # only used with hard negatives. null = full corpus
  top_k: 50                # top-k for hard negative mining

lora:
  r: 8
  alpha: 16
  dropout: 0.1
  target_modules: null     # auto-detect based on model architecture
  # override example: target_modules: [query, key, value]

train:
  epochs: 2
  batch_size: 4
  grad_accum_steps: 4      # effective batch size = 4 * 4 = 16
  lr: 2e-5
  weight_decay: 0.01       # AdamW weight decay
  warmup_steps: 10
  max_grad_norm: 1.0       # gradient clipping. null = disabled
  max_length: 512
  loss: triplet            # "triplet", "infonce", or "contrastive"
  margin: 0.2              # only for triplet loss
  temperature: 0.05        # only for infonce loss
  mixed_precision: null    # "fp16", "bf16", or null (disabled)
  overfit_batches: null    # set to 1 to overfit on 1 batch for debugging
  sanity_check_samples: 10 # check N training samples before/after training
  save_every_n_steps: null # save checkpoint every N optimizer steps. null = disabled
  keep_all_checkpoints: false  # true = keep all, false = keep only latest

seed: null                 # global seed for reproducibility. null = non-deterministic

eval:
  dataset: null            # eval dataset (BEIR name or local path). null = use data.dataset
  k_values: [1, 5, 10]
  split: test
  n_queries: 20            # small subset for fast evaluation. null = all
  corpus_size: 500         # small corpus for fast evaluation. null = full
  run_before: false        # evaluate baseline before training
  run_after: false         # evaluate after training

output_dir: ./forge-output/fiqa-quick
""",
    "fiqa_full.yaml": """\
# Full training config — uses all data
model:
  name: BAAI/bge-base-en-v1.5
  adapter_path: null       # path to existing adapter (for continued training)
  dtype: null              # "fp16", "bf16", or null (fp32). Load base model in this precision

data:
  dataset: fiqa
  split: train
  negatives: hard          # "random" or "hard"
  n_negatives: 3
  n_queries: null          # all queries
  corpus_size: null        # full corpus
  top_k: 50                # top-k for hard negative mining

lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: null     # auto-detect based on model architecture

train:
  epochs: 5
  batch_size: 32
  grad_accum_steps: 1      # effective batch size = 32
  lr: 2e-5
  weight_decay: 0.01       # AdamW weight decay
  warmup_steps: 100
  max_grad_norm: 1.0       # gradient clipping. null = disabled
  max_length: 512
  loss: infonce            # "triplet", "infonce", or "contrastive"
  margin: 0.2              # only for triplet loss
  temperature: 0.05        # only for infonce loss
  mixed_precision: bf16    # "fp16", "bf16", or null (disabled)
  overfit_batches: null    # set to 1 to overfit on 1 batch for debugging
  sanity_check_samples: 10 # check N training samples before/after training
  save_every_n_steps: 200  # checkpoint every 200 optimizer steps
  keep_all_checkpoints: false  # keep only latest checkpoint

seed: 42                   # global seed for reproducibility

eval:
  dataset: null            # eval dataset (BEIR name or local path). null = use data.dataset
  k_values: [1, 5, 10]
  split: test
  n_queries: null          # all test queries
  corpus_size: null        # full corpus
  run_before: true         # evaluate baseline before training
  run_after: true          # evaluate after training

output_dir: ./forge-output/fiqa-full
""",
    "fiqa_overfit.yaml": """\
# Overfit debug config — verify training pipeline works
model:
  name: BAAI/bge-base-en-v1.5
  adapter_path: null       # path to existing adapter (for continued training)
  dtype: null              # "fp16", "bf16", or null (fp32). Load base model in this precision

data:
  dataset: fiqa
  split: train
  negatives: random        # "random" or "hard"
  n_negatives: 1
  n_queries: 5             # tiny subset
  corpus_size: null        # not used with random negatives
  top_k: 50                # top-k for hard negative mining

lora:
  r: 8
  alpha: 16
  dropout: 0.0             # no dropout for overfitting
  target_modules: null     # auto-detect based on model architecture

train:
  epochs: 50
  batch_size: 4
  grad_accum_steps: 1      # effective batch size = 4
  lr: 1e-3                 # high LR to overfit fast
  weight_decay: 0.0        # no weight decay for overfitting
  warmup_steps: 0
  max_grad_norm: 1.0       # gradient clipping. null = disabled
  max_length: 512
  loss: triplet            # "triplet", "infonce", or "contrastive"
  margin: 0.2              # only for triplet loss
  temperature: 0.05        # only for infonce loss
  mixed_precision: null    # "fp16", "bf16", or null (disabled)
  overfit_batches: 1       # overfit on 1 batch
  sanity_check_samples: 10 # check N training samples before/after training
  save_every_n_steps: null # disabled for debug
  keep_all_checkpoints: false

seed: 42                   # fixed seed for reproducible debug runs

eval:
  dataset: null            # eval dataset (BEIR name or local path). null = use data.dataset
  k_values: [1, 5, 10]
  split: test
  n_queries: null          # not used (eval disabled)
  corpus_size: null        # not used (eval disabled)
  run_before: false        # skip eval for debug
  run_after: false         # skip eval for debug

output_dir: ./forge-output/fiqa-overfit
""",
}
