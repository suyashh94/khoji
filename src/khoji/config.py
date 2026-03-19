"""Configuration for training runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    name: str = "BAAI/bge-base-en-v1.5"
    adapter_path: str | None = None  # Path to existing adapter (for continued training)
    dtype: str | None = None  # "fp16", "bf16", or null (fp32). Load base model weights in this precision.


@dataclass
class DataConfig:
    dataset: str = "fiqa"
    split: str = "train"
    negatives: str = "random"  # "random", "hard", or "mixed"
    n_negatives: int = 1  # negatives per pair (used by "random" and "hard" modes)
    n_random: int = 1  # random negatives per pair (only used when negatives: mixed)
    n_hard: int = 1  # hard negatives per pair (only used when negatives: mixed)
    n_queries: int | None = None
    corpus_size: int | None = None
    top_k: int = 50  # for hard negative mining
    skip_top: int = 0  # skip top N non-relevant docs before picking hard negatives
    mining_rounds: int = 1  # iterative mining rounds (hard/mixed only)


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: list[str] | None = None


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    max_length: int = 512
    loss: str = "triplet"  # "triplet", "infonce", or "contrastive"
    margin: float = 0.2  # for triplet loss
    temperature: float = 0.05  # for infonce loss
    mixed_precision: str | None = None  # "fp16", "bf16", or null (disabled)
    overfit_batches: int | None = None  # set to 1 to overfit on 1 batch for debugging
    sanity_check_samples: int = 10  # number of training samples to check before/after training
    save_every_n_steps: int | None = None  # save checkpoint every N optimizer steps
    keep_all_checkpoints: bool = False  # True = keep all, False = keep only latest


@dataclass
class EvalConfig:
    dataset: str | None = None  # eval dataset (BEIR name or local path). None = use data.dataset
    k_values: list[int] = field(default_factory=lambda: [1, 5, 10])
    split: str = "test"
    n_queries: int | None = None
    corpus_size: int | None = None
    run_before: bool = True  # evaluate baseline before training
    run_after: bool = True  # evaluate after training


def _coerce_train_config(raw: dict) -> TrainConfig:
    """Coerce YAML values to correct types for TrainConfig.

    YAML safe_load parses values like 2e-5 as strings, not floats.
    """
    float_fields = {"lr", "margin", "temperature", "max_grad_norm", "weight_decay"}
    int_fields = {"epochs", "batch_size", "grad_accum_steps", "warmup_steps", "max_length", "overfit_batches", "sanity_check_samples", "save_every_n_steps"}
    coerced = {}
    for k, v in raw.items():
        if k in float_fields and isinstance(v, str):
            coerced[k] = float(v)
        elif k in int_fields and isinstance(v, str):
            coerced[k] = int(v)
        else:
            coerced[k] = v
    return TrainConfig(**coerced)


@dataclass
class ForgeConfig:
    """Top-level configuration for a retriever-forge training run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig | None = field(default_factory=LoRAConfig)  # null = full fine-tuning
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int | None = None  # global seed for reproducibility
    output_dir: str = "./forge-output"

    @staticmethod
    def from_yaml(path: str) -> ForgeConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        config = ForgeConfig()

        if "model" in raw:
            config.model = ModelConfig(**raw["model"])
        if "data" in raw:
            config.data = DataConfig(**raw["data"])
        if "lora" in raw:
            if raw["lora"] is None:
                config.lora = None
            else:
                config.lora = LoRAConfig(**raw["lora"])
        if "train" in raw:
            config.train = _coerce_train_config(raw["train"])
        if "eval" in raw:
            config.eval = EvalConfig(**raw["eval"])
        if "seed" in raw and raw["seed"] is not None:
            config.seed = int(raw["seed"])
        if "output_dir" in raw:
            config.output_dir = raw["output_dir"]

        return config

    @staticmethod
    def _default() -> ForgeConfig:
        """Return a config with all defaults."""
        return ForgeConfig()

    def to_yaml(self, path: str) -> None:
        """Save config to a YAML file."""
        from dataclasses import asdict

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
