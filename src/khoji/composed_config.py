"""Configuration for composed (image+text → image) training runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from khoji.config import EvalConfig, LoRAConfig, TrainConfig, _coerce_train_config


@dataclass
class ComposedModelConfig:
    """Model configuration for composed retrieval.

    For BLIP-2 models, set ``name`` to the HuggingFace model ID.

    Args:
        name: HuggingFace model identifier (e.g., "Salesforce/blip2-itm-vit-g").
        adapter_path: Path to a previously saved LoRA adapter.
        dtype: Load base model in "fp16", "bf16", or null (fp32).
    """

    name: str = "Salesforce/blip2-itm-vit-g"
    adapter_path: str | None = None
    dtype: str | None = None


@dataclass
class ComposedDataConfig:
    """Data configuration for composed retrieval.

    Args:
        dataset: Path to local dataset directory (with queries.jsonl, corpus.jsonl, qrels.tsv).
        split: Dataset split for training data.
        negatives: Negative mining strategy: "random", "hard", or "mixed".
        n_negatives: Number of negatives per (query, positive) pair.
        n_queries: Number of queries to use. None = all.
        corpus_size: Gallery size limit. None = all.
        top_k: Top-k for hard negative mining.
        skip_top: Skip top N non-relevant images before picking hard negatives.
        mining_rounds: Iterative mining rounds (hard/mixed only).
        cache_dir: Directory to cache downloaded images. None = no caching.
    """

    dataset: str = "./data/composed"
    split: str = "train"
    negatives: str = "random"  # "random", "hard", or "mixed"
    n_negatives: int = 1  # negatives per pair (random/hard modes)
    n_random: int = 1  # random negatives per pair (mixed mode only)
    n_hard: int = 1  # hard negatives per pair (mixed mode only)
    n_queries: int | None = None
    corpus_size: int | None = None
    top_k: int = 50
    skip_top: int = 0
    mining_rounds: int = 1
    cache_dir: str | None = None


@dataclass
class ComposedForgeConfig:
    """Top-level configuration for a composed retrieval training run."""

    model: ComposedModelConfig = field(default_factory=ComposedModelConfig)
    data: ComposedDataConfig = field(default_factory=ComposedDataConfig)
    lora: LoRAConfig | None = field(default_factory=LoRAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int | None = None
    output_dir: str = "./forge-output"

    @staticmethod
    def from_yaml(path: str) -> ComposedForgeConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        config = ComposedForgeConfig()

        if "model" in raw:
            config.model = ComposedModelConfig(**raw["model"])
        if "data" in raw:
            config.data = ComposedDataConfig(**raw["data"])
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

        config.validate()
        return config

    def validate(self) -> None:
        """Validate config for common mistakes."""
        if self.data.negatives not in {"random", "hard", "mixed"}:
            raise ValueError(
                f"data.negatives must be 'random', 'hard', or 'mixed', "
                f"got {self.data.negatives!r}"
            )

    def to_yaml(self, path: str) -> None:
        """Save config to a YAML file."""
        from dataclasses import asdict

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
