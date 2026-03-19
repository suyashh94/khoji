"""Configuration for multimodal (text-to-image) training runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from khoji.config import EvalConfig, LoRAConfig, TrainConfig, _coerce_train_config


@dataclass
class MultimodalModelConfig:
    """Model configuration for multimodal retrieval.

    For CLIP/SigLIP models, set ``name`` to the HuggingFace model ID.
    The text and vision encoders are loaded from the same model.

    Args:
        name: HuggingFace model identifier (e.g., "openai/clip-vit-base-patch32").
        adapter_path: Path to a previously saved LoRA adapter.
        dtype: Load base model in "fp16", "bf16", or null (fp32).
        lora_target: Which encoder(s) to apply LoRA to: "vision", "text", or "both".
    """

    name: str = "openai/clip-vit-base-patch32"
    adapter_path: str | None = None
    dtype: str | None = None
    lora_target: str = "both"  # "vision", "text", or "both"


@dataclass
class MultimodalDataConfig:
    """Data configuration for multimodal retrieval.

    Args:
        dataset: HuggingFace dataset name or path to local dataset directory.
        split: Dataset split for training data.
        negatives: Negative mining strategy: "random", "hard", or "mixed".
        n_negatives: Number of negatives per (query, positive_image) pair.
        n_queries: Number of queries to use. None = all.
        corpus_size: Corpus size limit. None = all. Relevant images always included.
        top_k: Top-k for hard negative mining.
        cache_dir: Directory to cache downloaded images. None = no caching.
    """

    dataset: str = "nlphuji/flickr30k"
    split: str = "train"
    negatives: str = "random"  # "random", "hard", or "mixed"
    n_negatives: int = 1  # negatives per pair (random/hard modes)
    n_random: int = 1  # random negatives per pair (mixed mode only)
    n_hard: int = 1  # hard negatives per pair (mixed mode only)
    n_queries: int | None = None
    corpus_size: int | None = None
    top_k: int = 50
    skip_top: int = 0  # skip top N non-relevant docs (avoids false negatives)
    mining_rounds: int = 1  # iterative mining rounds (hard/mixed only)
    cache_dir: str | None = None


@dataclass
class ImagePreprocessConfig:
    """Override image preprocessing parameters.

    When null, values are auto-detected from the HuggingFace model.

    Args:
        image_size: Override image size (shortest edge).
        mean: Override normalization mean [R, G, B].
        std: Override normalization std [R, G, B].
    """

    image_size: int | None = None
    mean: list[float] | None = None
    std: list[float] | None = None


@dataclass
class MultimodalForgeConfig:
    """Top-level configuration for a multimodal training run."""

    model: MultimodalModelConfig = field(default_factory=MultimodalModelConfig)
    data: MultimodalDataConfig = field(default_factory=MultimodalDataConfig)
    lora: LoRAConfig | None = field(default_factory=LoRAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    preprocess: ImagePreprocessConfig | None = None
    seed: int | None = None
    output_dir: str = "./forge-output"

    @staticmethod
    def from_yaml(path: str) -> MultimodalForgeConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        config = MultimodalForgeConfig()

        if "model" in raw:
            config.model = MultimodalModelConfig(**raw["model"])
        if "data" in raw:
            config.data = MultimodalDataConfig(**raw["data"])
        if "lora" in raw:
            if raw["lora"] is None:
                config.lora = None
            else:
                config.lora = LoRAConfig(**raw["lora"])
        if "train" in raw:
            config.train = _coerce_train_config(raw["train"])
        if "eval" in raw:
            config.eval = EvalConfig(**raw["eval"])
        if "preprocess" in raw and raw["preprocess"] is not None:
            config.preprocess = ImagePreprocessConfig(**raw["preprocess"])
        if "seed" in raw and raw["seed"] is not None:
            config.seed = int(raw["seed"])
        if "output_dir" in raw:
            config.output_dir = raw["output_dir"]

        config.validate()
        return config

    def validate(self) -> None:
        """Validate config for common mistakes."""
        # Validate lora_target
        valid_targets = {"vision", "text", "both"}
        if self.model.lora_target not in valid_targets:
            raise ValueError(
                f"model.lora_target must be one of {valid_targets}, "
                f"got {self.model.lora_target!r}"
            )

        # Validate lora_target requires lora config
        if self.lora is None and self.model.lora_target != "both":
            raise ValueError(
                "model.lora_target is set but lora is null (full fine-tuning). "
                "Either set lora config or remove lora_target."
            )

        # Validate negatives
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
