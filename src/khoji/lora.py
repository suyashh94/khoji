"""LoRA adapter configuration and setup."""

from __future__ import annotations

from dataclasses import dataclass, field

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, PreTrainedModel


# Default LoRA target modules per model architecture.
# These target the attention projection layers where LoRA is most effective.
DEFAULT_TARGET_MODULES: dict[str, list[str]] = {
    "bert": ["query", "key", "value"],
    "roberta": ["query", "key", "value"],
    "distilbert": ["q_lin", "k_lin", "v_lin"],
    "xlm-roberta": ["query", "key", "value"],
    "deberta": ["query_proj", "key_proj", "value_proj"],
    "deberta-v2": ["query_proj", "key_proj", "value_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj"],
    "llama": ["q_proj", "k_proj", "v_proj"],
}


@dataclass
class LoRASettings:
    """Configuration for LoRA adapters.

    Args:
        r: LoRA rank. Higher = more parameters, more capacity.
        alpha: LoRA scaling factor. Typically set to r or 2*r.
        dropout: Dropout probability on LoRA layers.
        target_modules: Which layers to apply LoRA to. None = auto-detect
            based on model architecture.
    """

    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: list[str] | None = None


def _get_target_modules(model: PreTrainedModel) -> list[str]:
    """Auto-detect LoRA target modules based on model architecture."""
    config = model.config
    model_type = getattr(config, "model_type", "")

    if model_type in DEFAULT_TARGET_MODULES:
        return DEFAULT_TARGET_MODULES[model_type]

    # Fallback: search for common attention layer names
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}

    for candidates in [
        ["query", "key", "value"],
        ["q_proj", "k_proj", "v_proj"],
        ["q_lin", "k_lin", "v_lin"],
        ["query_proj", "key_proj", "value_proj"],
    ]:
        if all(c in module_names for c in candidates):
            return candidates

    raise ValueError(
        f"Could not auto-detect LoRA target modules for model type '{model_type}'. "
        f"Please specify target_modules explicitly in LoRASettings. "
        f"Available modules: {sorted(module_names)}"
    )


def apply_lora(model: PreTrainedModel, settings: LoRASettings | None = None) -> PreTrainedModel:
    """Attach LoRA adapters to a model.

    Args:
        model: The base HuggingFace model.
        settings: LoRA configuration. None = use defaults.

    Returns:
        The model with LoRA adapters attached. Only LoRA parameters are trainable.
    """
    if settings is None:
        settings = LoRASettings()

    target_modules = settings.target_modules or _get_target_modules(model)

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=target_modules,
    )

    peft_model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"LoRA applied | target: {target_modules} | "
          f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return peft_model
