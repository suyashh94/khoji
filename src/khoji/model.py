"""Embedding model wrapper using transformers directly."""

from __future__ import annotations

import json
from typing import Callable

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from khoji.device import get_device


def _resolve_dtype(dtype: str | None) -> torch.dtype | None:
    """Map dtype string to torch dtype.

    Args:
        dtype: "fp16", "bf16", or None (fp32 default).

    Returns:
        torch.dtype or None.
    """
    if dtype is None:
        return None
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(
            f"Unknown dtype: {dtype!r}. Use 'fp16', 'bf16', or null."
        )
    return mapping[dtype]


def _detect_pooling(model_name: str) -> str:
    """Detect pooling strategy from sentence-transformers config.

    Looks for 1_Pooling/config.json in the model repo. Falls back to "cls".
    """
    try:
        config_path = hf_hub_download(model_name, "1_Pooling/config.json")
        with open(config_path) as f:
            config = json.load(f)
    except Exception:
        return "cls"

    mode_map = {
        "pooling_mode_cls_token": "cls",
        "pooling_mode_mean_tokens": "mean",
        "pooling_mode_max_tokens": "max",
        "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len",
        "pooling_mode_weightedmean_tokens": "weightedmean",
        "pooling_mode_lasttoken": "lasttoken",
    }

    for key, mode in mode_map.items():
        if config.get(key, False):
            return mode

    return "cls"


def _pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Apply pooling to transformer outputs.

    Args:
        last_hidden_state: (batch, seq_len, hidden_dim)
        attention_mask: (batch, seq_len)
        mode: Pooling strategy.

    Returns:
        (batch, hidden_dim)
    """
    if mode == "cls":
        return last_hidden_state[:, 0, :]

    if mode == "lasttoken":
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(
            last_hidden_state.size(0), device=last_hidden_state.device
        )
        return last_hidden_state[batch_indices, seq_lengths, :]

    mask = attention_mask.unsqueeze(-1).float()
    masked = last_hidden_state * mask

    if mode == "mean":
        return masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    if mode == "mean_sqrt_len":
        return masked.sum(dim=1) / torch.sqrt(
            mask.sum(dim=1).clamp(min=1e-9)
        )

    if mode == "weightedmean":
        seq_len = last_hidden_state.size(1)
        weights = torch.arange(
            1, seq_len + 1,
            dtype=torch.float, device=last_hidden_state.device,
        )
        weights = weights.unsqueeze(0).unsqueeze(-1) * mask
        return (
            (last_hidden_state * weights).sum(dim=1)
            / weights.sum(dim=1).clamp(min=1e-9)
        )

    if mode == "max":
        masked = last_hidden_state.masked_fill(
            ~attention_mask.unsqueeze(-1).bool(), -1e9
        )
        return masked.max(dim=1).values

    raise ValueError(f"Unknown pooling mode: {mode}")


class EmbeddingModel:
    """Wraps a text encoder for producing embeddings.

    **HuggingFace models** (auto-detects pooling, tokenizer):

        model = EmbeddingModel("BAAI/bge-base-en-v1.5")
        embeddings = model.encode(["What is compound interest?"])

    **Custom encoder** — provide your own encode function:

        model = EmbeddingModel(
            encoder=my_encode_fn,  # (texts: list[str]) -> Tensor
        )

    The encoder function receives a batch of texts and returns a tensor
    of shape ``(batch, embed_dim)``. The library handles batching,
    L2 normalization, and device transfer.

    Args:
        model_name: HuggingFace model name. Auto-detects pooling,
            loads tokenizer and model weights.
        adapter_path: Path to a saved LoRA adapter directory.
        encoder: Custom encoding function. Signature:
            ``(texts: list[str]) -> Tensor``. Returns raw embeddings
            (library normalizes). Takes priority over model_name.
        device: Device to load the model on. None = auto-detect.
        max_length: Maximum token length for tokenization.
        dtype: Load base model weights in "fp16", "bf16", or None.
    """

    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        encoder: Callable[[list[str]], torch.Tensor] | None = None,
        device: torch.device | None = None,
        max_length: int = 512,
        dtype: str | None = None,
    ):
        self.device = device or get_device()
        self.max_length = max_length
        self._torch_dtype = _resolve_dtype(dtype)

        if encoder is not None:
            # Custom encoder path
            self._encoder_fn = encoder
            self.model = None
            self.tokenizer = None
            self.pooling_mode = None
            print(f"Loaded custom encoder | device: {self.device}")

        elif model_name is not None:
            # HuggingFace model path
            self._load_hf_model(model_name, adapter_path, dtype)

        else:
            raise ValueError(
                "Provide either model_name or encoder."
            )

    def _load_hf_model(
        self,
        model_name: str,
        adapter_path: str | None,
        dtype: str | None,
    ) -> None:
        """Load HuggingFace model and wire up encoder function."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs = {}
        if self._torch_dtype is not None:
            load_kwargs["torch_dtype"] = self._torch_dtype
        model = AutoModel.from_pretrained(
            model_name, **load_kwargs
        ).to(self.device)
        pooling_mode = _detect_pooling(model_name)

        if adapter_path is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model, adapter_path
            ).to(self.device)

        model.eval()

        # Expose for trainer (needs direct model access for gradients)
        self.model = model
        self.tokenizer = tokenizer
        self.pooling_mode = pooling_mode

        # Build encoder function via closure
        device = self.device
        max_length = self.max_length

        def _encode(texts: list[str]) -> torch.Tensor:
            encoded = tokenizer(
                texts, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            outputs = model(**encoded)
            return _pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
                pooling_mode,
            )

        self._encoder_fn = _encode

        dtype_str = f" | dtype: {dtype}" if dtype else ""
        adapter_str = (
            f" + adapter from {adapter_path}" if adapter_path else ""
        )
        print(
            f"Loaded {model_name}{adapter_str} | "
            f"pooling: {pooling_mode} | "
            f"device: {self.device}{dtype_str}"
        )

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Encode a list of texts into L2-normalized embeddings.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for encoding.
            show_progress: Whether to show a progress bar.

        Returns:
            Tensor of shape (len(texts), embed_dim), L2-normalized.
        """
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Encoding", unit="batch")

        for start in iterator:
            batch_texts = texts[start: start + batch_size]
            embeddings = self._encoder_fn(batch_texts)
            embeddings = torch.nn.functional.normalize(
                embeddings, p=2, dim=1
            )
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
