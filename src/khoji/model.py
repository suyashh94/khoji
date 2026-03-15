"""Embedding model wrapper using transformers directly."""

from __future__ import annotations

import json

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
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if dtype not in mapping:
        raise ValueError(f"Unknown dtype: {dtype!r}. Use 'fp16', 'bf16', or null.")
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
        # Index of last non-padding token per sequence
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        return last_hidden_state[batch_indices, seq_lengths, :]

    # For mean variants, mask out padding tokens
    mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
    masked = last_hidden_state * mask

    if mode == "mean":
        return masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    if mode == "mean_sqrt_len":
        return masked.sum(dim=1) / torch.sqrt(mask.sum(dim=1).clamp(min=1e-9))

    if mode == "weightedmean":
        # Weights: position index (1-indexed), higher weight for later tokens
        seq_len = last_hidden_state.size(1)
        weights = torch.arange(1, seq_len + 1, dtype=torch.float, device=last_hidden_state.device)
        weights = weights.unsqueeze(0).unsqueeze(-1) * mask  # (batch, seq_len, 1)
        return (last_hidden_state * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)

    if mode == "max":
        masked = last_hidden_state.masked_fill(~attention_mask.unsqueeze(-1).bool(), -1e9)
        return masked.max(dim=1).values

    raise ValueError(f"Unknown pooling mode: {mode}")


class EmbeddingModel:
    """Wraps a transformer model for producing embeddings.

    Can load from HuggingFace by name, or accept a pre-built model and tokenizer
    for custom architectures.
    """

    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        device: torch.device | None = None,
        model: torch.nn.Module | None = None,
        tokenizer: object | None = None,
        pooling: str = "cls",
        max_length: int = 512,
        dtype: str | None = None,
    ):
        """Load an embedding model, optionally with a LoRA adapter.

        **HuggingFace models** — pass ``model_name``:

            EmbeddingModel("BAAI/bge-base-en-v1.5")

        **Custom PyTorch models** — pass ``model``, ``tokenizer``, and ``pooling``:

            EmbeddingModel(
                model=my_encoder,
                tokenizer=my_tokenizer,
                pooling="mean",
            )

        Args:
            model_name: HuggingFace model name. Ignored if ``model`` is provided.
            adapter_path: Path to a saved LoRA adapter directory.
            device: Device to load the model on. None = auto-detect.
            model: A pre-built PyTorch model. Must return an object with
                ``last_hidden_state`` when called with tokenizer outputs.
            tokenizer: A tokenizer compatible with the model. Must support
                ``(texts, padding=True, truncation=True, return_tensors="pt")``.
            pooling: Pooling strategy when using a custom model. One of:
                "cls", "mean", "max", "mean_sqrt_len", "weightedmean", "lasttoken".
                Ignored for HuggingFace models (auto-detected).
            max_length: Maximum token length for tokenization. Default: 512.
            dtype: Load base model weights in this precision. One of:
                ``"fp16"``, ``"bf16"``, or ``None`` (fp32, default).
                Reduces memory usage — LoRA weights are always kept in fp32.
        """
        self.device = device or get_device()

        self.max_length = max_length
        self._torch_dtype = _resolve_dtype(dtype)

        if model is not None:
            # Custom model path
            if tokenizer is None:
                raise ValueError("Must provide tokenizer when passing a custom model.")
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
            self.pooling_mode = pooling
            self.model.eval()
            dtype_str = f" | dtype: {dtype}" if dtype else ""
            print(f"Loaded custom model | pooling: {self.pooling_mode} | device: {self.device}{dtype_str}")
        elif model_name is not None:
            # HuggingFace model path
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            load_kwargs = {}
            if self._torch_dtype is not None:
                load_kwargs["torch_dtype"] = self._torch_dtype
            self.model = AutoModel.from_pretrained(model_name, **load_kwargs).to(self.device)
            self.pooling_mode = _detect_pooling(model_name)

            if adapter_path is not None:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(self.model, adapter_path).to(self.device)
                self.model.eval()
                dtype_str = f" | dtype: {dtype}" if dtype else ""
                print(f"Loaded {model_name} + adapter from {adapter_path} | "
                      f"pooling: {self.pooling_mode} | device: {self.device}{dtype_str}")
            else:
                self.model.eval()
                dtype_str = f" | dtype: {dtype}" if dtype else ""
                print(f"Loaded {model_name} | pooling: {self.pooling_mode} | device: {self.device}{dtype_str}")
        else:
            raise ValueError("Provide either model_name or model.")

    @torch.no_grad()
    def encode(
        self, texts: list[str], batch_size: int = 64, show_progress: bool = True
    ) -> torch.Tensor:
        """Encode a list of texts into embeddings.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for encoding.
            show_progress: Whether to show a progress bar.

        Returns:
            Tensor of shape (len(texts), hidden_dim), L2-normalized.
        """
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Encoding", unit="batch")

        for start in iterator:
            batch_texts = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded)
            embeddings = _pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
                self.pooling_mode,
            )
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
