"""Multimodal embedding model for text-to-image retrieval."""

from __future__ import annotations

from typing import Callable

import torch
from PIL import Image
from tqdm import tqdm

from khoji.device import get_device
from khoji.image_utils import build_image_processor, load_images_batch
from khoji.model import _resolve_dtype


def _detect_model_type(model_name: str) -> str:
    """Detect whether a HuggingFace model is CLIP, SigLIP, etc."""
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_name)
        return getattr(config, "model_type", "unknown")
    except Exception:
        return "unknown"


class MultimodalEmbeddingModel:
    """Wraps text and vision encoders for cross-modal retrieval.

    **HuggingFace CLIP/SigLIP models:**

        model = MultimodalEmbeddingModel("openai/clip-vit-base-patch32")

    **Custom models:**

        model = MultimodalEmbeddingModel(
            text_model=my_text_encoder,
            vision_model=my_vision_encoder,
            tokenizer=my_tokenizer,
            image_processor=my_processor,  # callable: list[PIL.Image] -> tensor
        )

    Args:
        model_name: HuggingFace model ID (CLIP or SigLIP).
        adapter_path: Path to saved LoRA adapter weights.
        device: Device to use. None = auto-detect.
        text_model: Custom text encoder (nn.Module).
        vision_model: Custom vision encoder (nn.Module).
        tokenizer: Tokenizer for text inputs.
        image_processor: Callable that takes list[PIL.Image] and returns
            a batched pixel tensor.
        preprocess_overrides: Dict with optional keys: image_size, mean, std.
        max_length: Maximum token length for text inputs.
        dtype: Load base model in "fp16", "bf16", or None (fp32).
    """

    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        device: torch.device | None = None,
        text_model: torch.nn.Module | None = None,
        vision_model: torch.nn.Module | None = None,
        tokenizer: object | None = None,
        image_processor: Callable[[list[Image.Image]], torch.Tensor] | None = None,
        preprocess_overrides: dict | None = None,
        max_length: int = 77,
        dtype: str | None = None,
    ):
        self.device = device or get_device()
        self.max_length = max_length
        self._torch_dtype = _resolve_dtype(dtype)

        if text_model is not None and vision_model is not None:
            # Custom model path
            if tokenizer is None:
                raise ValueError("Must provide tokenizer with custom models.")
            if image_processor is None:
                raise ValueError("Must provide image_processor with custom models.")
            self.text_encoder = text_model.to(self.device)
            self.vision_encoder = vision_model.to(self.device)
            self.tokenizer = tokenizer
            self.image_processor = image_processor
            self.model_type = "custom"
            self._full_model = None
            dtype_str = f" | dtype: {dtype}" if dtype else ""
            print(f"Loaded custom multimodal model | device: {self.device}{dtype_str}")

        elif model_name is not None:
            # HuggingFace model path
            self.model_type = _detect_model_type(model_name)
            self._load_hf_model(model_name, adapter_path, dtype, preprocess_overrides)
            # Allow custom image_processor to override the auto-detected one
            if image_processor is not None:
                self.image_processor = image_processor

        else:
            raise ValueError(
                "Provide either model_name or (text_model + vision_model)."
            )

    def _load_hf_model(
        self,
        model_name: str,
        adapter_path: str | None,
        dtype: str | None,
        preprocess_overrides: dict | None,
    ) -> None:
        """Load a CLIP or SigLIP model from HuggingFace."""
        from transformers import AutoModel, AutoTokenizer

        load_kwargs = {}
        if self._torch_dtype is not None:
            load_kwargs["torch_dtype"] = self._torch_dtype

        if self.model_type == "clip":
            from transformers import CLIPModel

            full_model = CLIPModel.from_pretrained(model_name, **load_kwargs)
        elif self.model_type == "siglip":
            from transformers import SiglipModel

            full_model = SiglipModel.from_pretrained(model_name, **load_kwargs)
        else:
            # Try generic AutoModel
            full_model = AutoModel.from_pretrained(model_name, **load_kwargs)

        if adapter_path is not None:
            from peft import PeftModel

            full_model = PeftModel.from_pretrained(full_model, adapter_path)
            full_model = full_model.merge_and_unload()

        full_model = full_model.to(self.device)
        full_model.eval()
        self._full_model = full_model

        # Access sub-models
        self.text_encoder = full_model.text_model
        self.vision_encoder = full_model.vision_model
        self.text_projection = getattr(full_model, "text_projection", None)
        self.visual_projection = getattr(full_model, "visual_projection", None)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Image processor
        self.image_processor = build_image_processor(
            model_name=model_name,
            overrides=preprocess_overrides,
        )

        dtype_str = f" | dtype: {dtype}" if dtype else ""
        adapter_str = f" + adapter from {adapter_path}" if adapter_path else ""
        print(
            f"Loaded {model_name}{adapter_str} | "
            f"type: {self.model_type} | device: {self.device}{dtype_str}"
        )

    @torch.no_grad()
    def encode_text(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Encode text queries into embeddings.

        Returns:
            L2-normalized tensor of shape (N, embedding_dim).
        """
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress and len(texts) > batch_size:
            iterator = tqdm(iterator, desc="Encoding text", unit="batch")

        for i in iterator:
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            embeddings = self._extract_text_features(encoded)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def encode_images(
        self,
        images: list[Image.Image],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Encode PIL images into embeddings.

        Returns:
            L2-normalized tensor of shape (N, embedding_dim).
        """
        all_embeddings = []
        iterator = range(0, len(images), batch_size)
        if show_progress and len(images) > batch_size:
            iterator = tqdm(iterator, desc="Encoding images", unit="batch")

        for i in iterator:
            batch_images = images[i : i + batch_size]
            pixel_values = self.image_processor(batch_images).to(self.device)

            embeddings = self._extract_image_features(pixel_values)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def encode_image_sources(
        self,
        sources: list[str],
        base_dir: str | None = None,
        cache_dir: str | None = None,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Load images from paths/URLs and encode them.

        Convenience wrapper that handles image loading.

        Args:
            sources: List of local paths or URLs.
            base_dir: Base directory for resolving relative paths.
            cache_dir: Optional cache directory for URL downloads.
            batch_size: Encoding batch size.
            show_progress: Show progress bar.

        Returns:
            L2-normalized tensor of shape (N, embedding_dim).
        """
        all_embeddings = []
        iterator = range(0, len(sources), batch_size)
        if show_progress and len(sources) > batch_size:
            iterator = tqdm(iterator, desc="Encoding images", unit="batch")

        for i in iterator:
            batch_sources = sources[i : i + batch_size]
            batch_images = load_images_batch(
                batch_sources, base_dir=base_dir, cache_dir=cache_dir
            )
            pixel_values = self.image_processor(batch_images).to(self.device)

            embeddings = self._extract_image_features(pixel_values)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def _extract_text_features(self, encoded: dict) -> torch.Tensor:
        """Extract text embeddings from tokenized inputs."""
        if self.model_type in ("clip", "siglip") and self._full_model is not None:
            outputs = self.text_encoder(**encoded)
            pooled = outputs.pooler_output
            if self.text_projection is not None:
                return self.text_projection(pooled)
            return pooled
        else:
            outputs = self.text_encoder(**encoded)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state[:, 0, :]
            return outputs

    def _extract_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract image embeddings from pixel values."""
        if self.model_type in ("clip", "siglip") and self._full_model is not None:
            outputs = self.vision_encoder(pixel_values=pixel_values)
            pooled = outputs.pooler_output
            if self.visual_projection is not None:
                return self.visual_projection(pooled)
            return pooled
        else:
            outputs = self.vision_encoder(pixel_values)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state[:, 0, :]
            return outputs
