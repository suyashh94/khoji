"""Multimodal embedding models for retrieval.

Two model classes for different retrieval paradigms:

- ``MultimodalEmbeddingModel``: Separate text and image encoders (CLIP, SigLIP).
  For text-to-image retrieval where queries are text and documents are images.

- ``JointEmbeddingModel``: Single encoder that handles text, images, or both
  together (BLIP-2, or any custom model with joint encoding).
  For composed image retrieval where queries are (image + text).
"""

from __future__ import annotations

from typing import Callable

import torch
from PIL import Image
from tqdm import tqdm

from khoji.device import get_device
from khoji.image_utils import build_image_processor, load_images_batch
from khoji.model import _resolve_dtype


def _detect_model_type(model_name: str) -> str:
    """Detect whether a HuggingFace model is CLIP, SigLIP, BLIP-2, etc."""
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_name)
        return getattr(config, "model_type", "unknown")
    except Exception:
        return "unknown"


# ──────────────────────────────────────────────────────────────
# MultimodalEmbeddingModel — separate text and image encoders
# ──────────────────────────────────────────────────────────────


class MultimodalEmbeddingModel:
    """Separate text and image encoders for cross-modal retrieval.

    For CLIP, SigLIP, and custom dual-encoder models where text and
    images are encoded independently.

    **HuggingFace models:**

        model = MultimodalEmbeddingModel("openai/clip-vit-base-patch32")
        text_emb = model.encode_text(["a red dress"])
        img_emb = model.encode_images([pil_image])

    **Custom models:**

        model = MultimodalEmbeddingModel(
            text_model=my_text_encoder,
            vision_model=my_vision_encoder,
            tokenizer=my_tokenizer,
            image_processor=my_processor,
        )
    """

    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        device: torch.device | None = None,
        text_model: torch.nn.Module | None = None,
        vision_model: torch.nn.Module | None = None,
        tokenizer: object | None = None,
        image_processor: (
            Callable[[list[Image.Image]], torch.Tensor] | None
        ) = None,
        preprocess_overrides: dict | None = None,
        max_length: int = 77,
        dtype: str | None = None,
    ):
        self.device = device or get_device()
        self.max_length = max_length
        self._torch_dtype = _resolve_dtype(dtype)

        if text_model is not None and vision_model is not None:
            if tokenizer is None:
                raise ValueError(
                    "Must provide tokenizer with custom models."
                )
            if image_processor is None:
                raise ValueError(
                    "Must provide image_processor with custom models."
                )
            self.text_encoder = text_model.to(self.device)
            self.vision_encoder = vision_model.to(self.device)
            self.tokenizer = tokenizer
            self.image_processor = image_processor
            self.model_type = "custom"
            self._full_model = None
            self.text_projection = None
            self.visual_projection = None
            dtype_str = f" | dtype: {dtype}" if dtype else ""
            print(
                f"Loaded custom multimodal model | "
                f"device: {self.device}{dtype_str}"
            )

        elif model_name is not None:
            self.model_type = _detect_model_type(model_name)
            self._load_hf_model(
                model_name, adapter_path, dtype, preprocess_overrides
            )
            if image_processor is not None:
                self.image_processor = image_processor

        else:
            raise ValueError(
                "Provide either model_name or "
                "(text_model + vision_model)."
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

            full_model = CLIPModel.from_pretrained(
                model_name, **load_kwargs
            )
        elif self.model_type == "siglip":
            from transformers import SiglipModel

            full_model = SiglipModel.from_pretrained(
                model_name, **load_kwargs
            )
        else:
            full_model = AutoModel.from_pretrained(
                model_name, **load_kwargs
            )

        if adapter_path is not None:
            from peft import PeftModel

            full_model = PeftModel.from_pretrained(
                full_model, adapter_path
            )
            full_model = full_model.merge_and_unload()

        full_model = full_model.to(self.device)
        full_model.eval()
        self._full_model = full_model

        self.text_encoder = full_model.text_model
        self.vision_encoder = full_model.vision_model
        self.text_projection = getattr(
            full_model, "text_projection", None
        )
        self.visual_projection = getattr(
            full_model, "visual_projection", None
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.image_processor = build_image_processor(
            model_name=model_name, overrides=preprocess_overrides,
        )

        dtype_str = f" | dtype: {dtype}" if dtype else ""
        adapter_str = (
            f" + adapter from {adapter_path}" if adapter_path else ""
        )
        print(
            f"Loaded {model_name}{adapter_str} | "
            f"type: {self.model_type} | "
            f"device: {self.device}{dtype_str}"
        )

    @torch.no_grad()
    def encode_text(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Encode texts into L2-normalized embeddings."""
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress and len(texts) > batch_size:
            iterator = tqdm(iterator, desc="Encoding text", unit="batch")

        for i in iterator:
            batch_texts = texts[i: i + batch_size]
            encoded = self.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)

            embeddings = self._extract_text_features(encoded)
            embeddings = torch.nn.functional.normalize(
                embeddings, p=2, dim=1
            )
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def encode_images(
        self,
        images: list[Image.Image],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Encode PIL images into L2-normalized embeddings."""
        all_embeddings = []
        iterator = range(0, len(images), batch_size)
        if show_progress and len(images) > batch_size:
            iterator = tqdm(
                iterator, desc="Encoding images", unit="batch"
            )

        for i in iterator:
            batch_images = images[i: i + batch_size]
            pixel_values = self.image_processor(batch_images).to(
                self.device
            )

            embeddings = self._extract_image_features(pixel_values)
            embeddings = torch.nn.functional.normalize(
                embeddings, p=2, dim=1
            )
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
        """Load images from paths/URLs and encode them."""
        all_embeddings = []
        iterator = range(0, len(sources), batch_size)
        if show_progress and len(sources) > batch_size:
            iterator = tqdm(
                iterator, desc="Encoding images", unit="batch"
            )

        for i in iterator:
            batch_sources = sources[i: i + batch_size]
            batch_images = load_images_batch(
                batch_sources, base_dir=base_dir, cache_dir=cache_dir
            )
            pixel_values = self.image_processor(batch_images).to(
                self.device
            )

            embeddings = self._extract_image_features(pixel_values)
            embeddings = torch.nn.functional.normalize(
                embeddings, p=2, dim=1
            )
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def _extract_text_features(self, encoded: dict) -> torch.Tensor:
        """Extract text embeddings from tokenized inputs."""
        if (
            self.model_type in ("clip", "siglip")
            and self._full_model is not None
        ):
            outputs = self.text_encoder(**encoded)
            pooled = outputs.pooler_output
            if self.text_projection is not None:
                return self.text_projection(pooled)
            return pooled
        else:
            outputs = self.text_encoder(**encoded)
            if (
                hasattr(outputs, "pooler_output")
                and outputs.pooler_output is not None
            ):
                return outputs.pooler_output
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state[:, 0, :]
            return outputs

    def _extract_image_features(
        self, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """Extract image embeddings from pixel values."""
        if (
            self.model_type in ("clip", "siglip")
            and self._full_model is not None
        ):
            outputs = self.vision_encoder(pixel_values=pixel_values)
            pooled = outputs.pooler_output
            if self.visual_projection is not None:
                return self.visual_projection(pooled)
            return pooled
        else:
            outputs = self.vision_encoder(pixel_values)
            if (
                hasattr(outputs, "pooler_output")
                and outputs.pooler_output is not None
            ):
                return outputs.pooler_output
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state[:, 0, :]
            return outputs


# ──────────────────────────────────────────────────────────────
# JointEmbeddingModel — single encoder for image, text, or both
# ──────────────────────────────────────────────────────────────


class JointEmbeddingModel:
    """Single encoder that handles text, images, or both together.

    For composed image retrieval and models with native joint encoding
    (BLIP-2, or custom models with cross-attention fusion).

    All three modes use the same ``encode()`` method:

        model.encode(images=[img])                          # image-only
        model.encode(texts=["red dress"])                    # text-only
        model.encode(images=[img], texts=["make it red"])    # joint

    **HuggingFace BLIP-2:**

        model = JointEmbeddingModel("Salesforce/blip2-itm-vit-g")

    **Custom model:**

        model = JointEmbeddingModel(
            encoder=my_encode_fn,  # (images, texts, device) -> tensor
        )

    Args:
        model_name: HuggingFace model ID (BLIP-2 variants).
        adapter_path: Path to saved LoRA adapter weights.
        encoder: Custom encoding function with signature:
            ``(images: list[Image] | None, texts: list[str] | None,
            device: torch.device) -> Tensor``.
            Must return L2-normalized embeddings.
        device: Device to use. None = auto-detect.
        max_length: Maximum token length for text inputs.
        dtype: Load model in "fp16", "bf16", or None (fp32).
    """

    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        encoder: Callable | None = None,
        device: torch.device | None = None,
        max_length: int = 77,
        dtype: str | None = None,
    ):
        self.device = device or get_device()
        self.max_length = max_length
        self._torch_dtype = _resolve_dtype(dtype)
        self._custom_encoder = encoder
        self._full_model = None

        if encoder is not None:
            self.model_type = "custom"
            print(
                f"Loaded custom joint encoder | device: {self.device}"
            )

        elif model_name is not None:
            self.model_type = _detect_model_type(model_name)
            if self.model_type != "blip-2":
                raise ValueError(
                    f"JointEmbeddingModel requires a BLIP-2 model, "
                    f"got model_type={self.model_type!r}. "
                    f"Use MultimodalEmbeddingModel for CLIP/SigLIP."
                )
            self._load_blip2(model_name, adapter_path, dtype)

        else:
            raise ValueError(
                "Provide either model_name (BLIP-2) or encoder callable."
            )

    def _load_blip2(
        self,
        model_name: str,
        adapter_path: str | None,
        dtype: str | None,
    ) -> None:
        """Load BLIP-2 for image-text retrieval."""
        from transformers import (
            AutoProcessor,
            Blip2ForImageTextRetrieval,
        )

        load_kwargs = {}
        if self._torch_dtype is not None:
            load_kwargs["torch_dtype"] = self._torch_dtype

        model = Blip2ForImageTextRetrieval.from_pretrained(
            model_name, **load_kwargs
        )

        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

        model = model.to(self.device)
        model.eval()
        self._full_model = model
        self._processor = AutoProcessor.from_pretrained(model_name)

        dtype_str = f" | dtype: {dtype}" if dtype else ""
        adapter_str = (
            f" + adapter from {adapter_path}" if adapter_path else ""
        )
        print(
            f"Loaded {model_name}{adapter_str} | "
            f"type: blip-2 | device: {self.device}{dtype_str}"
        )

    @torch.no_grad()
    def encode(
        self,
        images: list[Image.Image] | None = None,
        texts: list[str] | None = None,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Encode images, texts, or both into L2-normalized embeddings.

        Args:
            images: List of PIL images. None for text-only.
            texts: List of text strings. None for image-only.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar.

        Returns:
            L2-normalized tensor of shape (N, embedding_dim).
        """
        if images is None and texts is None:
            raise ValueError("Provide at least one of images or texts.")

        if self._custom_encoder is not None:
            return self._custom_encoder(images, texts, self.device)

        return self._encode_blip2(
            images, texts, batch_size, show_progress
        )

    @torch.no_grad()
    def _encode_blip2(
        self,
        images: list[Image.Image] | None,
        texts: list[str] | None,
        batch_size: int,
        show_progress: bool,
    ) -> torch.Tensor:
        """Encode via BLIP-2 ITC mode in shared 256-dim space."""
        n = len(images) if images is not None else len(texts)  # type: ignore
        joint_mode = images is not None and texts is not None
        all_embeddings = []
        iterator = range(0, n, batch_size)
        if show_progress and n > batch_size:
            iterator = tqdm(iterator, desc="Encoding", unit="batch")

        for start in iterator:
            bs = min(batch_size, n - start)

            if images is not None:
                batch_imgs = images[start: start + bs]
                batch_texts = (
                    texts[start: start + bs]
                    if texts is not None
                    else [""] * bs
                )
            else:
                dummy = Image.new("RGB", (224, 224), "black")
                batch_imgs = [dummy] * bs
                batch_texts = texts[start: start + bs]  # type: ignore

            inputs = self._processor(
                images=batch_imgs, text=batch_texts,
                return_tensors="pt", padding=True, truncation=True,
            ).to(self.device)

            out = self._full_model(
                **inputs, use_image_text_matching_head=False,
            )

            if joint_mode:
                img_emb = out.image_embeds.max(dim=1).values
                txt_emb = out.text_embeds
                embeddings = img_emb + txt_emb
            elif images is not None:
                embeddings = out.image_embeds.max(dim=1).values
            else:
                embeddings = out.text_embeds

            embeddings = torch.nn.functional.normalize(
                embeddings, p=2, dim=1
            )
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
