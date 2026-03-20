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
    """Detect whether a HuggingFace model is CLIP, SigLIP, BLIP-2, etc."""
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

    **HuggingFace BLIP-2 models (supports joint image+text encoding):**

        model = MultimodalEmbeddingModel("Salesforce/blip2-itm-vit-g")
        emb = model.encode(images=[img], texts=["make it red"])

    **Custom models:**

        model = MultimodalEmbeddingModel(
            text_model=my_text_encoder,
            vision_model=my_vision_encoder,
            tokenizer=my_tokenizer,
            image_processor=my_processor,
        )

    **Custom models with joint encoding:**

        model = MultimodalEmbeddingModel(
            text_model=my_text_encoder,
            vision_model=my_vision_encoder,
            tokenizer=my_tokenizer,
            image_processor=my_processor,
            joint_encoder=my_joint_fn,  # (images, texts, device) -> tensor
        )

    Args:
        model_name: HuggingFace model ID (CLIP, SigLIP, or BLIP-2).
        adapter_path: Path to saved LoRA adapter weights.
        device: Device to use. None = auto-detect.
        text_model: Custom text encoder (nn.Module).
        vision_model: Custom vision encoder (nn.Module).
        tokenizer: Tokenizer for text inputs.
        image_processor: Callable that takes list[PIL.Image] and returns
            a batched pixel tensor.
        joint_encoder: Callable for joint (image+text) encoding. Signature:
            ``(images: list[PIL.Image], texts: list[str], device) -> Tensor``.
            When provided, ``encode(images=..., texts=...)`` uses this instead
            of additive fusion. The returned tensor should be L2-normalized.
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
        joint_encoder: Callable | None = None,
        preprocess_overrides: dict | None = None,
        max_length: int = 77,
        dtype: str | None = None,
    ):
        self.device = device or get_device()
        self.max_length = max_length
        self._torch_dtype = _resolve_dtype(dtype)
        self._joint_encoder = joint_encoder

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
            joint_str = " | joint_encoder: yes" if joint_encoder else ""
            print(
                f"Loaded custom multimodal model | "
                f"device: {self.device}{dtype_str}{joint_str}"
            )

        elif model_name is not None:
            # HuggingFace model path
            self.model_type = _detect_model_type(model_name)
            if self.model_type == "blip-2":
                self._load_blip2_model(
                    model_name, adapter_path, dtype
                )
            else:
                self._load_hf_model(
                    model_name, adapter_path, dtype, preprocess_overrides
                )
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

    def _load_blip2_model(
        self,
        model_name: str,
        adapter_path: str | None,
        dtype: str | None,
    ) -> None:
        """Load a BLIP-2 model with Q-Former for joint image+text encoding."""
        from transformers import AutoProcessor, Blip2Model

        load_kwargs = {}
        if self._torch_dtype is not None:
            load_kwargs["torch_dtype"] = self._torch_dtype

        model = Blip2Model.from_pretrained(model_name, **load_kwargs)

        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

        model = model.to(self.device)
        model.eval()
        self._full_model = model
        self._blip2_processor = AutoProcessor.from_pretrained(model_name)

        self.text_encoder = None
        self.vision_encoder = None
        self.text_projection = None
        self.visual_projection = None
        self.tokenizer = self._blip2_processor.tokenizer
        self.image_processor = self._blip2_processor.image_processor

        dtype_str = f" | dtype: {dtype}" if dtype else ""
        adapter_str = (
            f" + adapter from {adapter_path}" if adapter_path else ""
        )
        print(
            f"Loaded {model_name}{adapter_str} | "
            f"type: blip-2 (Q-Former) | device: {self.device}{dtype_str}"
        )

    @torch.no_grad()
    def encode(
        self,
        images: list[Image.Image] | None = None,
        texts: list[str] | None = None,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Unified encoding: image-only, text-only, or joint image+text.

        For BLIP-2: passes both through the Q-Former for joint encoding.
        For CLIP/SigLIP: encodes separately; joint falls back to additive fusion.

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

        # Joint encoding: image + text together
        if images is not None and texts is not None:
            if self.model_type == "blip-2":
                return self._encode_blip2(
                    images, texts, batch_size, show_progress
                )
            if self._joint_encoder is not None:
                return self._joint_encoder(images, texts, self.device)
            # Additive fusion fallback for CLIP/SigLIP/custom
            img_emb = self.encode_images(images, batch_size, show_progress)
            txt_emb = self.encode_text(texts, batch_size, show_progress)
            fused = img_emb + txt_emb
            return torch.nn.functional.normalize(fused, p=2, dim=1)

        # Single modality
        if self.model_type == "blip-2":
            return self._encode_blip2(
                images, texts, batch_size, show_progress
            )
        if images is not None:
            return self.encode_images(images, batch_size, show_progress)
        return self.encode_text(texts, batch_size, show_progress)

    @torch.no_grad()
    def _encode_blip2(
        self,
        images: list[Image.Image] | None,
        texts: list[str] | None,
        batch_size: int,
        show_progress: bool,
    ) -> torch.Tensor:
        """Encode via BLIP-2 Q-Former — supports joint image+text."""
        n = len(images) if images is not None else len(texts)  # type: ignore
        all_embeddings = []
        iterator = range(0, n, batch_size)
        if show_progress and n > batch_size:
            iterator = tqdm(iterator, desc="Encoding", unit="batch")

        for start in iterator:
            kwargs = {}
            if images is not None:
                batch_imgs = images[start: start + batch_size]
                proc = self._blip2_processor(
                    images=batch_imgs, return_tensors="pt",
                )
                kwargs["pixel_values"] = proc["pixel_values"].to(self.device)

            if texts is not None:
                batch_texts = texts[start: start + batch_size]
                tok = self._blip2_processor.tokenizer(
                    batch_texts, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                )
                kwargs["input_ids"] = tok["input_ids"].to(self.device)
                kwargs["attention_mask"] = (
                    tok["attention_mask"].to(self.device)
                )

            qformer_out = self._full_model.get_qformer_features(**kwargs)
            embeddings = qformer_out.mean(dim=1)
            embeddings = torch.nn.functional.normalize(
                embeddings, p=2, dim=1
            )
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

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
