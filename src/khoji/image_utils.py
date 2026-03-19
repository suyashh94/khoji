"""Image loading and preprocessing utilities for multimodal retrieval."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Callable

import torch
from PIL import Image


def load_image(
    source: str,
    base_dir: str | None = None,
    cache_dir: str | None = None,
) -> Image.Image:
    """Load a single image from a local path or URL.

    Args:
        source: Local file path (absolute or relative) or HTTP(S) URL.
        base_dir: Base directory for resolving relative paths.
        cache_dir: If set, downloaded URL images are cached here.

    Returns:
        PIL Image in RGB mode.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return _load_from_url(source, cache_dir)
    else:
        path = Path(base_dir) / source if base_dir else Path(source)
        return Image.open(path).convert("RGB")


def load_images_batch(
    sources: list[str],
    base_dir: str | None = None,
    cache_dir: str | None = None,
) -> list[Image.Image]:
    """Batch load images. Downloads all URLs before returning.

    All images for the batch are loaded into memory at once so that
    the forward pass can run on the full batch without I/O stalls.

    Args:
        sources: List of local paths or URLs.
        base_dir: Base directory for resolving relative paths.
        cache_dir: Optional cache directory for URL downloads.

    Returns:
        List of PIL Images in RGB mode, same order as sources.
    """
    return [load_image(s, base_dir=base_dir, cache_dir=cache_dir) for s in sources]


def _load_from_url(url: str, cache_dir: str | None = None) -> Image.Image:
    """Download and load an image from a URL.

    If cache_dir is set, the image is cached locally using a hash of the URL
    as the filename. Subsequent calls for the same URL skip the download.
    """
    import requests

    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cached_file = cache_path / f"{url_hash}.jpg"
        if cached_file.exists():
            return Image.open(cached_file).convert("RGB")

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")

    if cache_dir is not None:
        img.save(cached_file)

    return img


def build_image_processor(
    model_name: str | None = None,
    overrides: dict | None = None,
    custom_fn: Callable[[list[Image.Image]], torch.Tensor] | None = None,
) -> Callable[[list[Image.Image]], torch.Tensor]:
    """Build an image preprocessing function.

    Three tiers:
        1. custom_fn: User-provided callable, used as-is.
        2. overrides: Auto-detect from HuggingFace + apply overrides
           (image_size, mean, std).
        3. auto: Pure auto from HuggingFace AutoProcessor.

    Args:
        model_name: HuggingFace model name for auto-detection.
        overrides: Dict with optional keys: image_size, mean, std.
        custom_fn: Custom preprocessing function.

    Returns:
        Callable that takes list[PIL.Image] and returns a batched tensor.
    """
    # Tier 1: custom function
    if custom_fn is not None:
        return custom_fn

    # Tier 2 & 3: HuggingFace processor
    if model_name is None:
        raise ValueError(
            "Must provide model_name for auto preprocessing, or custom_fn."
        )

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)

    # Apply overrides if provided
    if overrides:
        if "image_size" in overrides and overrides["image_size"] is not None:
            if hasattr(processor, "image_processor"):
                processor.image_processor.size = {
                    "shortest_edge": overrides["image_size"],
                }
        if "mean" in overrides and overrides["mean"] is not None:
            if hasattr(processor, "image_processor"):
                processor.image_processor.image_mean = overrides["mean"]
        if "std" in overrides and overrides["std"] is not None:
            if hasattr(processor, "image_processor"):
                processor.image_processor.image_std = overrides["std"]

    def process_images(images: list[Image.Image]) -> torch.Tensor:
        inputs = processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]

    return process_images
