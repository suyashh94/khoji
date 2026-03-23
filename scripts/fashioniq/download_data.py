"""Download FashionIQ captions and image splits from the official GitHub repo.

Downloads annotation JSONs and image split files. Does NOT download images
(those need to be sourced separately from the fashion-iq-metadata repo).

Usage:
    python scripts/fashioniq/download_data.py [output_dir]
    # Default output_dir: ./data/fashioniq
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = (
    "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master"
)
METADATA_URL = (
    "https://raw.githubusercontent.com/"
    "hongwang600/fashion-iq-metadata/master/image_url"
)

CATEGORIES = ["dress", "shirt", "toptee"]
SPLITS = ["train", "val", "test"]


def download_file(url: str, dest: Path) -> None:
    """Download a file if it doesn't already exist."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest)


def download_captions(output_dir: Path) -> None:
    """Download caption annotation files."""
    print("\nDownloading captions...")
    captions_dir = output_dir / "captions"
    for cat in CATEGORIES:
        for split in SPLITS:
            filename = f"cap.{cat}.{split}.json"
            url = f"{BASE_URL}/captions/{filename}"
            download_file(url, captions_dir / filename)


def download_image_splits(output_dir: Path) -> None:
    """Download image split files."""
    print("\nDownloading image splits...")
    splits_dir = output_dir / "image_splits"
    for cat in CATEGORIES:
        for split in SPLITS:
            filename = f"split.{cat}.{split}.json"
            url = f"{BASE_URL}/image_splits/{filename}"
            download_file(url, splits_dir / filename)


def download_url_mappings(output_dir: Path) -> None:
    """Download ASIN → image URL mapping files."""
    print("\nDownloading image URL mappings...")
    url_dir = output_dir / "image_url"
    for cat in CATEGORIES:
        filename = f"asin2url.{cat}.txt"
        url = f"{METADATA_URL}/{filename}"
        download_file(url, url_dir / filename)


def summarize(output_dir: Path) -> None:
    """Print summary of downloaded data."""
    print("\n" + "=" * 60)
    print("FashionIQ Data Summary")
    print("=" * 60)

    captions_dir = output_dir / "captions"
    splits_dir = output_dir / "image_splits"

    for cat in CATEGORIES:
        print(f"\n  {cat.upper()}")
        for split in SPLITS:
            cap_file = captions_dir / f"cap.{cat}.{split}.json"
            split_file = splits_dir / f"split.{cat}.{split}.json"

            n_caps = 0
            n_images = 0
            if cap_file.exists():
                with open(cap_file) as f:
                    n_caps = len(json.load(f))
            if split_file.exists():
                with open(split_file) as f:
                    n_images = len(json.load(f))

            print(f"    {split:>5}: {n_caps:>6} annotations, {n_images:>6} images")

    # URL mapping stats
    url_dir = output_dir / "image_url"
    for cat in CATEGORIES:
        url_file = url_dir / f"asin2url.{cat}.txt"
        if url_file.exists():
            n_urls = sum(1 for _ in open(url_file))
            print(f"\n  {cat} URL mappings: {n_urls}")

    print(f"\nData saved to: {output_dir}")
    print("\nNext step:")
    print("  python scripts/fashioniq/train.py --category dress --epochs 5")
    print("  (images are loaded via URL and cached automatically)")


def main():
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "./data/fashioniq")
    print(f"Downloading FashionIQ data to: {output_dir}")

    download_captions(output_dir)
    download_image_splits(output_dir)
    download_url_mappings(output_dir)
    summarize(output_dir)


if __name__ == "__main__":
    main()
