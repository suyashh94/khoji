"""Dataset loading for multimodal (text-to-image) retrieval."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MultimodalRetrievalDataset:
    """Retrieval dataset where queries are text and corpus entries are images.

    Construct directly for custom datasets::

        dataset = MultimodalRetrievalDataset(
            queries={"q1": "a dog playing fetch", "q2": "sunset over ocean"},
            corpus={"d1": "images/dog.jpg", "d2": "https://example.com/sunset.jpg"},
            qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
            base_dir="./my_dataset",
        )

    Args:
        queries: query_id -> query text.
        corpus: doc_id -> image source (local path or URL).
        qrels: query_id -> {doc_id -> relevance_score}.
        base_dir: Base directory for resolving relative image paths.
    """

    queries: dict[str, str]
    corpus: dict[str, str]  # values are image paths/URLs, not text
    qrels: dict[str, dict[str, int]]
    base_dir: str | None = None


def load_flickr30k(
    split: str = "test",
    n_samples: int | None = None,
    cache_dir: str | None = None,
) -> MultimodalRetrievalDataset:
    """Load Flickr30k from HuggingFace for text-to-image retrieval.

    Downloads the annotations CSV and images zip from HuggingFace,
    extracts images to a local cache, and builds a dataset where
    each caption is a query and the corresponding image is the document.

    Args:
        split: Dataset split ("train", "val", or "test").
        n_samples: Limit number of images. None = all in split.
        cache_dir: Where to cache extracted images. Defaults to
            ``~/.cache/khoji/flickr30k``.

    Returns:
        MultimodalRetrievalDataset with text queries and image file paths.
    """
    import zipfile

    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    # Set up cache dir for extracted images
    if cache_dir is None:
        cache_dir = str(Path.home() / ".cache" / "khoji" / "flickr30k")
    images_dir = Path(cache_dir) / "images"

    # Download and extract images if not already cached
    if not images_dir.exists() or not any(images_dir.glob("*.jpg")):
        print("Downloading Flickr30k images (this may take a while)...")
        zip_path = hf_hub_download(
            "nlphuji/flickr30k", "flickr30k-images.zip", repo_type="dataset"
        )
        print(f"Extracting images to {images_dir}...")
        images_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract only .jpg files, flatten into images_dir
            for member in zf.namelist():
                if member.endswith(".jpg"):
                    filename = Path(member).name
                    target = images_dir / filename
                    if not target.exists():
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
        print(f"Extracted {len(list(images_dir.glob('*.jpg')))} images")

    # Load annotations CSV
    ds = load_dataset(
        "csv",
        data_files=f"hf://datasets/nlphuji/flickr30k/flickr_annotations_30k.csv",
        split="train",  # CSV loads as single split
    )

    # Filter to requested split
    ds = ds.filter(lambda row: row["split"] == split)

    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    for idx, row in enumerate(ds):
        filename = row["filename"]
        doc_id = f"img_{idx}"

        # Corpus entry points to local image file
        corpus[doc_id] = filename

        # Parse captions (stored as JSON string in CSV)
        captions_raw = row["raw"]
        try:
            captions = json.loads(captions_raw)
        except (json.JSONDecodeError, TypeError):
            captions = [str(captions_raw)]

        if isinstance(captions, str):
            captions = [captions]

        for cap_idx, caption in enumerate(captions):
            qid = f"q_{idx}_{cap_idx}"
            queries[qid] = caption
            qrels[qid] = {doc_id: 1}

    print(
        f"Loaded Flickr30k ({split}): "
        f"{len(queries)} queries, {len(corpus)} images"
    )

    return MultimodalRetrievalDataset(
        queries=queries, corpus=corpus, qrels=qrels, base_dir=str(images_dir)
    )


def load_rsicd(
    split: str = "test",
    n_samples: int | None = None,
    cache_dir: str | None = None,
) -> MultimodalRetrievalDataset:
    """Load RSICD (Remote Sensing Image Captioning Dataset) from HuggingFace.

    RSICD contains ~10k satellite/aerial images, each with 5 captions.
    This is a domain where CLIP has not been specifically trained,
    making it ideal for demonstrating domain adaptation via fine-tuning.

    Each caption becomes a query and its image becomes the relevant document.

    Args:
        split: Dataset split ("train", "test", or "valid").
        n_samples: Limit number of images. None = all in split.
        cache_dir: Where to cache extracted images. Defaults to
            ``~/.cache/khoji/rsicd``.

    Returns:
        MultimodalRetrievalDataset with text queries and image file paths.
    """
    from datasets import load_dataset

    if cache_dir is None:
        cache_dir = str(Path.home() / ".cache" / "khoji" / "rsicd")
    images_dir = Path(cache_dir) / "images" / split
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading RSICD ({split})...")
    ds = load_dataset("arampacha/rsicd", split=split)

    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    for idx, row in enumerate(ds):
        doc_id = f"img_{idx}"
        filename = row.get("filename", f"rsicd_{idx}.jpg")
        # Use just the basename
        basename = Path(filename).name
        if not basename:
            basename = f"rsicd_{idx}.jpg"

        # Save image to cache if not already there
        img_path = images_dir / basename
        if not img_path.exists():
            row["image"].save(img_path)

        corpus[doc_id] = basename

        # Each image has multiple captions
        captions = row.get("captions", [])
        if isinstance(captions, str):
            captions = [captions]

        for cap_idx, caption in enumerate(captions):
            qid = f"q_{idx}_{cap_idx}"
            queries[qid] = caption
            qrels[qid] = {doc_id: 1}

    print(
        f"Loaded RSICD ({split}): "
        f"{len(queries)} queries, {len(corpus)} images, "
        f"cached at {images_dir}"
    )

    return MultimodalRetrievalDataset(
        queries=queries, corpus=corpus, qrels=qrels, base_dir=str(images_dir)
    )


def load_custom_multimodal(path: str) -> MultimodalRetrievalDataset:
    """Load a custom multimodal dataset from a local directory.

    Expected directory structure::

        my_dataset/
            queries.jsonl   # {"_id": "q1", "text": "a dog playing"}
            corpus.jsonl    # {"_id": "d1", "image": "images/dog.jpg"}
            qrels.tsv       # q1\\td1\\t1 (tab-separated, no header)

    Image paths in corpus.jsonl are relative to the dataset directory.
    URLs (http/https) are also supported.

    Args:
        path: Path to the dataset directory.

    Returns:
        MultimodalRetrievalDataset with text queries and image sources.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    base = Path(path)

    # --- Queries ---
    queries_path = base / "queries.jsonl"
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    queries: dict[str, str] = {}
    with open(queries_path) as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]

    # --- Corpus (images) ---
    corpus_path = base / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

    corpus: dict[str, str] = {}
    with open(corpus_path) as f:
        for line in f:
            row = json.loads(line)
            corpus[str(row["_id"])] = row["image"]

    # --- Qrels ---
    qrels_path = base / "qrels.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"Missing qrels file: {qrels_path}")

    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_path) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            qid, did, score = str(row[0]), str(row[1]), int(row[2])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = score

    # Filter to queries with qrels and text
    queries = {qid: queries[qid] for qid in qrels if qid in queries}
    qrels = {qid: docs for qid, docs in qrels.items() if qid in queries}

    print(
        f"Loaded custom multimodal dataset from {base}: "
        f"{len(queries)} queries, {len(corpus)} images, "
        f"{sum(len(d) for d in qrels.values())} relevance judgments"
    )

    return MultimodalRetrievalDataset(
        queries=queries, corpus=corpus, qrels=qrels, base_dir=str(base)
    )
