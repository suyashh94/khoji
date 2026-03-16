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


def load_flickr30k(split: str = "test", n_samples: int | None = None) -> MultimodalRetrievalDataset:
    """Load Flickr30k from HuggingFace for text-to-image retrieval.

    Each image has 5 captions. We treat each caption as a separate query
    and the image as the relevant document.

    Args:
        split: Dataset split ("test" or "train").
        n_samples: Limit number of images to load. None = all.

    Returns:
        MultimodalRetrievalDataset with text queries and image paths.
    """
    from datasets import load_dataset

    ds = load_dataset("nlphuji/flickr30k", split=split)

    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    for idx, row in enumerate(ds):
        doc_id = f"img_{idx}"

        # Save image to a temp location (Flickr30k provides PIL images)
        img = row["image"]
        corpus[doc_id] = f"__hf_image__{idx}"  # placeholder, handled specially

        # Each image has multiple captions
        captions = row.get("caption", [])
        if isinstance(captions, str):
            captions = [captions]

        for cap_idx, caption in enumerate(captions):
            qid = f"q_{idx}_{cap_idx}"
            queries[qid] = caption
            qrels[qid] = {doc_id: 1}

    # Store the HF dataset reference for image access
    dataset = MultimodalRetrievalDataset(
        queries=queries, corpus=corpus, qrels=qrels
    )
    dataset._hf_dataset = ds  # type: ignore[attr-defined]

    print(
        f"Loaded Flickr30k ({split}): "
        f"{len(queries)} queries, {len(corpus)} images"
    )
    return dataset


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
