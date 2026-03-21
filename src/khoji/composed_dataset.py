"""Dataset loading for composed (image+text → image) retrieval."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ComposedRetrievalDataset:
    """Retrieval dataset for composed queries: (reference image + text) → target image.

    Each query is a pair of (reference_image, modification_text), and the
    goal is to retrieve the correct target image from the gallery.

    Construct directly for custom datasets::

        dataset = ComposedRetrievalDataset(
            queries={"q1": ("ref_img.jpg", "make it red")},
            corpus={"d1": "images/target.jpg", "d2": "images/other.jpg"},
            qrels={"q1": {"d1": 1}},
            base_dir="./my_dataset",
        )

    Args:
        queries: query_id -> (reference_image_source, modification_text).
        corpus: doc_id -> image source (local path or URL).
        qrels: query_id -> {doc_id -> relevance_score}.
        base_dir: Base directory for resolving relative image paths.
    """

    queries: dict[str, tuple[str, str]]  # query_id -> (image_source, text)
    corpus: dict[str, str]  # doc_id -> image source
    qrels: dict[str, dict[str, int]]
    base_dir: str | None = None


def load_custom_composed(path: str) -> ComposedRetrievalDataset:
    """Load a custom composed retrieval dataset from a local directory.

    Expected directory structure::

        my_dataset/
            queries.jsonl   # {"_id": "q1", "image": "imgs/ref.jpg", "text": "make it red"}
            corpus.jsonl    # {"_id": "d1", "image": "imgs/target.jpg"}
            qrels.tsv       # q1\\td1\\t1 (tab-separated, no header)

    Image paths in queries.jsonl and corpus.jsonl are relative to the
    dataset directory. URLs (http/https) are also supported.

    Args:
        path: Path to the dataset directory.

    Returns:
        ComposedRetrievalDataset.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    base = Path(path)

    # --- Queries ---
    queries_path = base / "queries.jsonl"
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    queries: dict[str, tuple[str, str]] = {}
    with open(queries_path) as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = (row["image"], row["text"])

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

    # Filter to queries with qrels
    queries = {qid: queries[qid] for qid in qrels if qid in queries}
    qrels = {qid: docs for qid, docs in qrels.items() if qid in queries}

    print(
        f"Loaded custom composed dataset from {base}: "
        f"{len(queries)} queries, {len(corpus)} gallery images, "
        f"{sum(len(d) for d in qrels.values())} relevance judgments"
    )

    return ComposedRetrievalDataset(
        queries=queries, corpus=corpus, qrels=qrels, base_dir=str(base)
    )
