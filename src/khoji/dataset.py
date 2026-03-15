"""Dataset loading for retrieval evaluation.

Loads datasets in BEIR format: queries, corpus, and qrels (relevance judgments).
Supports both HuggingFace BEIR datasets and custom local datasets.
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


@dataclass
class RetrievalDataset:
    """Standard retrieval dataset with queries, corpus, and relevance judgments.

    This is the core data structure for retriever-forge. You can construct it
    directly for custom datasets:

        dataset = RetrievalDataset(
            queries={"q1": "What is Python?", "q2": "How does GC work?"},
            corpus={"d1": "Python is a programming language.", "d2": "Garbage collection..."},
            qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
        )
    """

    queries: dict[str, str]  # query_id -> query_text
    corpus: dict[str, str]  # doc_id -> doc_text
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id -> relevance_score}


def _load_jsonl_gz(repo_id: str, filename: str) -> list[dict]:
    """Download and parse a gzipped JSONL file from a HuggingFace dataset repo."""
    path = hf_hub_download(repo_id, filename, repo_type="dataset")
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def load_beir(dataset_name: str, split: str = "test") -> RetrievalDataset:
    """Load a BEIR-format dataset from HuggingFace.

    Loads queries and corpus from raw JSONL files in the BeIR/{name} repo,
    and qrels from the BeIR/{name}-qrels dataset.

    Args:
        dataset_name: Name of the BEIR dataset (e.g. "fiqa").
        split: Which split to use for qrels ("train", "validation", "test").

    Returns:
        A RetrievalDataset with queries, corpus, and qrels populated.
    """
    repo_id = f"BeIR/{dataset_name}"

    # Load queries from raw JSONL
    queries_raw = _load_jsonl_gz(repo_id, "queries.jsonl.gz")
    all_queries = {str(row["_id"]): row["text"] for row in queries_raw}

    # Load corpus from raw JSONL
    corpus_raw = _load_jsonl_gz(repo_id, "corpus.jsonl.gz")
    corpus = {}
    for row in corpus_raw:
        text = row["text"]
        if row.get("title"):
            text = f"{row['title']} {text}"
        corpus[str(row["_id"])] = text

    # Load relevance judgments
    qrels_ds = load_dataset(f"{repo_id}-qrels", split=split)

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"]) # type: ignore
        did = str(row["corpus-id"]) # type: ignore
        score = int(row["score"]) # type: ignore
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][did] = score

    # Filter to only queries that appear in qrels
    queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}

    # Filter qrels to only queries we have text for
    qrels = {qid: docs for qid, docs in qrels.items() if qid in queries}

    return RetrievalDataset(queries=queries, corpus=corpus, qrels=qrels)


def load_custom(path: str) -> RetrievalDataset:
    """Load a custom dataset from a local directory.

    Expected directory structure::

        my_dataset/
            queries.jsonl      # Required: {"_id": "q1", "text": "query text"}
            corpus.jsonl       # Required: {"_id": "d1", "text": "doc text", "title": "optional"}
            qrels.tsv          # Required: query_id \\t corpus_id \\t score (tab-separated, no header)

    All three files are required. IDs are treated as strings.

    The qrels file uses tab-separated values with three columns:
    ``query_id``, ``corpus_id``, ``relevance_score``. No header row.

    Example qrels.tsv::

        q1	d1	1
        q1	d3	2
        q2	d2	1

    Args:
        path: Path to the directory containing queries.jsonl, corpus.jsonl, and qrels.tsv.

    Returns:
        A RetrievalDataset ready for training or evaluation.

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

    # --- Corpus ---
    corpus_path = base / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

    corpus: dict[str, str] = {}
    with open(corpus_path) as f:
        for line in f:
            row = json.loads(line)
            text = row["text"]
            if row.get("title"):
                text = f"{row['title']} {text}"
            corpus[str(row["_id"])] = text

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

    print(f"Loaded custom dataset from {base}: "
          f"{len(queries)} queries, {len(corpus)} docs, "
          f"{sum(len(d) for d in qrels.values())} relevance judgments")

    return RetrievalDataset(queries=queries, corpus=corpus, qrels=qrels)
