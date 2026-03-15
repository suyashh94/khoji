"""Evaluator: end-to-end retrieval evaluation pipeline."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import torch

from khoji.dataset import RetrievalDataset, load_beir
from khoji.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from khoji.model import EmbeddingModel


@dataclass
class EvalResult:
    """Structured evaluation result with metadata."""

    metrics: dict[str, float]
    model_name: str
    dataset_name: str
    split: str
    num_queries: int
    num_corpus: int
    k_values: list[int]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def print(self) -> None:
        """Pretty-print results as a table."""
        print(f"\n{'=' * 50}")
        print(f"Model:   {self.model_name}")
        print(f"Dataset: {self.dataset_name} ({self.split})")
        print(f"Queries: {self.num_queries} | Corpus: {self.num_corpus}")
        print(f"{'=' * 50}")

        # Header
        header = f"{'k':>5} | {'nDCG':>8} | {'MRR':>8} | {'Recall':>8}"
        print(header)
        print("-" * len(header))

        for k in self.k_values:
            ndcg = self.metrics.get(f"ndcg@{k}", 0.0)
            mrr = self.metrics.get(f"mrr@{k}", 0.0)
            recall = self.metrics.get(f"recall@{k}", 0.0)
            print(f"{k:>5} | {ndcg:>8.4f} | {mrr:>8.4f} | {recall:>8.4f}")

        print()

    def save(self, path: str) -> None:
        """Save results to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Results saved to {path}")

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "num_queries": self.num_queries,
            "num_corpus": self.num_corpus,
            "k_values": self.k_values,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }


def _build_test_corpus(
    dataset: RetrievalDataset,
    query_ids: list[str],
    corpus_size: int,
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Build a smaller corpus for testing mode.

    Includes all relevant docs for the selected queries, then fills
    the rest with random corpus docs. This ensures metrics are meaningful.

    Returns:
        (subset_corpus, filtered_qrels)
    """
    # Collect all relevant doc IDs for selected queries
    relevant_doc_ids: set[str] = set()
    qrels_subset: dict[str, dict[str, int]] = {}
    for qid in query_ids:
        if qid in dataset.qrels:
            qrels_subset[qid] = dataset.qrels[qid]
            relevant_doc_ids.update(dataset.qrels[qid].keys())

    # Start with all relevant docs
    subset_corpus = {did: dataset.corpus[did] for did in relevant_doc_ids if did in dataset.corpus}

    # Fill remaining slots with random non-relevant docs
    remaining = corpus_size - len(subset_corpus)
    if remaining > 0:
        filler_ids = [did for did in dataset.corpus if did not in relevant_doc_ids]
        if len(filler_ids) > remaining:
            random.seed(42)
            filler_ids = random.sample(filler_ids, remaining)
        for did in filler_ids:
            subset_corpus[did] = dataset.corpus[did]

    return subset_corpus, qrels_subset


class Evaluator:
    """Evaluate an embedding model on a retrieval dataset.

    **HuggingFace models:**

        Evaluator("BAAI/bge-base-en-v1.5", adapter_path="./adapter")

    **Custom models** — pass an ``EmbeddingModel`` you've already built:

        model = EmbeddingModel(model=my_encoder, tokenizer=my_tok, pooling="mean")
        Evaluator(embedding_model=model)
    """

    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        embedding_model: EmbeddingModel | None = None,
        max_length: int = 512,
        dtype: str | None = None,
    ):
        if embedding_model is not None:
            self.model = embedding_model
            self.model_name = model_name or "custom"
        elif model_name is not None:
            self.model = EmbeddingModel(model_name, adapter_path=adapter_path, max_length=max_length, dtype=dtype)
            self.model_name = model_name
        else:
            raise ValueError("Provide either model_name or embedding_model.")
        self.adapter_path = adapter_path

    def evaluate(
        self,
        dataset_name: str | None = None,
        split: str = "test",
        k_values: list[int] | None = None,
        batch_size: int = 64,
        n_queries: int | None = None,
        corpus_size: int | None = None,
        dataset: RetrievalDataset | None = None,
        extra_metrics: dict[str, Callable[[list[str], dict[str, int], int], float]] | None = None,
    ) -> EvalResult:
        """Evaluate the model on a retrieval dataset.

        Provide either ``dataset_name`` to load a BEIR dataset from HuggingFace,
        or ``dataset`` to use your own RetrievalDataset directly.

        Args:
            dataset_name: BEIR dataset name (e.g. "fiqa"). Ignored if ``dataset``
                is provided.
            split: Dataset split for qrels (only used with ``dataset_name``).
            k_values: List of k values to compute metrics at. Defaults to [1, 5, 10].
            batch_size: Batch size for encoding.
            n_queries: Number of queries to evaluate. None = all queries.
            corpus_size: Size of corpus to use. None = full corpus. When set,
                builds a smaller corpus that includes all relevant docs for the
                selected queries plus random filler docs.
            dataset: A RetrievalDataset to evaluate on directly. When provided,
                ``dataset_name`` and ``split`` are ignored for loading.
            extra_metrics: Additional metric functions to compute. Dict mapping
                metric name to a function with signature
                ``(ranked_doc_ids: list[str], qrel: dict[str, int], k: int) -> float``.
                These are computed alongside the built-in nDCG, MRR, and Recall.
                Example::

                    def precision_at_k(ranked_doc_ids, qrel, k):
                        relevant = {d for d, s in qrel.items() if s > 0}
                        found = sum(1 for d in ranked_doc_ids[:k] if d in relevant)
                        return found / k

                    evaluator.evaluate(..., extra_metrics={"precision": precision_at_k})

        Returns:
            EvalResult with metrics and metadata.
        """
        if k_values is None:
            k_values = [1, 5, 10]

        # Load dataset
        if dataset is None:
            if dataset_name is None:
                raise ValueError("Provide either dataset_name or dataset.")
            dataset = load_beir(dataset_name, split=split)
        else:
            if dataset_name is None:
                dataset_name = "custom"

        # Select queries
        query_ids = list(dataset.queries.keys())
        if n_queries is not None and n_queries < len(query_ids):
            random.seed(42)
            query_ids = random.sample(query_ids, n_queries)

        # Build corpus (full or subset)
        if corpus_size is not None and corpus_size < len(dataset.corpus):
            corpus, qrels = _build_test_corpus(dataset, query_ids, corpus_size)
        else:
            corpus = dataset.corpus
            qrels = dataset.qrels

        query_texts = [dataset.queries[qid] for qid in query_ids]
        corpus_ids = list(corpus.keys())
        corpus_texts = list(corpus.values())

        print(f"Encoding {len(corpus_texts)} corpus documents...")
        corpus_embeddings = self.model.encode(corpus_texts, batch_size=batch_size)

        print(f"Encoding {len(query_texts)} queries...")
        query_embeddings = self.model.encode(query_texts, batch_size=batch_size)

        # Compute similarity and rank
        max_k = max(k_values)
        metrics = _compute_metrics(
            query_ids=query_ids,
            query_embeddings=query_embeddings,
            corpus_ids=corpus_ids,
            corpus_embeddings=corpus_embeddings,
            qrels=qrels,
            k_values=k_values,
            max_k=max_k,
            extra_metrics=extra_metrics,
        )

        return EvalResult(
            metrics=metrics,
            model_name=self.model_name,
            dataset_name=dataset_name,
            split=split,
            num_queries=len(query_ids),
            num_corpus=len(corpus_ids),
            k_values=k_values,
        )


def _compute_metrics(
    query_ids: list[str],
    query_embeddings: torch.Tensor,
    corpus_ids: list[str],
    corpus_embeddings: torch.Tensor,
    qrels: dict[str, dict[str, int]],
    k_values: list[int],
    max_k: int,
    extra_metrics: dict[str, Callable[[list[str], dict[str, int], int], float]] | None = None,
) -> dict[str, float]:
    """Compute retrieval metrics over all queries."""
    metric_sums: dict[str, float] = {}
    for k in k_values:
        metric_sums[f"ndcg@{k}"] = 0.0
        metric_sums[f"mrr@{k}"] = 0.0
        metric_sums[f"recall@{k}"] = 0.0
        if extra_metrics:
            for name in extra_metrics:
                metric_sums[f"{name}@{k}"] = 0.0

    num_queries = 0

    for i, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        num_queries += 1

        # Cosine similarity (embeddings are already L2-normalized)
        query_emb = query_embeddings[i].unsqueeze(0)  # (1, dim)
        scores = torch.mm(query_emb, corpus_embeddings.t()).squeeze(0)  # (num_corpus,)

        # Get top-k indices
        topk_indices = torch.topk(scores, min(max_k, len(corpus_ids))).indices.tolist()
        ranked_doc_ids = [corpus_ids[idx] for idx in topk_indices]

        qrel = qrels[qid]
        for k in k_values:
            metric_sums[f"ndcg@{k}"] += ndcg_at_k(ranked_doc_ids, qrel, k)
            metric_sums[f"mrr@{k}"] += mrr_at_k(ranked_doc_ids, qrel, k)
            metric_sums[f"recall@{k}"] += recall_at_k(ranked_doc_ids, qrel, k)
            if extra_metrics:
                for name, fn in extra_metrics.items():
                    metric_sums[f"{name}@{k}"] += fn(ranked_doc_ids, qrel, k)

    # Average
    results = {}
    for metric, total in metric_sums.items():
        results[metric] = round(total / max(num_queries, 1), 4)

    return results
