"""Retrieval metrics: nDCG, MRR, Recall."""

from __future__ import annotations

import math


def _dcg(relevances: list[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def ndcg_at_k(
    ranked_doc_ids: list[str], qrel: dict[str, int], k: int
) -> float:
    """Compute nDCG@k for a single query.

    Args:
        ranked_doc_ids: Document IDs ranked by similarity (most similar first).
        qrel: Ground truth relevance: {doc_id: relevance_score}.
        k: Cutoff.

    Returns:
        nDCG@k score.
    """
    # Actual relevances in ranked order
    relevances = [qrel.get(doc_id, 0) for doc_id in ranked_doc_ids[:k]]
    dcg = _dcg(relevances, k)

    # Ideal: sort all relevant docs by relevance descending
    ideal_relevances = sorted(qrel.values(), reverse=True)
    idcg = _dcg(ideal_relevances, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr_at_k(
    ranked_doc_ids: list[str], qrel: dict[str, int], k: int
) -> float:
    """Compute MRR@k (Mean Reciprocal Rank) for a single query.

    Args:
        ranked_doc_ids: Document IDs ranked by similarity (most similar first).
        qrel: Ground truth relevance: {doc_id: relevance_score}.
        k: Cutoff.

    Returns:
        Reciprocal rank (1/rank of first relevant doc), or 0 if none in top-k.
    """
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        if qrel.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(
    ranked_doc_ids: list[str], qrel: dict[str, int], k: int
) -> float:
    """Compute Recall@k for a single query.

    Args:
        ranked_doc_ids: Document IDs ranked by similarity (most similar first).
        qrel: Ground truth relevance: {doc_id: relevance_score}.
        k: Cutoff.

    Returns:
        Fraction of relevant documents found in top-k.
    """
    relevant = {doc_id for doc_id, score in qrel.items() if score > 0}
    if not relevant:
        return 0.0
    retrieved_relevant = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant)
    return retrieved_relevant / len(relevant)
