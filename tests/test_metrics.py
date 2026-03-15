"""Tests for retrieval metrics."""

import math

from khoji.metrics import mrr_at_k, ndcg_at_k, recall_at_k


class TestNDCG:
    def test_perfect_ranking(self):
        """All relevant docs at the top → nDCG = 1.0."""
        ranked = ["a", "b", "c", "d", "e"]
        qrel = {"a": 1, "b": 1}
        assert ndcg_at_k(ranked, qrel, 5) == 1.0

    def test_worst_ranking(self):
        """Relevant docs not in ranked list → nDCG = 0.0."""
        ranked = ["x", "y", "z"]
        qrel = {"a": 1, "b": 1}
        assert ndcg_at_k(ranked, qrel, 3) == 0.0

    def test_partial_ranking(self):
        """One relevant doc at position 2 (0-indexed)."""
        ranked = ["x", "y", "a", "b"]
        qrel = {"a": 1}
        # DCG = 1/log2(4) = 0.5, IDCG = 1/log2(2) = 1.0
        expected = (1 / math.log2(4)) / (1 / math.log2(2))
        assert abs(ndcg_at_k(ranked, qrel, 4) - expected) < 1e-9

    def test_graded_relevance(self):
        """Graded relevance scores (not just binary)."""
        ranked = ["a", "b", "c"]
        qrel = {"a": 2, "c": 1}
        # DCG = 2/log2(2) + 0/log2(3) + 1/log2(4) = 2.0 + 0 + 0.5
        dcg = 2 / math.log2(2) + 0 / math.log2(3) + 1 / math.log2(4)
        # IDCG = 2/log2(2) + 1/log2(3)
        idcg = 2 / math.log2(2) + 1 / math.log2(3)
        expected = dcg / idcg
        assert abs(ndcg_at_k(ranked, qrel, 3) - expected) < 1e-9

    def test_k_cutoff(self):
        """Only consider top-k results."""
        ranked = ["x", "y", "a"]
        qrel = {"a": 1}
        # At k=2, "a" is not included
        assert ndcg_at_k(ranked, qrel, 2) == 0.0
        # At k=3, "a" is included
        assert ndcg_at_k(ranked, qrel, 3) > 0.0

    def test_no_relevant_docs(self):
        """Empty qrel → nDCG = 0.0."""
        ranked = ["a", "b", "c"]
        qrel = {}
        assert ndcg_at_k(ranked, qrel, 3) == 0.0


class TestMRR:
    def test_first_position(self):
        """Relevant doc at rank 1 → MRR = 1.0."""
        ranked = ["a", "b", "c"]
        qrel = {"a": 1}
        assert mrr_at_k(ranked, qrel, 3) == 1.0

    def test_second_position(self):
        """First relevant doc at rank 2 → MRR = 0.5."""
        ranked = ["x", "a", "c"]
        qrel = {"a": 1, "c": 1}
        assert mrr_at_k(ranked, qrel, 3) == 0.5

    def test_not_in_top_k(self):
        """No relevant doc in top-k → MRR = 0.0."""
        ranked = ["x", "y", "z"]
        qrel = {"a": 1}
        assert mrr_at_k(ranked, qrel, 3) == 0.0

    def test_k_cutoff(self):
        """Relevant doc at position 3, but k=2 → MRR = 0.0."""
        ranked = ["x", "y", "a"]
        qrel = {"a": 1}
        assert mrr_at_k(ranked, qrel, 2) == 0.0
        assert mrr_at_k(ranked, qrel, 3) == pytest.approx(1 / 3)


class TestRecall:
    def test_all_found(self):
        """All relevant docs in top-k → Recall = 1.0."""
        ranked = ["a", "b", "c"]
        qrel = {"a": 1, "b": 1}
        assert recall_at_k(ranked, qrel, 3) == 1.0

    def test_none_found(self):
        """No relevant docs in top-k → Recall = 0.0."""
        ranked = ["x", "y", "z"]
        qrel = {"a": 1, "b": 1}
        assert recall_at_k(ranked, qrel, 3) == 0.0

    def test_partial(self):
        """1 of 3 relevant docs found → Recall = 1/3."""
        ranked = ["a", "x", "y"]
        qrel = {"a": 1, "b": 1, "c": 1}
        assert recall_at_k(ranked, qrel, 3) == pytest.approx(1 / 3)

    def test_k_cutoff(self):
        """Relevant doc outside k is not counted."""
        ranked = ["x", "y", "a"]
        qrel = {"a": 1}
        assert recall_at_k(ranked, qrel, 2) == 0.0
        assert recall_at_k(ranked, qrel, 3) == 1.0

    def test_ignores_zero_relevance(self):
        """Docs with relevance=0 in qrel are not considered relevant."""
        ranked = ["a", "b"]
        qrel = {"a": 0, "b": 1}
        assert recall_at_k(ranked, qrel, 2) == 1.0

    def test_no_relevant_docs(self):
        """Empty qrel → Recall = 0.0."""
        ranked = ["a", "b"]
        qrel = {}
        assert recall_at_k(ranked, qrel, 2) == 0.0


# Need pytest import for approx
import pytest
