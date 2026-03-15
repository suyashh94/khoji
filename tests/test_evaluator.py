"""Tests for the evaluator."""

import pytest

from khoji.dataset import RetrievalDataset
from khoji.evaluator import Evaluator


@pytest.fixture(scope="module")
def small_dataset():
    """Tiny dataset for evaluator testing."""
    return RetrievalDataset(
        queries={
            "q1": "What is compound interest?",
            "q2": "How do stock dividends work?",
        },
        corpus={
            "d1": "Compound interest is interest calculated on the initial principal and accumulated interest.",
            "d2": "Dividends are payments made by a corporation to its shareholders.",
            "d3": "The weather today is sunny with clear skies.",
            "d4": "Index funds track a market index like the S&P 500.",
        },
        qrels={
            "q1": {"d1": 1},
            "q2": {"d2": 1},
        },
    )


class TestEvaluator:
    @pytest.fixture(scope="class")
    def evaluator(self):
        return Evaluator("BAAI/bge-base-en-v1.5")

    def test_evaluate_returns_eval_result(self, evaluator, small_dataset):
        result = evaluator.evaluate(
            dataset=small_dataset,
            k_values=[1, 3],
            batch_size=4,
        )
        assert "ndcg@1" in result.metrics
        assert "mrr@1" in result.metrics
        assert "recall@1" in result.metrics
        assert result.num_queries == 2
        assert result.num_corpus == 4

    def test_relevant_docs_rank_high(self, evaluator, small_dataset):
        """With such a small corpus, relevant docs should be easy to find."""
        result = evaluator.evaluate(
            dataset=small_dataset,
            k_values=[3],
            batch_size=4,
        )
        # Recall@3 out of 4 docs should be high for an obvious match
        assert result.metrics["recall@3"] >= 0.5

    def test_extra_metrics(self, evaluator, small_dataset):
        def precision_at_k(ranked_doc_ids, qrel, k):
            relevant = {d for d, s in qrel.items() if s > 0}
            found = sum(1 for d in ranked_doc_ids[:k] if d in relevant)
            return found / k

        result = evaluator.evaluate(
            dataset=small_dataset,
            k_values=[1, 3],
            batch_size=4,
            extra_metrics={"precision": precision_at_k},
        )
        assert "precision@1" in result.metrics
        assert "precision@3" in result.metrics

    def test_eval_result_save_and_to_dict(self, evaluator, small_dataset, tmp_path):
        result = evaluator.evaluate(
            dataset=small_dataset,
            k_values=[1],
            batch_size=4,
        )
        # to_dict
        d = result.to_dict()
        assert "metrics" in d
        assert "model_name" in d

        # save
        path = str(tmp_path / "result.json")
        result.save(path)
        import json
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["metrics"] == d["metrics"]
