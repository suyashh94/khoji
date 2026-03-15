"""Tests for training data preparation."""

import random

import pytest

from khoji.data import (
    Triplet,
    TripletDataset,
    build_random_negatives,
    mine_hard_negatives,
)
from khoji.dataset import RetrievalDataset


@pytest.fixture(scope="module")
def small_dataset():
    """A small synthetic dataset for fast tests."""
    queries = {
        "q1": "What is compound interest?",
        "q2": "How do stock dividends work?",
        "q3": "Best way to save for retirement?",
    }
    corpus = {
        "d1": "Compound interest is interest calculated on the initial principal and accumulated interest.",
        "d2": "Dividends are payments made by a corporation to its shareholders from profits.",
        "d3": "A 401k is a retirement savings plan sponsored by an employer.",
        "d4": "The stock market saw gains today driven by tech sector rally.",
        "d5": "Mortgage rates have risen significantly over the past year.",
        "d6": "Index funds are a type of mutual fund that tracks a market index.",
        "d7": "Credit scores range from 300 to 850 in the FICO model.",
        "d8": "Tax-loss harvesting is a strategy to offset capital gains.",
    }
    qrels = {
        "q1": {"d1": 1},
        "q2": {"d2": 1},
        "q3": {"d3": 1, "d6": 1},
    }
    return RetrievalDataset(queries=queries, corpus=corpus, qrels=qrels)


@pytest.fixture(scope="module")
def model():
    from khoji.model import EmbeddingModel
    return EmbeddingModel("BAAI/bge-base-en-v1.5")


class TestTripletDataset:
    def test_len(self):
        triplets = [Triplet("q", "p", "n"), Triplet("q2", "p2", "n2")]
        ds = TripletDataset(triplets)
        assert len(ds) == 2

    def test_getitem_returns_three_strings(self):
        ds = TripletDataset([Triplet("query", "pos", "neg")])
        q, p, n = ds[0]
        assert q == "query"
        assert p == "pos"
        assert n == "neg"


class TestRandomNegatives:
    def test_produces_triplets(self, small_dataset):
        triplets = build_random_negatives(small_dataset, n_negatives=1)
        assert len(triplets) > 0

    def test_triplet_count_matches_positives(self, small_dataset):
        """One negative per positive → one triplet per positive doc."""
        triplets = build_random_negatives(small_dataset, n_negatives=1)
        # q1 has 1 positive, q2 has 1, q3 has 2 → 4 triplets
        assert len(triplets) == 4

    def test_multiple_negatives(self, small_dataset):
        triplets = build_random_negatives(small_dataset, n_negatives=3)
        # 4 positives * 3 negatives = 12 triplets
        assert len(triplets) == 12

    def test_negatives_are_not_relevant(self, small_dataset):
        triplets = build_random_negatives(small_dataset, n_negatives=2)
        for t in triplets:
            # Find which query this is
            for qid, qtext in small_dataset.queries.items():
                if qtext == t.query:
                    relevant_texts = {
                        small_dataset.corpus[did]
                        for did in small_dataset.qrels[qid]
                    }
                    assert t.negative not in relevant_texts, (
                        f"Negative should not be a relevant doc for query {qid}"
                    )

    def test_positives_are_relevant(self, small_dataset):
        triplets = build_random_negatives(small_dataset, n_negatives=1)
        for t in triplets:
            for qid, qtext in small_dataset.queries.items():
                if qtext == t.query:
                    relevant_texts = {
                        small_dataset.corpus[did]
                        for did in small_dataset.qrels[qid]
                    }
                    assert t.positive in relevant_texts

    def test_deterministic_with_seed(self, small_dataset):
        t1 = build_random_negatives(small_dataset, n_negatives=2, seed=42)
        t2 = build_random_negatives(small_dataset, n_negatives=2, seed=42)
        assert [(t.query, t.positive, t.negative) for t in t1] == \
               [(t.query, t.positive, t.negative) for t in t2]

    def test_wraps_in_torch_dataset(self, small_dataset):
        triplets = build_random_negatives(small_dataset, n_negatives=1)
        ds = TripletDataset(triplets)
        assert len(ds) == len(triplets)
        q, p, n = ds[0]
        assert isinstance(q, str)
        assert isinstance(p, str)
        assert isinstance(n, str)


class TestHardNegativeMining:
    def test_produces_triplets(self, small_dataset, model):
        triplets = mine_hard_negatives(
            small_dataset, model, n_negatives=1, top_k=5
        )
        assert len(triplets) > 0

    def test_triplet_count(self, small_dataset, model):
        triplets = mine_hard_negatives(
            small_dataset, model, n_negatives=1, top_k=5
        )
        # 4 positives * 1 negative = 4 triplets
        assert len(triplets) == 4

    def test_negatives_are_not_relevant(self, small_dataset, model):
        triplets = mine_hard_negatives(
            small_dataset, model, n_negatives=2, top_k=5
        )
        for t in triplets:
            for qid, qtext in small_dataset.queries.items():
                if qtext == t.query:
                    relevant_texts = {
                        small_dataset.corpus[did]
                        for did in small_dataset.qrels[qid]
                    }
                    assert t.negative not in relevant_texts

    def test_negatives_come_from_corpus(self, small_dataset, model):
        triplets = mine_hard_negatives(
            small_dataset, model, n_negatives=1, top_k=5
        )
        corpus_texts = set(small_dataset.corpus.values())
        for t in triplets:
            assert t.negative in corpus_texts

    def test_hard_negatives_are_semantically_closer(self, small_dataset, model):
        """Hard negatives should on average score higher than random negatives."""
        import torch

        hard_triplets = mine_hard_negatives(
            small_dataset, model, n_negatives=2, top_k=5
        )
        random_triplets = build_random_negatives(
            small_dataset, n_negatives=2
        )

        def avg_neg_similarity(triplets_list):
            scores = []
            for t in triplets_list:
                embs = model.encode([t.query, t.negative], show_progress=False)
                score = torch.dot(embs[0], embs[1]).item()
                scores.append(score)
            return sum(scores) / len(scores)

        hard_avg = avg_neg_similarity(hard_triplets)
        random_avg = avg_neg_similarity(random_triplets)
        # Hard negatives should be more similar to query (harder to distinguish)
        assert hard_avg >= random_avg, (
            f"Hard negatives ({hard_avg:.4f}) should score >= random ({random_avg:.4f})"
        )
