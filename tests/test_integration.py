"""Integration tests: dataset loading + model encoding + retrieval sanity checks."""

import random

import torch
import pytest

from khoji.dataset import load_beir
from khoji.model import EmbeddingModel


@pytest.fixture(scope="module")
def fiqa_dataset():
    return load_beir("fiqa", split="test")


@pytest.fixture(scope="module")
def model():
    return EmbeddingModel("BAAI/bge-base-en-v1.5")


class TestDatasetLoading:
    def test_has_queries(self, fiqa_dataset):
        assert len(fiqa_dataset.queries) > 0

    def test_has_corpus(self, fiqa_dataset):
        assert len(fiqa_dataset.corpus) > 0

    def test_has_qrels(self, fiqa_dataset):
        assert len(fiqa_dataset.qrels) > 0

    def test_qrel_queries_have_text(self, fiqa_dataset):
        """Every query in qrels should have corresponding text."""
        for qid in fiqa_dataset.qrels:
            assert qid in fiqa_dataset.queries

    def test_qrel_docs_exist_in_corpus(self, fiqa_dataset):
        """Every doc referenced in qrels should exist in corpus."""
        for qid, docs in fiqa_dataset.qrels.items():
            for doc_id in docs:
                assert doc_id in fiqa_dataset.corpus, (
                    f"Doc {doc_id} from qrels not found in corpus"
                )


class TestRetrievalSanity:
    def test_relevant_doc_scores_higher_than_random(self, fiqa_dataset, model):
        """A relevant doc should score higher than a random irrelevant doc."""
        # Pick a query that has at least one relevant doc
        qid = next(iter(fiqa_dataset.qrels))
        query_text = fiqa_dataset.queries[qid]
        relevant_doc_id = next(iter(fiqa_dataset.qrels[qid]))
        relevant_doc_text = fiqa_dataset.corpus[relevant_doc_id]

        # Pick a random doc that is NOT relevant to this query
        relevant_ids = set(fiqa_dataset.qrels[qid].keys())
        irrelevant_ids = [
            did for did in fiqa_dataset.corpus if did not in relevant_ids
        ]
        random.seed(42)
        irrelevant_doc_id = random.choice(irrelevant_ids)
        irrelevant_doc_text = fiqa_dataset.corpus[irrelevant_doc_id]

        # Encode all three
        embeddings = model.encode(
            [query_text, relevant_doc_text, irrelevant_doc_text],
            show_progress=False,
        )
        query_emb = embeddings[0]
        relevant_emb = embeddings[1]
        irrelevant_emb = embeddings[2]

        relevant_score = torch.dot(query_emb, relevant_emb).item()
        irrelevant_score = torch.dot(query_emb, irrelevant_emb).item()

        assert relevant_score > irrelevant_score, (
            f"Relevant doc scored {relevant_score:.4f} vs "
            f"irrelevant {irrelevant_score:.4f}"
        )

    def test_batch_of_queries_vs_their_relevant_docs(self, fiqa_dataset, model):
        """For 5 queries, the relevant doc should be in top-50% when scored
        against a small pool of 20 random docs + the relevant doc."""
        random.seed(42)
        query_ids = list(fiqa_dataset.qrels.keys())[:5]
        passed = 0

        for qid in query_ids:
            query_text = fiqa_dataset.queries[qid]
            relevant_doc_id = next(iter(fiqa_dataset.qrels[qid]))
            relevant_doc_text = fiqa_dataset.corpus[relevant_doc_id]

            # Build a small pool: relevant doc + 20 random docs
            relevant_ids = set(fiqa_dataset.qrels[qid].keys())
            pool_ids = random.sample(
                [d for d in fiqa_dataset.corpus if d not in relevant_ids],
                20,
            )
            pool_texts = [fiqa_dataset.corpus[d] for d in pool_ids]
            pool_texts.insert(0, relevant_doc_text)  # index 0 is the relevant doc

            embeddings = model.encode(
                [query_text] + pool_texts, show_progress=False
            )
            query_emb = embeddings[0]
            doc_embs = embeddings[1:]

            scores = torch.mv(doc_embs, query_emb)
            rank = (scores >= scores[0]).sum().item()  # rank of relevant doc (1-based)

            if rank <= len(pool_texts) // 2:
                passed += 1

        assert passed >= 3, f"Only {passed}/5 queries had relevant doc in top half"
