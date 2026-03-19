"""Training data preparation: triplet construction and hard negative mining."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from khoji.dataset import RetrievalDataset
from khoji.model import EmbeddingModel


@dataclass
class Triplet:
    """A single training triplet."""

    query: str
    positive: str
    negative: str


class TripletDataset(Dataset):
    """PyTorch Dataset of (query, positive, negative) triplets."""

    def __init__(self, triplets: list[Triplet]):
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        t = self.triplets[idx]
        return t.query, t.positive, t.negative


def _subset_dataset(
    dataset: RetrievalDataset,
    n_queries: int | None = None,
    corpus_size: int | None = None,
    seed: int = 42,
) -> RetrievalDataset:
    """Build a smaller version of a dataset for quick experiments.

    Selects a subset of queries and builds a corpus that includes all
    relevant docs for those queries plus random filler.
    """
    rng = random.Random(seed)

    # Subset queries
    query_ids = list(dataset.queries.keys())
    if n_queries is not None and n_queries < len(query_ids):
        query_ids = rng.sample(query_ids, n_queries)

    queries = {qid: dataset.queries[qid] for qid in query_ids}
    qrels = {qid: dataset.qrels[qid] for qid in query_ids if qid in dataset.qrels}

    # Subset corpus
    if corpus_size is not None and corpus_size < len(dataset.corpus):
        # Always include relevant docs
        relevant_ids: set[str] = set()
        for docs in qrels.values():
            relevant_ids.update(docs.keys())

        corpus = {did: dataset.corpus[did] for did in relevant_ids if did in dataset.corpus}

        # Fill with random non-relevant docs
        remaining = corpus_size - len(corpus)
        if remaining > 0:
            filler_ids = [did for did in dataset.corpus if did not in relevant_ids]
            if len(filler_ids) > remaining:
                filler_ids = rng.sample(filler_ids, remaining)
            for did in filler_ids:
                corpus[did] = dataset.corpus[did]
    else:
        corpus = dataset.corpus

    return RetrievalDataset(queries=queries, corpus=corpus, qrels=qrels)


def mine_hard_negatives(
    dataset: RetrievalDataset,
    model: EmbeddingModel,
    n_negatives: int = 1,
    top_k: int = 50,
    skip_top: int = 0,
    batch_size: int = 64,
    n_queries: int | None = None,
    corpus_size: int | None = None,
) -> list[Triplet]:
    """Build training triplets with hard negatives mined from the model.

    For each (query, positive_doc) pair, finds the top-k most similar corpus
    docs according to the model, filters out actually relevant ones, and
    picks the hardest remaining docs as negatives.

    Args:
        dataset: RetrievalDataset with queries, corpus, and qrels.
        model: Embedding model to use for mining.
        n_negatives: Number of hard negatives per (query, positive) pair.
        top_k: How many top corpus docs to consider when mining negatives.
        skip_top: Skip the top N most similar non-relevant docs before
            picking negatives. Useful when qrels are incomplete — the
            very top-ranked "negatives" are often unlabeled positives.
            E.g., skip_top=5 skips ranks 1-5 and picks from rank 6+.
        batch_size: Batch size for encoding.
        n_queries: Number of queries to use. None = all.
        corpus_size: Number of corpus docs to use. None = all. Relevant docs
            for selected queries are always included.

    Returns:
        List of Triplet(query, positive, negative).
    """
    if n_queries is not None or corpus_size is not None:
        dataset = _subset_dataset(dataset, n_queries, corpus_size)

    # Encode corpus
    corpus_ids = list(dataset.corpus.keys())
    corpus_texts = list(dataset.corpus.values())
    print(f"Encoding {len(corpus_texts)} corpus documents for negative mining...")
    corpus_embeddings = model.encode(corpus_texts, batch_size=batch_size)

    # Encode queries
    query_ids = list(dataset.queries.keys())
    query_texts = list(dataset.queries.values())
    print(f"Encoding {len(query_texts)} queries for negative mining...")
    query_embeddings = model.encode(query_texts, batch_size=batch_size)

    # Fetch enough candidates to account for skip_top + filtering
    fetch_k = top_k + skip_top

    if skip_top > 0:
        print(f"Hard negative mining: skipping top {skip_top}, "
              f"picking from ranks {skip_top + 1}-{fetch_k}")

    triplets: list[Triplet] = []

    for qi, qid in enumerate(query_ids):
        if qid not in dataset.qrels:
            continue

        query_text = query_texts[qi]
        qrel = dataset.qrels[qid]
        relevant_ids = set(qrel.keys())

        # Get top-(k + skip_top) similar corpus docs
        query_emb = query_embeddings[qi].unsqueeze(0)
        scores = torch.mm(query_emb, corpus_embeddings.t()).squeeze(0)
        topk_indices = torch.topk(
            scores, min(fetch_k, len(corpus_ids))
        ).indices.tolist()

        # Filter to non-relevant docs, then skip the top N
        hard_neg_ids = [
            corpus_ids[idx] for idx in topk_indices
            if corpus_ids[idx] not in relevant_ids
        ]
        hard_neg_ids = hard_neg_ids[skip_top:]

        if not hard_neg_ids:
            # Fallback: random negatives
            all_non_relevant = [
                cid for cid in corpus_ids if cid not in relevant_ids
            ]
            hard_neg_ids = random.sample(
                all_non_relevant, min(n_negatives, len(all_non_relevant))
            )

        # Create triplets: one per (positive, negative) pair
        for pos_id in relevant_ids:
            if pos_id not in dataset.corpus:
                continue
            pos_text = dataset.corpus[pos_id]
            for neg_id in hard_neg_ids[:n_negatives]:
                neg_text = dataset.corpus[neg_id]
                triplets.append(Triplet(
                    query=query_text, positive=pos_text, negative=neg_text
                ))

    print(f"Mined {len(triplets)} training triplets "
          f"({len(query_ids)} queries, {n_negatives} negatives each)")
    return triplets


def build_mixed_negatives(
    dataset: RetrievalDataset,
    model: EmbeddingModel,
    n_random: int = 1,
    n_hard: int = 1,
    top_k: int = 50,
    skip_top: int = 0,
    batch_size: int = 64,
    n_queries: int | None = None,
    corpus_size: int | None = None,
    seed: int = 42,
) -> list[Triplet]:
    """Build training triplets with a mix of random and hard negatives.

    For each (query, positive) pair, creates triplets with both random negatives
    (easy) and hard negatives (mined from model). This provides a curriculum-like
    signal: random negatives teach basic discrimination, hard negatives push for
    fine-grained ranking.

    Args:
        dataset: RetrievalDataset with queries, corpus, and qrels.
        model: Embedding model to use for hard negative mining.
        n_random: Number of random negatives per (query, positive) pair.
        n_hard: Number of hard negatives per (query, positive) pair.
        top_k: Top-k docs to consider for hard negative mining.
        skip_top: Skip top N non-relevant docs (avoids false negatives).
        batch_size: Batch size for encoding.
        n_queries: Number of queries to use. None = all.
        corpus_size: Corpus size for hard negative mining. None = full.
        seed: Random seed for random negative selection.

    Returns:
        List of Triplet(query, positive, negative) — contains both random and hard triplets.
    """
    # Mine hard negatives
    hard_triplets = mine_hard_negatives(
        dataset, model,
        n_negatives=n_hard,
        top_k=top_k,
        skip_top=skip_top,
        batch_size=batch_size,
        n_queries=n_queries,
        corpus_size=corpus_size,
    )

    # Build random negatives for the same queries
    random_triplets = build_random_negatives(
        dataset,
        n_negatives=n_random,
        n_queries=n_queries,
        seed=seed,
    )

    combined = hard_triplets + random_triplets
    random.Random(seed).shuffle(combined)

    print(f"Mixed negatives: {len(hard_triplets)} hard + {len(random_triplets)} random "
          f"= {len(combined)} total triplets")
    return combined


def build_random_negatives(
    dataset: RetrievalDataset,
    n_negatives: int = 1,
    n_queries: int | None = None,
    seed: int = 42,
) -> list[Triplet]:
    """Build training triplets with random negatives (no model needed).

    Faster alternative to hard negative mining for initial experiments.

    Args:
        dataset: RetrievalDataset with queries, corpus, and qrels.
        n_negatives: Number of random negatives per (query, positive) pair.
        n_queries: Number of queries to use. None = all.
        seed: Random seed.

    Returns:
        List of Triplet(query, positive, negative).
    """
    if n_queries is not None:
        dataset = _subset_dataset(dataset, n_queries, seed=seed)

    rng = random.Random(seed)
    corpus_ids = list(dataset.corpus.keys())
    triplets: list[Triplet] = []

    for qid, qrel in dataset.qrels.items():
        if qid not in dataset.queries:
            continue
        query_text = dataset.queries[qid]
        relevant_ids = set(qrel.keys())

        non_relevant = [cid for cid in corpus_ids if cid not in relevant_ids]

        for pos_id in relevant_ids:
            if pos_id not in dataset.corpus:
                continue
            pos_text = dataset.corpus[pos_id]
            neg_sample = rng.sample(non_relevant, min(n_negatives, len(non_relevant)))
            for neg_id in neg_sample:
                neg_text = dataset.corpus[neg_id]
                triplets.append(Triplet(query=query_text, positive=pos_text, negative=neg_text))

    print(f"Built {len(triplets)} training triplets with random negatives")
    return triplets
