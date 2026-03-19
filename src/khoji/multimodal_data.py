"""Training data preparation for multimodal (text-to-image) retrieval."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from khoji.multimodal_dataset import MultimodalRetrievalDataset


@dataclass
class MultimodalTriplet:
    """A training triplet: text query, positive image source, negative image source."""

    query: str  # text
    positive: str  # image path/URL
    negative: str  # image path/URL


class MultimodalTripletDataset(Dataset):
    """PyTorch Dataset of (text_query, positive_image, negative_image) triplets."""

    def __init__(self, triplets: list[MultimodalTriplet]):
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        t = self.triplets[idx]
        return t.query, t.positive, t.negative


def _subset_multimodal_dataset(
    dataset: MultimodalRetrievalDataset,
    n_queries: int | None = None,
    corpus_size: int | None = None,
    seed: int = 42,
) -> MultimodalRetrievalDataset:
    """Build a smaller version of a multimodal dataset."""
    rng = random.Random(seed)

    # Subset queries
    query_ids = list(dataset.queries.keys())
    if n_queries is not None and n_queries < len(query_ids):
        query_ids = rng.sample(query_ids, n_queries)

    queries = {qid: dataset.queries[qid] for qid in query_ids}
    qrels = {qid: dataset.qrels[qid] for qid in query_ids if qid in dataset.qrels}

    # Subset corpus
    if corpus_size is not None and corpus_size < len(dataset.corpus):
        relevant_ids: set[str] = set()
        for docs in qrels.values():
            relevant_ids.update(docs.keys())

        corpus = {did: dataset.corpus[did] for did in relevant_ids if did in dataset.corpus}

        remaining = corpus_size - len(corpus)
        if remaining > 0:
            filler_ids = [did for did in dataset.corpus if did not in relevant_ids]
            if len(filler_ids) > remaining:
                filler_ids = rng.sample(filler_ids, remaining)
            for did in filler_ids:
                corpus[did] = dataset.corpus[did]
    else:
        corpus = dataset.corpus

    return MultimodalRetrievalDataset(
        queries=queries, corpus=corpus, qrels=qrels, base_dir=dataset.base_dir
    )


def build_random_negatives_multimodal(
    dataset: MultimodalRetrievalDataset,
    n_negatives: int = 1,
    n_queries: int | None = None,
    seed: int = 42,
) -> list[MultimodalTriplet]:
    """Build training triplets with random negative images.

    Args:
        dataset: MultimodalRetrievalDataset with text queries and image corpus.
        n_negatives: Number of random negatives per (query, positive_image) pair.
        n_queries: Number of queries to use. None = all.
        seed: Random seed.

    Returns:
        List of MultimodalTriplet(query_text, positive_image, negative_image).
    """
    if n_queries is not None:
        dataset = _subset_multimodal_dataset(dataset, n_queries, seed=seed)

    rng = random.Random(seed)
    corpus_ids = list(dataset.corpus.keys())
    triplets: list[MultimodalTriplet] = []

    for qid, qrel in dataset.qrels.items():
        if qid not in dataset.queries:
            continue
        query_text = dataset.queries[qid]
        relevant_ids = set(qrel.keys())

        non_relevant = [cid for cid in corpus_ids if cid not in relevant_ids]

        for pos_id in relevant_ids:
            if pos_id not in dataset.corpus:
                continue
            pos_source = dataset.corpus[pos_id]
            neg_sample = rng.sample(non_relevant, min(n_negatives, len(non_relevant)))
            for neg_id in neg_sample:
                neg_source = dataset.corpus[neg_id]
                triplets.append(
                    MultimodalTriplet(query=query_text, positive=pos_source, negative=neg_source)
                )

    print(f"Built {len(triplets)} multimodal triplets with random negatives")
    return triplets


def mine_hard_negatives_multimodal(
    dataset: MultimodalRetrievalDataset,
    model,  # MultimodalEmbeddingModel
    n_negatives: int = 1,
    top_k: int = 50,
    skip_top: int = 0,
    batch_size: int = 64,
    n_queries: int | None = None,
    corpus_size: int | None = None,
    cache_dir: str | None = None,
) -> list[MultimodalTriplet]:
    """Build training triplets with hard negative images mined from the model.

    Encodes all queries (text) and corpus images, then finds the most
    similar non-relevant images as hard negatives.

    Args:
        dataset: MultimodalRetrievalDataset.
        model: MultimodalEmbeddingModel for encoding.
        n_negatives: Number of hard negatives per (query, positive_image) pair.
        top_k: Top-k corpus images to consider for mining.
        skip_top: Skip top N non-relevant images before picking negatives.
        batch_size: Batch size for encoding.
        n_queries: Number of queries to use. None = all.
        corpus_size: Corpus size limit. None = all.
        cache_dir: Image cache directory.

    Returns:
        List of MultimodalTriplet.
    """
    if n_queries is not None or corpus_size is not None:
        dataset = _subset_multimodal_dataset(dataset, n_queries, corpus_size)

    # Encode corpus images
    corpus_ids = list(dataset.corpus.keys())
    corpus_sources = list(dataset.corpus.values())
    print(f"Encoding {len(corpus_sources)} corpus images for negative mining...")
    corpus_embeddings = model.encode_image_sources(
        corpus_sources,
        base_dir=dataset.base_dir,
        cache_dir=cache_dir,
        batch_size=batch_size,
    )

    # Encode queries (text)
    query_ids = list(dataset.queries.keys())
    query_texts = list(dataset.queries.values())
    print(f"Encoding {len(query_texts)} queries for negative mining...")
    query_embeddings = model.encode_text(query_texts, batch_size=batch_size)

    fetch_k = top_k + skip_top

    if skip_top > 0:
        print(
            f"Hard negative mining: skipping top {skip_top}, "
            f"picking from ranks {skip_top + 1}-{fetch_k}"
        )

    triplets: list[MultimodalTriplet] = []

    for qi, qid in enumerate(query_ids):
        if qid not in dataset.qrels:
            continue

        query_text = query_texts[qi]
        qrel = dataset.qrels[qid]
        relevant_ids = set(qrel.keys())

        # Get top-(k + skip_top) similar corpus images
        query_emb = query_embeddings[qi].unsqueeze(0)
        scores = torch.mm(query_emb, corpus_embeddings.t()).squeeze(0)
        topk_indices = torch.topk(
            scores, min(fetch_k, len(corpus_ids))
        ).indices.tolist()

        # Filter to non-relevant, then skip top N
        hard_neg_ids = [
            corpus_ids[idx]
            for idx in topk_indices
            if corpus_ids[idx] not in relevant_ids
        ]
        hard_neg_ids = hard_neg_ids[skip_top:]

        if not hard_neg_ids:
            all_non_relevant = [cid for cid in corpus_ids if cid not in relevant_ids]
            hard_neg_ids = random.sample(
                all_non_relevant, min(n_negatives, len(all_non_relevant))
            )

        for pos_id in relevant_ids:
            if pos_id not in dataset.corpus:
                continue
            pos_source = dataset.corpus[pos_id]
            for neg_id in hard_neg_ids[:n_negatives]:
                neg_source = dataset.corpus[neg_id]
                triplets.append(
                    MultimodalTriplet(
                        query=query_text, positive=pos_source, negative=neg_source
                    )
                )

    print(
        f"Mined {len(triplets)} multimodal triplets "
        f"({len(query_ids)} queries, {n_negatives} negatives each)"
    )
    return triplets


def build_mixed_negatives_multimodal(
    dataset: MultimodalRetrievalDataset,
    model,  # MultimodalEmbeddingModel
    n_random: int = 1,
    n_hard: int = 1,
    top_k: int = 50,
    skip_top: int = 0,
    batch_size: int = 64,
    n_queries: int | None = None,
    corpus_size: int | None = None,
    cache_dir: str | None = None,
    seed: int = 42,
) -> list[MultimodalTriplet]:
    """Build multimodal triplets with a mix of random and hard negatives.

    Args:
        dataset: MultimodalRetrievalDataset.
        model: MultimodalEmbeddingModel for hard negative mining.
        n_random: Number of random negatives per pair.
        n_hard: Number of hard negatives per pair.
        top_k: Top-k for hard negative mining.
        skip_top: Skip top N non-relevant images.
        batch_size: Batch size for encoding.
        n_queries: Number of queries. None = all.
        corpus_size: Corpus size limit. None = all.
        cache_dir: Image cache directory.
        seed: Random seed.

    Returns:
        List of MultimodalTriplet — both random and hard.
    """
    hard_triplets = mine_hard_negatives_multimodal(
        dataset, model,
        n_negatives=n_hard,
        top_k=top_k,
        skip_top=skip_top,
        batch_size=batch_size,
        n_queries=n_queries,
        corpus_size=corpus_size,
        cache_dir=cache_dir,
    )

    random_triplets = build_random_negatives_multimodal(
        dataset,
        n_negatives=n_random,
        n_queries=n_queries,
        seed=seed,
    )

    combined = hard_triplets + random_triplets
    random.Random(seed).shuffle(combined)

    print(
        f"Mixed negatives: {len(hard_triplets)} hard + "
        f"{len(random_triplets)} random = {len(combined)} total"
    )
    return combined
