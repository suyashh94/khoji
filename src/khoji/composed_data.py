"""Training data preparation for composed (image+text → image) retrieval."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from khoji.composed_dataset import ComposedRetrievalDataset


@dataclass
class ComposedTriplet:
    """A training triplet for composed retrieval.

    The query is a (reference_image, modification_text) pair.
    Positive and negative are target image sources.
    """

    query_image: str  # reference image path/URL
    query_text: str  # modification caption
    positive: str  # target image path/URL
    negative: str  # negative image path/URL


class ComposedTripletDataset(Dataset):
    """PyTorch Dataset of composed retrieval triplets."""

    def __init__(self, triplets: list[ComposedTriplet]):
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[str, str, str, str]:
        t = self.triplets[idx]
        return t.query_image, t.query_text, t.positive, t.negative


def _subset_composed_dataset(
    dataset: ComposedRetrievalDataset,
    n_queries: int | None = None,
    corpus_size: int | None = None,
    seed: int = 42,
) -> ComposedRetrievalDataset:
    """Build a smaller version of a composed dataset."""
    rng = random.Random(seed)

    query_ids = list(dataset.queries.keys())
    if n_queries is not None and n_queries < len(query_ids):
        query_ids = rng.sample(query_ids, n_queries)

    queries = {qid: dataset.queries[qid] for qid in query_ids}
    qrels = {qid: dataset.qrels[qid] for qid in query_ids if qid in dataset.qrels}

    if corpus_size is not None and corpus_size < len(dataset.corpus):
        relevant_ids: set[str] = set()
        for docs in qrels.values():
            relevant_ids.update(docs.keys())
        # Also include query reference images in relevant set
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

    return ComposedRetrievalDataset(
        queries=queries, corpus=corpus, qrels=qrels, base_dir=dataset.base_dir
    )


def build_random_negatives_composed(
    dataset: ComposedRetrievalDataset,
    n_negatives: int = 1,
    n_queries: int | None = None,
    seed: int = 42,
) -> list[ComposedTriplet]:
    """Build composed triplets with random negative images.

    Args:
        dataset: ComposedRetrievalDataset with (image, text) queries and image corpus.
        n_negatives: Number of random negatives per (query, positive) pair.
        n_queries: Number of queries to use. None = all.
        seed: Random seed.

    Returns:
        List of ComposedTriplet.
    """
    if n_queries is not None:
        dataset = _subset_composed_dataset(dataset, n_queries, seed=seed)

    rng = random.Random(seed)
    corpus_ids = list(dataset.corpus.keys())
    triplets: list[ComposedTriplet] = []

    for qid, qrel in dataset.qrels.items():
        if qid not in dataset.queries:
            continue
        query_image, query_text = dataset.queries[qid]
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
                    ComposedTriplet(
                        query_image=query_image,
                        query_text=query_text,
                        positive=pos_source,
                        negative=neg_source,
                    )
                )

    print(f"Built {len(triplets)} composed triplets with random negatives")
    return triplets


def mine_hard_negatives_composed(
    dataset: ComposedRetrievalDataset,
    model,  # JointEmbeddingModel
    n_negatives: int = 1,
    top_k: int = 50,
    skip_top: int = 0,
    batch_size: int = 64,
    n_queries: int | None = None,
    corpus_size: int | None = None,
    cache_dir: str | None = None,
) -> list[ComposedTriplet]:
    """Build composed triplets with hard negatives mined from the model.

    Encodes all gallery images and composed queries (image + text),
    then finds the most similar non-relevant gallery images as hard negatives.

    Args:
        dataset: ComposedRetrievalDataset.
        model: JointEmbeddingModel for encoding.
        n_negatives: Number of hard negatives per (query, positive) pair.
        top_k: Top-k gallery images to consider for mining.
        skip_top: Skip top N non-relevant images before picking negatives.
        batch_size: Batch size for encoding.
        n_queries: Number of queries to use. None = all.
        corpus_size: Gallery size limit. None = all.
        cache_dir: Image cache directory.

    Returns:
        List of ComposedTriplet.
    """
    from khoji.image_utils import load_image

    if n_queries is not None or corpus_size is not None:
        dataset = _subset_composed_dataset(dataset, n_queries, corpus_size)

    # Encode gallery images
    corpus_ids = list(dataset.corpus.keys())
    corpus_sources = list(dataset.corpus.values())
    print(f"Encoding {len(corpus_sources)} gallery images for negative mining...")

    gallery_images = []
    valid_corpus_ids = []
    for cid, src in zip(corpus_ids, corpus_sources):
        img = load_image(src, base_dir=dataset.base_dir, cache_dir=cache_dir)
        if img is not None:
            gallery_images.append(img)
            valid_corpus_ids.append(cid)

    gallery_embeddings = model.encode(
        images=gallery_images, batch_size=batch_size
    )

    fetch_k = top_k + skip_top

    if skip_top > 0:
        print(
            f"Hard negative mining: skipping top {skip_top}, "
            f"picking from ranks {skip_top + 1}-{fetch_k}"
        )

    triplets: list[ComposedTriplet] = []

    query_ids = list(dataset.queries.keys())
    for qid in query_ids:
        if qid not in dataset.qrels:
            continue

        query_image_src, query_text = dataset.queries[qid]
        qrel = dataset.qrels[qid]
        relevant_ids = set(qrel.keys())

        # Load query reference image
        q_img = load_image(query_image_src, base_dir=dataset.base_dir, cache_dir=cache_dir)
        if q_img is None:
            continue

        # Encode composed query (image + text)
        q_emb = model.encode(images=[q_img], texts=[query_text], show_progress=False)
        scores = torch.mm(q_emb, gallery_embeddings.t()).squeeze(0)
        topk_indices = torch.topk(
            scores, min(fetch_k, len(valid_corpus_ids))
        ).indices.tolist()

        # Filter to non-relevant, then skip top N
        hard_neg_ids = [
            valid_corpus_ids[idx]
            for idx in topk_indices
            if valid_corpus_ids[idx] not in relevant_ids
        ]
        hard_neg_ids = hard_neg_ids[skip_top:]

        if not hard_neg_ids:
            all_non_relevant = [cid for cid in valid_corpus_ids if cid not in relevant_ids]
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
                    ComposedTriplet(
                        query_image=query_image_src,
                        query_text=query_text,
                        positive=pos_source,
                        negative=neg_source,
                    )
                )

    print(
        f"Mined {len(triplets)} composed triplets "
        f"({len(query_ids)} queries, {n_negatives} negatives each)"
    )
    return triplets


def build_mixed_negatives_composed(
    dataset: ComposedRetrievalDataset,
    model,  # JointEmbeddingModel
    n_random: int = 1,
    n_hard: int = 1,
    top_k: int = 50,
    skip_top: int = 0,
    batch_size: int = 64,
    n_queries: int | None = None,
    corpus_size: int | None = None,
    cache_dir: str | None = None,
    seed: int = 42,
) -> list[ComposedTriplet]:
    """Build composed triplets with a mix of random and hard negatives.

    Args:
        dataset: ComposedRetrievalDataset.
        model: JointEmbeddingModel for hard negative mining.
        n_random: Number of random negatives per pair.
        n_hard: Number of hard negatives per pair.
        top_k: Top-k for hard negative mining.
        skip_top: Skip top N non-relevant images.
        batch_size: Batch size for encoding.
        n_queries: Number of queries. None = all.
        corpus_size: Gallery size limit. None = all.
        cache_dir: Image cache directory.
        seed: Random seed.

    Returns:
        List of ComposedTriplet — both random and hard.
    """
    hard_triplets = mine_hard_negatives_composed(
        dataset, model,
        n_negatives=n_hard,
        top_k=top_k,
        skip_top=skip_top,
        batch_size=batch_size,
        n_queries=n_queries,
        corpus_size=corpus_size,
        cache_dir=cache_dir,
    )

    random_triplets = build_random_negatives_composed(
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
