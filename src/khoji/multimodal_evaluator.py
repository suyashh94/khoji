"""Evaluator for multimodal (text-to-image) retrieval."""

from __future__ import annotations

import random
from typing import Callable

import torch

from khoji.evaluator import EvalResult, _compute_metrics
from khoji.image_utils import load_images_batch
from khoji.multimodal_dataset import MultimodalRetrievalDataset, load_flickr30k
from khoji.multimodal_model import MultimodalEmbeddingModel


def _build_test_corpus_multimodal(
    dataset: MultimodalRetrievalDataset,
    query_ids: list[str],
    corpus_size: int,
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Build a smaller corpus for testing mode.

    Includes all relevant images for the selected queries, then fills
    the rest with random corpus images.
    """
    relevant_doc_ids: set[str] = set()
    qrels_subset: dict[str, dict[str, int]] = {}
    for qid in query_ids:
        if qid in dataset.qrels:
            qrels_subset[qid] = dataset.qrels[qid]
            relevant_doc_ids.update(dataset.qrels[qid].keys())

    subset_corpus = {did: dataset.corpus[did] for did in relevant_doc_ids if did in dataset.corpus}

    remaining = corpus_size - len(subset_corpus)
    if remaining > 0:
        filler_ids = [did for did in dataset.corpus if did not in relevant_doc_ids]
        if len(filler_ids) > remaining:
            random.seed(42)
            filler_ids = random.sample(filler_ids, remaining)
        for did in filler_ids:
            subset_corpus[did] = dataset.corpus[did]

    return subset_corpus, qrels_subset


class MultimodalEvaluator:
    """Evaluate a multimodal model on text-to-image retrieval.

    **HuggingFace models:**

        MultimodalEvaluator("openai/clip-vit-base-patch32")

    **Custom models:**

        model = MultimodalEmbeddingModel(
            text_model=my_text_enc, vision_model=my_vis_enc,
            tokenizer=tok, image_processor=proc,
        )
        MultimodalEvaluator(embedding_model=model)
    """

    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        embedding_model: MultimodalEmbeddingModel | None = None,
        max_length: int = 77,
        dtype: str | None = None,
    ):
        if embedding_model is not None:
            self.model = embedding_model
            self.model_name = model_name or "custom"
        elif model_name is not None:
            self.model = MultimodalEmbeddingModel(
                model_name, adapter_path=adapter_path, max_length=max_length, dtype=dtype
            )
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
        dataset: MultimodalRetrievalDataset | None = None,
        cache_dir: str | None = None,
        extra_metrics: dict[str, Callable[[list[str], dict[str, int], int], float]] | None = None,
    ) -> EvalResult:
        """Evaluate text-to-image retrieval.

        Encodes queries as text and corpus as images, then computes
        standard retrieval metrics (nDCG, MRR, Recall).

        Args:
            dataset_name: Dataset name (for loading). Ignored if dataset is provided.
            split: Dataset split.
            k_values: K values for metrics. Defaults to [1, 5, 10].
            batch_size: Batch size for encoding.
            n_queries: Number of queries. None = all.
            corpus_size: Corpus size limit. None = full.
            dataset: A MultimodalRetrievalDataset directly.
            cache_dir: Image cache directory.
            extra_metrics: Additional metric functions.

        Returns:
            EvalResult with metrics and metadata.
        """
        if k_values is None:
            k_values = [1, 5, 10]

        if dataset is None:
            if dataset_name is None:
                raise ValueError("Provide either dataset_name or dataset.")
            dataset = load_flickr30k(split=split)
        else:
            if dataset_name is None:
                dataset_name = "custom"

        # Select queries
        query_ids = list(dataset.queries.keys())
        if n_queries is not None and n_queries < len(query_ids):
            random.seed(42)
            query_ids = random.sample(query_ids, n_queries)

        # Build corpus
        if corpus_size is not None and corpus_size < len(dataset.corpus):
            corpus, qrels = _build_test_corpus_multimodal(dataset, query_ids, corpus_size)
        else:
            corpus = dataset.corpus
            qrels = dataset.qrels

        query_texts = [dataset.queries[qid] for qid in query_ids]
        corpus_ids = list(corpus.keys())
        corpus_sources = list(corpus.values())

        # Encode corpus images
        print(f"Encoding {len(corpus_sources)} corpus images...")
        corpus_embeddings = self.model.encode_image_sources(
            corpus_sources,
            base_dir=dataset.base_dir,
            cache_dir=cache_dir,
            batch_size=batch_size,
        )

        # Encode queries (text)
        print(f"Encoding {len(query_texts)} queries...")
        query_embeddings = self.model.encode_text(query_texts, batch_size=batch_size)

        # Compute metrics (reused from text evaluator)
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
