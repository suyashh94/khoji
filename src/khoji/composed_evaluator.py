"""Evaluator for composed (image+text → image) retrieval."""

from __future__ import annotations

import random
from typing import Callable

import torch

from khoji.composed_dataset import ComposedRetrievalDataset
from khoji.evaluator import EvalResult, _compute_metrics
from khoji.image_utils import load_image
from khoji.multimodal_model import JointEmbeddingModel


def _build_test_corpus_composed(
    dataset: ComposedRetrievalDataset,
    query_ids: list[str],
    corpus_size: int,
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Build a smaller gallery for testing mode.

    Includes all relevant images for the selected queries, then fills
    the rest with random gallery images.
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


class ComposedEvaluator:
    """Evaluate a joint model on composed (image+text → image) retrieval.

    Queries are (reference_image, modification_text) pairs encoded jointly.
    The gallery is encoded as image-only embeddings.

    **HuggingFace BLIP-2:**

        ComposedEvaluator("Salesforce/blip2-itm-vit-g")

    **With adapter:**

        ComposedEvaluator(
            "Salesforce/blip2-itm-vit-g",
            adapter_path="./output/adapter"
        )

    **Custom models:**

        model = JointEmbeddingModel(encoder=my_fn)
        ComposedEvaluator(embedding_model=model)
    """

    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        embedding_model: JointEmbeddingModel | None = None,
        max_length: int = 77,
        dtype: str | None = None,
    ):
        if embedding_model is not None:
            self.model = embedding_model
            self.model_name = model_name or "custom"
        elif model_name is not None:
            self.model = JointEmbeddingModel(
                model_name, adapter_path=adapter_path, max_length=max_length, dtype=dtype
            )
            self.model_name = model_name
        else:
            raise ValueError("Provide either model_name or embedding_model.")
        self.adapter_path = adapter_path

    def evaluate(
        self,
        dataset_name: str | None = None,
        split: str = "val",
        k_values: list[int] | None = None,
        batch_size: int = 64,
        n_queries: int | None = None,
        corpus_size: int | None = None,
        dataset: ComposedRetrievalDataset | None = None,
        cache_dir: str | None = None,
        extra_metrics: dict[str, Callable[[list[str], dict[str, int], int], float]] | None = None,
    ) -> EvalResult:
        """Evaluate composed retrieval.

        Encodes queries as (image + text) jointly and gallery as images,
        then computes standard retrieval metrics (nDCG, MRR, Recall).

        Args:
            dataset_name: Dataset name (for metadata). Ignored if dataset is provided.
            split: Dataset split.
            k_values: K values for metrics. Defaults to [1, 5, 10, 50].
            batch_size: Batch size for encoding.
            n_queries: Number of queries. None = all.
            corpus_size: Gallery size limit. None = full.
            dataset: A ComposedRetrievalDataset directly.
            cache_dir: Image cache directory.
            extra_metrics: Additional metric functions.

        Returns:
            EvalResult with metrics and metadata.
        """
        if k_values is None:
            k_values = [1, 5, 10, 50]

        if dataset is None:
            raise ValueError("Provide a dataset for composed evaluation.")
        if dataset_name is None:
            dataset_name = "custom"

        # Select queries
        query_ids = list(dataset.queries.keys())
        if n_queries is not None and n_queries < len(query_ids):
            random.seed(42)
            query_ids = random.sample(query_ids, n_queries)

        # Build corpus
        if corpus_size is not None and corpus_size < len(dataset.corpus):
            corpus, qrels = _build_test_corpus_composed(dataset, query_ids, corpus_size)
        else:
            corpus = dataset.corpus
            qrels = dataset.qrels

        corpus_ids = list(corpus.keys())
        corpus_sources = list(corpus.values())

        # Encode gallery images
        print(f"Encoding {len(corpus_sources)} gallery images...")
        gallery_images = []
        valid_corpus_ids = []
        for cid, src in zip(corpus_ids, corpus_sources):
            img = load_image(src, base_dir=dataset.base_dir, cache_dir=cache_dir)
            if img is not None:
                gallery_images.append(img)
                valid_corpus_ids.append(cid)

        gallery_embeddings = self.model.encode(
            images=gallery_images, batch_size=batch_size
        )

        # Encode composed queries (image + text)
        print(f"Encoding {len(query_ids)} composed queries...")
        query_embeddings_list = []
        valid_query_ids = []
        for qid in query_ids:
            img_src, text = dataset.queries[qid]
            q_img = load_image(img_src, base_dir=dataset.base_dir, cache_dir=cache_dir)
            if q_img is None:
                continue
            q_emb = self.model.encode(
                images=[q_img], texts=[text], show_progress=False
            )
            query_embeddings_list.append(q_emb)
            valid_query_ids.append(qid)

        if not query_embeddings_list:
            print("Warning: No valid queries found.")
            return EvalResult(
                metrics={}, model_name=self.model_name,
                dataset_name=dataset_name, split=split,
                num_queries=0, num_corpus=len(valid_corpus_ids),
                k_values=k_values,
            )

        query_embeddings = torch.cat(query_embeddings_list, dim=0)

        # Compute metrics
        max_k = max(k_values)
        metrics = _compute_metrics(
            query_ids=valid_query_ids,
            query_embeddings=query_embeddings,
            corpus_ids=valid_corpus_ids,
            corpus_embeddings=gallery_embeddings,
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
            num_queries=len(valid_query_ids),
            num_corpus=len(valid_corpus_ids),
            k_values=k_values,
        )
