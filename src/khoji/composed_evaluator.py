"""Evaluator for composed retrieval (mixed-mode)."""

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
) -> tuple[dict[str, tuple[str, str]], dict[str, dict[str, int]]]:
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
    """Evaluate a joint model on mixed-mode retrieval.

    Queries and corpus items can each be image-only, text-only, or
    image+text. Encoding mode is dispatched per item based on which
    modalities are present.

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

    def _encode_items(
        self,
        items: dict[str, tuple[str, str]],
        base_dir: str | None,
        cache_dir: str | None,
        batch_size: int,
        label: str = "items",
    ) -> tuple[torch.Tensor, list[str]]:
        """Encode mixed-mode items by grouping them by modality for efficient batching.

        Returns (embeddings, valid_ids) with failed items skipped.
        """
        # Load images and classify items by mode
        img_only_ids, img_only_imgs = [], []
        txt_only_ids, txt_only_txts = [], []
        joint_ids, joint_imgs, joint_txts = [], [], []

        for item_id, (img_src, txt) in items.items():
            img = load_image(img_src, base_dir=base_dir, cache_dir=cache_dir) if img_src != "" else None
            if img_src != "" and img is None:
                continue  # failed to load
            has_img = img is not None
            has_txt = txt != ""
            if has_img and has_txt:
                joint_ids.append(item_id)
                joint_imgs.append(img)
                joint_txts.append(txt)
            elif has_img:
                img_only_ids.append(item_id)
                img_only_imgs.append(img)
            elif has_txt:
                txt_only_ids.append(item_id)
                txt_only_txts.append(txt)

        total = len(img_only_ids) + len(txt_only_ids) + len(joint_ids)
        print(f"Encoding {total} {label} (img={len(img_only_ids)}, txt={len(txt_only_ids)}, joint={len(joint_ids)})...")

        all_embs = []
        all_ids = []

        if img_only_imgs:
            emb = self.model.encode(images=img_only_imgs, batch_size=batch_size)
            all_embs.append(emb)
            all_ids.extend(img_only_ids)

        if txt_only_txts:
            emb = self.model.encode(texts=txt_only_txts, batch_size=batch_size)
            all_embs.append(emb)
            all_ids.extend(txt_only_ids)

        if joint_imgs:
            emb = self.model.encode(images=joint_imgs, texts=joint_txts, batch_size=batch_size)
            all_embs.append(emb)
            all_ids.extend(joint_ids)

        if not all_embs:
            return torch.empty(0), []

        return torch.cat(all_embs, dim=0), all_ids

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

        # Encode corpus items — group by mode for efficient batching
        gallery_embeddings, valid_corpus_ids = self._encode_items(
            {cid: corpus[cid] for cid in corpus_ids},
            base_dir=dataset.base_dir, cache_dir=cache_dir,
            batch_size=batch_size, label="corpus",
        )

        # Encode queries — group by mode for efficient batching
        query_embeddings, valid_query_ids = self._encode_items(
            {qid: dataset.queries[qid] for qid in query_ids},
            base_dir=dataset.base_dir, cache_dir=cache_dir,
            batch_size=batch_size, label="queries",
        )

        if len(valid_query_ids) == 0:
            print("Warning: No valid queries found.")
            return EvalResult(
                metrics={}, model_name=self.model_name,
                dataset_name=dataset_name, split=split,
                num_queries=0, num_corpus=len(valid_corpus_ids),
                k_values=k_values,
            )

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
