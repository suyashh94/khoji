"""Loss functions for retrieval model training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def triplet_margin_loss(
    query_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    negative_emb: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Triplet margin loss with cosine distance.

    Pushes the query closer to the positive and farther from the negative
    by at least `margin` in cosine distance.

    Args:
        query_emb: (batch, dim), L2-normalized.
        positive_emb: (batch, dim), L2-normalized.
        negative_emb: (batch, dim), L2-normalized.
        margin: Minimum margin between positive and negative distances.

    Returns:
        Scalar loss.
    """
    pos_dist = 1.0 - F.cosine_similarity(query_emb, positive_emb)
    neg_dist = 1.0 - F.cosine_similarity(query_emb, negative_emb)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def infonce_loss(
    query_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    negative_emb: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """InfoNCE (contrastive) loss.

    Treats the positive as the correct match and the negative plus all other
    in-batch positives as distractors. This gives (batch_size) negatives per
    query: 1 explicit hard negative + (batch_size - 1) in-batch negatives.

    Args:
        query_emb: (batch, dim), L2-normalized.
        positive_emb: (batch, dim), L2-normalized.
        negative_emb: (batch, dim), L2-normalized.
        temperature: Scaling temperature.

    Returns:
        Scalar loss.
    """
    # Negative scores from explicit hard negatives: (batch,)
    neg_scores = (query_emb * negative_emb).sum(dim=1) / temperature

    # In-batch negatives: each query against all other positives: (batch, batch)
    in_batch_scores = torch.mm(query_emb, positive_emb.t()) / temperature

    # Logits: positive score vs [in-batch scores + hard negative]
    # For each query i, the positive is at index i in in_batch_scores
    # We add the hard negative as an extra column
    logits = torch.cat([in_batch_scores, neg_scores.unsqueeze(1)], dim=1)  # (batch, batch+1)

    # Labels: the positive for query i is at index i
    labels = torch.arange(query_emb.size(0), device=query_emb.device)

    return F.cross_entropy(logits, labels)


def contrastive_loss(
    query_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    negative_emb: torch.Tensor,
) -> torch.Tensor:
    """Contrastive loss.

    Directly maximizes cosine similarity to the positive and minimizes
    cosine similarity to the negative.

    loss = -cos_sim(query, positive) + cos_sim(query, negative)

    Args:
        query_emb: (batch, dim), L2-normalized.
        positive_emb: (batch, dim), L2-normalized.
        negative_emb: (batch, dim), L2-normalized.

    Returns:
        Scalar loss.
    """
    pos_sim = F.cosine_similarity(query_emb, positive_emb)
    neg_sim = F.cosine_similarity(query_emb, negative_emb)
    loss = -pos_sim + neg_sim
    return loss.mean()
