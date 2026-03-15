"""Tests for loss functions."""

import torch
import pytest

from khoji.loss import triplet_margin_loss, infonce_loss, contrastive_loss


@pytest.fixture
def identical_embeddings():
    """Query = positive, orthogonal negative."""
    query = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    negative = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    return query, positive, negative


@pytest.fixture
def hard_case():
    """Positive and negative are both close to query."""
    query = torch.nn.functional.normalize(torch.tensor([[1.0, 0.1, 0.0]]), dim=1)
    positive = torch.nn.functional.normalize(torch.tensor([[1.0, 0.2, 0.0]]), dim=1)
    negative = torch.nn.functional.normalize(torch.tensor([[1.0, 0.3, 0.0]]), dim=1)
    return query, positive, negative


class TestTripletMarginLoss:
    def test_returns_scalar(self, identical_embeddings):
        q, p, n = identical_embeddings
        loss = triplet_margin_loss(q, p, n)
        assert loss.shape == ()

    def test_zero_loss_when_well_separated(self, identical_embeddings):
        """cos_sim(q,p)=1, cos_sim(q,n)=0 → pos_dist=0, neg_dist=1 → relu(0-1+0.2) = 0."""
        q, p, n = identical_embeddings
        loss = triplet_margin_loss(q, p, n, margin=0.2)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_when_close(self, hard_case):
        """When positive and negative are close, loss should be > 0 with large margin."""
        q, p, n = hard_case
        loss = triplet_margin_loss(q, p, n, margin=0.5)
        assert loss.item() > 0

    def test_larger_margin_increases_loss(self, hard_case):
        q, p, n = hard_case
        loss_small = triplet_margin_loss(q, p, n, margin=0.1)
        loss_large = triplet_margin_loss(q, p, n, margin=0.5)
        assert loss_large.item() >= loss_small.item()

    def test_swapped_pos_neg_increases_loss(self, identical_embeddings):
        q, p, n = identical_embeddings
        normal_loss = triplet_margin_loss(q, p, n)
        swapped_loss = triplet_margin_loss(q, n, p)  # negative as positive
        assert swapped_loss.item() >= normal_loss.item()


class TestInfoNCELoss:
    def test_returns_scalar(self, identical_embeddings):
        q, p, n = identical_embeddings
        loss = infonce_loss(q, p, n)
        assert loss.shape == ()

    def test_low_loss_for_obvious_positives(self, identical_embeddings):
        q, p, n = identical_embeddings
        loss = infonce_loss(q, p, n, temperature=0.05)
        # With identical q-p and orthogonal n, loss should be small
        assert loss.item() < 1.0

    def test_lower_temperature_sharpens(self, hard_case):
        q, p, n = hard_case
        loss_warm = infonce_loss(q, p, n, temperature=1.0)
        loss_cold = infonce_loss(q, p, n, temperature=0.01)
        # Lower temp should give more extreme logits (different loss)
        assert loss_warm.item() != pytest.approx(loss_cold.item(), abs=0.01)

    def test_batch_size_one(self):
        """Should work with batch_size=1."""
        q = torch.tensor([[1.0, 0.0]])
        p = torch.tensor([[0.9, 0.1]])
        n = torch.tensor([[0.0, 1.0]])
        loss = infonce_loss(q, p, n)
        assert loss.shape == ()
        assert not torch.isnan(loss)


class TestContrastiveLoss:
    def test_returns_scalar(self, identical_embeddings):
        q, p, n = identical_embeddings
        loss = contrastive_loss(q, p, n)
        assert loss.shape == ()

    def test_negative_loss_when_well_separated(self, identical_embeddings):
        """cos_sim(q,p)=1, cos_sim(q,n)=0 → loss = -1 + 0 = -1."""
        q, p, n = identical_embeddings
        loss = contrastive_loss(q, p, n)
        assert loss.item() == pytest.approx(-1.0, abs=1e-6)

    def test_swapped_pos_neg_increases_loss(self, identical_embeddings):
        q, p, n = identical_embeddings
        normal_loss = contrastive_loss(q, p, n)
        swapped_loss = contrastive_loss(q, n, p)
        assert swapped_loss.item() > normal_loss.item()


class TestLossGradients:
    """Verify all losses produce valid gradients."""

    @pytest.mark.parametrize("loss_fn", [triplet_margin_loss, infonce_loss, contrastive_loss])
    def test_gradients_flow(self, loss_fn):
        q = torch.randn(4, 8, requires_grad=True)
        p = torch.randn(4, 8)
        n = torch.randn(4, 8)
        loss = loss_fn(q, p, n)
        loss.backward()
        assert q.grad is not None
        assert not torch.isnan(q.grad).any()
