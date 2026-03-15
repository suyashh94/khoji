"""Tests for the embedding model wrapper."""

import torch
import pytest

from khoji.model import _pool


class TestPooling:
    """Test pooling functions with synthetic tensors."""

    def _make_inputs(self, batch=2, seq_len=4, dim=3):
        hidden = torch.randn(batch, seq_len, dim)
        mask = torch.ones(batch, seq_len, dtype=torch.long)
        return hidden, mask

    def test_cls_pooling(self):
        hidden, mask = self._make_inputs()
        result = _pool(hidden, mask, "cls")
        assert result.shape == (2, 3)
        # Should equal the first token
        torch.testing.assert_close(result, hidden[:, 0, :])

    def test_mean_pooling_no_padding(self):
        hidden, mask = self._make_inputs()
        result = _pool(hidden, mask, "mean")
        expected = hidden.mean(dim=1)
        torch.testing.assert_close(result, expected)

    def test_mean_pooling_with_padding(self):
        hidden, mask = self._make_inputs(batch=1, seq_len=4, dim=2)
        # Mask out last 2 tokens
        mask[0, 2:] = 0
        result = _pool(hidden, mask, "mean")
        # Should only average first 2 tokens
        expected = hidden[0, :2, :].mean(dim=0, keepdim=True)
        torch.testing.assert_close(result, expected)

    def test_max_pooling(self):
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 1.0], [0.0, 4.0]]])
        mask = torch.ones(1, 3, dtype=torch.long)
        result = _pool(hidden, mask, "max")
        expected = torch.tensor([[3.0, 4.0]])
        torch.testing.assert_close(result, expected)

    def test_max_pooling_ignores_padding(self):
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 1.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]])
        result = _pool(hidden, mask, "max")
        expected = torch.tensor([[3.0, 2.0]])
        torch.testing.assert_close(result, expected)

    def test_lasttoken_pooling(self):
        hidden = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # last real = idx 2
        ])
        mask = torch.ones(1, 3, dtype=torch.long)
        result = _pool(hidden, mask, "lasttoken")
        expected = torch.tensor([[5.0, 6.0]])
        torch.testing.assert_close(result, expected)

    def test_lasttoken_with_padding(self):
        hidden = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],  # last real = idx 1
        ])
        mask = torch.tensor([[1, 1, 0]])
        result = _pool(hidden, mask, "lasttoken")
        expected = torch.tensor([[3.0, 4.0]])
        torch.testing.assert_close(result, expected)

    def test_unknown_mode_raises(self):
        hidden, mask = self._make_inputs()
        with pytest.raises(ValueError, match="Unknown pooling mode"):
            _pool(hidden, mask, "nonexistent")


class TestPoolingDetection:
    """Test that pooling mode is correctly detected from HF model configs."""

    def test_bge_is_cls(self):
        from khoji.model import _detect_pooling
        assert _detect_pooling("BAAI/bge-base-en-v1.5") == "cls"

    def test_minilm_is_mean(self):
        from khoji.model import _detect_pooling
        assert _detect_pooling("sentence-transformers/all-MiniLM-L6-v2") == "mean"

    def test_unknown_model_falls_back_to_cls(self):
        from khoji.model import _detect_pooling
        assert _detect_pooling("some/nonexistent-model-xyz") == "cls"


class TestEmbeddingModel:
    """Integration tests for the full EmbeddingModel (requires model download)."""

    @pytest.fixture(scope="class")
    def model(self):
        from khoji.model import EmbeddingModel
        return EmbeddingModel("BAAI/bge-base-en-v1.5")

    def test_encode_shape(self, model):
        embs = model.encode(["hello", "world"], show_progress=False)
        assert embs.shape == (2, 768)

    def test_embeddings_are_normalized(self, model):
        embs = model.encode(["test sentence"], show_progress=False)
        norm = embs.norm(dim=1)
        torch.testing.assert_close(norm, torch.ones(1), atol=1e-5, rtol=0)

    def test_different_texts_get_different_embeddings(self, model):
        embs = model.encode(
            ["the cat sat on the mat", "quantum computing research"],
            show_progress=False,
        )
        similarity = torch.mm(embs, embs.t())[0, 1].item()
        assert similarity < 0.99  # Should not be identical
