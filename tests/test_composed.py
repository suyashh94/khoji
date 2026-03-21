"""Tests for composed (image+text → image) retrieval."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from khoji.composed_config import ComposedForgeConfig, ComposedModelConfig, ComposedDataConfig
from khoji.composed_data import (
    ComposedTriplet,
    ComposedTripletDataset,
    build_random_negatives_composed,
)
from khoji.composed_dataset import ComposedRetrievalDataset, load_custom_composed
from khoji.composed_evaluator import ComposedEvaluator
from khoji.composed_trainer import ComposedTrainer, ComposedTrainingConfig
from khoji.multimodal_model import JointEmbeddingModel
from khoji.lora import LoRASettings


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def image_dir():
    """Create a temp directory with test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        colors = {
            "ref1.jpg": "orange", "ref2.jpg": "brown",
            "target1.jpg": "red", "target2.jpg": "green",
            "neg1.jpg": "blue", "neg2.jpg": "yellow",
            "neg3.jpg": "purple", "neg4.jpg": "white",
        }
        for name, color in colors.items():
            Image.new("RGB", (224, 224), color=color).save(Path(tmpdir) / name)
        yield tmpdir


@pytest.fixture(scope="module")
def small_composed_dataset(image_dir):
    """Tiny composed dataset for testing."""
    return ComposedRetrievalDataset(
        queries={
            "q1": ("ref1.jpg", "make it red"),
            "q2": ("ref2.jpg", "make it green"),
        },
        corpus={
            "d1": "target1.jpg",
            "d2": "target2.jpg",
            "d3": "neg1.jpg",
            "d4": "neg2.jpg",
            "d5": "neg3.jpg",
            "d6": "neg4.jpg",
        },
        qrels={
            "q1": {"d1": 1},
            "q2": {"d2": 1},
        },
        base_dir=image_dir,
    )


# ── Dataset tests ─────────────────────────────────────────────────


class TestComposedDataset:
    def test_dataset_structure(self, small_composed_dataset):
        ds = small_composed_dataset
        assert len(ds.queries) == 2
        assert len(ds.corpus) == 6
        assert len(ds.qrels) == 2

    def test_query_format(self, small_composed_dataset):
        ds = small_composed_dataset
        img, text = ds.queries["q1"]
        assert isinstance(img, str)
        assert isinstance(text, str)
        assert img == "ref1.jpg"
        assert text == "make it red"

    def test_load_custom(self, image_dir):
        """Test loading from the standard file format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write queries.jsonl
            with open(Path(tmpdir) / "queries.jsonl", "w") as f:
                f.write(json.dumps({"_id": "q1", "image": "ref1.jpg", "text": "make it red"}) + "\n")
                f.write(json.dumps({"_id": "q2", "image": "ref2.jpg", "text": "make it green"}) + "\n")

            # Write corpus.jsonl
            with open(Path(tmpdir) / "corpus.jsonl", "w") as f:
                f.write(json.dumps({"_id": "d1", "image": "target1.jpg"}) + "\n")
                f.write(json.dumps({"_id": "d2", "image": "target2.jpg"}) + "\n")
                f.write(json.dumps({"_id": "d3", "image": "neg1.jpg"}) + "\n")

            # Write qrels.tsv
            with open(Path(tmpdir) / "qrels.tsv", "w") as f:
                f.write("q1\td1\t1\n")
                f.write("q2\td2\t1\n")

            ds = load_custom_composed(tmpdir)
            assert len(ds.queries) == 2
            assert len(ds.corpus) == 3
            assert ds.queries["q1"] == ("ref1.jpg", "make it red")
            assert ds.base_dir == tmpdir

    def test_load_custom_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_custom_composed(tmpdir)


# ── Triplet data tests ────────────────────────────────────────────


class TestComposedData:
    def test_triplet_fields(self):
        t = ComposedTriplet("ref.jpg", "make it blue", "target.jpg", "neg.jpg")
        assert t.query_image == "ref.jpg"
        assert t.query_text == "make it blue"
        assert t.positive == "target.jpg"
        assert t.negative == "neg.jpg"

    def test_triplet_dataset(self):
        triplets = [
            ComposedTriplet("r1.jpg", "caption 1", "t1.jpg", "n1.jpg"),
            ComposedTriplet("r2.jpg", "caption 2", "t2.jpg", "n2.jpg"),
        ]
        ds = ComposedTripletDataset(triplets)
        assert len(ds) == 2
        q_img, q_text, pos, neg = ds[0]
        assert q_img == "r1.jpg"
        assert q_text == "caption 1"
        assert pos == "t1.jpg"
        assert neg == "n1.jpg"

    def test_random_negatives(self, small_composed_dataset):
        triplets = build_random_negatives_composed(
            small_composed_dataset, n_negatives=2, seed=42
        )
        assert len(triplets) > 0

        for t in triplets:
            # Negative must not be the target
            qid = None
            for q, (img, text) in small_composed_dataset.queries.items():
                if img == t.query_image and text == t.query_text:
                    qid = q
                    break
            assert qid is not None
            relevant = set(small_composed_dataset.qrels[qid].keys())
            neg_id = None
            for did, src in small_composed_dataset.corpus.items():
                if src == t.negative:
                    neg_id = did
                    break
            assert neg_id not in relevant

    def test_random_negatives_deterministic(self, small_composed_dataset):
        t1 = build_random_negatives_composed(
            small_composed_dataset, n_negatives=2, seed=42
        )
        t2 = build_random_negatives_composed(
            small_composed_dataset, n_negatives=2, seed=42
        )
        assert len(t1) == len(t2)
        for a, b in zip(t1, t2):
            assert a.query_image == b.query_image
            assert a.query_text == b.query_text
            assert a.positive == b.positive
            assert a.negative == b.negative


# ── Config tests ──────────────────────────────────────────────────


class TestComposedConfig:
    def test_config_defaults(self):
        config = ComposedForgeConfig()
        assert config.model.name == "Salesforce/blip2-itm-vit-g"
        assert config.data.negatives == "random"
        assert config.lora is not None
        assert config.lora.r == 8

    def test_config_yaml_roundtrip(self):
        config = ComposedForgeConfig()
        config.data.negatives = "mixed"
        config.data.n_random = 3
        config.data.n_hard = 2

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            config.to_yaml(f.name)
            loaded = ComposedForgeConfig.from_yaml(f.name)

        assert loaded.data.negatives == "mixed"
        assert loaded.data.n_random == 3
        assert loaded.data.n_hard == 2
        assert loaded.model.name == config.model.name

    def test_config_validation_bad_negatives(self):
        config = ComposedForgeConfig()
        config.data.negatives = "invalid"
        with pytest.raises(ValueError, match="negatives"):
            config.validate()


# ── Model tests (using custom encoder) ───────────────────────────


class TestComposedModel:
    def test_joint_custom_encoder(self):
        """Test JointEmbeddingModel with a custom encoder for composed retrieval."""
        embed_dim = 32

        def mock_encoder(images, texts, device):
            if images is not None:
                n = len(images)
            else:
                n = len(texts)
            emb = torch.randn(n, embed_dim)
            return torch.nn.functional.normalize(emb, p=2, dim=1)

        model = JointEmbeddingModel(encoder=mock_encoder)

        # Image-only
        imgs = [Image.new("RGB", (64, 64), "red")]
        img_emb = model.encode(images=imgs)
        assert img_emb.shape == (1, embed_dim)

        # Joint mode
        joint_emb = model.encode(images=imgs, texts=["make it blue"])
        assert joint_emb.shape == (1, embed_dim)


# ── Trainer tests (using custom encoder) ─────────────────────────


class TestComposedTrainer:
    def test_custom_model_training(self, image_dir, small_composed_dataset):
        """Test training with custom encode functions."""
        embed_dim = 32

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(embed_dim, embed_dim)

            def forward(self, x):
                return self.linear(x)

        model = DummyModel()

        def encode_query(images, texts):
            device = next(model.parameters()).device
            emb = torch.randn(len(images), embed_dim, device=device)
            return model(emb)

        def encode_image(images):
            device = next(model.parameters()).device
            emb = torch.randn(len(images), embed_dim, device=device)
            return model(emb)

        triplets = build_random_negatives_composed(
            small_composed_dataset, n_negatives=1, seed=42
        )

        config = ComposedTrainingConfig(
            epochs=2,
            batch_size=2,
            grad_accum_steps=1,
            lr=1e-3,
            warmup_steps=1,
            sanity_check_samples=0,
            base_dir=image_dir,
        )

        trainer = ComposedTrainer(
            model=model,
            encode_query_fn=encode_query,
            encode_image_fn=encode_image,
            config=config,
        )
        ds = ComposedTripletDataset(triplets)
        history = trainer.train(ds)

        assert len(history.epoch_loss) == 2
        assert len(history.step_loss) > 0


# ── Evaluator tests (using custom encoder) ───────────────────────


class TestComposedEvaluator:
    def test_custom_model_evaluation(self, image_dir, small_composed_dataset):
        """Test evaluation with a custom JointEmbeddingModel."""
        embed_dim = 32

        def mock_encoder(images, texts, device):
            if images is not None:
                n = len(images)
            else:
                n = len(texts)
            emb = torch.randn(n, embed_dim)
            return torch.nn.functional.normalize(emb, p=2, dim=1)

        model = JointEmbeddingModel(encoder=mock_encoder)
        evaluator = ComposedEvaluator(embedding_model=model)

        result = evaluator.evaluate(
            dataset=small_composed_dataset,
            k_values=[1, 5],
        )

        assert result.num_queries > 0
        assert "recall@1" in result.metrics
        assert "mrr@1" in result.metrics
        assert "ndcg@1" in result.metrics
