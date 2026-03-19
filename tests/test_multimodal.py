"""Tests for multimodal (text-to-image) retrieval."""

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from khoji.image_utils import load_image, load_images_batch, build_image_processor
from khoji.multimodal_config import MultimodalForgeConfig, MultimodalModelConfig, ImagePreprocessConfig
from khoji.multimodal_data import MultimodalTriplet, MultimodalTripletDataset, build_random_negatives_multimodal
from khoji.multimodal_dataset import MultimodalRetrievalDataset, load_custom_multimodal
from khoji.multimodal_model import MultimodalEmbeddingModel
from khoji.multimodal_trainer import MultimodalTrainer, MultimodalTrainingConfig
from khoji.multimodal_evaluator import MultimodalEvaluator
from khoji.lora import LoRASettings


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def image_dir():
    """Create a temp directory with test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        colors = {"cat.jpg": "orange", "dog.jpg": "brown", "car.jpg": "red",
                  "tree.jpg": "green", "sky.jpg": "blue", "sun.jpg": "yellow"}
        for name, color in colors.items():
            Image.new("RGB", (224, 224), color=color).save(Path(tmpdir) / name)
        yield tmpdir


@pytest.fixture(scope="module")
def small_multimodal_dataset(image_dir):
    """Tiny multimodal dataset for testing."""
    return MultimodalRetrievalDataset(
        queries={
            "q1": "a photo of a cat",
            "q2": "a photo of a dog",
            "q3": "a red car",
        },
        corpus={
            "d1": "cat.jpg",
            "d2": "dog.jpg",
            "d3": "car.jpg",
            "d4": "tree.jpg",
            "d5": "sky.jpg",
            "d6": "sun.jpg",
        },
        qrels={
            "q1": {"d1": 1},
            "q2": {"d2": 1},
            "q3": {"d3": 1},
        },
        base_dir=image_dir,
    )


@pytest.fixture(scope="module")
def clip_model():
    return MultimodalEmbeddingModel("openai/clip-vit-base-patch32")


# ── Image Utils ───────────────────────────────────────────────────


class TestImageUtils:
    def test_load_local_image(self, image_dir):
        img = load_image("cat.jpg", base_dir=image_dir)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_load_images_batch(self, image_dir):
        images = load_images_batch(["cat.jpg", "dog.jpg"], base_dir=image_dir)
        assert len(images) == 2
        assert all(isinstance(i, Image.Image) for i in images)

    def test_build_image_processor_auto(self):
        proc = build_image_processor(model_name="openai/clip-vit-base-patch32")
        img = Image.new("RGB", (300, 300), color="red")
        tensor = proc([img])
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 1  # batch size
        assert tensor.shape[1] == 3  # RGB channels

    def test_build_image_processor_custom(self):
        def my_proc(images):
            return torch.zeros(len(images), 3, 224, 224)

        proc = build_image_processor(custom_fn=my_proc)
        result = proc([Image.new("RGB", (100, 100))])
        assert result.shape == (1, 3, 224, 224)


# ── Config ────────────────────────────────────────────────────────


class TestMultimodalConfig:
    def test_default_config(self):
        config = MultimodalForgeConfig()
        assert config.model.name == "openai/clip-vit-base-patch32"
        assert config.model.lora_target == "both"

    def test_yaml_roundtrip(self):
        config = MultimodalForgeConfig()
        config.model.lora_target = "vision"
        config.seed = 42
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config.to_yaml(f.name)
            loaded = MultimodalForgeConfig.from_yaml(f.name)
        assert loaded.model.lora_target == "vision"
        assert loaded.seed == 42
        Path(f.name).unlink()

    def test_validation_invalid_lora_target(self):
        config = MultimodalForgeConfig()
        config.model.lora_target = "invalid"
        with pytest.raises(ValueError, match="lora_target"):
            config.validate()


# ── Dataset ───────────────────────────────────────────────────────


class TestMultimodalDataset:
    def test_direct_construction(self, small_multimodal_dataset):
        ds = small_multimodal_dataset
        assert len(ds.queries) == 3
        assert len(ds.corpus) == 6
        assert ds.corpus["d1"] == "cat.jpg"

    def test_load_custom(self, image_dir):
        base = Path(image_dir)
        (base / "queries.jsonl").write_text(
            '{"_id": "q1", "text": "a cat"}\n'
            '{"_id": "q2", "text": "a dog"}\n'
        )
        (base / "corpus.jsonl").write_text(
            '{"_id": "d1", "image": "cat.jpg"}\n'
            '{"_id": "d2", "image": "dog.jpg"}\n'
        )
        (base / "qrels.tsv").write_text("q1\td1\t1\nq2\td2\t1\n")

        ds = load_custom_multimodal(image_dir)
        assert len(ds.queries) == 2
        assert ds.corpus["d1"] == "cat.jpg"
        assert ds.base_dir == image_dir


# ── Data (Triplets) ──────────────────────────────────────────────


class TestMultimodalData:
    def test_triplet_dataset(self):
        triplets = [MultimodalTriplet("query", "pos.jpg", "neg.jpg")]
        ds = MultimodalTripletDataset(triplets)
        assert len(ds) == 1
        q, p, n = ds[0]
        assert q == "query"

    def test_random_negatives(self, small_multimodal_dataset):
        triplets = build_random_negatives_multimodal(
            small_multimodal_dataset, n_negatives=1
        )
        assert len(triplets) == 3  # 3 queries, 1 positive each, 1 negative
        for t in triplets:
            assert isinstance(t.query, str)
            assert t.positive.endswith(".jpg")
            assert t.negative.endswith(".jpg")
            assert t.positive != t.negative


# ── Model ─────────────────────────────────────────────────────────


class TestMultimodalModel:
    def test_encode_text(self, clip_model):
        embs = clip_model.encode_text(["a cat", "a dog"], show_progress=False)
        assert embs.shape == (2, 512)
        # L2 normalized
        norms = embs.norm(dim=1)
        torch.testing.assert_close(norms, torch.ones(2), atol=1e-4, rtol=0)

    def test_encode_images(self, clip_model):
        imgs = [Image.new("RGB", (224, 224), c) for c in ["red", "blue"]]
        embs = clip_model.encode_images(imgs, show_progress=False)
        assert embs.shape == (2, 512)
        norms = embs.norm(dim=1)
        torch.testing.assert_close(norms, torch.ones(2), atol=1e-4, rtol=0)

    def test_encode_image_sources(self, clip_model, image_dir):
        embs = clip_model.encode_image_sources(
            ["cat.jpg", "dog.jpg"], base_dir=image_dir, show_progress=False
        )
        assert embs.shape == (2, 512)

    def test_cross_modal_similarity(self, clip_model):
        text_emb = clip_model.encode_text(["a red square"], show_progress=False)
        imgs = [
            Image.new("RGB", (224, 224), "red"),
            Image.new("RGB", (224, 224), "blue"),
        ]
        img_embs = clip_model.encode_images(imgs, show_progress=False)
        sims = torch.mm(text_emb, img_embs.t()).squeeze(0)
        # Both are simple color blocks, similarity won't be meaningful
        # but shapes and values should be valid
        assert sims.shape == (2,)
        assert not torch.isnan(sims).any()


# ── Trainer ───────────────────────────────────────────────────────


class TestMultimodalTrainer:
    def test_training_runs(self, image_dir):
        triplets = MultimodalTripletDataset([
            MultimodalTriplet("a photo of a cat", "cat.jpg", "car.jpg"),
            MultimodalTriplet("a photo of a dog", "dog.jpg", "tree.jpg"),
        ])
        config = MultimodalTrainingConfig(
            epochs=2, batch_size=2, grad_accum_steps=1, lr=1e-4,
            warmup_steps=0, max_length=77,
            lora=LoRASettings(r=4, alpha=8, dropout=0.0),
            lora_target="both", base_dir=image_dir,
        )
        trainer = MultimodalTrainer("openai/clip-vit-base-patch32", config)
        history = trainer.train(triplets)

        assert len(history.step_loss) == 2
        assert len(history.epoch_loss) == 2
        assert all(isinstance(l, float) for l in history.step_loss)

    def test_lora_target_vision_only(self, image_dir):
        config = MultimodalTrainingConfig(
            epochs=1, batch_size=2, grad_accum_steps=1, lr=1e-4,
            warmup_steps=0, max_length=77,
            lora=LoRASettings(r=4, alpha=8, dropout=0.0),
            lora_target="vision", base_dir=image_dir,
        )
        trainer = MultimodalTrainer("openai/clip-vit-base-patch32", config)
        # Check that text model LoRA params are frozen
        text_lora_params = [
            p for n, p in trainer._full_model.named_parameters()
            if "text_model" in n and "lora" in n
        ]
        assert all(not p.requires_grad for p in text_lora_params)


# ── Evaluator ─────────────────────────────────────────────────────


class TestMultimodalEvaluator:
    def test_evaluate(self, small_multimodal_dataset):
        evaluator = MultimodalEvaluator("openai/clip-vit-base-patch32")
        result = evaluator.evaluate(
            dataset=small_multimodal_dataset,
            k_values=[1, 3],
            batch_size=4,
            cache_dir=None,
        )
        assert "ndcg@1" in result.metrics
        assert "mrr@1" in result.metrics
        assert "recall@1" in result.metrics
        assert result.num_queries == 3
        assert result.num_corpus == 6
