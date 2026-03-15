"""Tests for the trainer."""

import pytest

from khoji.data import Triplet, TripletDataset
from khoji.lora import LoRASettings
from khoji.trainer import Trainer, TrainHistory, TrainingConfig


@pytest.fixture(scope="module")
def small_triplets():
    """Small set of triplets for training tests."""
    return TripletDataset([
        Triplet("What is compound interest?",
                "Compound interest is interest on interest accumulated over time.",
                "The weather today is sunny."),
        Triplet("How do dividends work?",
                "Dividends are payments from a corporation to its shareholders.",
                "A recipe for chocolate cake."),
        Triplet("What is an index fund?",
                "An index fund tracks a market index like the S&P 500.",
                "The history of ancient Rome."),
        Triplet("How to save for retirement?",
                "A 401k is a retirement savings plan sponsored by an employer.",
                "Tips for growing tomatoes in your garden."),
    ])


class TestTrainHistory:
    def test_to_dict(self):
        history = TrainHistory()
        history.step_loss = [0.5, 0.3]
        history.epoch_loss = [0.4]
        d = history.to_dict()
        assert d["step_loss"] == [0.5, 0.3]
        assert d["epoch_loss"] == [0.4]

    def test_save(self, tmp_path):
        history = TrainHistory()
        history.step_loss = [0.5, 0.3]
        path = str(tmp_path / "history.json")
        history.save(path)

        import json
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["step_loss"] == [0.5, 0.3]


class TestTrainer:
    def test_overfit_loss_decreases(self, small_triplets):
        """Training in overfit mode should drive loss down."""
        config = TrainingConfig(
            epochs=30,
            batch_size=4,
            grad_accum_steps=1,
            lr=5e-3,
            warmup_steps=0,
            max_length=128,
            lora=LoRASettings(r=8, alpha=16, dropout=0.0),
            overfit_batches=1,
        )
        trainer = Trainer("BAAI/bge-base-en-v1.5", config)
        history = trainer.train(small_triplets)

        # Loss should decrease from start to end
        assert len(history.step_loss) > 0
        first_losses = history.step_loss[:3]
        last_losses = history.step_loss[-3:]
        avg_first = sum(first_losses) / len(first_losses)
        avg_last = sum(last_losses) / len(last_losses)
        assert avg_last < avg_first, (
            f"Loss should decrease: first {avg_first:.4f} -> last {avg_last:.4f}"
        )

    def test_history_has_all_fields(self, small_triplets):
        config = TrainingConfig(
            epochs=2,
            batch_size=4,
            grad_accum_steps=1,
            lr=2e-5,
            warmup_steps=0,
            max_length=128,
            lora=LoRASettings(r=4, alpha=8),
        )
        trainer = Trainer("BAAI/bge-base-en-v1.5", config)
        history = trainer.train(small_triplets)

        assert len(history.step_loss) > 0
        assert len(history.step_lr) > 0
        assert len(history.step_grad_norm) > 0
        assert len(history.epoch_loss) == 2
