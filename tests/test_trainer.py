"""Tests for the trainer."""

import pytest

from khoji.data import Triplet, TripletDataset
from khoji.lora import LoRASettings
from khoji.trainer import Trainer, TrainHistory, TrainingConfig


@pytest.fixture(scope="module")
def small_triplets():
    """Small set of triplets with hard negatives (semantically similar but wrong)."""
    return TripletDataset([
        Triplet("What is compound interest?",
                "Compound interest is interest calculated on the initial principal and accumulated interest from previous periods.",
                "Simple interest is calculated only on the original principal amount of a loan."),
        Triplet("How do stock dividends work?",
                "Stock dividends are payments made by a corporation to its shareholders from company profits.",
                "Stock splits divide existing shares into multiple shares but do not distribute profits."),
        Triplet("What is an index fund?",
                "An index fund is a type of mutual fund that tracks a specific market index like the S&P 500.",
                "An actively managed fund has a portfolio manager who picks individual stocks to beat the market."),
        Triplet("How to save for retirement?",
                "A 401k is a tax-advantaged retirement savings plan sponsored by an employer with matching contributions.",
                "A health savings account is a tax-advantaged account for medical expenses with employer contributions."),
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
