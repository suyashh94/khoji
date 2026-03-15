"""Tests for configuration loading and dtype resolution."""

import tempfile
from pathlib import Path

import pytest
import torch

from khoji.config import ForgeConfig, _coerce_train_config
from khoji.model import _resolve_dtype


class TestResolveDtype:
    def test_none_returns_none(self):
        assert _resolve_dtype(None) is None

    def test_fp16(self):
        assert _resolve_dtype("fp16") == torch.float16

    def test_bf16(self):
        assert _resolve_dtype("bf16") == torch.bfloat16

    def test_fp32(self):
        assert _resolve_dtype("fp32") == torch.float32

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            _resolve_dtype("int8")


class TestCoerceTrainConfig:
    def test_string_lr_becomes_float(self):
        config = _coerce_train_config({"lr": "2e-5"})
        assert config.lr == pytest.approx(2e-5)
        assert isinstance(config.lr, float)

    def test_string_epochs_becomes_int(self):
        config = _coerce_train_config({"epochs": "10"})
        assert config.epochs == 10
        assert isinstance(config.epochs, int)

    def test_none_values_pass_through(self):
        config = _coerce_train_config({"overfit_batches": None})
        assert config.overfit_batches is None

    def test_normal_values_unchanged(self):
        config = _coerce_train_config({"lr": 0.001, "epochs": 5})
        assert config.lr == 0.001
        assert config.epochs == 5


class TestForgeConfigYaml:
    def test_roundtrip(self):
        """Save and load config should produce equivalent config."""
        config = ForgeConfig()
        config.seed = 42
        config.model.dtype = "bf16"
        config.train.mixed_precision = "fp16"
        config.train.save_every_n_steps = 100

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "config.yaml")
            config.to_yaml(path)
            loaded = ForgeConfig.from_yaml(path)

        assert loaded.seed == 42
        assert loaded.model.dtype == "bf16"
        assert loaded.train.mixed_precision == "fp16"
        assert loaded.train.save_every_n_steps == 100
        assert loaded.model.name == config.model.name

    def test_defaults_when_empty(self):
        """Loading an empty YAML should give defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("{}\n")
            path = f.name

        config = ForgeConfig.from_yaml(path)
        assert config.model.name == "BAAI/bge-base-en-v1.5"
        assert config.train.epochs == 3
        assert config.seed is None
        Path(path).unlink()

    def test_partial_override(self):
        """Only specified fields should override defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  name: custom/model\nseed: 123\n")
            path = f.name

        config = ForgeConfig.from_yaml(path)
        assert config.model.name == "custom/model"
        assert config.seed == 123
        assert config.train.lr == pytest.approx(2e-5)  # default
        Path(path).unlink()

    def test_lr_as_string_in_yaml(self):
        """YAML safe_load parses 2e-5 as string; should be coerced to float."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("train:\n  lr: 2e-5\n")
            path = f.name

        config = ForgeConfig.from_yaml(path)
        assert config.train.lr == pytest.approx(2e-5)
        assert isinstance(config.train.lr, float)
        Path(path).unlink()
