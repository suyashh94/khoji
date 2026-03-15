"""Tests for LoRA configuration and application."""

import pytest
import torch

from khoji.lora import LoRASettings, apply_lora, _get_target_modules


class TestLoRASettings:
    def test_defaults(self):
        settings = LoRASettings()
        assert settings.r == 8
        assert settings.alpha == 16
        assert settings.dropout == 0.1
        assert settings.target_modules is None


class TestApplyLoRA:
    @pytest.fixture(scope="class")
    def base_model(self):
        from transformers import AutoModel
        return AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")

    def test_reduces_trainable_params(self, base_model):
        """After LoRA, most params should be frozen."""
        total_before = sum(p.numel() for p in base_model.parameters())
        trainable_before = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

        lora_model = apply_lora(base_model, LoRASettings(r=4, alpha=8))
        trainable_after = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

        assert trainable_after < trainable_before
        assert trainable_after > 0

    def test_auto_detects_bert_modules(self, base_model):
        modules = _get_target_modules(base_model)
        assert modules == ["query", "key", "value"]

    def test_custom_target_modules(self, base_model):
        settings = LoRASettings(target_modules=["query", "value"])
        lora_model = apply_lora(base_model, settings)
        trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        assert trainable > 0
