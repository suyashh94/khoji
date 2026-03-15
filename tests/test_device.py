"""Tests for device selection."""

import torch

from khoji.device import get_device


def test_returns_valid_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "mps", "cpu")


def test_device_is_usable():
    """Can actually create a tensor on the selected device."""
    device = get_device()
    t = torch.zeros(2, 3, device=device)
    assert t.device.type == device.type
