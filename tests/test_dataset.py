"""Unit tests for data utilities."""

import torch
import pytest


def test_collate_fn():
    from src.data.collate import collate_fn
    batch = [torch.randn(16000) for _ in range(4)]
    out = collate_fn(batch)
    assert out.shape == (4, 16000)


def test_collate_fn_single():
    from src.data.collate import collate_fn
    batch = [torch.randn(16000)]
    out = collate_fn(batch)
    assert out.shape == (1, 16000)
