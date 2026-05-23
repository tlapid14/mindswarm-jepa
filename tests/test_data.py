"""Tests for the Dataset loader, focused on its shape-validation guard
(the thing that catches a stale dataset after a config.py edit)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from config import PAST_WINDOW, STATE_DIM
from data import SwarmDataset


def _write_npz(path, *, n=4, past_window=PAST_WINDOW, state_dim=STATE_DIM):
    x_past = np.zeros((n, past_window, state_dim), dtype=np.float32)
    x_future = np.zeros((n, state_dim), dtype=np.float32)
    y = (np.arange(n) % 2).astype(np.uint8)  # alternating 0/1
    np.savez_compressed(path, X_past=x_past, X_future=x_future, y_failure=y)


def test_loads_valid_dataset(tmp_path):
    p = tmp_path / "train.npz"
    _write_npz(p, n=4)
    ds = SwarmDataset(p)
    assert len(ds) == 4
    assert ds.positive_rate == pytest.approx(0.5)
    x_past, x_future, label = ds[0]
    assert x_past.shape == (PAST_WINDOW, STATE_DIM)
    assert x_future.shape == (STATE_DIM,)
    assert label.dtype == torch.float32  # cast once at load time


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SwarmDataset(tmp_path / "does_not_exist.npz")


def test_wrong_past_window_raises(tmp_path):
    p = tmp_path / "train.npz"
    _write_npz(p, past_window=PAST_WINDOW + 1)
    with pytest.raises(ValueError):
        SwarmDataset(p)


def test_wrong_state_dim_raises(tmp_path):
    p = tmp_path / "train.npz"
    _write_npz(p, state_dim=STATE_DIM + 1)
    with pytest.raises(ValueError):
        SwarmDataset(p)
