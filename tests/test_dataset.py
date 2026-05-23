"""Tests for dataset construction: window indexing, label timing, and
the episode-level train/val split that prevents leakage."""

from __future__ import annotations

import numpy as np
import pytest

import generate_dataset as gd
from config import (
    FAILURE_HORIZON,
    FUTURE_OFFSET,
    MAX_EPISODE_STEPS,
    PAST_WINDOW,
    STATE_DIM,
)


def _const_histories():
    """state_history[t] is a row filled with the value t, so window
    indexing is directly checkable. Fully connected => no failure."""
    n_steps = MAX_EPISODE_STEPS + 1
    state = (np.arange(n_steps, dtype=np.float32)[:, None]
             * np.ones((1, STATE_DIM), dtype=np.float32))
    conn = np.ones(n_steps, dtype=np.float32)
    return state, conn


def test_samples_per_episode_matches_anchor_count():
    assert gd.SAMPLES_PER_EPISODE == gd._ANCHOR_TIMES.size
    assert gd.SAMPLES_PER_EPISODE > 0


def test_every_anchor_window_is_in_bounds():
    n_steps = MAX_EPISODE_STEPS + 1
    for t in gd._ANCHOR_TIMES:
        assert t - PAST_WINDOW + 1 >= 0          # past window fits
        assert t + FUTURE_OFFSET < n_steps       # future target fits
        assert t + FAILURE_HORIZON < n_steps     # label window fits


class TestExtractSamples:
    def test_output_shapes(self):
        state, conn = _const_histories()
        x_past, x_future, y = gd._extract_samples(state, conn)
        assert x_past.shape == (gd.SAMPLES_PER_EPISODE, PAST_WINDOW, STATE_DIM)
        assert x_future.shape == (gd.SAMPLES_PER_EPISODE, STATE_DIM)
        assert y.shape == (gd.SAMPLES_PER_EPISODE,)

    def test_past_and_future_windows_use_correct_indices(self):
        state, conn = _const_histories()
        x_past, x_future, _ = gd._extract_samples(state, conn)
        for i, t in enumerate(gd._ANCHOR_TIMES):
            expected_past = np.arange(t - PAST_WINDOW + 1, t + 1)
            assert np.array_equal(x_past[i][:, 0], expected_past.astype(np.float32))
            assert x_future[i][0] == pytest.approx(float(t + FUTURE_OFFSET))

    def test_no_future_failure_gives_all_zero_labels(self):
        state, conn = _const_histories()
        _, _, y = gd._extract_samples(state, conn)
        assert y.sum() == 0

    def test_future_failure_sets_label(self):
        state, conn = _const_histories()
        t = int(gd._ANCHOR_TIMES[5])
        conn[t + 1 : t + 4] = 0.0  # sustained drop inside the label window
        _, _, y = gd._extract_samples(state, conn)
        assert y[5] == 1

    def test_failure_at_anchor_is_not_labelled(self):
        # A failure at or before the anchor step is INPUT, not a future
        # prediction. The label window starts at t+1, so this must NOT
        # flip the anchor's label.
        state, conn = _const_histories()
        t = int(gd._ANCHOR_TIMES[5])
        conn[t - 2 : t + 1] = 0.0  # three sustained, ending exactly at t
        _, _, y = gd._extract_samples(state, conn)
        assert y[5] == 0


class TestSplitEpisodeIds:
    def test_train_and_val_disjoint_and_cover_all(self):
        # The leakage guarantee: every episode is in exactly one split.
        n = 50
        val = gd._split_episode_ids(n, val_frac=0.2, seed=0)
        train = set(range(n)) - val
        assert val.isdisjoint(train)
        assert (val | train) == set(range(n))
        assert len(val) == round(n * 0.2)

    def test_split_is_deterministic_for_a_seed(self):
        assert gd._split_episode_ids(50, 0.2, 0) == gd._split_episode_ids(50, 0.2, 0)

    def test_different_seed_gives_different_split(self):
        assert gd._split_episode_ids(50, 0.2, 0) != gd._split_episode_ids(50, 0.2, 1)
