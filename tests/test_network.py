"""Tests for the communication-graph core: connectivity, the failure
rule, and the jamming model. These are the correctness-critical pure
functions every other module trusts."""

from __future__ import annotations

import numpy as np
import pytest

from config import CONNECTIVITY_FAILURE_THRESHOLD, JAM_STRENGTH
from network import (
    HOPS_DISCONNECTED,
    _jam_intensity_at,
    build_graph_state,
    is_failure_sustained,
)

_NO_REDS = np.zeros((0, 2), dtype=np.float32)


class TestIsFailureSustained:
    def test_too_short_history_is_false(self):
        # Fewer samples than the sustain window can't constitute failure.
        assert is_failure_sustained(np.array([0.0, 0.0])) is False

    def test_all_connected_is_false(self):
        assert is_failure_sustained(np.ones(20)) is False

    def test_three_consecutive_below_is_true(self):
        h = np.ones(10)
        h[4:7] = 0.0
        assert is_failure_sustained(h) is True

    def test_two_below_then_reset_is_false(self):
        h = np.ones(10)
        h[2:4] = 0.0  # only two in a row
        assert is_failure_sustained(h) is False

    def test_run_resets_then_reaches_threshold(self):
        # Two below, recover, then three below -> the run must reset and
        # only the trailing three-in-a-row counts.
        h = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        assert is_failure_sustained(h) is True

    def test_exactly_threshold_is_not_below(self):
        # The rule uses a strict <, so sitting exactly at threshold is OK.
        h = np.full(5, CONNECTIVITY_FAILURE_THRESHOLD)
        assert is_failure_sustained(h) is False


class TestBuildGraphState:
    def test_single_blue_connected_to_relay(self):
        blue = np.array([[100.0, 100.0]], dtype=np.float32)
        relay = np.array([[150.0, 100.0]], dtype=np.float32)  # in range
        g = build_graph_state(blue, relay, _NO_REDS)
        assert g.connected[0]
        assert g.connectivity_ratio == 1.0
        assert g.hops_to_relay[0] == 1

    def test_blue_out_of_range_is_disconnected(self):
        blue = np.array([[0.0, 0.0]], dtype=np.float32)
        relay = np.array([[1000.0, 1000.0]], dtype=np.float32)  # ~1414 px
        g = build_graph_state(blue, relay, _NO_REDS)
        assert not g.connected[0]
        assert g.connectivity_ratio == 0.0
        assert g.hops_to_relay[0] == HOPS_DISCONNECTED

    def test_jammer_at_edge_midpoint_breaks_link(self):
        blue = np.array([[100.0, 100.0]], dtype=np.float32)
        relay = np.array([[150.0, 100.0]], dtype=np.float32)
        red = np.array([[125.0, 100.0]], dtype=np.float32)  # on the midpoint
        g = build_graph_state(blue, relay, red)
        # Intensity at the midpoint is JAM_STRENGTH (>= break threshold),
        # so the only link breaks and the blue is cut off.
        assert not g.connected[0]
        assert g.blue_jammed[0]

    def test_multi_hop_relay_reachability(self):
        # blue -> blue -> relay, each leg in range but the far blue is
        # not directly in range of the relay.
        blue = np.array([[100.0, 100.0], [500.0, 100.0]], dtype=np.float32)
        relay = np.array([[900.0, 100.0]], dtype=np.float32)
        g = build_graph_state(blue, relay, _NO_REDS)
        assert g.connected.all()
        assert g.hops_to_relay[0] == 2
        assert g.hops_to_relay[1] == 1


class TestJamIntensity:
    def test_no_reds_is_zero_everywhere(self):
        pts = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
        out = _jam_intensity_at(pts, _NO_REDS)
        assert np.all(out == 0.0)

    def test_peak_intensity_at_red_location(self):
        pt = np.array([[50.0, 50.0]], dtype=np.float32)
        red = np.array([[50.0, 50.0]], dtype=np.float32)  # distance 0
        out = _jam_intensity_at(pt, red)
        assert out[0] == pytest.approx(JAM_STRENGTH)

    def test_intensity_falls_off_with_distance(self):
        red = np.array([[0.0, 0.0]], dtype=np.float32)
        near = _jam_intensity_at(np.array([[10.0, 0.0]], dtype=np.float32), red)
        far = _jam_intensity_at(np.array([[200.0, 0.0]], dtype=np.float32), red)
        assert near[0] > far[0]
