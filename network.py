"""Comm graph: edges, jamming, connectivity, scalar graph metrics.

Pure numpy, stateless. The simulator owns positions and time; this
module just answers "given these positions now, what does the graph
look like?". Imported by the live sim, the dataset generator, and
the demo overlay so all three agree on the connectivity definition.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Final

import numpy as np

from config import (
    COMM_RANGE,
    JAM_REFERENCE,
    JAM_STRENGTH,
    JAM_BREAK_THRESHOLD,
    CONNECTIVITY_FAILURE_THRESHOLD,
    FAILURE_SUSTAIN_STEPS,
    WORLD_WIDTH,
    WORLD_HEIGHT,
)

# Coarse sampling grid for the scalar jam_coverage feature.
_JAM_GRID_NX: Final[int] = 30
_JAM_GRID_NY: Final[int] = 20

# Sentinel meaning "no path to any relay". Kept as int so downstream
# arithmetic doesn't accidentally introduce NaNs.
HOPS_DISCONNECTED: Final[int] = -1


@dataclass(frozen=True)
class GraphState:
    """Graph snapshot for one timestep.

    Node indexing is unified:
        [0, n_blue)            -> Blues
        [n_blue, n_blue+n_relay) -> Relays
    """

    # Per edge (length E)
    edge_index:     np.ndarray   # (E, 2) int64, u < v
    edge_midpoint:  np.ndarray   # (E, 2) float32
    edge_intensity: np.ndarray   # (E,)   float32
    edge_active:    np.ndarray   # (E,)   bool

    # Per Blue
    connected:     np.ndarray    # (n_blue,) bool
    hops_to_relay: np.ndarray    # (n_blue,) int32, HOPS_DISCONNECTED if unreachable
    blue_jammed:   np.ndarray    # (n_blue,) bool, has any adjacent broken edge

    # Scalars for the state vector
    connectivity_ratio:    float
    avg_path_length:       float
    jam_coverage:          float
    mean_red_distance:     float
    fraction_blues_jammed: float


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(M,2), (N,2) -> (M,N) float32 Euclidean distance matrix."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1)).astype(np.float32)


def _candidate_edges(
    blue_pos: np.ndarray, relay_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """All node pairs within COMM_RANGE, ignoring jamming.

    Separating geometric reachability from jamming lets the metrics
    distinguish "edge never possible" from "edge possible but jammed".
    """
    n_blue  = blue_pos.shape[0]
    n_relay = relay_pos.shape[0]
    nodes = np.concatenate([blue_pos, relay_pos], axis=0).astype(np.float32)
    n_total = n_blue + n_relay

    i_idx, j_idx = np.triu_indices(n_total, k=1)
    diffs = nodes[i_idx] - nodes[j_idx]
    dists = np.sqrt((diffs * diffs).sum(axis=-1))

    mask = dists <= COMM_RANGE
    i_sel = i_idx[mask]
    j_sel = j_idx[mask]
    edge_index = np.stack([i_sel, j_sel], axis=1).astype(np.int64)
    midpoints  = ((nodes[i_sel] + nodes[j_sel]) * 0.5).astype(np.float32)
    return edge_index, midpoints


def _jam_intensity_at(points: np.ndarray, red_pos: np.ndarray) -> np.ndarray:
    """Summed jam intensity at each point. Lorentzian falloff per Red."""
    if red_pos.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=np.float32)
    dists = _pairwise_distances(points.astype(np.float32),
                                red_pos.astype(np.float32))
    contrib = JAM_STRENGTH / (1.0 + (dists / JAM_REFERENCE) ** 2)
    return contrib.sum(axis=1).astype(np.float32)


def _bfs_hops_from_relays(
    edge_index: np.ndarray,
    edge_active: np.ndarray,
    n_blue: int,
    n_relay: int,
) -> np.ndarray:
    """Multi-source BFS seeded at every relay over active edges."""
    n_total = n_blue + n_relay
    adj: list[list[int]] = [[] for _ in range(n_total)]
    for u, v in edge_index[edge_active]:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))

    hops = np.full(n_total, HOPS_DISCONNECTED, dtype=np.int32)
    queue: collections.deque[int] = collections.deque()
    for r in range(n_blue, n_total):
        hops[r] = 0
        queue.append(r)
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if hops[v] == HOPS_DISCONNECTED:
                hops[v] = hops[u] + 1
                queue.append(v)
    return hops[:n_blue]


def _jam_coverage(red_pos: np.ndarray) -> float:
    """Fraction of a coarse world grid above the jam-break threshold."""
    if red_pos.shape[0] == 0:
        return 0.0
    xs = np.linspace(0.0, float(WORLD_WIDTH),  _JAM_GRID_NX, dtype=np.float32)
    ys = np.linspace(0.0, float(WORLD_HEIGHT), _JAM_GRID_NY, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    grid_points = np.stack([gx.ravel(), gy.ravel()], axis=1)
    intensities = _jam_intensity_at(grid_points, red_pos)
    return float((intensities >= JAM_BREAK_THRESHOLD).mean())


def build_graph_state(
    blue_pos: np.ndarray,
    relay_pos: np.ndarray,
    red_pos: np.ndarray,
) -> GraphState:
    """Compute the full graph snapshot for one timestep."""
    n_blue = blue_pos.shape[0]

    edge_index, edge_midpoint = _candidate_edges(blue_pos, relay_pos)
    # Midpoint sampling: a long edge whose midpoint is clean but whose
    # body passes near a Red survives. OK for current ranges.
    edge_intensity = _jam_intensity_at(edge_midpoint, red_pos)
    edge_active    = edge_intensity < JAM_BREAK_THRESHOLD

    hops_to_relay = _bfs_hops_from_relays(
        edge_index, edge_active, n_blue, relay_pos.shape[0]
    )
    connected = hops_to_relay != HOPS_DISCONNECTED

    blue_jammed = np.zeros(n_blue, dtype=bool)
    for u, v in edge_index[~edge_active]:
        if u < n_blue:
            blue_jammed[int(u)] = True
        if v < n_blue:
            blue_jammed[int(v)] = True

    connectivity_ratio = float(connected.mean()) if n_blue > 0 else 0.0
    avg_path_length = (
        float(hops_to_relay[connected].mean()) if connected.any() else 0.0
    )
    if red_pos.shape[0] > 0 and n_blue > 0:
        d_br = _pairwise_distances(blue_pos.astype(np.float32),
                                   red_pos.astype(np.float32))
        mean_red_distance = float(d_br.min(axis=1).mean())
    else:
        mean_red_distance = 0.0

    return GraphState(
        edge_index=edge_index,
        edge_midpoint=edge_midpoint,
        edge_intensity=edge_intensity,
        edge_active=edge_active,
        connected=connected,
        hops_to_relay=hops_to_relay,
        blue_jammed=blue_jammed,
        connectivity_ratio=connectivity_ratio,
        avg_path_length=avg_path_length,
        jam_coverage=_jam_coverage(red_pos),
        mean_red_distance=mean_red_distance,
        fraction_blues_jammed=(
            float(blue_jammed.mean()) if n_blue > 0 else 0.0
        ),
    )


def is_failure_sustained(
    connectivity_history: np.ndarray,
    threshold: float = CONNECTIVITY_FAILURE_THRESHOLD,
    sustain_steps: int = FAILURE_SUSTAIN_STEPS,
) -> bool:
    """True iff connectivity stayed below threshold for sustain_steps in a row.

    Both the live simulator and the dataset labeller use this so the
    two definitions of "failure" can't drift apart.
    """
    if connectivity_history.size < sustain_steps:
        return False
    run = 0
    for v in connectivity_history < threshold:
        if v:
            run += 1
            if run >= sustain_steps:
                return True
        else:
            run = 0
    return False
