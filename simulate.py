"""2D swarm sim with stale-observation latency, plus a pygame renderer.

The Simulator is headless (pure numpy + GraphState). The Renderer is
isolated so headless callers can import simulate without pygame.

Latency model is OBSERVATION-only: the world advances in real time, but
the state vector exposed to callers is delayed by COMM_LATENCY_STEPS.
Both JEPA and the baseline see the same stale stream.
"""

from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass
from typing import Final, Optional

import numpy as np

from config import (
    BLUE_ACCEL,
    BLUE_MAX_SPEED,
    COMM_LATENCY_STEPS,
    FPS,
    MAX_BLUE,
    MAX_EPISODE_STEPS,
    MAX_RELAY,
    N_BLUE,
    N_OBJECTIVES,
    N_RED,
    N_RELAY,
    PER_BLUE_FEATURES,
    PER_RELAY_FEATURES,
    RED_MAX_SPEED,
    SCALAR_FEATURES,
    SEED,
    SIM_DT,
    STATE_DIM,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    COLOR_BG,
    COLOR_BLUE,
    COLOR_BLUE_DISCONNECTED,
    COLOR_EDGE_BROKEN,
    COLOR_EDGE_OK,
    COLOR_JAM_FIELD,
    COLOR_RED,
    COLOR_RELAY,
    COLOR_SPARK_BASELINE,
    COLOR_SPARK_JEPA,
    COLOR_TEXT,
    JAM_BREAK_THRESHOLD,
)
from network import (
    GraphState,
    HOPS_DISCONNECTED,
    build_graph_state,
    is_failure_sustained,
    _jam_intensity_at,
)

_OBJECTIVE_REACHED_RADIUS: Final[float] = 60.0
_BLUE_SPAWN_RADIUS: Final[float] = 80.0
_RED_NOISE_STD: Final[float] = 0.4   # keeps Reds from stacking on the centroid
_SPAWN_MARGIN: Final[float] = 60.0

# Normalizers for the scalar block of the state vector.
_WORLD_DIAG: Final[float] = float(np.hypot(WORLD_WIDTH, WORLD_HEIGHT))
_HOPS_NORM: Final[float] = float(MAX_BLUE + MAX_RELAY)


@dataclass(frozen=True)
class _Snapshot:
    blue_pos:  np.ndarray
    blue_vel:  np.ndarray
    relay_pos: np.ndarray
    red_pos:   np.ndarray
    graph:     GraphState


class Simulator:
    """Headless 2D swarm sim. No pygame dependency."""

    def __init__(self, seed: int = SEED) -> None:
        self._rng = np.random.default_rng(seed)
        self.t: int = 0
        self.blue_pos:  np.ndarray = np.zeros((N_BLUE, 2), dtype=np.float32)
        self.blue_vel:  np.ndarray = np.zeros((N_BLUE, 2), dtype=np.float32)
        self.relay_pos: np.ndarray = np.zeros((N_RELAY, 2), dtype=np.float32)
        self.red_pos:   np.ndarray = np.zeros((N_RED, 2),  dtype=np.float32)
        self.red_vel:   np.ndarray = np.zeros((N_RED, 2),  dtype=np.float32)
        self.objectives:   np.ndarray = np.zeros((N_OBJECTIVES, 2), dtype=np.float32)
        self.objective_idx: int = 0
        self.current_graph: GraphState
        # Buffer of recent snapshots so [0] is always K-step-delayed.
        self._obs_buffer: collections.deque[_Snapshot] = collections.deque(
            maxlen=COMM_LATENCY_STEPS + 1
        )
        # True (non-stale) connectivity ratio per step, used for labelling.
        self._conn_history: list[float] = []
        self.reset()

    def reset(self) -> None:
        """Re-randomize placements using the existing RNG (not reseeded)."""
        self.t = 0
        self.objective_idx = 0

        # Relays on a jittered grid. Deterministic backbone, per-episode jitter.
        cols = max(1, N_RELAY // 2)
        rows = max(1, (N_RELAY + cols - 1) // cols)
        xs = np.linspace(_SPAWN_MARGIN, WORLD_WIDTH  - _SPAWN_MARGIN, cols)
        ys = np.linspace(_SPAWN_MARGIN, WORLD_HEIGHT - _SPAWN_MARGIN, rows)
        gx, gy = np.meshgrid(xs, ys)
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)[:N_RELAY]
        jitter = self._rng.uniform(-40.0, 40.0, size=grid.shape)
        self.relay_pos = (grid + jitter).astype(np.float32)

        # Blues spawn clustered in the left third so they have somewhere to go.
        spawn = np.array([
            self._rng.uniform(_SPAWN_MARGIN, WORLD_WIDTH * 0.3),
            self._rng.uniform(_SPAWN_MARGIN, WORLD_HEIGHT - _SPAWN_MARGIN),
        ], dtype=np.float32)
        offsets = self._rng.normal(0.0, _BLUE_SPAWN_RADIUS / 2.0,
                                   size=(N_BLUE, 2))
        self.blue_pos = (spawn[None, :] + offsets).astype(np.float32)
        self.blue_vel = np.zeros((N_BLUE, 2), dtype=np.float32)

        # Objectives in the right two-thirds, ordered closest-to-spawn first.
        objs = np.stack([
            self._rng.uniform(WORLD_WIDTH * 0.4, WORLD_WIDTH  - _SPAWN_MARGIN,
                              size=N_OBJECTIVES),
            self._rng.uniform(_SPAWN_MARGIN,     WORLD_HEIGHT - _SPAWN_MARGIN,
                              size=N_OBJECTIVES),
        ], axis=1).astype(np.float32)
        order = np.argsort(np.linalg.norm(objs - spawn[None, :], axis=1))
        self.objectives = objs[order]

        # Reds biased toward the swarm/objective midline so they actually
        # interact with the trajectory.
        bias = (spawn + self.objectives.mean(axis=0)) / 2.0
        red_offsets = self._rng.normal(0.0, 200.0, size=(N_RED, 2))
        self.red_pos = np.clip(
            bias[None, :] + red_offsets,
            _SPAWN_MARGIN,
            [WORLD_WIDTH - _SPAWN_MARGIN, WORLD_HEIGHT - _SPAWN_MARGIN],
        ).astype(np.float32)
        self.red_vel = np.zeros((N_RED, 2), dtype=np.float32)

        self.current_graph = build_graph_state(
            self.blue_pos, self.relay_pos, self.red_pos
        )
        self._obs_buffer.clear()
        self._obs_buffer.append(self._snapshot())
        self._conn_history = [self.current_graph.connectivity_ratio]

    def step(self) -> None:
        self._step_reds()
        self._step_blues()
        self._maybe_advance_objective()
        self.current_graph = build_graph_state(
            self.blue_pos, self.relay_pos, self.red_pos
        )
        self._obs_buffer.append(self._snapshot())
        self._conn_history.append(self.current_graph.connectivity_ratio)
        self.t += 1

    def _step_reds(self) -> None:
        """Reds drift toward the Blue centroid with noise."""
        centroid = self.blue_pos.mean(axis=0)
        direction = centroid[None, :] - self.red_pos
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = direction / np.maximum(norm, 1e-6)
        noise = self._rng.normal(0.0, _RED_NOISE_STD,
                                 size=self.red_pos.shape).astype(np.float32)
        self.red_vel = (direction * RED_MAX_SPEED + noise).astype(np.float32)
        self._cap_speed(self.red_vel, RED_MAX_SPEED)
        self.red_pos = self.red_pos + self.red_vel * SIM_DT
        self._clip_to_world(self.red_pos)

    def _step_blues(self) -> None:
        """Steer toward the current objective with capped accel + speed."""
        target = self.objectives[self.objective_idx]
        direction = target[None, :] - self.blue_pos
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = direction / np.maximum(norm, 1e-6)
        desired_vel = direction * BLUE_MAX_SPEED
        delta_v = desired_vel - self.blue_vel
        delta_norm = np.linalg.norm(delta_v, axis=1, keepdims=True)
        scale = np.minimum(1.0, BLUE_ACCEL / np.maximum(delta_norm, 1e-6))
        self.blue_vel = (self.blue_vel + delta_v * scale).astype(np.float32)
        self._cap_speed(self.blue_vel, BLUE_MAX_SPEED)
        self.blue_pos = self.blue_pos + self.blue_vel * SIM_DT
        self._clip_to_world(self.blue_pos)

    def _maybe_advance_objective(self) -> None:
        if self.objective_idx >= len(self.objectives) - 1:
            return
        centroid = self.blue_pos.mean(axis=0)
        target = self.objectives[self.objective_idx]
        if float(np.linalg.norm(centroid - target)) < _OBJECTIVE_REACHED_RADIUS:
            self.objective_idx += 1

    def flatten_state(self, *, delayed: bool = True) -> np.ndarray:
        """Pack a snapshot into the (STATE_DIM,) state vector.

        delayed=True returns the K-step-stale snapshot (what models see);
        delayed=False uses the current truth (debugging / labelling only).
        """
        snap = self._obs_buffer[0] if delayed else self._snapshot()
        out = np.zeros(STATE_DIM, dtype=np.float32)

        n = snap.blue_pos.shape[0]
        blue_block = out[: MAX_BLUE * PER_BLUE_FEATURES].reshape(
            MAX_BLUE, PER_BLUE_FEATURES
        )
        blue_block[:n, 0] = snap.blue_pos[:, 0] / WORLD_WIDTH
        blue_block[:n, 1] = snap.blue_pos[:, 1] / WORLD_HEIGHT
        blue_block[:n, 2] = snap.blue_vel[:, 0] / BLUE_MAX_SPEED
        blue_block[:n, 3] = snap.blue_vel[:, 1] / BLUE_MAX_SPEED
        blue_block[:n, 4] = snap.graph.connected.astype(np.float32)

        m = snap.relay_pos.shape[0]
        relay_offset = MAX_BLUE * PER_BLUE_FEATURES
        relay_block = out[
            relay_offset : relay_offset + MAX_RELAY * PER_RELAY_FEATURES
        ].reshape(MAX_RELAY, PER_RELAY_FEATURES)
        relay_block[:m, 0] = snap.relay_pos[:, 0] / WORLD_WIDTH
        relay_block[:m, 1] = snap.relay_pos[:, 1] / WORLD_HEIGHT

        s = relay_offset + MAX_RELAY * PER_RELAY_FEATURES
        g = snap.graph
        out[s + 0] = g.connectivity_ratio
        out[s + 1] = g.avg_path_length / _HOPS_NORM
        out[s + 2] = g.jam_coverage
        out[s + 3] = g.mean_red_distance / _WORLD_DIAG
        out[s + 4] = g.fraction_blues_jammed
        assert s + SCALAR_FEATURES == STATE_DIM, "state-vector layout drift"
        return out

    def connectivity_history(self) -> np.ndarray:
        return np.asarray(self._conn_history, dtype=np.float32)

    def is_failed(self) -> bool:
        return is_failure_sustained(self.connectivity_history())

    def is_done(self) -> bool:
        return self.is_failed() or self.t >= MAX_EPISODE_STEPS

    def _snapshot(self) -> _Snapshot:
        # Copies, not references, so later in-place updates don't corrupt
        # entries already in the buffer.
        return _Snapshot(
            blue_pos=self.blue_pos.copy(),
            blue_vel=self.blue_vel.copy(),
            relay_pos=self.relay_pos.copy(),
            red_pos=self.red_pos.copy(),
            graph=self.current_graph,
        )

    @staticmethod
    def _cap_speed(vel: np.ndarray, max_speed: float) -> None:
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_speed / np.maximum(speed, 1e-6))
        vel *= scale

    @staticmethod
    def _clip_to_world(pos: np.ndarray) -> None:
        np.clip(pos[:, 0], 0.0, WORLD_WIDTH,  out=pos[:, 0])
        np.clip(pos[:, 1], 0.0, WORLD_HEIGHT, out=pos[:, 1])


# Renderer is separate so the headless sim doesn't pull in pygame.

_VIZ_JAM_NX: Final[int] = 60
_VIZ_JAM_NY: Final[int] = 45

# Risk sparkline geometry. 60 steps ~= 2s at FPS=30.
_SPARKLINE_LENGTH: Final[int] = 60
_SPARKLINE_BOX_W:  Final[int] = 240
_SPARKLINE_BOX_H:  Final[int] = 60
_SPARKLINE_X:      Final[int] = 10
_SPARKLINE_Y:      Final[int] = 160


class Renderer:
    """Pygame visualization of a Simulator instance."""

    def __init__(self) -> None:
        import pygame
        self._pg = pygame
        pygame.init()
        self._screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
        pygame.display.set_caption("MindSwarm V2")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 14)
        self._jam_surf = pygame.Surface(
            (_VIZ_JAM_NX, _VIZ_JAM_NY), flags=pygame.SRCALPHA
        )
        self._risk_history: collections.deque[tuple[float, float]] = (
            collections.deque(maxlen=_SPARKLINE_LENGTH)
        )
        self._last_pushed_t: Optional[int] = None

    def render(
        self,
        sim: Simulator,
        jepa_risk: Optional[float] = None,
        baseline_risk: Optional[float] = None,
    ) -> None:
        """Draw one frame. Risk overlays are optional."""
        self._update_sparkline(sim.t, jepa_risk, baseline_risk)
        self._screen.fill(COLOR_BG)
        self._draw_jam(sim.red_pos)
        self._draw_edges(sim)
        self._draw_relays(sim.relay_pos)
        self._draw_objectives(sim.objectives, sim.objective_idx)
        self._draw_blues(sim.blue_pos, sim.current_graph.connected)
        self._draw_reds(sim.red_pos)
        self._draw_hud(sim, jepa_risk, baseline_risk)
        self._draw_sparkline()
        self._pg.display.flip()
        self._clock.tick(FPS)

    def poll_events(self) -> dict[str, bool]:
        flags = {"quit": False, "reset": False}
        for ev in self._pg.event.get():
            if ev.type == self._pg.QUIT:
                flags["quit"] = True
            elif ev.type == self._pg.KEYDOWN:
                if ev.key == self._pg.K_ESCAPE:
                    flags["quit"] = True
                elif ev.key == self._pg.K_r:
                    flags["reset"] = True
        return flags

    def close(self) -> None:
        self._pg.quit()

    def _draw_jam(self, red_pos: np.ndarray) -> None:
        xs = np.linspace(0.0, float(WORLD_WIDTH),  _VIZ_JAM_NX, dtype=np.float32)
        ys = np.linspace(0.0, float(WORLD_HEIGHT), _VIZ_JAM_NY, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        intensities = _jam_intensity_at(pts, red_pos).reshape(_VIZ_JAM_NY, _VIZ_JAM_NX)
        alpha = np.clip(intensities / JAM_BREAK_THRESHOLD, 0.0, 1.0) * 140.0
        rgba = np.zeros((_VIZ_JAM_NY, _VIZ_JAM_NX, 4), dtype=np.uint8)
        rgba[..., 0] = COLOR_JAM_FIELD[0]
        rgba[..., 1] = COLOR_JAM_FIELD[1]
        rgba[..., 2] = COLOR_JAM_FIELD[2]
        rgba[..., 3] = alpha.astype(np.uint8)
        # surfarray uses (width, height, channels); blit_array drops alpha.
        self._pg.surfarray.blit_array(
            self._jam_surf, rgba.transpose(1, 0, 2)[..., :3]
        )
        alpha_surf = self._pg.surfarray.pixels_alpha(self._jam_surf)
        alpha_surf[...] = rgba[..., 3].T
        del alpha_surf  # release surface lock before blitting
        scaled = self._pg.transform.smoothscale(
            self._jam_surf, (WORLD_WIDTH, WORLD_HEIGHT)
        )
        self._screen.blit(scaled, (0, 0))

    def _draw_edges(self, sim: Simulator) -> None:
        nodes = np.concatenate([sim.blue_pos, sim.relay_pos], axis=0)
        g = sim.current_graph
        # Broken first so active edges paint on top.
        for active in (False, True):
            mask = (g.edge_active == active)
            color = COLOR_EDGE_OK if active else COLOR_EDGE_BROKEN
            width = 2 if active else 1
            for u, v in g.edge_index[mask]:
                self._pg.draw.line(
                    self._screen, color,
                    nodes[int(u)].astype(int), nodes[int(v)].astype(int),
                    width,
                )

    def _draw_relays(self, relay_pos: np.ndarray) -> None:
        for p in relay_pos:
            x, y = int(p[0]), int(p[1])
            self._pg.draw.rect(
                self._screen, COLOR_RELAY, (x - 6, y - 6, 12, 12)
            )

    def _draw_objectives(
        self, objectives: np.ndarray, current_idx: int
    ) -> None:
        for i, p in enumerate(objectives):
            x, y = int(p[0]), int(p[1])
            color = COLOR_TEXT if i == current_idx else (90, 90, 110)
            self._pg.draw.circle(self._screen, color, (x, y), 14, width=2)

    def _draw_blues(
        self, blue_pos: np.ndarray, connected: np.ndarray
    ) -> None:
        for p, alive in zip(blue_pos, connected):
            color = COLOR_BLUE if bool(alive) else COLOR_BLUE_DISCONNECTED
            self._pg.draw.circle(
                self._screen, color, (int(p[0]), int(p[1])), 6
            )

    def _draw_reds(self, red_pos: np.ndarray) -> None:
        for p in red_pos:
            x, y = int(p[0]), int(p[1])
            self._pg.draw.polygon(
                self._screen, COLOR_RED,
                [(x, y - 8), (x - 7, y + 6), (x + 7, y + 6)],
            )

    def _draw_hud(
        self,
        sim: Simulator,
        jepa_risk: Optional[float],
        baseline_risk: Optional[float],
    ) -> None:
        g = sim.current_graph
        lines = [
            f"t={sim.t:4d}/{MAX_EPISODE_STEPS}",
            f"conn={g.connectivity_ratio:.2f}  jam_cov={g.jam_coverage:.2f}",
            f"obj={sim.objective_idx + 1}/{len(sim.objectives)}",
        ]
        if jepa_risk is not None:
            lines.append(f"JEPA   risk={jepa_risk:.2f}")
        if baseline_risk is not None:
            lines.append(f"LSTM   risk={baseline_risk:.2f}")
        if sim.is_failed():
            lines.append("FAILED (press R to reset)")
        for i, line in enumerate(lines):
            surf = self._font.render(line, True, COLOR_TEXT)
            self._screen.blit(surf, (10, 8 + 18 * i))

    def _update_sparkline(
        self,
        sim_t: int,
        jepa_risk: Optional[float],
        baseline_risk: Optional[float],
    ) -> None:
        # sim_t going backward = sim was reset; clear history.
        if self._last_pushed_t is not None and sim_t < self._last_pushed_t:
            self._risk_history.clear()
            self._last_pushed_t = None
        # Warm-up: buffer not full yet.
        if jepa_risk is None or baseline_risk is None:
            return
        # Sim frozen on failure frame: don't duplicate the last sample.
        if sim_t == self._last_pushed_t:
            return
        self._risk_history.append((jepa_risk, baseline_risk))
        self._last_pushed_t = sim_t

    def _draw_sparkline(self) -> None:
        if len(self._risk_history) < 2:
            return

        pg = self._pg
        right_edge = _SPARKLINE_X + _SPARKLINE_BOX_W - 1
        bottom_edge = _SPARKLINE_Y + _SPARKLINE_BOX_H - 1
        step_w = (_SPARKLINE_BOX_W - 1) / (_SPARKLINE_LENGTH - 1)

        panel = pg.Surface(
            (_SPARKLINE_BOX_W, _SPARKLINE_BOX_H), flags=pg.SRCALPHA
        )
        panel.fill((30, 30, 45, 200))
        self._screen.blit(panel, (_SPARKLINE_X, _SPARKLINE_Y))
        pg.draw.rect(
            self._screen, COLOR_TEXT,
            (_SPARKLINE_X, _SPARKLINE_Y, _SPARKLINE_BOX_W, _SPARKLINE_BOX_H),
            width=1,
        )
        # Decision-threshold reference line at risk = 0.5.
        mid_y = _SPARKLINE_Y + _SPARKLINE_BOX_H // 2
        pg.draw.line(
            self._screen, (110, 110, 130),
            (_SPARKLINE_X, mid_y), (right_edge, mid_y), 1,
        )

        history = list(self._risk_history)
        n = len(history)
        for series_idx, color in (
            (0, COLOR_SPARK_JEPA),
            (1, COLOR_SPARK_BASELINE),
        ):
            pts: list[tuple[int, int]] = []
            for i, pair in enumerate(history):
                steps_back = (n - 1) - i               # 0 = newest, at right edge
                x = round(right_edge - steps_back * step_w)
                y = round(bottom_edge - pair[series_idx] * (_SPARKLINE_BOX_H - 1))
                pts.append((x, y))
            pg.draw.lines(self._screen, color, False, pts, 2)

        title = self._font.render(
            "risk forecast (last 60 steps)", True, COLOR_TEXT
        )
        self._screen.blit(title, (_SPARKLINE_X, _SPARKLINE_Y - 18))
        legend_x = _SPARKLINE_X + _SPARKLINE_BOX_W + 8
        self._screen.blit(
            self._font.render("JEPA", True, COLOR_SPARK_JEPA),
            (legend_x, _SPARKLINE_Y + 6),
        )
        self._screen.blit(
            self._font.render("LSTM", True, COLOR_SPARK_BASELINE),
            (legend_x, _SPARKLINE_Y + 6 + 18),
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MindSwarm V2 interactive sim.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS,
                   help="Hard cap on episode length (auto-reset on hit).")
    return p.parse_args()


def _main() -> None:
    args = _parse_args()
    sim = Simulator(seed=args.seed)
    renderer = Renderer()
    try:
        while True:
            sim.step()
            events = renderer.poll_events()
            if events["quit"]:
                break
            if events["reset"] or sim.t >= args.max_steps or sim.is_failed():
                renderer.render(sim)   # one final frame so user sees end state
                sim.reset()
                continue
            renderer.render(sim)
    finally:
        renderer.close()


if __name__ == "__main__":
    _main()
