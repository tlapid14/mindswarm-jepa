"""Run headless episodes and build the supervised dataset.

For each step t we record:
    state_history[t]  -- delayed state vector (what the model would see)
    conn_history[t]   -- true connectivity ratio (used only for labels)

Sliding windows then produce (X_past, X_future, y_failure) tuples. The
inputs are stale because that's what the model sees at inference; the
label uses true connectivity so the supervisor isn't itself fooled by
the latency. Train/val split is by episode so windows can't leak across.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from config import (
    DATA_DIR,
    DATASET_STRIDE,
    FAILURE_HORIZON,
    FUTURE_OFFSET,
    MAX_EPISODE_STEPS,
    PAST_WINDOW,
    SEED,
    STATE_DIM,
)
from network import is_failure_sustained
from simulate import Simulator

# Valid anchor times t for a (past, future, label) tuple:
#   t >= PAST_WINDOW - 1                          (past window fits)
#   t <= MAX_EPISODE_STEPS - max(FUTURE_OFFSET, FAILURE_HORIZON)
_T_MIN: int = PAST_WINDOW - 1
_T_MAX: int = MAX_EPISODE_STEPS - max(FUTURE_OFFSET, FAILURE_HORIZON)
_ANCHOR_TIMES: np.ndarray = np.arange(_T_MIN, _T_MAX + 1, DATASET_STRIDE)
SAMPLES_PER_EPISODE: int = int(_ANCHOR_TIMES.size)

_PROGRESS_EVERY: int = 50


def _run_episode(sim: Simulator) -> tuple[np.ndarray, np.ndarray]:
    """Return (delayed-state history, true connectivity history)."""
    # +1 because we record both the initial state and the post-step
    # state for each of MAX_EPISODE_STEPS steps.
    states = np.zeros((MAX_EPISODE_STEPS + 1, STATE_DIM), dtype=np.float32)
    states[0] = sim.flatten_state(delayed=True)
    for t in range(1, MAX_EPISODE_STEPS + 1):
        sim.step()
        states[t] = sim.flatten_state(delayed=True)
    return states, sim.connectivity_history()


def _extract_samples(
    state_history: np.ndarray,
    conn_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = SAMPLES_PER_EPISODE
    X_past   = np.zeros((n, PAST_WINDOW, STATE_DIM), dtype=np.float32)
    X_future = np.zeros((n, STATE_DIM),              dtype=np.float32)
    y        = np.zeros(n,                            dtype=np.uint8)
    for i, t in enumerate(_ANCHOR_TIMES):
        X_past[i]   = state_history[t - PAST_WINDOW + 1 : t + 1]
        X_future[i] = state_history[t + FUTURE_OFFSET]
        # Label window starts at t+1: a failure already at t is input,
        # not a prediction.
        future_conn = conn_history[t + 1 : t + FAILURE_HORIZON + 1]
        y[i] = 1 if is_failure_sustained(future_conn) else 0
    return X_past, X_future, y


def _save_split(
    path: Path,
    X_past: np.ndarray,
    X_future: np.ndarray,
    y: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X_past=X_past, X_future=X_future, y_failure=y)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-episodes", type=int, default=500,
                   help="Episodes to run; total samples = N * %d" % SAMPLES_PER_EPISODE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Fraction of EPISODES held out for validation.")
    p.add_argument("--output-dir", type=Path, default=DATA_DIR)
    return p.parse_args()


def _main() -> None:
    args = _parse_args()

    sim = Simulator(seed=args.seed)
    total_samples = args.n_episodes * SAMPLES_PER_EPISODE

    X_past   = np.zeros((total_samples, PAST_WINDOW, STATE_DIM), dtype=np.float32)
    X_future = np.zeros((total_samples, STATE_DIM),              dtype=np.float32)
    y        = np.zeros(total_samples,                            dtype=np.uint8)
    episode_id = np.zeros(total_samples, dtype=np.int32)

    start_time = time.time()
    for ep in range(args.n_episodes):
        sim.reset()
        states, conn = _run_episode(sim)
        Xp, Xf, y_ep = _extract_samples(states, conn)

        slc = slice(ep * SAMPLES_PER_EPISODE, (ep + 1) * SAMPLES_PER_EPISODE)
        X_past[slc]     = Xp
        X_future[slc]   = Xf
        y[slc]          = y_ep
        episode_id[slc] = ep

        if (ep + 1) % _PROGRESS_EVERY == 0:
            elapsed = time.time() - start_time
            rate = (ep + 1) / max(elapsed, 1e-9)
            so_far = y[: slc.stop].mean()
            print(f"ep {ep + 1:4d}/{args.n_episodes}  "
                  f"{rate:5.1f} ep/s  "
                  f"failure_rate so far: {so_far:.2%}")

    # Episode-level split: shuffle episode indices, hold out val_frac.
    rng = np.random.default_rng(args.seed)
    ep_order = np.arange(args.n_episodes)
    rng.shuffle(ep_order)
    n_val_eps = int(round(args.n_episodes * args.val_frac))
    val_episode_ids = set(int(e) for e in ep_order[:n_val_eps])
    is_val = np.fromiter(
        (eid in val_episode_ids for eid in episode_id),
        dtype=bool,
        count=total_samples,
    )

    train_path = args.output_dir / "train.npz"
    val_path   = args.output_dir / "val.npz"
    _save_split(train_path, X_past[~is_val], X_future[~is_val], y[~is_val])
    _save_split(val_path,   X_past[ is_val], X_future[ is_val], y[ is_val])

    print()
    print(f"wrote {train_path}  ({(~is_val).sum():6d} samples, "
          f"failure_rate={y[~is_val].mean():.2%})")
    print(f"wrote {val_path}    ({ is_val .sum():6d} samples, "
          f"failure_rate={y[ is_val].mean():.2%})")
    print(f"overall failure rate: {y.mean():.2%}  "
          f"(target ~30-50%; tune JAM_BREAK_THRESHOLD if far off)")


if __name__ == "__main__":
    _main()
