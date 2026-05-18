"""Live demo: run the simulator with JEPA + LSTM risk predictions overlaid.

Controls:
    R              -- reset to a new random episode
    ESC / close    -- quit
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from config import CHECKPOINT_DIR, PAST_WINDOW, SEED
from models import BaselineLSTM, JEPAModel
from simulate import Renderer, Simulator


def _load(
    model_factory: Callable[[], torch.nn.Module],
    path: Path,
    device: torch.device,
) -> torch.nn.Module:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Train it first: python train_{path.stem}.py"
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = model_factory().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _infer(
    buffer: collections.deque,
    jepa: JEPAModel,
    baseline: BaselineLSTM,
    device: torch.device,
) -> tuple[float | None, float | None]:
    """Returns (None, None) while the buffer is still filling."""
    if len(buffer) < PAST_WINDOW:
        return None, None
    past_np = np.stack(list(buffer), axis=0)
    past = torch.from_numpy(past_np).unsqueeze(0).to(device)
    with torch.inference_mode():
        j = float(jepa.predict_risk(past).item())
        b = float(baseline.predict_risk(past).item())
    return j, b


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live JEPA + baseline risk demo.")
    p.add_argument("--seed",     type=int,  default=SEED)
    p.add_argument("--ckpt-dir", type=Path, default=CHECKPOINT_DIR)
    p.add_argument("--device",   type=str,  default=None,
                   help="cpu | cuda. Default: cuda if available, else cpu.")
    return p.parse_args()


def _main() -> None:
    args = _parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load both checkpoints before opening the window so a missing
    # checkpoint surfaces as a clean error.
    jepa     = _load(JEPAModel,    args.ckpt_dir / "jepa.pt",     device)
    baseline = _load(BaselineLSTM, args.ckpt_dir / "baseline.pt", device)

    print("controls:  R = reset    ESC / close-window = quit")

    sim = Simulator(seed=args.seed)
    renderer = Renderer()

    # Rolling buffer of the last PAST_WINDOW delayed states; one row of
    # this matches one row of training X_past.
    buffer: collections.deque[np.ndarray] = collections.deque(
        maxlen=PAST_WINDOW
    )
    buffer.append(sim.flatten_state(delayed=True))

    jepa_risk:     float | None = None
    baseline_risk: float | None = None

    try:
        while True:
            events = renderer.poll_events()
            if events["quit"]:
                break

            if events["reset"]:
                sim.reset()
                buffer.clear()
                buffer.append(sim.flatten_state(delayed=True))
                jepa_risk = baseline_risk = None
            elif not sim.is_done():
                sim.step()
                buffer.append(sim.flatten_state(delayed=True))
                jepa_risk, baseline_risk = _infer(buffer, jepa, baseline, device)
            # else: sim is done. Freeze on the failure frame; the last
            # risk values stay on the HUD until the user hits R.

            renderer.render(
                sim, jepa_risk=jepa_risk, baseline_risk=baseline_risk
            )
    finally:
        renderer.close()


if __name__ == "__main__":
    _main()
