"""Compare the trained JEPA model and LSTM baseline on the val set.

Prints a per-metric winner table and writes three plots to results/:
calibration.png, metrics.png, learning_curves.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")   # must come before pyplot; headless backend
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, RESULTS_DIR
from data import load_split, make_dataloader
from models import BaselineLSTM, JEPAModel

_DEFAULT_BATCH_SIZE: int   = 512
_CALIBRATION_BINS:   int   = 10
_DECISION_THRESHOLD: float = 0.5


def _load_ckpt(
    path: Path,
    model_factory: Callable[[], torch.nn.Module],
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Train it first: python train_{path.stem}.py"
        )
    # weights_only=False because the dict has plain-Python metadata
    # (args, history) alongside the state_dict.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = model_factory().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.inference_mode()
def _predict_all(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    is_jepa: bool,
) -> tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Return (probs, labels, latent_mse_or_None)."""
    all_probs: list[np.ndarray] = []
    all_ys:    list[np.ndarray] = []
    sum_latent_mse = 0.0
    n_seen = 0
    for past, future, y in loader:
        past = past.to(device)
        y    = y.to(device)
        if is_jepa:
            future = future.to(device)
            out = model(past, future)
            probs = torch.sigmoid(out["risk_logit"])
            mse = F.mse_loss(out["z_pred_future"], out["z_future"])
            sum_latent_mse += mse.item() * y.shape[0]
        else:
            probs = torch.sigmoid(model(past))
        all_probs.append(probs.cpu().numpy())
        all_ys.append(y.cpu().numpy())
        n_seen += y.shape[0]
    probs_arr = np.concatenate(all_probs)
    ys_arr    = np.concatenate(all_ys)
    latent_mse = (sum_latent_mse / n_seen) if is_jepa else None
    return probs_arr, ys_arr, latent_mse


def _classification_metrics(
    probs: np.ndarray,
    ys: np.ndarray,
    threshold: float = _DECISION_THRESHOLD,
) -> dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    y_int = ys.astype(np.int64)
    tp = int(((preds == 1) & (y_int == 1)).sum())
    tn = int(((preds == 0) & (y_int == 0)).sum())
    fp = int(((preds == 1) & (y_int == 0)).sum())
    fn = int(((preds == 0) & (y_int == 1)).sum())
    n = tp + tn + fp + fn
    accuracy  = (tp + tn) / max(n, 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1_denom  = precision + recall
    f1 = (2 * precision * recall / f1_denom) if f1_denom > 0 else 0.0
    return {
        "accuracy":  float(accuracy),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
    }


def _calibration(
    probs: np.ndarray,
    ys: np.ndarray,
    n_bins: int = _CALIBRATION_BINS,
) -> dict[str, np.ndarray | float]:
    """Reliability diagram inputs + Expected Calibration Error."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)

    bin_conf  = np.zeros(n_bins, dtype=np.float64)
    bin_acc   = np.zeros(n_bins, dtype=np.float64)
    bin_count = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = bin_ids == b
        bin_count[b] = int(mask.sum())
        if bin_count[b] > 0:
            bin_conf[b] = float(probs[mask].mean())
            bin_acc[b]  = float(ys[mask].mean())

    n_total = int(bin_count.sum())
    nonempty = bin_count > 0
    ece = float(np.sum(
        (bin_count[nonempty] / max(n_total, 1))
        * np.abs(bin_conf[nonempty] - bin_acc[nonempty])
    ))
    return {
        "bin_conf":  bin_conf,
        "bin_acc":   bin_acc,
        "bin_count": bin_count,
        "ece":       ece,
    }


def _plot_calibration(jepa_cal: dict, baseline_cal: dict, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    for cal, marker, label in (
        (jepa_cal,     "o-", "JEPA"),
        (baseline_cal, "s-", "LSTM baseline"),
    ):
        nz = cal["bin_count"] > 0
        ax.plot(
            cal["bin_conf"][nz], cal["bin_acc"][nz], marker,
            label=f"{label}  (ECE={cal['ece']:.3f})",
        )
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical positive rate")
    ax.set_title("Reliability diagram (val set)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_metrics_bar(
    jepa_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    save_path: Path,
) -> None:
    keys = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(keys))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, [jepa_metrics[k] for k in keys],
           width=width, label="JEPA")
    ax.bar(x + width / 2, [baseline_metrics[k] for k in keys],
           width=width, label="LSTM baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Val-set classification metrics")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_learning_curves(
    jepa_history: dict[str, list[float]],
    baseline_history: dict[str, list[float]],
    save_path: Path,
) -> None:
    # JEPA's val_loss combines MSE and BCE, so we plot val_bce against
    # baseline's val_loss (which is just BCE) to keep it comparable.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(range(1, len(jepa_history["val_acc"]) + 1),
            jepa_history["val_acc"], "o-", label="JEPA")
    ax.plot(range(1, len(baseline_history["val_acc"]) + 1),
            baseline_history["val_acc"], "s-", label="LSTM baseline")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val accuracy")
    ax.set_title("Validation accuracy")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(range(1, len(jepa_history["val_bce"]) + 1),
            jepa_history["val_bce"], "o-", label="JEPA val_bce")
    ax.plot(range(1, len(baseline_history["val_loss"]) + 1),
            baseline_history["val_loss"], "s-", label="LSTM val_loss (=BCE)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE")
    ax.set_title("Validation BCE")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_comparison(
    jepa_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    jepa_latent_mse: Optional[float],
) -> None:
    rows = [
        ("accuracy",  True),
        ("precision", True),
        ("recall",    True),
        ("f1",        True),
        ("ece",       False),   # lower is better
    ]
    print()
    print(f"{'metric':<12} {'JEPA':>10} {'Baseline':>10}   winner")
    print("-" * 48)
    jepa_wins = 0
    baseline_wins = 0
    for key, higher_is_better in rows:
        j = jepa_metrics[key]
        b = baseline_metrics[key]
        if abs(j - b) < 1e-9:
            winner = "tie"
        elif (j > b) == higher_is_better:
            winner = "JEPA"
            jepa_wins += 1
        else:
            winner = "Baseline"
            baseline_wins += 1
        print(f"{key:<12} {j:>10.4f} {b:>10.4f}   {winner}")

    if jepa_latent_mse is not None:
        print(f"{'latent_mse':<12} {jepa_latent_mse:>10.4f} {'  N/A':>10}   JEPA-only")

    print("-" * 48)
    print(f"JEPA wins {jepa_wins} metric(s); baseline wins {baseline_wins}.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare JEPA vs LSTM baseline on val set.")
    p.add_argument("--batch-size",     type=int,  default=_DEFAULT_BATCH_SIZE)
    p.add_argument("--ckpt-dir",       type=Path, default=CHECKPOINT_DIR)
    p.add_argument("--output-dir",     type=Path, default=RESULTS_DIR)
    p.add_argument("--device",         type=str,  default=None,
                   help="cpu | cuda. Default: cuda if available, else cpu.")
    return p.parse_args()


def _main() -> None:
    args = _parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = load_split("val")
    val_loader = make_dataloader(
        val_ds, args.batch_size, shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    jepa,     jepa_ckpt     = _load_ckpt(args.ckpt_dir / "jepa.pt",
                                         JEPAModel,    device)
    baseline, baseline_ckpt = _load_ckpt(args.ckpt_dir / "baseline.pt",
                                         BaselineLSTM, device)

    jepa_probs, ys, jepa_latent_mse = _predict_all(
        jepa, val_loader, device, is_jepa=True
    )
    baseline_probs, _, _ = _predict_all(
        baseline, val_loader, device, is_jepa=False
    )

    jepa_metrics     = _classification_metrics(jepa_probs, ys)
    baseline_metrics = _classification_metrics(baseline_probs, ys)
    jepa_cal     = _calibration(jepa_probs, ys)
    baseline_cal = _calibration(baseline_probs, ys)
    jepa_metrics["ece"]     = jepa_cal["ece"]
    baseline_metrics["ece"] = baseline_cal["ece"]

    print(f"device:   {device}")
    print(f"val set:  {val_ds}")
    print(f"JEPA      best_epoch={jepa_ckpt['best_epoch']}, "
          f"saved_val_loss={jepa_ckpt['val_metrics']['loss']:.4f}")
    print(f"Baseline  best_epoch={baseline_ckpt['best_epoch']}, "
          f"saved_val_loss={baseline_ckpt['val_metrics']['loss']:.4f}")
    _print_comparison(jepa_metrics, baseline_metrics, jepa_latent_mse)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cal_path   = args.output_dir / "calibration.png"
    bar_path   = args.output_dir / "metrics.png"
    curve_path = args.output_dir / "learning_curves.png"
    _plot_calibration(jepa_cal, baseline_cal, cal_path)
    _plot_metrics_bar(jepa_metrics, baseline_metrics, bar_path)
    _plot_learning_curves(
        jepa_ckpt["history"], baseline_ckpt["history"], curve_path
    )
    print()
    print(f"plots saved:")
    print(f"  {cal_path}")
    print(f"  {bar_path}")
    print(f"  {curve_path}")


if __name__ == "__main__":
    _main()
