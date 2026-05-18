"""Train the LSTM baseline failure predictor.

Mirrors train_jepa.py on optimizer, defaults, eval logic, and checkpoint
format so the JEPA-vs-baseline comparison isn't a tuning-knob comparison.
Loss is just BCE; the future state is loaded but unused.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, SEED
from data import load_split, make_dataloader
from models import BaselineLSTM

# Match train_jepa.py defaults.
_DEFAULT_EPOCHS:         int   = 30
_DEFAULT_BATCH_SIZE:     int   = 256
_DEFAULT_LR:             float = 1e-3
_DEFAULT_WEIGHT_DECAY:   float = 0.0
_DEFAULT_GRAD_CLIP_NORM: float = 1.0   # pass <=0 to disable

_CKPT_NAME: str = "baseline.pt"


@torch.inference_mode()
def _evaluate(
    model: BaselineLSTM,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sum_loss = 0.0
    sum_correct = 0
    n = 0
    for past, _future, y in loader:
        past = past.to(device)
        y    = y.to(device)
        logit = model(past)
        bce = F.binary_cross_entropy_with_logits(logit, y)
        bs = y.shape[0]
        sum_loss += bce.item() * bs
        preds = (logit > 0).float()
        sum_correct += int((preds == y).sum().item())
        n += bs
    return {"loss": sum_loss / n, "acc": sum_correct / n}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM baseline failure predictor.")
    p.add_argument("--epochs",       type=int,   default=_DEFAULT_EPOCHS)
    p.add_argument("--batch-size",   type=int,   default=_DEFAULT_BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=_DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=_DEFAULT_WEIGHT_DECAY)
    p.add_argument("--grad-clip-norm", type=float, default=_DEFAULT_GRAD_CLIP_NORM,
                   help="Max L2 grad norm before optimizer step. <=0 disables.")
    p.add_argument("--seed",         type=int,   default=SEED)
    p.add_argument("--output-dir",   type=Path,  default=CHECKPOINT_DIR)
    p.add_argument("--device",       type=str,   default=None,
                   help="cpu | cuda. Default: cuda if available, else cpu.")
    return p.parse_args()


def _main() -> None:
    args = _parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = load_split("train")
    val_ds   = load_split("val")
    val_pos = val_ds.positive_rate
    majority_acc = max(val_pos, 1.0 - val_pos)

    pin = (device.type == "cuda")
    train_loader = make_dataloader(train_ds, args.batch_size,
                                   shuffle=True,  pin_memory=pin)
    val_loader   = make_dataloader(val_ds,   args.batch_size,
                                   shuffle=False, pin_memory=pin)

    model = BaselineLSTM().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"device:    {device}")
    print(f"train:     {train_ds}")
    print(f"val:       {val_ds}")
    print(f"majority-class baseline val acc: {majority_acc:.2%}")
    print(f"params:    {n_params:,}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / _CKPT_NAME
    best_val_loss = float("inf")
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss":   [],
        "val_acc":    [],
    }

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sum_loss = 0.0
        n_batches = 0
        for past, _future, y in train_loader:
            past = past.to(device)
            y    = y.to(device)
            logit = model(past)
            loss = F.binary_cross_entropy_with_logits(logit, y)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip_norm
                )
            optimizer.step()

            sum_loss += loss.item()
            n_batches += 1

        train_loss = sum_loss / n_batches
        val = _evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])

        marker = ""
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            torch.save({
                "state_dict":  model.state_dict(),
                "best_epoch":  epoch + 1,
                "val_metrics": val,
                "history":     history,
                "args": {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in vars(args).items()},
            }, ckpt_path)
            marker = "  *best (saved)"

        print(
            f"ep {epoch + 1:3d}/{args.epochs} "
            f"({elapsed:4.1f}s) | "
            f"train loss={train_loss:.4f} | "
            f"val loss={val['loss']:.4f} acc={val['acc']:.2%}{marker}"
        )

    print()
    print(f"training done. best val loss = {best_val_loss:.4f}")
    print(f"checkpoint:    {ckpt_path}")


if __name__ == "__main__":
    _main()
