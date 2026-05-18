"""Train the JEPA-style failure predictor.

Loss is MSE(z_pred_future, z_future.detach()) + lambda * BCE(risk, y).
Saves the best-by-val-loss checkpoint to checkpoints/jepa.pt.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import BCE_LOSS_WEIGHT, CHECKPOINT_DIR, SEED
from data import load_split, make_dataloader
from models import JEPAModel

_DEFAULT_EPOCHS:         int   = 30
_DEFAULT_BATCH_SIZE:     int   = 256
_DEFAULT_LR:             float = 1e-3
_DEFAULT_WEIGHT_DECAY:   float = 0.0
_DEFAULT_GRAD_CLIP_NORM: float = 1.0   # pass <=0 to disable

_CKPT_NAME: str = "jepa.pt"


def _compute_loss(
    outputs: dict[str, torch.Tensor],
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # detach() here is the load-bearing JEPA detail: without it, the
    # shared embedder can collapse to a constant and drive MSE to zero.
    z_target = outputs["z_future"].detach()
    mse = F.mse_loss(outputs["z_pred_future"], z_target)
    bce = F.binary_cross_entropy_with_logits(outputs["risk_logit"], y)
    total = mse + BCE_LOSS_WEIGHT * bce
    return total, mse, bce


@torch.inference_mode()
def _evaluate(
    model: JEPAModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sum_mse = sum_bce = 0.0
    sum_correct = 0
    n = 0
    for past, future, y in loader:
        past   = past.to(device)
        future = future.to(device)
        y      = y.to(device)
        out = model(past, future)
        mse = F.mse_loss(out["z_pred_future"], out["z_future"])
        bce = F.binary_cross_entropy_with_logits(out["risk_logit"], y)
        bs = y.shape[0]
        sum_mse += mse.item() * bs
        sum_bce += bce.item() * bs
        preds = (out["risk_logit"] > 0).float()   # sigmoid(0) = 0.5 threshold
        sum_correct += int((preds == y).sum().item())
        n += bs
    mse_mean = sum_mse / n
    bce_mean = sum_bce / n
    return {
        "mse":  mse_mean,
        "bce":  bce_mean,
        "loss": mse_mean + BCE_LOSS_WEIGHT * bce_mean,
        "acc":  sum_correct / n,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train JEPA-style failure predictor.")
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

    model = JEPAModel().to(device)
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
    print(f"lambda (BCE weight): {BCE_LOSS_WEIGHT}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / _CKPT_NAME
    best_val_loss = float("inf")
    history: dict[str, list[float]] = {
        "train_loss": [], "train_mse": [], "train_bce": [],
        "val_loss":   [], "val_mse":   [], "val_bce":   [], "val_acc": [],
    }

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sum_loss = sum_mse = sum_bce = 0.0
        n_batches = 0
        for past, future, y in train_loader:
            past   = past.to(device)
            future = future.to(device)
            y      = y.to(device)
            out = model(past, future)
            loss, mse, bce = _compute_loss(out, y)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip_norm
                )
            optimizer.step()
            sum_loss += loss.item()
            sum_mse  += mse.item()
            sum_bce  += bce.item()
            n_batches += 1

        train_loss = sum_loss / n_batches
        train_mse  = sum_mse  / n_batches
        train_bce  = sum_bce  / n_batches
        val = _evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_mse"].append(train_mse)
        history["train_bce"].append(train_bce)
        history["val_loss"].append(val["loss"])
        history["val_mse"].append(val["mse"])
        history["val_bce"].append(val["bce"])
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
            f"train loss={train_loss:.4f} (mse={train_mse:.4f}, bce={train_bce:.4f}) | "
            f"val loss={val['loss']:.4f} (mse={val['mse']:.4f}, bce={val['bce']:.4f}) "
            f"acc={val['acc']:.2%}{marker}"
        )

    print()
    print(f"training done. best val loss = {best_val_loss:.4f}")
    print(f"checkpoint:    {ckpt_path}")


if __name__ == "__main__":
    _main()
