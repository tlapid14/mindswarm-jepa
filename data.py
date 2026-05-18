"""PyTorch Dataset + DataLoader for the generated .npz splits."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import DATA_DIR, PAST_WINDOW, STATE_DIM


class SwarmDataset(Dataset):
    """Loads one split (.npz) eagerly into memory as torch tensors."""

    def __init__(self, npz_path: Path) -> None:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {npz_path}.\n"
                f"Run: python generate_dataset.py"
            )

        with np.load(npz_path) as data:
            X_past_np   = data["X_past"]
            X_future_np = data["X_future"]
            y_np        = data["y_failure"]

        # Catch stale datasets after a config.py edit.
        if X_past_np.shape[1:] != (PAST_WINDOW, STATE_DIM):
            raise ValueError(
                f"X_past shape {X_past_np.shape[1:]} does not match "
                f"config (PAST_WINDOW={PAST_WINDOW}, STATE_DIM={STATE_DIM}). "
                f"Regenerate the dataset."
            )
        if X_future_np.shape[1:] != (STATE_DIM,):
            raise ValueError(
                f"X_future shape {X_future_np.shape[1:]} does not match "
                f"config (STATE_DIM={STATE_DIM}). Regenerate the dataset."
            )
        if not (y_np.shape[0] == X_past_np.shape[0] == X_future_np.shape[0]):
            raise ValueError(
                f"Sample-count mismatch: X_past={X_past_np.shape[0]}, "
                f"X_future={X_future_np.shape[0]}, y={y_np.shape[0]}."
            )

        self.X_past   = torch.from_numpy(X_past_np)        # (N, T, D) float32
        self.X_future = torch.from_numpy(X_future_np)      # (N, D)    float32
        # Cast once so the training loop doesn't redo it every batch.
        self.y        = torch.from_numpy(y_np).float()     # (N,)      float32

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_past[idx], self.X_future[idx], self.y[idx]

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean())

    def __repr__(self) -> str:
        return (
            f"SwarmDataset(n={len(self)}, "
            f"positive_rate={self.positive_rate:.2%})"
        )


def load_split(
    split: Literal["train", "val"],
    data_dir: Path = DATA_DIR,
) -> SwarmDataset:
    return SwarmDataset(data_dir / f"{split}.npz")


def make_dataloader(
    dataset: SwarmDataset,
    batch_size: int,
    *,
    shuffle: bool,
    pin_memory: bool = False,
) -> DataLoader:
    # num_workers=0 because the dataset is already in RAM; forking
    # workers would just duplicate it and add IPC overhead.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=0,
        drop_last=False,
    )
