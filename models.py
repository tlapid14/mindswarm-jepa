"""JEPA-style failure predictor and the LSTM baseline it's compared against.

JEPA stack:

    past   -> PastEncoder   -> z_context -> Predictor -> z_pred
                                                          |
                                                          +-> RiskHead -> logit
                                                          |
    future -> TargetEncoder -> z_future ---- (detach) ----+

Loss is MSE(z_pred, z_future.detach()) + lambda * BCE(logit, y). The
detach is what stops the shared embedder from collapsing both sides
to a constant and calling the prediction task solved.
"""

from __future__ import annotations

import torch
from torch import nn

from config import (
    BASELINE_LSTM_HIDDEN_DIM,
    GRU_HIDDEN_DIM,
    LATENT_DIM,
    PREDICTOR_HIDDEN_DIM,
    RISK_HEAD_HIDDEN_DIM,
    STATE_DIM,
)

_EMBEDDER_HIDDEN = LATENT_DIM * 2


class _StateEmbedder(nn.Module):
    """One state vector -> one latent. Shared between both encoders."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, _EMBEDDER_HIDDEN),
            nn.ReLU(),
            nn.Linear(_EMBEDDER_HIDDEN, LATENT_DIM),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PastEncoder(nn.Module):
    """(B, PAST_WINDOW, STATE_DIM) -> (B, LATENT_DIM)."""

    def __init__(self, embedder: _StateEmbedder) -> None:
        super().__init__()
        self.embedder = embedder
        self.gru = nn.GRU(LATENT_DIM, GRU_HIDDEN_DIM, batch_first=True)
        # Project the GRU hidden back to LATENT_DIM so z_context and
        # z_future live in the same space (MSE needs that).
        self.proj = nn.Linear(GRU_HIDDEN_DIM, LATENT_DIM)

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        B, T, _ = past.shape
        emb = self.embedder(past.reshape(B * T, STATE_DIM)).reshape(B, T, LATENT_DIM)
        _, h = self.gru(emb)
        return self.proj(h[-1])


class TargetEncoder(nn.Module):
    """(B, STATE_DIM) -> (B, LATENT_DIM). Wraps the shared embedder.

    The embedder is the same nn.Module instance PastEncoder uses
    (SimSiam-style weight sharing). The stop-gradient that prevents
    representation collapse is applied at the loss site, not here.
    """

    def __init__(self, embedder: _StateEmbedder) -> None:
        super().__init__()
        self.embedder = embedder

    def forward(self, future: torch.Tensor) -> torch.Tensor:
        return self.embedder(future)


class Predictor(nn.Module):
    """z_context -> z_pred. Both in LATENT_DIM."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, PREDICTOR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(PREDICTOR_HIDDEN_DIM, LATENT_DIM),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class RiskHead(nn.Module):
    """Latent -> failure logit (no sigmoid; use BCEWithLogitsLoss)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, RISK_HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(RISK_HEAD_HIDDEN_DIM, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class JEPAModel(nn.Module):
    """JEPA-style: predict the future latent, plus a small risk head.

    Pure JEPA is self-supervised, so nothing forces the latent to
    encode anything task-relevant. We have free failure labels from
    the sim, so a supervised head on z_pred shapes the latent toward
    the question we actually care about. Hence "JEPA-style".
    """

    def __init__(self) -> None:
        super().__init__()
        self.embedder = _StateEmbedder()
        self.past_encoder = PastEncoder(self.embedder)
        self.target_encoder = TargetEncoder(self.embedder)
        self.predictor = Predictor()
        self.risk_head = RiskHead()

    def forward(
        self, past: torch.Tensor, future: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Returns raw z_future; the training loop detaches it before MSE.
        z_context = self.past_encoder(past)
        z_future = self.target_encoder(future)
        z_pred = self.predictor(z_context)
        risk_logit = self.risk_head(z_pred)
        return {
            "z_context": z_context,
            "z_future": z_future,
            "z_pred_future": z_pred,
            "risk_logit": risk_logit,
        }

    def predict_risk(self, past: torch.Tensor) -> torch.Tensor:
        """Inference path: past -> failure probability."""
        z_context = self.past_encoder(past)
        z_pred = self.predictor(z_context)
        return torch.sigmoid(self.risk_head(z_pred))


class BaselineLSTM(nn.Module):
    """Predicts failure directly from the past window. No latent prediction.

    Hidden size matches GRU_HIDDEN_DIM so the comparison isn't a
    capacity comparison in disguise.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(STATE_DIM, BASELINE_LSTM_HIDDEN_DIM, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(BASELINE_LSTM_HIDDEN_DIM, RISK_HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(RISK_HEAD_HIDDEN_DIM, 1),
        )

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(past)
        return self.head(h[-1]).squeeze(-1)

    def predict_risk(self, past: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(past))
