"""Tests for the two models' shapes and, crucially, the two JEPA design
invariants: the shared embedder and the stop-gradient on the target."""

from __future__ import annotations

import torch

from config import LATENT_DIM, PAST_WINDOW, STATE_DIM
from models import BaselineLSTM, JEPAModel
from train_jepa import _compute_loss

_BATCH = 3


def _past(b=_BATCH):
    return torch.randn(b, PAST_WINDOW, STATE_DIM)


def _future(b=_BATCH):
    return torch.randn(b, STATE_DIM)


def test_jepa_forward_shapes():
    torch.manual_seed(0)
    out = JEPAModel()(_past(), _future())
    assert out["z_context"].shape == (_BATCH, LATENT_DIM)
    assert out["z_future"].shape == (_BATCH, LATENT_DIM)
    assert out["z_pred_future"].shape == (_BATCH, LATENT_DIM)
    assert out["risk_logit"].shape == (_BATCH,)


def test_jepa_predict_risk_is_a_probability():
    model = JEPAModel().eval()
    probs = model.predict_risk(_past())
    assert probs.shape == (_BATCH,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_baseline_forward_and_risk():
    model = BaselineLSTM().eval()
    assert model(_past()).shape == (_BATCH,)
    probs = model.predict_risk(_past())
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_past_and_target_encoders_share_one_embedder():
    # SimSiam-style weight sharing. If these were separate instances,
    # MSE(z_pred, z_future) would compare points in two unrelated spaces.
    model = JEPAModel()
    assert model.past_encoder.embedder is model.target_encoder.embedder


def test_target_latent_receives_no_gradient():
    # The load-bearing JEPA detail. In _compute_loss the future latent is
    # detached before the MSE, and it feeds nothing else in the loss, so
    # the loss gradient w.r.t. z_future must be exactly zero. Without the
    # stop-gradient the shared embedder could collapse both sides.
    torch.manual_seed(0)
    model = JEPAModel()
    out = model(_past(), _future())
    out["z_future"].retain_grad()
    assert out["z_future"].requires_grad  # it IS part of the graph...

    total, _, _ = _compute_loss(out, torch.zeros(_BATCH))
    total.backward()

    grad = out["z_future"].grad           # ...but gets no gradient.
    assert grad is None or torch.all(grad == 0.0)


def test_compute_loss_backward_trains_predictor():
    torch.manual_seed(0)
    model = JEPAModel()
    out = model(_past(), _future())
    total, mse, bce = _compute_loss(out, torch.zeros(_BATCH))
    total.backward()
    assert model.predictor.net[0].weight.grad is not None
    assert mse.item() >= 0.0 and bce.item() >= 0.0
