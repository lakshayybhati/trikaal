"""The KV-cache rollout predictor runs, is deterministic (greedy), and scales with σ_t."""

from __future__ import annotations

import numpy as np

from trikaal.data.config import BAR_MS
from trikaal.eval.predict import predict_mu, sigma_hat
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.utils.seeding import set_determinism


def _tiny():
    set_determinism(0)
    model = TrikaalAR(n_layers=1, d_model=32, d_ff=64, n_heads=2, mtp_depths=0, max_len=64).eval()
    tok = TokenizerAE(d_model=32, n_layers=1, n_heads=2, d_ff=64).eval()
    n = 80
    rng = np.random.default_rng(0)
    b_c = rng.integers(0, model.v_c, n)
    b_f = rng.integers(0, model.v_f, n)
    ts = (1_600_000_000_000 + np.arange(n) * BAR_MS).astype(np.int64)
    return model, tok, b_c, b_f, ts


def test_predict_mu_shape_finite_deterministic():
    model, tok, b_c, b_f, ts = _tiny()
    sigma = np.full(80, 0.01)
    dec = np.array([20, 40, 60])
    mu1 = predict_mu(model, tok, b_c, b_f, ts, sigma, dec, h=5, seq_len=16)
    mu2 = predict_mu(model, tok, b_c, b_f, ts, sigma, dec, h=5, seq_len=16)
    assert mu1.shape == (3,) and np.isfinite(mu1).all()
    assert np.array_equal(mu1, mu2)  # greedy rollout is deterministic


def test_predict_mu_scales_linearly_with_sigma():
    model, tok, b_c, b_f, ts = _tiny()
    dec = np.array([20, 40, 60])
    mu1 = predict_mu(model, tok, b_c, b_f, ts, np.full(80, 0.01), dec, h=5, seq_len=16)
    mu2 = predict_mu(model, tok, b_c, b_f, ts, np.full(80, 0.02), dec, h=5, seq_len=16)
    assert np.allclose(mu2, 2.0 * mu1)  # μ̂ = cum · σ_t (the §3.4 decode contract)


def test_predict_mu_excludes_t_to_t1_move():
    """Contract: μ̂ covers [t+1, t+h] (entry at C_{t+1}). At h=1 the window is empty → μ̂ ≡ 0,
    matching the harness label log(C_{t+1}/C_{t+1}) = 0 (the k=1 t→t+1 move is excluded)."""
    model, tok, b_c, b_f, ts = _tiny()
    dec = np.array([20, 40, 60])
    mu_h1 = predict_mu(model, tok, b_c, b_f, ts, np.full(80, 0.01), dec, h=1, seq_len=16)
    assert np.allclose(mu_h1, 0.0)  # one-bar entry-next-bar hold is 0 by construction


def test_sigma_hat_random_walk():
    assert np.allclose(sigma_hat(np.array([0.01, 0.02]), 4), [0.02, 0.04])  # σ_t·√h
