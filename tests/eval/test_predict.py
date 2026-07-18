"""The KV-cache rollout predictor runs, is deterministic (greedy), and scales with σ_t."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_predict_mu_chunking_contract():
    """Decision-chunking caps KV-cache memory (~35k uncapped money-grid decisions/symbol would
    be tens of GB unchunked). Contract, honestly scoped: (a) at a FIXED chunk size replay is
    bit-exact — chunk is a recorded recipe constant, never varied inside a run; (b) across
    DIFFERENT chunk sizes agreement is approximate only (GEMM kernels are batch-shape-dependent
    at the ULP level, and an argmax near-tie could amplify that — same class as the recorded
    attention-mode nondeterminism, handled by fixing the constant, not by claiming exactness)."""
    model, tok, b_c, b_f, ts = _tiny()
    sigma = np.full(80, 0.01)
    dec = np.arange(20, 70, 3, dtype=np.int64)  # 17 decisions
    ref = predict_mu(model, tok, b_c, b_f, ts, sigma, dec, h=5, seq_len=16, chunk=10_000)
    for chunk in (1, 7, 16):
        a = predict_mu(model, tok, b_c, b_f, ts, sigma, dec, h=5, seq_len=16, chunk=chunk)
        b = predict_mu(model, tok, b_c, b_f, ts, sigma, dec, h=5, seq_len=16, chunk=chunk)
        assert np.array_equal(a, b), f"fixed chunk={chunk} must replay bit-exactly"
        np.testing.assert_allclose(a, ref, rtol=1e-4, atol=1e-6)  # cross-size: approximate


def test_predict_mu_refuses_a_rollout_past_the_rope_cache():
    """seq_len + h − 1 > max_len would index past the RoPE cos/sin cache mid-eval (the money
    config's seq 512 + h=60 reaches position 571 vs the 512 default — caught at rehearsal
    sizing). The guard must fail LOUDLY up front, naming the fix."""
    model, tok, b_c, b_f, ts = _tiny()  # max_len=64
    sigma = np.full(80, 0.01)
    with pytest.raises(ValueError, match="max_len"):
        predict_mu(model, tok, b_c, b_f, ts, sigma, np.array([70]), h=10, seq_len=60)
    # at the boundary (seq_len + h - 1 == max_len) it must run
    mu = predict_mu(model, tok, b_c, b_f, ts, sigma, np.array([60]), h=5, seq_len=60)
    assert mu.shape == (1,) and np.isfinite(mu).all()
