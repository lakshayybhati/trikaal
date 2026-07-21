"""The KV-cache rollout predictor runs, is deterministic (greedy), and scales with σ_t."""

from __future__ import annotations

import typing

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


# ---- §7 v1.4.2: the MEAN estimator (μ̂ = conditional mean, not mode) ------------------------
class _SkewToy:
    """A degenerate model+tokenizer with a KNOWN skewed per-step return distribution, so the
    argmax (mode) μ̂ and the true mean μ̂ have hand-computable, DIFFERENT values.

    Construction: coarse vocab of 3 ids whose decoded ret_close (dim 0) values are fixed to a
    right-skewed set with p = [0.6, 0.3, 0.1] independent of context. The MODE is the p=0.6 id;
    the MEAN is Σ p·value. Choose values so mode < mean (a left-heavy mode under right skew):
    values [-1, 0, +5] → mode −1, mean 0.6·(−1)+0.3·0+0.1·5 = −0.1. Over (h−1) accumulated
    steps the argmax μ̂ = (h−1)·(−1)·σ while the mean μ̂ = (h−1)·(−0.1)·σ — the mode is
    6.7σ-scale more negative per step, the exact acceptance-run pathology in miniature."""

    P: typing.ClassVar = [0.6, 0.3, 0.1]
    VAL: typing.ClassVar = [-1.0, 0.0, 5.0]
    V_C = 3

    class _Out:
        def __init__(self, logits_c, logits_f, coarse_cond, h_final):
            self.logits_c, self.logits_f = logits_c, logits_f
            self.coarse_cond, self.h_final = coarse_cond, h_final

    def __init__(self):
        import torch

        self.v_c, self.v_f = self.V_C, 1
        self._logits_c = torch.log(torch.tensor([self.P]))  # [1,3] fixed
        self._config = {"max_len": 4096}

    def eval(self):
        return self

    def init_caches(self):
        return []

    def step(self, b_c_t, b_f_t, ts_t, caches, pos):
        import torch

        n = b_c_t.shape[0]
        lc = self._logits_c.to(b_c_t.device).expand(n, 1, self.V_C)
        cc = lc.argmax(-1)
        lf = torch.zeros(n, 1, 1, device=b_c_t.device)
        return self._Out(lc, lf, cc, torch.zeros(n, 1, 1, device=b_c_t.device))

    def fine_logits(self, h_final, coarse_ids):
        import torch

        return torch.zeros(*coarse_ids.shape, 1, device=coarse_ids.device)


class _SkewTok:
    """Tokenizer whose decode maps coarse id → the fixed VAL[id] on dim 0; fine ignored."""

    quantizer = "fsq"

    class _Q:
        coarse_idx: typing.ClassVar = [0]
        fine_idx: typing.ClassVar = [1]
        dim = 2

    def __init__(self):
        self.quant = self._Q()
        self.v_c, self.v_f = 3, 1

    def eval(self):
        return self

    def _group_radix(self, group):
        return [3] if group == [0] else [1]

    def group_grid(self, which, *, device=None):
        import torch

        g = torch.tensor([[-1.0], [0.0], [5.0]]) if which == "coarse" else torch.zeros(1, 1)
        return g.to(device) if device is not None else g

    def decode_tokens(self, cidx, fidx):
        import torch

        vals = torch.tensor(_SkewToy.VAL, device=cidx.device)
        x = torch.zeros(*cidx.shape, 16, device=cidx.device)
        x[..., 0] = vals[cidx]
        return x

    def decode_latent(self, z):
        import torch

        x = torch.zeros(*z.shape[:-1], 16, device=z.device)
        x[..., 0] = z[..., 0]  # dim-0 latent IS the ret_close (coarse grid value)
        return x


def test_estimator_argmax_is_biased_expectation_is_not():
    """Known-answer: under the right-skewed toy, argmax μ̂ = (h−1)·(−1)·σ (the mode) while
    expectation μ̂ = (h−1)·(−0.1)·σ (the true mean). This is the acceptance sign-saturation
    pathology with an exact target — argmax must be wrong and expectation must be right."""
    m, tok = _SkewToy(), _SkewTok()
    ts = (1_600_000_000_000 + np.arange(40) * BAR_MS).astype(np.int64)
    b_c = np.zeros(40, dtype=np.int64)
    b_f = np.zeros(40, dtype=np.int64)
    sigma = np.full(40, 0.01)
    dec = np.array([20], dtype=np.int64)
    h = 6
    mu_arg = predict_mu(m, tok, b_c, b_f, ts, sigma, dec, h=h, seq_len=8, estimator="argmax")
    mu_exp = predict_mu(m, tok, b_c, b_f, ts, sigma, dec, h=h, seq_len=8, estimator="expectation")
    steps = h - 1  # k=1 excluded
    assert np.allclose(mu_arg, steps * (-1.0) * 0.01), mu_arg  # mode
    assert np.allclose(mu_exp, steps * (-0.1) * 0.01), mu_exp  # true mean
    assert mu_exp[0] > mu_arg[0]  # the mode is biased LOW under this right skew


def test_mc_mean_approaches_expectation_on_the_toy():
    """The MC-mean fallback is unbiased-in-expectation: at enough samples its μ̂ approaches the
    exact mean, not the mode (loose tolerance — it is the noisy estimator, by design)."""
    m, tok = _SkewToy(), _SkewTok()
    ts = (1_600_000_000_000 + np.arange(40) * BAR_MS).astype(np.int64)
    b_c, b_f, sigma = np.zeros(40, np.int64), np.zeros(40, np.int64), np.full(40, 0.01)
    mu_mc = predict_mu(
        m,
        tok,
        b_c,
        b_f,
        ts,
        sigma,
        np.array([20]),
        h=6,
        seq_len=8,
        estimator="mc_mean",
        mc_samples=400,
        mc_seed=7,
    )
    assert abs(mu_mc[0] - 5 * (-0.1) * 0.01) < abs(mu_mc[0] - 5 * (-1.0) * 0.01)  # nearer mean


def test_estimators_are_all_deterministic_and_default_is_expectation():
    """All three estimators replay bit-exactly (mc under its pinned seed), and the DEFAULT is
    the mean estimator (expectation) — μ̂'s pre-registered meaning."""
    model, tok, b_c, b_f, ts = _tiny()
    sigma = np.full(80, 0.01)
    dec = np.array([20, 40, 60])
    default = predict_mu(model, tok, b_c, b_f, ts, sigma, dec, h=5, seq_len=16)
    exp = predict_mu(model, tok, b_c, b_f, ts, sigma, dec, h=5, seq_len=16, estimator="expectation")
    assert np.array_equal(default, exp)  # default IS expectation
    for est in ("argmax", "expectation", "mc_mean"):
        a = predict_mu(model, tok, b_c, b_f, ts, sigma, dec, h=5, seq_len=16, estimator=est)
        b = predict_mu(model, tok, b_c, b_f, ts, sigma, dec, h=5, seq_len=16, estimator=est)
        assert np.array_equal(a, b), f"{est} must replay bit-exactly"


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
