"""Known-answer tests for DSR / PBO and the from-scratch normal CDF/inverse (§8.E.4)."""

from __future__ import annotations

import math

import numpy as np

from trikaal.eval.dsr import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    norm_cdf,
    norm_ppf,
    pbo_cscv,
    psr_from_moments,
)


def test_normal_cdf_known_values():
    assert np.isclose(norm_cdf(0.0), 0.5)
    assert np.isclose(norm_cdf(1.959963985), 0.975, atol=1e-6)
    assert np.isclose(norm_cdf(-1.0), 0.158655254, atol=1e-7)


def test_normal_ppf_known_values_and_roundtrip():
    assert np.isclose(norm_ppf(0.5), 0.0, atol=1e-9)
    assert np.isclose(norm_ppf(0.975), 1.959963985, atol=1e-5)
    assert np.isclose(norm_ppf(0.9), 1.281551566, atol=1e-5)
    for p in (0.01, 0.2, 0.5, 0.8, 0.99):
        assert np.isclose(norm_cdf(norm_ppf(p)), p, atol=1e-9)


def test_psr_formula_hand_computed():
    # SR̂=0.1, SR*=0, n=101, skew 0, kurt 3 → arg = 0.1·√100 / √(1+0.005) ≈ 0.99751 → Φ ≈ 0.8407
    val = psr_from_moments(0.1, 0.0, 101, 0.0, 3.0)
    assert np.isclose(val, norm_cdf(1.0 / math.sqrt(1.005)))  # self-consistent
    assert np.isclose(val, 0.8407, atol=1e-3)  # independent table value


def test_expected_max_sharpe_grows_with_trials():
    s10 = expected_max_sharpe(1.0, 10)
    assert np.isclose(s10, 1.576, atol=2e-2)  # hand-computed for var_sr=1, N=10
    assert expected_max_sharpe(1.0, 100) > s10  # more trials → higher bar to clear
    assert expected_max_sharpe(4.0, 10) > s10  # more cross-trial variance → higher bar


def test_dsr_decreases_with_more_trials():
    rng = np.random.default_rng(0)
    r = 0.05 + 0.1 * rng.standard_normal(500)  # a positive-Sharpe series
    dsr_few = deflated_sharpe_ratio(r, n_trials=2, var_sr=0.01)
    dsr_many = deflated_sharpe_ratio(r, n_trials=1000, var_sr=0.01)
    assert dsr_few > dsr_many  # deflation penalizes more trials
    assert 0.0 <= dsr_many <= dsr_few <= 1.0


def test_pbo_consistent_vs_noise():
    rng = np.random.default_rng(1)
    base = rng.standard_normal((400, 1))
    # config 0 strictly dominates every period → IS-best is always OOS-best → PBO = 0
    consistent = np.hstack([base + 0.5, base, base - 0.5])
    assert pbo_cscv(consistent, n_splits=8) == 0.0
    # pure noise → IS-best is selection luck → PBO near 0.5
    noise = rng.standard_normal((400, 10))
    assert 0.2 < pbo_cscv(noise, n_splits=8) < 0.8
