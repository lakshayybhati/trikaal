"""IC-screen statistics — does the screen have teeth?

A perfectly-aligned feature must read as carries-signal (RankIC≈1, CI excludes 0); a shuffled
feature must read as no-signal (RankIC≈0, CI overlaps 0). Plus effective-N / MDE sanity. All
synthetic and fast — no lake, no network.
"""

from __future__ import annotations

import numpy as np

from trikaal.eval.ic_screen import (
    effective_sample_size,
    mde_from_neff,
    moving_block_bootstrap_ci,
    pearson,
    rankdata_average,
    spearman_rankic,
)


def _ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """An AR(1) series so the moving-block bootstrap faces real serial dependence."""
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_rankdata_average_handles_ties():
    r = rankdata_average(np.array([10.0, 10.0, 20.0, 5.0]))
    # the two 10s share ranks 2 and 3 → 2.5 each; 5 → 1; 20 → 4
    assert np.allclose(r, [2.5, 2.5, 4.0, 1.0])


def test_perfect_feature_carries_signal():
    rng = np.random.default_rng(0)
    y = _ar1(5000, 0.6, rng)
    feat = y.copy()  # perfectly aligned
    ric = spearman_rankic(feat, y)
    assert ric > 0.999
    lo, _hi = moving_block_bootstrap_ci(
        rankdata_average(feat), rankdata_average(y), block_len=60, n_boot=400, rng=rng
    )
    assert lo > 0.0  # CI excludes 0 → carries-signal


def test_monotone_feature_is_rank_perfect_but_not_pearson_perfect():
    rng = np.random.default_rng(1)
    y = _ar1(4000, 0.5, rng)
    feat = np.exp(y)  # strictly monotone transform: preserves ranks, not linearity
    assert spearman_rankic(feat, y) > 0.999
    assert pearson(feat, y) < 0.999


def test_shuffled_feature_has_no_signal():
    rng = np.random.default_rng(2)
    y = _ar1(5000, 0.6, rng)
    feat = rng.permutation(y)  # destroy alignment
    ric = spearman_rankic(feat, y)
    assert abs(ric) < 0.05
    lo, hi = moving_block_bootstrap_ci(
        rankdata_average(feat), rankdata_average(y), block_len=60, n_boot=400, rng=rng
    )
    assert lo < 0.0 < hi  # CI overlaps 0 → no standalone signal


def test_effective_n_white_vs_autocorrelated():
    rng = np.random.default_rng(3)
    white = rng.standard_normal(20000)
    eff_w = effective_sample_size(white, max_lag=500)
    assert eff_w.ratio > 0.9  # near-white → little deflation

    strong = _ar1(20000, 0.8, rng)
    eff_s = effective_sample_size(strong, max_lag=500)
    assert eff_s.n_eff < eff_s.n  # positive autocorrelation deflates N_eff
    assert eff_s.ratio < eff_w.ratio


def test_mde_scaling():
    mde = mde_from_neff(10_000.0)
    assert np.isclose(mde.se, 0.01, atol=1e-9)
    assert np.isclose(mde.mde_single, 1.96 * 0.01)
    assert np.isclose(mde.mde_two_cell, 1.96 * np.sqrt(2.0) * 0.01)
    # a smaller effective-N must raise the bar
    assert mde_from_neff(1_000.0).mde_single > mde.mde_single
