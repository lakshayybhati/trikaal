"""Causal information-coefficient screen — pure statistics (Milestone-2 diagnostic).

Every metric here is written from scratch over NumPy (no scipy/statsmodels) — IC/RankIC are
listed in docs/ENGINEERING.md as eval metrics we own. The functions are deliberately I/O-free and
deterministic so they unit-test without the lake: lake reads, feature/horizon assembly, and
reporting live in ``scripts/m2_feature_ic.py``.

Definitions used by the screen (spec §8.B / §8.E.8):

* **RankIC** (primary) = Pearson correlation of the *ranks* of a feature and a forward return.
* **Pearson IC** (secondary) = Pearson correlation of the raw values.
* **Significance** = a moving-block bootstrap 95% CI on RankIC. Naive n≈5e5 p-values are
  meaningless under the heavy serial autocorrelation of 1-minute features, so blocks of ≈1 day
  tile the (single) segment and are resampled with replacement. A feature "carries standalone
  signal" only if its RankIC CI excludes 0.
* **Effective N / MDE** = the autocorrelation-deflated independent-sample count of the per-bar
  forward returns, and the smallest RankIC (and two-cell IC gap) distinguishable from noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------------------
# Rank / correlation primitives
# --------------------------------------------------------------------------------------


def rankdata_average(a: np.ndarray) -> np.ndarray:
    """Ranks of ``a`` with ties assigned their average rank (the Spearman tie rule)."""
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    order = np.argsort(a, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    # average tied groups: find runs of equal values in sorted order
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0  # mean of ranks (i+1)..j (1-based)
            ranks[order[i:j]] = avg
        i = j
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; ``nan`` if either input is (near-)constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[0] < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def spearman_rankic(x: np.ndarray, y: np.ndarray) -> float:
    """RankIC = Pearson correlation of average-tie ranks."""
    return pearson(rankdata_average(x), rankdata_average(y))


# --------------------------------------------------------------------------------------
# Moving-block bootstrap CI (autocorrelation-aware significance)
# --------------------------------------------------------------------------------------


def moving_block_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    block_len: int,
    n_boot: int,
    rng: np.random.Generator,
    ci: tuple[float, float] = (2.5, 97.5),
) -> tuple[float, float]:
    """Percentile CI of ``pearson(a, b)`` under an overlapping moving-block bootstrap.

    ``a`` and ``b`` are paired, time-ordered series (pass the *rank* series for a RankIC CI).
    Resampling preserves within-block serial dependence; blocks tile the series to length
    ``ceil(n / block_len) * block_len``. Accelerated via per-block sufficient statistics so the
    per-resample cost is O(blocks), not O(n) — seed-pinned through ``rng``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.shape[0]
    block_len = int(min(block_len, n))
    if block_len < 1 or n < 2:
        return (float("nan"), float("nan"))
    n_starts = n - block_len + 1
    k = (n + block_len - 1) // block_len  # blocks per resample (ceil)
    total = k * block_len

    # cumulative sums (length n+1) → per-block sufficient stats by slicing
    def block_sums(v: np.ndarray) -> np.ndarray:
        cs = np.concatenate(([0.0], np.cumsum(v)))
        return cs[block_len : block_len + n_starts] - cs[0:n_starts]

    s_a = block_sums(a)
    s_b = block_sums(b)
    s_aa = block_sums(a * a)
    s_bb = block_sums(b * b)
    s_ab = block_sums(a * b)

    idx = rng.integers(0, n_starts, size=(n_boot, k))
    t_a = s_a[idx].sum(axis=1)
    t_b = s_b[idx].sum(axis=1)
    t_aa = s_aa[idx].sum(axis=1)
    t_bb = s_bb[idx].sum(axis=1)
    t_ab = s_ab[idx].sum(axis=1)

    cov = t_ab / total - (t_a / total) * (t_b / total)
    var_a = t_aa / total - (t_a / total) ** 2
    var_b = t_bb / total - (t_b / total) ** 2
    denom = np.sqrt(np.clip(var_a, 0.0, None) * np.clip(var_b, 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 0.0, cov / denom, np.nan)
    lo, hi = np.nanpercentile(r, ci)
    return (float(lo), float(hi))


# --------------------------------------------------------------------------------------
# Effective sample size & minimum detectable effect (temporal, single symbol)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectiveN:
    n: int
    n_eff: float
    ratio: float
    lags_used: int
    capped_at_n: bool


def effective_sample_size(
    returns: np.ndarray, *, max_lag: int = 1440, sig_z: float = 1.96
) -> EffectiveN:
    """Autocorrelation-deflated effective N of a per-bar forward-return series.

    ``N_eff = N / (1 + 2 Σ_{k=1..K} (1 − k/N) ρ_k)`` where ρ_k is the sample autocorrelation and
    ``K`` is truncated at the first lag falling inside the white-noise band ``|ρ_k| < z/√N``
    (Bartlett), capped at ``max_lag``. Negative net autocorrelation would give ``N_eff > N``;
    we conservatively cap at ``N`` (never claim more independence than raw count).
    """
    r = np.asarray(returns, dtype=np.float64)
    n = r.shape[0]
    if n < 3:
        return EffectiveN(n=n, n_eff=float(n), ratio=1.0, lags_used=0, capped_at_n=False)
    r = r - r.mean()
    denom0 = float(np.dot(r, r))
    if denom0 == 0.0:
        return EffectiveN(n=n, n_eff=float(n), ratio=1.0, lags_used=0, capped_at_n=False)
    band = sig_z / np.sqrt(n)
    acc = 0.0
    lags = 0
    kmax = int(min(max_lag, n - 1))
    for k in range(1, kmax + 1):
        rho_k = float(np.dot(r[k:], r[:-k]) / denom0)
        if abs(rho_k) < band:  # first non-significant lag → truncate
            break
        acc += (1.0 - k / n) * rho_k
        lags = k
    factor = 1.0 + 2.0 * acc
    n_eff_raw = n / factor if factor > 0.0 else float(n)
    capped = n_eff_raw > n
    n_eff = float(min(n_eff_raw, n)) if n_eff_raw > 0 else float(n)
    return EffectiveN(n=n, n_eff=n_eff, ratio=n_eff / n, lags_used=lags, capped_at_n=capped)


@dataclass(frozen=True)
class MDE:
    se: float
    mde_single: float
    mde_two_cell: float


def mde_from_neff(n_eff: float, *, z: float = 1.96) -> MDE:
    """Minimum detectable RankIC at the given confidence from an effective-N.

    ``SE(RankIC) ≈ 1/√N_eff``; single-feature ``MDE = z·SE``; a two-cell IC-gap (e.g.
    Cell 4 − Cell 1) adds the two cells' variances → ``MDE_gap = z·√2·SE``.
    """
    se = 1.0 / np.sqrt(max(n_eff, 1.0))
    return MDE(se=float(se), mde_single=float(z * se), mde_two_cell=float(z * np.sqrt(2.0) * se))


__all__ = [
    "MDE",
    "EffectiveN",
    "effective_sample_size",
    "mde_from_neff",
    "moving_block_bootstrap_ci",
    "pearson",
    "rankdata_average",
    "spearman_rankic",
]
