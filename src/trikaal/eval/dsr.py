"""Deflated Sharpe Ratio + Probability of Backtest Overfitting (§8.E.4, Bailey & López de Prado).

Multiple-testing discipline: with 5 cells × ≥3 seeds × horizons × the κ-filter search we run many
trials, so a high Sharpe can be a selection artifact. The **DSR** discounts the observed Sharpe by
the *expected maximum* Sharpe under that many independent trials (a result that does not survive
deflation is reported not-significant, not dropped). **PBO** (CSCV) estimates the probability that
the in-sample-best config underperforms the OOS median.

Normal CDF/inverse-CDF are written from scratch (no scipy): CDF via ``math.erf``; inverse via
Acklam's rational approximation (~1e-9). Known-answer-tested in ``tests/eval/test_dsr.py``.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

GAMMA_EULER = 0.5772156649015329


def norm_cdf(x: float) -> float:
    """Standard normal CDF Φ(x) = ½(1 + erf(x/√2))."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation), p ∈ (0,1)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf requires p in (0,1), got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def _sharpe(returns: np.ndarray) -> float:
    """Per-period (non-annualized) Sharpe = mean/std (population)."""
    r = np.asarray(returns, dtype=np.float64)
    sd = r.std(ddof=0)
    return float(r.mean() / sd) if sd > 0 else 0.0


def psr_from_moments(sr_hat: float, sr_star: float, n: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio from precomputed moments (the pure formula; ``kurt`` non-excess).

    PSR = Φ( (SR̂ − SR*)·√(n−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) ).
    """
    denom = math.sqrt(max(1e-300, 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat * sr_hat))
    return norm_cdf((sr_hat - sr_star) * math.sqrt(max(0, n - 1)) / denom)


def probabilistic_sharpe_ratio(returns: np.ndarray, sr_star: float) -> float:
    """PSR of a return series against a benchmark Sharpe ``sr_star`` (per-period units)."""
    r = np.asarray(returns, dtype=np.float64)
    n = r.shape[0]
    if n < 2:
        return float("nan")
    mean = r.mean()
    sd = r.std(ddof=0)
    if sd == 0:
        return float("nan")
    z = (r - mean) / sd
    skew = float((z**3).mean())
    kurt = float((z**4).mean())  # non-excess (normal → 3)
    return psr_from_moments(_sharpe(r), sr_star, n, skew, kurt)


def expected_max_sharpe(var_sr: float, n_trials: int, *, gamma: float = GAMMA_EULER) -> float:
    """Expected maximum Sharpe under ``n_trials`` independent trials (the deflation benchmark SR₀).

    SR₀ = √Var(SR) · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ].
    """
    n = max(2, int(n_trials))
    term = (1.0 - gamma) * norm_ppf(1.0 - 1.0 / n) + gamma * norm_ppf(1.0 - 1.0 / (n * math.e))
    return math.sqrt(max(0.0, var_sr)) * term


def deflated_sharpe_ratio(returns: np.ndarray, *, n_trials: int, var_sr: float) -> float:
    """DSR = PSR(SR₀) where SR₀ is the expected max Sharpe under ``n_trials`` (var_sr = cross-trial
    variance of the trial Sharpe estimates). Survives deflation iff DSR > 0.95 (the usual bar)."""
    sr0 = expected_max_sharpe(var_sr, n_trials)
    return probabilistic_sharpe_ratio(returns, sr0)


def pbo_cscv(return_matrix: np.ndarray, *, n_splits: int = 8) -> float:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric Cross-Validation.

    ``return_matrix`` is ``[T_periods, N_configs]``. Rows are split into ``n_splits`` (even)
    disjoint blocks; over each IS/OOS partition (S/2 blocks each) we pick the IS-best config, rank
    it OOS.
    PBO = fraction of partitions whose IS-best config lands at-or-below the OOS median (logit ≤ 0).
    """
    m = np.asarray(return_matrix, dtype=np.float64)
    t, n_cfg = m.shape
    if n_cfg < 2 or n_splits < 2 or n_splits % 2 != 0:
        raise ValueError("need ≥2 configs and an even n_splits ≥ 2")
    bounds = np.linspace(0, t, n_splits + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_splits)]
    half = n_splits // 2
    logits: list[float] = []
    for is_idx in combinations(range(n_splits), half):
        oos_idx = [b for b in range(n_splits) if b not in is_idx]
        is_rows = np.concatenate([blocks[b] for b in is_idx])
        oos_rows = np.concatenate([blocks[b] for b in oos_idx])
        is_sr = np.array([_sharpe(m[is_rows, j]) for j in range(n_cfg)])
        oos_sr = np.array([_sharpe(m[oos_rows, j]) for j in range(n_cfg)])
        best = int(np.argmax(is_sr))
        # relative OOS rank of the IS-best config (1 = best OOS, →0 = worst)
        rank = float((oos_sr <= oos_sr[best]).sum()) / (n_cfg + 1)
        rank = min(max(rank, 1.0 / (n_cfg + 1)), n_cfg / (n_cfg + 1))
        logits.append(math.log(rank / (1.0 - rank)))
    return float(np.mean(np.array(logits) <= 0.0))


__all__ = [
    "GAMMA_EULER",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "norm_cdf",
    "norm_ppf",
    "pbo_cscv",
    "probabilistic_sharpe_ratio",
    "psr_from_moments",
]
