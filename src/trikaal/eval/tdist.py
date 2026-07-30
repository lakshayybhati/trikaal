"""Student-t CDF / quantiles, written from scratch (prereg §7 v1.5 item B).

WHY THIS EXISTS. The v1.5 MDE carries an explicit across-seed training-variance term whose variance
estimate has only ``S − 1`` degrees of freedom (2 at S=3, 4 at S=5). At that df the normal quantile
is not conservative: ``z_0.95 = 1.6449`` against ``t_0.95,2 = 2.9200``. Using z would understate the
MDE precisely where the new term is least well estimated, so the v1.5 rule uses t with
Welch–Satterthwaite effective df.

No scipy (project rule: statistics on the research surface are self-written). The regularized
incomplete beta is a Lentz continued fraction; the CDF uses the standard identity

    F(t; ν) = 1 − ½·I_{ν/(ν+t²)}(ν/2, ½)   for t ≥ 0,    F(−t; ν) = 1 − F(t; ν)

and the quantile is a bracketed bisection on that CDF — slow in principle, exact enough in practice,
and impossible to get subtly wrong in a way a KAT would miss. Known-answer-tested against published
t-table values in ``tests/eval/test_tdist.py``, including the ν → ∞ limit against ``norm_ppf``.
"""

from __future__ import annotations

import math

_TINY = 1e-300
_EPS = 3e-16


def _betacf(a: float, b: float, x: float, itmax: int = 300) -> float:
    """Continued fraction for the incomplete beta (modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) ∈ [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, nu: float) -> float:
    """P(T ≤ t) for T ~ Student-t with ``nu`` degrees of freedom (nu > 0, real)."""
    if nu <= 0:
        raise ValueError(f"degrees of freedom must be positive, got {nu}")
    x = nu / (nu + t * t)
    half = 0.5 * betainc(0.5 * nu, 0.5, x)
    return 1.0 - half if t > 0 else half


def student_t_ppf(p: float, nu: float) -> float:
    """Inverse Student-t CDF by bracketed bisection on ``student_t_cdf``.

    Bisection rather than Newton: the derivative is cheap but the tails are where this is used and a
    bracketed method cannot diverge there. ~60 iterations gives ~1e-12 on the bracket.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"student_t_ppf requires p in (0,1), got {p}")
    if nu <= 0:
        raise ValueError(f"degrees of freedom must be positive, got {nu}")
    if p == 0.5:
        return 0.0
    lo, hi = -1.0, 1.0
    while student_t_cdf(lo, nu) > p:
        lo *= 2.0
        if lo < -1e12:
            return lo
    while student_t_cdf(hi, nu) < p:
        hi *= 2.0
        if hi > 1e12:
            return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


def welch_satterthwaite_df(se_a: float, nu_a: float, se_b: float, nu_b: float) -> float:
    """Effective df for SE_total = √(se_a² + se_b²) — the standard WS combination.

    ν = (se_a² + se_b²)² / (se_a⁴/ν_a + se_b⁴/ν_b). Degenerate inputs fall back to the df of
    whichever term survives, so a zero training term does not produce a 0/0.
    """
    va, vb = se_a * se_a, se_b * se_b
    num = (va + vb) ** 2
    den = 0.0
    if nu_a > 0 and va > 0:
        den += va * va / nu_a
    if nu_b > 0 and vb > 0:
        den += vb * vb / nu_b
    if den <= 0.0:
        return max(nu_a, nu_b, 1.0)
    return num / den


__all__ = [
    "betainc",
    "student_t_cdf",
    "student_t_ppf",
    "welch_satterthwaite_df",
]
