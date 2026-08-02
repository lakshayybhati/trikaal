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


def beta_cf_coefficient(a: float, b: float, x: float, k: int) -> float:
    """The k-th partial numerator d_k of the incomplete-beta continued fraction (k ≥ 1).

    Straight from the published definition (Abramowitz & Stegun 26.5.8; DLMF 8.17.22):

        d_{2m}   = + m (b − m) x / ((a + 2m − 1)(a + 2m))
        d_{2m+1} = − (a + m)(a + b + m) x / ((a + 2m)(a + 2m + 1))

    Exposed as a named callable so the coefficients can be checked term-by-term against the
    reference formula, independently of how the fraction is then summed.
    """
    if k % 2 == 0:
        m = k // 2
        return m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
    m = (k - 1) // 2
    return -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))


def beta_continued_fraction(a: float, b: float, x: float, *, max_depth: int = 4096) -> float:
    """Evaluate 1 / (1 + d₁/(1 + d₂/(1 + d₃/…))) by BACKWARD recurrence, to convergence.

    §7 v1.6.17 (audit S-4). The predecessor was the *Numerical Recipes* ``betacf`` in structure —
    same variable names (``qab/qap/qam/c/d/h/aa/m2``) and the same forward modified-Lentz loop.
    Neither of us is a lawyer and no licence violation is asserted; the point is that a released
    artifact should not carry a transcription of NR code when the underlying mathematics is
    public and the safe version costs an afternoon.

    This is a genuinely different algorithm for the same quantity: the coefficients come from
    :func:`beta_cf_coefficient` (the published definition) and the fraction is summed from the
    TAIL INWARDS at a doubling depth until the value stops moving, rather than forwards with
    Lentz's rescaling. Backward evaluation needs no ``_TINY`` guards on the running scale factors
    because there are none — only a division by the running tail, which is guarded.
    """
    prev: float | None = None
    depth = 16
    val = 1.0
    while depth <= max_depth:
        f = 1.0
        for k in range(depth, 0, -1):
            f = 1.0 + beta_cf_coefficient(a, b, x, k) / f
            if abs(f) < _TINY:
                f = _TINY
        val = 1.0 / f
        if prev is not None and abs(val - prev) <= _EPS * max(1.0, abs(val)):
            return val
        prev = val
        depth *= 2
    return val


# Back-compat name for the one internal caller; the implementation above is the definition.
def _betacf(a: float, b: float, x: float) -> float:
    return beta_continued_fraction(a, b, x)


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
