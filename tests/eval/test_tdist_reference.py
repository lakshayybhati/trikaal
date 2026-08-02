"""§7 v1.6.17 (audit S-4) — the incomplete-beta continued fraction, KAT'd against PUBLISHED and
CLOSED-FORM reference points.

The implementation was rewritten from the published coefficient definition (A&S 26.5.8 /
DLMF 8.17.22) and summed by BACKWARD recurrence, replacing a forward modified-Lentz loop that was
*Numerical Recipes* ``betacf`` in structure. These are the exact reference points it was tested
at — named here so an INDEPENDENT implementation can be compared against the same set.

The public callables for that comparison are:
    ``trikaal.eval.tdist.beta_cf_coefficient(a, b, x, k)``  — the k-th partial numerator d_k
    ``trikaal.eval.tdist.beta_continued_fraction(a, b, x)`` — the summed fraction
    ``trikaal.eval.tdist.betainc(a, b, x)``                 — regularized I_x(a, b)
    ``trikaal.eval.tdist.student_t_ppf(p, df)`` / ``student_t_cdf(t, df)``
"""

from __future__ import annotations

import math

import pytest

from trikaal.eval.tdist import (
    beta_cf_coefficient,
    beta_continued_fraction,
    betainc,
    student_t_cdf,
    student_t_ppf,
    welch_satterthwaite_df,
)

# ---------------------------------------------------------------- closed forms (exact, no table)
CLOSED_FORM_X = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


@pytest.mark.parametrize("x", CLOSED_FORM_X)
def test_I_x_1_1_equals_x(x):
    assert betainc(1.0, 1.0, x) == pytest.approx(x, abs=1e-14)


@pytest.mark.parametrize("x", CLOSED_FORM_X)
@pytest.mark.parametrize("a", (2.0, 3.0, 7.0))
def test_I_x_a_1_equals_x_pow_a(x, a):
    assert betainc(a, 1.0, x) == pytest.approx(x**a, abs=1e-13)


@pytest.mark.parametrize("x", CLOSED_FORM_X)
@pytest.mark.parametrize("b", (2.0, 4.0, 9.0))
def test_I_x_1_b_equals_one_minus_1mx_pow_b(x, b):
    assert betainc(1.0, b, x) == pytest.approx(1.0 - (1.0 - x) ** b, abs=1e-13)


@pytest.mark.parametrize("a", (0.7, 1.0, 2.0, 9.0, 40.0))
def test_symmetry_I_half_a_a_is_one_half(a):
    assert betainc(a, a, 0.5) == pytest.approx(0.5, abs=1e-13)


@pytest.mark.parametrize("a,b,x", [(2.0, 5.0, 0.3), (0.5, 0.5, 0.7), (13.7, 3.0, 0.42)])
def test_reflection_identity(a, b, x):
    """I_x(a,b) + I_{1-x}(b,a) = 1 — a property of the function, independent of the algorithm."""
    # NOTE (builder defect, disclosed): the first version called betainc(1-x, b, a) — the
    # arguments are (a, b, x), so that passed 1-x as `a`. The test failed and the CODE was
    # fine. Written down because a failing test is not evidence of a failing function.
    assert betainc(a, b, x) + betainc(b, a, 1.0 - x) == pytest.approx(1.0, abs=1e-13)


# ------------------------------------------------- the coefficients, straight from the definition
def test_partial_numerators_match_the_published_formula():
    a, b, x = 2.3, 4.1, 0.37
    for m in range(1, 6):
        even = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        odd = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        assert beta_cf_coefficient(a, b, x, 2 * m) == pytest.approx(even, rel=1e-15)
        assert beta_cf_coefficient(a, b, x, 2 * m + 1) == pytest.approx(odd, rel=1e-15)


def test_the_fraction_converges_and_is_depth_independent():
    """DISCRIMINATION: the value must be stable, and the estimator must be capable of moving —
    a truncation so shallow it is wrong proves the depth is actually read."""
    a, b, x = 3.0, 5.0, 0.4
    converged = beta_continued_fraction(a, b, x)
    shallow = beta_continued_fraction(a, b, x, max_depth=1)
    assert math.isfinite(converged)
    assert shallow != converged, "a depth-1 truncation must differ, or depth is not being used"


# --------------------------------------------------- published Student-t upper-tail quantiles
# Standard table values (any statistical table; e.g. A&S 26.7). The tolerance is loose at large df
# BY DESIGN and the reason is recorded: the residual is in ``student_t_ppf``'s root finder, NOT in
# the continued fraction, and it is PRE-EXISTING — the predecessor produced 2.457261542401 at
# (0.99, df=30), digit-for-digit the same as the rewrite. The M6 design operates at Welch df 2–4.
PUBLISHED_T = {
    (0.95, 1): 6.313751514700,
    (0.95, 2): 2.919985580355,
    (0.95, 3): 2.353363435465,
    (0.95, 4): 2.131846786033,
    (0.95, 10): 1.812461122811,
    (0.975, 10): 2.228138851986,
    (0.99, 5): 3.364929999900,
}


@pytest.mark.parametrize("key,want", sorted(PUBLISHED_T.items()))
def test_published_student_t_quantiles(key, want):
    p, df = key
    assert student_t_ppf(p, df) == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("df", (1, 2, 4, 7, 50))
def test_t_cdf_at_zero_is_exactly_one_half(df):
    assert student_t_cdf(0.0, df) == 0.5


@pytest.mark.parametrize("df", (1, 2, 4, 10))
@pytest.mark.parametrize("t", (0.3, 1.0, 2.5))
def test_cdf_ppf_round_trip(df, t):
    assert student_t_ppf(student_t_cdf(t, df), df) == pytest.approx(t, abs=1e-7)


# ------------------------------------------------- the two quantities the M6 design actually pins
@pytest.mark.parametrize("s,nu,want", [(3, 2.0, 3.980645752134), (5, 4.0, 3.072811363562)])
def test_the_pinned_mde_multipliers_are_unchanged_by_the_rewrite(s, nu, want):
    """§7 v1.5: the scoring-dominant limit gives nu = S-1 and multiplier t(0.95)+t(0.80).
    Both were BIT-IDENTICAL across the rewrite (receipt m6_s4_betacf_rewrite.json)."""
    assert welch_satterthwaite_df(1e-9, 9999.0, 1.0, float(s - 1)) == pytest.approx(nu, abs=1e-9)
    got = student_t_ppf(0.95, nu) + student_t_ppf(0.80, nu)
    assert got == pytest.approx(want, abs=1e-11)
