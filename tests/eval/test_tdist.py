"""KATs for the self-written Student-t (prereg §7 v1.5 item B).

The v1.5 MDE uses t-quantiles at the Welch-Satterthwaite df because the training-variance term has
only S-1 degrees of freedom. These are known-answer tests against published t-table values plus the
nu -> infinity limit against the project's own norm_ppf — the multiplier this feeds is a
pre-registered decision threshold, so it is not allowed to be approximately right.
"""

from __future__ import annotations

import math

import pytest

from trikaal.eval.dsr import norm_ppf
from trikaal.eval.tdist import (
    betainc,
    student_t_cdf,
    student_t_ppf,
    welch_satterthwaite_df,
)

# published two-decimal-table values, carried to 6 dp
TABLE = [
    (0.95, 1, 6.313752),
    (0.95, 2, 2.919986),
    (0.80, 2, 1.060660),
    (0.95, 4, 2.131847),
    (0.80, 4, 0.940965),
    (0.975, 10, 2.228139),
    (0.99, 30, 2.457262),
    (0.995, 100, 2.625891),
]


@pytest.mark.parametrize(("p", "nu", "want"), TABLE)
def test_ppf_matches_published_tables(p, nu, want):
    assert abs(student_t_ppf(p, nu) - want) < 1e-5


@pytest.mark.parametrize(("p", "nu", "want"), TABLE)
def test_cdf_inverts_the_ppf(p, nu, want):
    assert abs(student_t_cdf(want, nu) - p) < 1e-7


def test_df2_against_the_closed_form():
    """At nu=2 the CDF is elementary: F(t) = 1/2 + t / (2*sqrt(2 + t^2)). An independent check of
    the incomplete-beta path."""
    for t in (-3.0, -1.0, 0.0, 0.5, 2.919986, 7.0):
        closed = 0.5 + t / (2.0 * math.sqrt(2.0 + t * t))
        assert abs(student_t_cdf(t, 2) - closed) < 1e-12


def test_converges_to_the_normal_as_df_grows():
    for p in (0.80, 0.95, 0.99):
        assert abs(student_t_ppf(p, 1e7) - norm_ppf(p)) < 1e-5
    # and it is strictly heavier-tailed at finite df
    assert student_t_ppf(0.95, 2) > student_t_ppf(0.95, 30) > norm_ppf(0.95)


def test_symmetry_and_median():
    for nu in (2, 4, 17.3):
        assert abs(student_t_cdf(0.0, nu) - 0.5) < 1e-12
        assert abs(student_t_ppf(0.3, nu) + student_t_ppf(0.7, nu)) < 1e-9


def test_betainc_endpoints_and_symmetry():
    assert betainc(2.0, 3.0, 0.0) == 0.0
    assert betainc(2.0, 3.0, 1.0) == 1.0
    for a, b, x in ((0.5, 2.5, 0.3), (3.0, 1.5, 0.7), (1.0, 1.0, 0.42)):
        assert abs(betainc(a, b, x) + betainc(b, a, 1.0 - x) - 1.0) < 1e-12
    assert abs(betainc(1.0, 1.0, 0.42) - 0.42) < 1e-12  # I_x(1,1) = x


def test_ppf_rejects_out_of_range_inputs():
    for bad in (0.0, 1.0, -0.1, 1.2):
        with pytest.raises(ValueError, match="p in"):
            student_t_ppf(bad, 4)
    with pytest.raises(ValueError, match="degrees of freedom"):
        student_t_ppf(0.5, 0)


def test_welch_satterthwaite_is_between_the_two_dfs():
    nu = welch_satterthwaite_df(1.0, 9999.0, 1.0, 4.0)
    assert 4.0 < nu < 9999.0
    assert abs(welch_satterthwaite_df(0.0, 9999.0, 0.0, 4.0) - 9999.0) < 1e-9  # degenerate: no 0/0
