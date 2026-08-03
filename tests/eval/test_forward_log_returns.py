"""§7 v1.6.25 (RE-AUDIT R10a) — THE MONEY LABEL, WHICH HAD NO TESTS AT ALL.

``forward_log_returns`` computes y_{t,h} = log(C_{t+h}/C_{t+1}): the realized return of a position
ENTERED AT THE NEXT BAR'S CLOSE. Every net IR in this study — the headline, the placebo, all five
cells, both dual-specification legs — is that array multiplied by a position and netted. It is the
single most load-bearing eight lines in the eval, and until now the entire suite contained not one
assertion about it.

**THE MUTATION THAT MOTIVATES THIS FILE.** Changing `clp[p + h] - clp[p + 1]` to
`clp[p + h] - clp[p]` moves the entry from C_{t+1} to C_t — the classic same-bar-entry lookahead,
which hands the strategy the t→t+1 move it formed its signal on. It passes 601 tests. It also
passes the Gate-A causal-safety file, because `exhaustive_truncation_sweep` tests that FEATURES at
bar t do not see the future; it says nothing about which bar the LABEL enters on. Two different
lookaheads, one of them unguarded.

The discriminating invariant is sharp: with `clp` a cumulative sum of per-bar log returns,
y_t = Σ_{i=t+2}^{t+h} r_i. So **y_t must be invariant to r_t AND to r_{t+1}** (r_{t+1} is the move
INTO the entry price) and must respond to every r_i in (t+1, t+h]. The lookahead mutant depends on
r_{t+1}; the "entry two bars later" variant loses r_{t+2}. Both are caught below.
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.eval.harness import forward_log_returns

H = 5


def _seg(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int64)


def _rand(n: int, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n) * 0.01


# ------------------------------------------------------------------ DISCRIMINATION FIRST
def test_the_fixture_can_tell_the_entry_conventions_apart():
    """Before asserting invariance, prove the three conventions actually differ on this input —
    otherwise every assertion below would hold against a function that ignored its arguments."""
    r = _rand(40)
    clp = np.cumsum(r)
    p = np.arange(0, 40 - H)
    correct = clp[p + H] - clp[p + 1]  # entry at C_{t+1}   (production)
    lookahead = clp[p + H] - clp[p]  # entry at C_t       (the mutant)
    late = clp[p + H] - clp[p + 2]  # entry at C_{t+2}
    assert not np.allclose(correct, lookahead)
    assert not np.allclose(correct, late)


# ------------------------------------------------------------------ the closed form
def test_matches_the_closed_form_on_a_known_path():
    r = _rand(60)
    y = forward_log_returns(r, _seg(60), H)
    clp = np.cumsum(r)
    for t in range(0, 60 - H):
        assert y[t] == pytest.approx(clp[t + H] - clp[t + 1], abs=1e-15)


def test_a_constant_drift_path_gives_exactly_h_minus_1_steps_of_drift():
    """A hand-checkable case: with every bar returning exactly d, a position entered at C_{t+1}
    and closed at C_{t+h} earns (h-1)·d — NOT h·d, which is what same-bar entry would pay."""
    d = 0.002
    n = 30
    y = forward_log_returns(np.full(n, d), _seg(n), H)
    assert y[0] == pytest.approx((H - 1) * d, abs=1e-12)
    assert y[5] == pytest.approx((H - 1) * d, abs=1e-12)
    assert y[0] != pytest.approx(H * d, abs=1e-9)  # the lookahead value


# ------------------------------------------------------------------ THE LOOKAHEAD INVARIANTS
@pytest.mark.parametrize("offset", [0, 1])
def test_the_label_at_t_IGNORES_the_bars_up_to_and_including_the_entry(offset):
    """y_t must not move when r_t or r_{t+1} moves.

    r_t is the bar the signal is formed on and r_{t+1} is the move INTO the entry price; a label
    that responds to either is paying the strategy for information it had at decision time. This
    is the assertion that fails under the `clp[p+h] - clp[p]` mutation (offset=1)."""
    t = 10
    r = _rand(60)
    base = forward_log_returns(r, _seg(60), H)[t]
    bumped = r.copy()
    bumped[t + offset] += 0.05  # a 5% shock the label must not see
    assert forward_log_returns(bumped, _seg(60), H)[t] == pytest.approx(base, abs=1e-15)


@pytest.mark.parametrize("offset", [2, 3, H])
def test_the_label_at_t_DOES_respond_to_every_bar_inside_the_holding_period(offset):
    """The other half — an invariance test that passes because the function returns a constant is
    not a test. Every bar in (t+1, t+h] must move the label by exactly the shock."""
    t = 10
    r = _rand(60)
    base = forward_log_returns(r, _seg(60), H)[t]
    bumped = r.copy()
    bumped[t + offset] += 0.05
    assert forward_log_returns(bumped, _seg(60), H)[t] == pytest.approx(base + 0.05, abs=1e-12)


def test_the_label_never_reaches_beyond_t_plus_h():
    """A bar past the exit must not move the label — the symmetric leak."""
    t = 10
    r = _rand(60)
    base = forward_log_returns(r, _seg(60), H)[t]
    bumped = r.copy()
    bumped[t + H + 1] += 0.05
    assert forward_log_returns(bumped, _seg(60), H)[t] == pytest.approx(base, abs=1e-15)


# ------------------------------------------------------------------ segments and out-of-bounds
def test_the_tail_of_every_segment_is_NaN_not_a_truncated_return():
    """The last h bars have no full holding period. They must be NaN — a shortened return there
    would be a systematically different (and cheaper) label at the end of every segment."""
    n = 40
    y = forward_log_returns(_rand(n), _seg(n), H)
    assert np.all(np.isnan(y[n - H :]))
    assert np.all(np.isfinite(y[: n - H - 1]))


def test_a_label_NEVER_crosses_a_segment_boundary():
    """Segments are delisting/gap boundaries. A label spanning one prices a move that did not
    happen in continuous trading; per-segment isolation is what prevents it."""
    n = 60
    seg = np.concatenate([np.zeros(30, np.int64), np.ones(30, np.int64)])
    r = _rand(n)
    y = forward_log_returns(r, seg, H)
    # bumping the FIRST bar of segment 2 must not move any label in segment 1
    bumped = r.copy()
    bumped[30] += 0.05
    y2 = forward_log_returns(bumped, seg, H)
    a, b = y[:30], y2[:30]
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], atol=1e-15)
    # and each segment's own tail is NaN
    assert np.all(np.isnan(y[30 - H : 30])) and np.all(np.isnan(y[60 - H :]))


def test_a_segment_shorter_than_the_horizon_is_all_NaN():
    seg = np.concatenate([np.zeros(3, np.int64), np.ones(20, np.int64)])
    y = forward_log_returns(_rand(23), seg, H)
    assert np.all(np.isnan(y[:3]))
