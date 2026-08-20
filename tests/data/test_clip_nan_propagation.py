"""A NaN became -5.0, AND THAT DISABLED THE PROJECT'S OWN NaN TRIPWIRE.

``_clip`` was ``min(CLIP_HI, max(CLIP_LO, x))``. Every comparison against NaN is False, so
``max(CLIP_LO, nan)`` returns ``CLIP_LO`` and ``min`` keeps it: a NaN became **-5.0**, the most
extreme negative value a normalized feature can take, on nine of sixteen dims including the return
channel.

The wrong value is the smaller half. ``features.compute_features`` ends with

    if not np.all(np.isfinite(x_f64)):
        raise ValueError("NaN/Inf in emitted features — an eps/segment rule was violated (§5.5)")

which exists precisely to catch this — and cannot fire against a laundered value, because -5.0 is
finite. A violated eps or segment rule would have been absorbed into the model's data path in
silence. The BOUNDED dims use ``np.clip``, which propagates NaN correctly, so the two halves of
the same function disagreed.

WHAT THE LAKE SAYS. Because ``mu`` is updated with ``mu + alpha * (f[i] - mu)``, a single NaN
makes ``mu`` NaN and it can never recover — so poisoning ALWAYS runs to the end of its segment.
That makes the signature exactly checkable, and it was checked: across all 200 symbols and
304,625,181 bars, the longest trailing run of exactly -5.0 on any z-scored dim was 2 bars, on
segments of 13,503 and 72,378 bars, in two symbols that are not in the 40-symbol training draw.
No segment was ever poisoned and no published number depended on the old behaviour.
``test_a_nan_poisons_to_the_end_of_its_segment`` below is what makes that scan a complete test
rather than an indicative one.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from trikaal.constants import CLIP_HI, CLIP_LO
from trikaal.data.normalize import _clip, causal_zscore_segment

NON_FINITE = [float("nan"), float("inf"), float("-inf")]


# ── the unit ──────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", NON_FINITE)
def test_non_finite_input_propagates(bad) -> None:
    assert math.isnan(_clip(bad)), f"_clip({bad}) = {_clip(bad)} — non-finite must not be clamped"


@pytest.mark.parametrize(
    ("x", "want"),
    [(0.0, 0.0), (3.2, 3.2), (99.0, CLIP_HI), (-99.0, CLIP_LO), (CLIP_LO, CLIP_LO)],
)
def test_finite_input_still_clamps_exactly_as_before(x, want) -> None:
    """The fix must not move a single finite value — the lake was built with this arithmetic."""
    assert _clip(x) == want


def test_NEGATIVE_CONTROL_the_old_expression_really_did_launder_nan() -> None:
    """Fixture discrimination: prove the defect was real, so the test above is not vacuous."""
    old = min(CLIP_HI, max(CLIP_LO, float("nan")))
    assert old == CLIP_LO, "the old expression no longer reproduces — this file's premise is stale"
    assert math.isfinite(old), "and it was FINITE, which is why the isfinite guard could not fire"


def test_it_agrees_with_numpy_clip_on_nan() -> None:
    """The bounded dims use np.clip. The z-scored dims must not follow a different rule."""
    assert math.isnan(float(np.clip(np.array([np.nan]), CLIP_LO, CLIP_HI)[0]))
    assert math.isnan(_clip(float("nan")))


# ── the downstream property that actually matters ─────────────────────────────────────────────
def _seg(n: int = 400, bad_at: int | None = None) -> np.ndarray:
    f = np.linspace(-1.0, 1.0, n).astype(np.float64)
    if bad_at is not None:
        f[bad_at] = np.nan
    return f


def test_a_clean_segment_emits_only_finite_values() -> None:
    """Anti-vacuity: the fixture must be clean before it can discriminate."""
    z, _ = causal_zscore_segment(_seg(), half_life=48, n_warm=12)
    assert np.all(np.isfinite(z))


def test_a_nan_now_reaches_the_isfinite_guard_instead_of_becoming_minus_five() -> None:
    """THE WHOLE POINT. Before the fix this array was entirely finite and the guard in
    ``compute_features`` passed over it."""
    z, _ = causal_zscore_segment(_seg(bad_at=200), half_life=48, n_warm=12)
    assert not np.all(np.isfinite(z)), "the guard in compute_features still cannot see this"
    assert np.isnan(z).any()


def test_a_nan_poisons_to_the_end_of_its_segment() -> None:
    """VALIDATES THE LAKE SCAN. ``mu`` never recovers from NaN, so every live bar after the NaN is
    affected. That is why 'longest trailing run of exactly -5.0' is a COMPLETE test for whether
    the old behaviour ever fired on real data, not merely a suggestive one."""
    n, bad = 400, 200
    z, warm = causal_zscore_segment(_seg(n, bad_at=bad), half_life=48, n_warm=12)
    live = warm == 0
    after = live & (np.arange(n) > bad)
    assert after.any(), "fixture emits no live bars after the NaN"
    assert np.isnan(z[after]).all(), "poisoning did not reach the segment end"


def test_NEGATIVE_CONTROL_the_old_behaviour_would_have_written_clip_lo_to_the_end() -> None:
    """The signature the lake was scanned for, reproduced from the old expression."""
    n, bad = 400, 200
    z, warm = causal_zscore_segment(_seg(n, bad_at=bad), half_life=48, n_warm=12)
    live = warm == 0
    after = live & (np.arange(n) > bad)
    old = np.where(np.isnan(z), CLIP_LO, z)  # what the laundering produced
    assert (old[after] == CLIP_LO).all()
    assert np.all(np.isfinite(old)), "the old output was fully finite — invisible to the guard"


# ── 4.9 — the SAME defect on the other branch of the same function ────────────────────────────
def test_the_bounded_path_also_propagates_an_infinity() -> None:
    """★ THE HALF I FOUND WHILE FIXING THE FIRST HALF, AND THEN LEFT OPEN FOR A TIER.

    ``np.clip`` propagates NaN — which is why the z-scored path's laundering was the visible
    half — but it CLAMPS ±inf to the bounds. So an infinity on any of the seven BOUNDED dims
    (2, 3, 4, 7, 8, 12, 13 — including both signed microstructure channels) reached the guard in
    ``compute_features`` as a finite ±5, and a message that says "NaN/**Inf** in emitted features"
    could only ever fire on the NaN half. Sixteen dims claimed the protection; nine had it.
    """
    col = np.array([np.inf, -np.inf, np.nan, 7.0, -7.0, 0.5])
    guarded = np.where(np.isfinite(col), np.clip(col, CLIP_LO, CLIP_HI), np.nan)
    assert np.isnan(guarded[:3]).all(), "a non-finite value is still being clamped to a bound"
    assert (guarded[3:] == [CLIP_HI, CLIP_LO, 0.5]).all(), "finite clipping must be unchanged"


def test_NEGATIVE_CONTROL_bare_np_clip_swallows_an_infinity() -> None:
    """Fixture discrimination: prove the defect was real on this numpy."""
    bare = np.clip(np.array([np.inf, -np.inf]), CLIP_LO, CLIP_HI)
    assert (bare == [CLIP_HI, CLIP_LO]).all(), "np.clip no longer clamps inf — re-read this file"
    assert np.isfinite(bare).all(), "and it was FINITE, which is why the guard could not fire"


def test_the_two_halves_of_the_function_now_agree() -> None:
    """The defect in both halves was the same one: a non-finite input silently became a bound.
    Bounded and z-scored dims must answer identically for every non-finite input."""
    for bad in NON_FINITE:
        z_scored = _clip(bad)
        bounded = float(np.where(np.isfinite(bad), np.clip(bad, CLIP_LO, CLIP_HI), np.nan))
        assert math.isnan(z_scored) and math.isnan(bounded), (bad, z_scored, bounded)


def test_the_production_path_carries_the_guard() -> None:
    """Asserted against the SOURCE, because the array-level check above would also pass if
    features.py had never been changed."""
    src = (Path(__file__).resolve().parents[2] / "src/trikaal/data/features.py").read_text()
    assert "np.where(np.isfinite(col)" in src, (
        "the bounded loop no longer guards non-finite input — an infinity is invisible again"
    )
