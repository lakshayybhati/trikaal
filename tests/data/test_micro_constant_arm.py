"""The ``micro_constant`` arm must actually CONSTANT the micro dims — cell 6 was silently cell 4.

THE DEFECT. ``arm_feature_idx`` returns the SAME 16 columns for ``micro``, ``micro_shuffled`` and
``micro_constant``, so the transform dispatch in ``build_symbol_windows`` is the ONLY thing that
distinguishes them — and it handled only the shuffled arm. ``micro_constant`` therefore
column-selected the full micro vector and returned it untouched. Cell 6 would have trained clean,
scored clean, and been CELL 4 UNDER A DIFFERENT NAME: ΔIR(4−6) ≈ 0, read as "the capacity
handicap is negligible". A fabricated lower end for the C-12 bracket, with no crash to notice.

WHY THESE ASSERTIONS AND NOT "IT RUNS". A no-op branch reproduces the defect exactly while looking
fixed, so any check that a no-op would also pass is not a check. Hence the three-part shape:
the constant arm must DIFFER from the micro arm (kills the no-op), the difference must be CONFINED
to the micro dims (kills a transform that corrupts the price channels), and those dims must
actually be CONSTANT in the output (kills a transform that perturbs rather than pins).

These run on SYNTHETIC arrays so they hold in CI with no lake. The lake-backed versions live in
``test_micro_constant_arm_real_bars`` below and skip when the lake is absent — the defect was found
on real bars and the fix was proven there before a cent was spent.
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.constants import N_FEATURES
from trikaal.data.universe_loader import build_symbol_windows
from trikaal.train.arms import ARM_MICRO, ARM_MICRO_CONSTANT, ARM_OHLCV, arm_feature_idx

MICRO_DIMS = (7, 8, 9, 10, 11, 12)
PERP_DIMS = (13, 14, 15)  # funding/OI — masked on 100% of v1 bars; constant_micro asserts this
SEQ_LEN = 32


def _synthetic(n_bars: int = 4096, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_bars, N_FEATURES)).astype(np.float32)
    # MASK SEMANTICS ARE INVERTED FROM THE OBVIOUS READING: placebo.py:47 — 0 = usable/active,
    # 1 = masked. I wrote this fixture the intuitive way round first (ones everywhere, perp False)
    # and `assert_perp_dims_masked` refused it: that is the funding/OI tripwire doing its job on a
    # TEST, which is the cheapest possible place to learn the convention.
    mask = np.zeros((n_bars, N_FEATURES), dtype=bool)  # 0 = usable
    mask[:, list(PERP_DIMS)] = True  # 1 = masked — funding/OI, the v1 lake's structural reality
    x[:, list(PERP_DIMS)] = 0.0
    segment_id = np.zeros(n_bars, dtype=np.int64)
    ts = (np.arange(n_bars, dtype=np.int64) + 1) * 60_000
    return x, mask, segment_id, ts


def _windows(arm: str, *, boundary_ms: int, seed: int = 0, arrays=None):
    x, mask, segment_id, ts = arrays if arrays is not None else _synthetic()
    return build_symbol_windows(
        "TESTUSDT",
        x,
        mask,
        segment_id,
        ts,
        arm=arm,
        seq_len=SEQ_LEN,
        boundary_ms=boundary_ms,
        seed=seed,
    )


def _boundary(ts) -> int:
    return int(ts[len(ts) // 2])


# --------------------------------------------------------------- 1. DISCRIMINATION
def test_constant_arm_differs_from_micro_arm():
    """Kills the no-op. This is the assertion the pre-fix source fails."""
    arrays = _synthetic()
    b = _boundary(arrays[3])
    micro = _windows(ARM_MICRO, boundary_ms=b, arrays=arrays)
    const = _windows(ARM_MICRO_CONSTANT, boundary_ms=b, arrays=arrays)
    assert micro.x.shape == const.x.shape, "the arms must stay dimension-matched (F=16)"
    assert not np.array_equal(micro.x, const.x), (
        "micro_constant produced BYTE-IDENTICAL windows to micro — the transform did not run, "
        "and cell 6 is cell 4 under a different name"
    )


def test_the_difference_is_confined_to_the_micro_dims():
    """Kills a transform that corrupts the price/volume channels while zeroing the micro ones."""
    arrays = _synthetic()
    b = _boundary(arrays[3])
    micro = _windows(ARM_MICRO, boundary_ms=b, arrays=arrays)
    const = _windows(ARM_MICRO_CONSTANT, boundary_ms=b, arrays=arrays)
    idx = list(arm_feature_idx(ARM_MICRO))
    keep = [i for i, col in enumerate(idx) if col not in MICRO_DIMS]
    changed = [i for i, col in enumerate(idx) if col in MICRO_DIMS]
    assert np.array_equal(micro.x[..., keep], const.x[..., keep]), (
        "a non-micro channel changed — the transform is touching columns it must not"
    )
    assert not np.array_equal(micro.x[..., changed], const.x[..., changed])


def test_the_micro_dims_are_actually_constant():
    """Kills a transform that perturbs rather than pins. Zero IS the causal mean (features are
    z-scored), which is why constant_micro uses it."""
    arrays = _synthetic()
    const = _windows(ARM_MICRO_CONSTANT, boundary_ms=_boundary(arrays[3]), arrays=arrays)
    idx = list(arm_feature_idx(ARM_MICRO))
    changed = [i for i, col in enumerate(idx) if col in MICRO_DIMS]
    sub = const.x[..., changed]
    assert np.all(sub == 0.0), f"micro dims are not pinned: {np.unique(sub)[:5]}"


def test_ohlcv_arm_is_untouched_by_any_of_this():
    """The ohlcv arm never sees the micro columns at all; it must be unaffected."""
    arrays = _synthetic()
    b = _boundary(arrays[3])
    a = _windows(ARM_OHLCV, boundary_ms=b, arrays=arrays)
    assert a.x.shape[-1] == len(arm_feature_idx(ARM_OHLCV))


# --------------------------------------------------------------- 2. REAL BARS
LAKE = "processed/universe_bars"


@pytest.mark.skipif(not __import__("pathlib").Path(LAKE).exists(), reason="no local lake")
def test_micro_constant_arm_real_bars():
    """The same three assertions on REAL lake bars — where the defect was found."""
    import json

    from trikaal.data.universe_loader import calendar_boundary_ms, connect_lake
    from trikaal.eval.conformance import MDE_INPUTS
    from trikaal.run.matrix import load_symbol_arrays, windows_by_arm

    pinned = json.loads(MDE_INPUTS.read_text())
    sym = next(iter(pinned["symbols_sampled"]))
    b = calendar_boundary_ms(pinned["window"][0], pinned["window"][1], float(pinned["train_frac"]))
    raw = {sym: load_symbol_arrays(connect_lake(LAKE), sym, boundary_ms=b, tail_bars=200_000)}
    w = windows_by_arm(
        raw, seq_len=SEQ_LEN, boundary_ms=b, seed=0, arms=(ARM_MICRO, ARM_MICRO_CONSTANT)
    )
    micro, const = w[ARM_MICRO][0], w[ARM_MICRO_CONSTANT][0]
    idx = list(arm_feature_idx(ARM_MICRO))
    keep = [i for i, c in enumerate(idx) if c not in MICRO_DIMS]
    changed = [i for i, c in enumerate(idx) if c in MICRO_DIMS]
    assert not np.array_equal(micro.x, const.x), "no-op on real bars"
    assert np.array_equal(micro.x[..., keep], const.x[..., keep]), "leaked outside the micro dims"
    assert np.all(const.x[..., changed] == 0.0), "micro dims not pinned on real bars"
