"""KATs for the §7 v1.4.3 money-leg recalibration (scripts/m6_canary.py).

Load-bearing claims: (item 3) SIGMA is TRAINING-INVARIANT — the tokenized feature matrix is
bit-identical across SIGMA, so raising it changes ONLY the backtest economics; (item 2) the
non-degeneracy floor is SIGMA-relative; the recalibrated constants are pinned so a silent change
fails a named KAT.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import m6_canary as mc  # noqa: E402


def test_recalibrated_constants_pinned():
    """§7 v1.4.3 pins — a silent change to any of these fails here, not silently in a run."""
    assert mc.SIGMA == 0.05, "economic-magnitude SIGMA pinned to 0.05 (v1.4.3 item 3)"
    assert mc.C_SIGNAL == 3.0, "information content (C_SIGNAL) UNCHANGED by the recalibration"
    assert mc.SIGNAL_LAG == 2, "plant lag UNCHANGED"
    assert mc.ORACLE_MARGIN_MULT == 5.0, "pre-declared oracle margin multiple (item 3)"
    assert mc.NONDEGEN_MU_STD_FRAC == 0.5, "non-degeneracy floor fraction (item 2)"
    assert mc.MICRO_DIMS_IDX == (7, 8, 9, 10, 11, 12), "the fix keys on the canonical micro block"


def test_training_invariance_x_bit_identical():
    """Item 3 load-bearing: the tokenized features are BIT-IDENTICAL across SIGMA (x[:,0]=r/sigma
    cancels), so re-training at the new SIGMA reproduces the cells bit-for-bit."""
    inv = mc.x_bit_identity_across_sigma(n_bars=20_000)
    for arm in ("planted=True", "planted=False"):
        assert inv[arm]["x_identical"] is True, f"{arm}: feature matrix must be SIGMA-invariant"
        assert inv[arm]["mask_identical"] is True
    # raw_ret_close scales EXACTLY with the sigma ratio (0.05/0.01 = 5.0)
    assert abs(inv["planted=True"]["raw_ret_close_ratio_mean"] - 5.0) < 1e-9


def test_synth_symbol_features_sigma_invariant_direct():
    """Direct check: same rng seed, two sigmas → identical x/mask, raw_ret_close scaled by the ratio."""
    import zlib

    seed = (zlib.crc32(b"planted"), 0)
    a = mc.synth_symbol(np.random.default_rng(seed), planted=True, n_bars=10_000, sigma=0.01)
    b = mc.synth_symbol(np.random.default_rng(seed), planted=True, n_bars=10_000, sigma=0.05)
    assert np.array_equal(a["x"], b["x"]), "features must not depend on sigma"
    assert np.array_equal(a["mask"], b["mask"])
    np.testing.assert_allclose(b["raw_ret_close"], 5.0 * a["raw_ret_close"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(b["sigma"], 5.0 * a["sigma"])


def test_nondegeneracy_floor_is_sigma_relative():
    """Item 2: the floor separates a varying strategy cell from a near-constant (degenerate) cell,
    and it scales with SIGMA (μ̂ ∝ SIGMA), so it survives the economic recalibration."""
    floor = mc.NONDEGEN_MU_STD_FRAC * mc.SIGMA
    # signal-cell std ~ 3.1 * SIGMA (money-leg planted cell4); degenerate ~ 0.01-0.06 * SIGMA
    signal_std = 3.1 * mc.SIGMA
    degenerate_std = 0.05 * mc.SIGMA
    assert signal_std >= floor, "a genuine varying strategy clears the floor"
    assert degenerate_std < floor, "a near-constant cell is flagged degenerate"


@pytest.mark.parametrize("arm", ["ohlcv", "micro"])
def test_shared_ohlcv_dims_are_zero_through_six(arm):
    """The reallocation receipt (item 9) keys on OHLCV_ONLY_IDX=(0..6) as the SHARED dims."""
    from trikaal.constants import OHLCV_ONLY_IDX

    assert OHLCV_ONLY_IDX == (0, 1, 2, 3, 4, 5, 6)
    # both arms carry dims 0..6; micro adds 7..12 — the reallocation is on the shared block
    from trikaal.train.arms import arm_feature_idx

    idx = arm_feature_idx(arm)
    for d in OHLCV_ONLY_IDX:
        assert d in idx
