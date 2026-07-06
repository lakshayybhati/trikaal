"""KATs + calibration for the paired moving-block ΔIR bootstrap (prereg §3/§3a; Item 8).

The two ``calibration`` tests ARE the pre-flight Item-8 evidence: on synthetic paired series
with a strong common factor (the crypto-market structure the pairing exists for),
(a) a true-null Δ rejects at ≈ the nominal α = 0.05, and (b) a planted Δ = MDE_paired is
detected with ≈ the nominal 0.80 power. Both run the PINNED recipe (B = 10,000, ⌈√T⌉ blocks,
percentile CI) and are deterministic (fixed trial seeds), so the measured rates are stable
numbers, not flaky draws. Stated tolerances: ±3·binomial-MC-σ around the nominal rate.
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.eval.metrics import information_ratio, periods_per_year
from trikaal.eval.paired_bootstrap import (
    BOOT_ALPHA,
    BOOT_POWER,
    block_length,
    expand_block_starts,
    paired_delta_ir_bootstrap,
)

H = 15  # the primary horizon

# DGP: cells share a strong common factor (pairwise corr ≈ 0.80, like the realized ρ̄_raw
# structure) + mildly autocorrelated idiosyncratic parts, so the BLOCK aspect is exercised.
COMMON_W, IDIO_W, PHI = 0.9, 0.45, 0.2
SIGMA_CELL = float(np.sqrt(COMMON_W**2 + IDIO_W**2))  # per-cell period std under the DGP


def _ar1(rng: np.random.Generator, n: int, phi: float = PHI) -> np.ndarray:
    e = rng.standard_normal(n)
    out = np.empty(n)
    out[0] = e[0]
    for i in range(1, n):  # stationary unit-variance AR(1)
        out[i] = phi * out[i - 1] + np.sqrt(1.0 - phi * phi) * e[i]
    return out


def _paired_null(rng: np.random.Generator, t: int) -> tuple[np.ndarray, np.ndarray]:
    common = COMMON_W * _ar1(rng, t)
    return common + IDIO_W * _ar1(rng, t), common + IDIO_W * _ar1(rng, t)


# --------------------------------------------------------------------- fast unit KATs
def test_expand_block_starts_hand_kat():
    # two blocks of length 2 tile T=4 exactly
    idx = expand_block_starts(np.array([[0, 2]]), 2, 4)
    np.testing.assert_array_equal(idx, [[0, 1, 2, 3]])
    # truncation: 2 blocks of length 3 → 6 rows, cut to T=5
    idx = expand_block_starts(np.array([[1, 0]]), 3, 5)
    np.testing.assert_array_equal(idx, [[1, 2, 3, 0, 1]])
    assert block_length(2000) == 45 and block_length(35_000) == 188


def test_row_ir_matches_information_ratio():
    from trikaal.eval.paired_bootstrap import _row_ir

    rows = np.random.default_rng(0).standard_normal((8, 500))
    vec = _row_ir(rows, H)
    ref = np.array([information_ratio(r, H) for r in rows])
    np.testing.assert_allclose(vec, ref, rtol=1e-12)


def test_deterministic_chunk_independent_seed_sensitive():
    a, b = _paired_null(np.random.default_rng(11), 1200)
    r1 = paired_delta_ir_bootstrap(a, b, h=H, b=2000)
    r2 = paired_delta_ir_bootstrap(a, b, h=H, b=2000)
    assert r1 == r2  # bit-identical rerun
    r3 = paired_delta_ir_bootstrap(a, b, h=H, b=2000, chunk=64)
    r4 = paired_delta_ir_bootstrap(a, b, h=H, b=2000, chunk=1024)
    assert r1 == r3 == r4  # chunking is memory-only, never statistical
    r5 = paired_delta_ir_bootstrap(a, b, h=H, b=2000, seed=1)
    assert r5.ci_lower != r1.ci_lower  # the seed genuinely drives the replicates


def test_pairing_cancels_the_common_factor():
    """The v1.1 rationale, measured: with shared blocks the common component cancels inside
    every replicate, so the paired SE is FAR tighter than an independent-draws SE."""
    a, b = _paired_null(np.random.default_rng(21), 1500)
    paired = paired_delta_ir_bootstrap(a, b, h=H, b=2000)
    # unpaired reference: same recipe but INDEPENDENT block draws per cell
    t, length = 1500, block_length(1500)
    nb = int(np.ceil(t / length))
    rng = np.random.default_rng(99)
    sa = expand_block_starts(rng.integers(0, t - length + 1, size=(2000, nb)), length, t)
    sb = expand_block_starts(rng.integers(0, t - length + 1, size=(2000, nb)), length, t)
    ppy = np.sqrt(periods_per_year(H))
    ir = lambda rows: rows.mean(1) / (rows.std(1, ddof=0) + 1e-12) * ppy  # noqa: E731
    se_unpaired = np.std(ir(a[sa]) - ir(b[sb]), ddof=1)
    assert paired.se_boot < 0.5 * se_unpaired  # the pairing must bind, hard


def test_planted_large_delta_fires_all_clauses():
    a, b = _paired_null(np.random.default_rng(31), 1200)
    r = paired_delta_ir_bootstrap(a + 0.5, b, h=H, b=2000)  # huge planted per-period edge
    assert r.delta_ir > 0 and r.passes_ci and r.passes_mde
    assert r.ci_lower < r.delta_ir < r.ci_upper
    null = paired_delta_ir_bootstrap(a, b, h=H, b=2000)
    assert not null.passes_mde  # a true null must not clear the MDE clause here
    assert null.ci_lower < null.ci_upper


def test_input_validation():
    a, b = _paired_null(np.random.default_rng(41), 400)
    with pytest.raises(ValueError, match="same-shape"):
        paired_delta_ir_bootstrap(a, b[:-1], h=H, b=100)
    with pytest.raises(ValueError, match="too short"):
        paired_delta_ir_bootstrap(a[:4], b[:4], h=H, b=100)
    bad = a.copy()
    bad[7] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        paired_delta_ir_bootstrap(bad, b, h=H, b=100)


# ------------------------------------------------------- Item-8 calibration (slow, pinned B)
T_CAL = 2000
M_NULL = 400  # 3σ binomial MC tolerance at p=0.05: ±3·√(.05·.95/400) = ±0.0327
M_POWER = 300  # 3σ binomial MC tolerance at p=0.80: ±3·√(.8·.2/300)  = ±0.0693


@pytest.mark.slow
def test_calibration_null_rejection_is_alpha():
    """Item 8 KAT (a): true Δ = 0 → clause-1 rejection rate ≈ α = 0.05, band 0.05 ± 0.033."""
    rejections = 0
    for i in range(M_NULL):
        a, b = _paired_null(np.random.default_rng(10_000 + i), T_CAL)
        rejections += paired_delta_ir_bootstrap(a, b, h=H).passes_ci
    rate = rejections / M_NULL
    print(f"\n[calibration] null rejection rate = {rate:.4f} (nominal {BOOT_ALPHA})")
    assert 0.05 - 0.033 <= rate <= 0.05 + 0.033, f"null rejection {rate:.4f} outside 0.05±0.033"


@pytest.mark.slow
def test_calibration_power_at_planted_mde():
    """Item 8 KAT (b): planted Δ = MDE_paired → clause-1 power ≈ 0.80, band 0.80 ± 0.069.

    MDE_paired is derived from the DGP's own typical SE_boot (median over a 50-trial null
    pilot) — the same quantity §3 clause 2 derives it from at money time."""
    pilot = [
        paired_delta_ir_bootstrap(*_paired_null(np.random.default_rng(20_000 + i), T_CAL), h=H)
        for i in range(50)
    ]
    se_typ = float(np.median([p.se_boot for p in pilot]))
    mde = (1.6448536269514722 + 0.8416212335729143) * se_typ  # (z_.95 + z_.80)·SE — §3 clause 2
    shift = mde * SIGMA_CELL / np.sqrt(periods_per_year(H))  # per-period edge worth ΔIR = MDE
    hits = 0
    for i in range(M_POWER):
        a, b = _paired_null(np.random.default_rng(30_000 + i), T_CAL)
        hits += paired_delta_ir_bootstrap(a + shift, b, h=H).passes_ci
    power = hits / M_POWER
    print(f"[calibration] power at planted MDE = {power:.4f} (nominal {BOOT_POWER})")
    assert 0.80 - 0.069 <= power <= 0.80 + 0.069, f"power {power:.4f} outside 0.80±0.069"
