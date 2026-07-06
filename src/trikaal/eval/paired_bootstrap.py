"""Paired moving-block bootstrap for ΔIR — the prereg §3 CI clause, §3a pinned recipe.

The primary M6 statistic is ΔIR_info = IR(Cell 4) − IR(Cell 5) on the pooled cross-sectional
portfolio period series (seed-mean, FULL calendar grid with flat periods as 0.0 — xsection's
convention, which keeps the two cells time-aligned regardless of per-cell activity). The CI is
**paired**: one set of moving-block indices per replicate is applied to BOTH cells' series
(equivalently, blocks of the Δr_p series are resampled), so the common market component cancels
inside every replicate — an unpaired CI would be wrongly wide (prereg §3, amendment v1.1).

Pinned recipe (prereg §3a — do not parameterize away): B = 10,000 replicates, bootstrap RNG seed
20260704, block length ⌈√T⌉, percentile CI (the one-sided lower bound is the α-quantile of the
bootstrap ΔIR distribution, α = 0.05), SE_boot = the standard deviation of the same B replicates,
MDE_paired = (z₀.₉₅ + z₀.₈₀)·SE_boot (§3 clause 2 — a variance/nuisance quantity, never a
function of the effect's sign). The per-replicate statistic is IR(a*) − IR(b*) with the SHARED
index draw — the exact ``metrics.information_ratio`` convention (population std, EPS_IR,
√(525600/h) annualization).

Determinism: ALL block starts are drawn in one RNG call before any chunking, so results are
bit-identical for a given (series, h, b, seed) regardless of the memory-chunk size.
Calibration-KAT'd in ``tests/eval/test_paired_bootstrap.py`` (pre-flight Item 8): null →
rejection ≈ α; planted Δ = MDE_paired → power ≈ 0.80.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trikaal.eval.dsr import norm_ppf
from trikaal.eval.metrics import EPS_IR, information_ratio, periods_per_year

BOOT_B = 10_000  # §3a: replicates
BOOT_SEED = 20260704  # §3a: bootstrap RNG seed
BOOT_ALPHA = 0.05  # §3a: one-sided level (the CI lower bound is the α-quantile)
BOOT_POWER = 0.80  # §3 clause 2: the power point inside MDE_paired


def block_length(t: int) -> int:
    """§3a block length: ⌈√T⌉."""
    return int(np.ceil(np.sqrt(t)))


def expand_block_starts(starts: np.ndarray, length: int, t: int) -> np.ndarray:
    """``[B, n_blocks]`` block starts → ``[B, T]`` row indices: concatenate blocks, truncate to T.

    Hand-KAT'd: the standard moving-block reconstruction (overlapping blocks drawn with
    replacement; the final partial block is truncated so every replicate has exactly T rows)."""
    b, nb = starts.shape
    idx = starts[:, :, None] + np.arange(length, dtype=starts.dtype)[None, None, :]
    return idx.reshape(b, nb * length)[:, :t]


def _row_ir(rows: np.ndarray, h: int) -> np.ndarray:
    """Vectorized ``metrics.information_ratio`` over ``[B, T]`` rows (same math, same eps)."""
    mean = rows.mean(axis=1)
    std = rows.std(axis=1, ddof=0)  # population std — the metrics.py convention
    return mean / (std + EPS_IR) * np.sqrt(periods_per_year(h))


@dataclass(frozen=True)
class PairedBootstrap:
    """One paired ΔIR bootstrap: the point estimate, CI bounds, SE, and the derived MDE."""

    delta_ir: float  # IR(a) − IR(b) on the actual series (metrics.information_ratio)
    ci_lower: float  # one-sided lower bound = α-quantile of the bootstrap ΔIR replicates
    ci_upper: float  # (1−α)-quantile — the §3 clause-3 placebo-health diagnostic needs it
    se_boot: float  # std (ddof=1) of the bootstrap ΔIR replicates
    mde_paired: float  # (z_{1−α} + z_power)·SE_boot — §3 clause 2
    t: int
    block_len: int
    b: int
    seed: int
    alpha: float
    passes_ci: bool  # clause 1: ci_lower > 0
    passes_mde: bool  # clause 2: delta_ir ≥ mde_paired


def paired_delta_ir_bootstrap(
    r_a: np.ndarray,
    r_b: np.ndarray,
    *,
    h: int,
    b: int = BOOT_B,
    seed: int = BOOT_SEED,
    alpha: float = BOOT_ALPHA,
    power: float = BOOT_POWER,
    chunk: int = 256,
) -> PairedBootstrap:
    """Paired moving-block bootstrap of ΔIR = IR(r_a) − IR(r_b) with SHARED block indices.

    ``r_a``/``r_b`` are the two cells' pooled FULL-calendar-grid period series (seed-mean,
    flat periods = 0.0) on the SAME grid — the §3 pairing contract. ``chunk`` only bounds
    memory; it cannot change the result (all starts are drawn before chunking)."""
    a = np.asarray(r_a, dtype=np.float64)
    bb = np.asarray(r_b, dtype=np.float64)
    if a.ndim != 1 or bb.ndim != 1 or a.shape != bb.shape:
        raise ValueError(f"paired series must be 1-d and same-shape; got {a.shape} vs {bb.shape}")
    t = int(a.shape[0])
    if t < 8:
        raise ValueError(f"series too short for a block bootstrap: T={t}")
    if not (np.isfinite(a).all() and np.isfinite(bb).all()):
        raise ValueError(
            "non-finite period returns — the full-calendar-grid convention fills flat "
            "periods with 0.0; NaN here is an upstream bug, not data"
        )
    length = block_length(t)
    n_blocks = int(np.ceil(t / length))
    rng = np.random.default_rng(seed)
    # ONE draw for all replicates → chunk-size-independent determinism
    starts = rng.integers(0, t - length + 1, size=(b, n_blocks), dtype=np.int64)
    deltas = np.empty(b, dtype=np.float64)
    for c0 in range(0, b, chunk):
        c1 = min(c0 + chunk, b)
        idx = expand_block_starts(starts[c0:c1], length, t)
        deltas[c0:c1] = _row_ir(a[idx], h) - _row_ir(bb[idx], h)  # SHARED idx = the pairing
    delta_ir = information_ratio(a, h) - information_ratio(bb, h)
    ci_lower = float(np.quantile(deltas, alpha, method="linear"))
    ci_upper = float(np.quantile(deltas, 1.0 - alpha, method="linear"))
    se_boot = float(np.std(deltas, ddof=1))
    mde_paired = float((norm_ppf(1.0 - alpha) + norm_ppf(power)) * se_boot)
    return PairedBootstrap(
        delta_ir=float(delta_ir),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        se_boot=se_boot,
        mde_paired=mde_paired,
        t=t,
        block_len=length,
        b=int(b),
        seed=int(seed),
        alpha=float(alpha),
        passes_ci=bool(ci_lower > 0.0),
        passes_mde=bool(delta_ir >= mde_paired),
    )


__all__ = [
    "BOOT_ALPHA",
    "BOOT_B",
    "BOOT_POWER",
    "BOOT_SEED",
    "PairedBootstrap",
    "block_length",
    "expand_block_starts",
    "paired_delta_ir_bootstrap",
]
