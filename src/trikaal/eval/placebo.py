"""Cell-5 microstructure placebo — the negative control machinery (§8.C.4).

The +micro cells carry more *input capacity* (16 dims vs 7), so a Cell-4-over-Cell-2 gain could be
microstructure *information* or merely capacity. Cell 5 is bit-for-bit Cell 4 except the
microstructure sub-vector is replaced by a **surrogate** that preserves its marginal distribution
and cross-channel structure while **destroying temporal alignment** with OHLCV and the target:

* **block time-permutation** (primary): within each segment, permute the microstructure block as a
  unit across time — within-bar cross-channel structure kept, alignment to OHLCV/target destroyed.
* **phase-randomized** (stricter alt): per channel, randomize Fourier phase preserving the power
  spectrum (Theiler surrogate) — keeps each channel's autocorrelation, destroys cross-correlation.

The surrogate is a control, *deliberately non-causal* and never shipped. The decision rule compares
``Δ_micro = IR(Cell4) − IR(Cell2)`` to ``Δ_placebo = IR(Cell5) − IR(Cell2)``. M5 exercises the
machinery; the real cell IRs (and the DSR-margin test) come from M6's 5 trained models.
Known-answer-tested in ``tests/eval/test_placebo.py``.
"""

from __future__ import annotations

import numpy as np

from trikaal.data.segments import segment_bounds

# Active aggTrades microstructure channels in the M2/M3 build (code dims 7–12: TFI,
# signed_count_imbalance, trade_count, mean_trade_size, trade_size_dispersion, large_trade_share).
# The spec's full microstructure sub-vector is dims 7–15 (incl. funding/OI), masked in this build.
MICRO_DIMS: tuple[int, ...] = (7, 8, 9, 10, 11, 12)
# Perp channels (funding, log_oi, d_oi) — masked historically (§1.2 OI retention trap), so
# MICRO_DIMS excludes them. The tripwire below is what makes that exclusion SAFE.
PERP_DIMS: tuple[int, ...] = (13, 14, 15)


def assert_perp_dims_masked(mask: np.ndarray, *, perp_dims: tuple[int, ...] = PERP_DIMS) -> None:
    """Tripwire (m6_design §6 item 10f): funding/OI must be FULLY masked in any eval slice.

    ``MICRO_DIMS`` deliberately stops at dim 12 because this build masks funding/OI everywhere.
    If those dims ever ACTIVATE (mask bit 0), the Cell-5 shuffle would silently under-shuffle —
    leaving real, temporally-aligned funding/OI information inside a "shuffled-micro" placebo cell
    and corrupting the Δ_micro-vs-Δ_placebo comparison. Fail loudly; never silently exclude."""
    m = np.asarray(mask)
    cols = list(perp_dims)
    active = m[:, cols] == 0  # mask semantics: 0 = usable/active, 1 = masked
    if bool(active.any()):
        n_active = int(active.sum())
        raise AssertionError(
            f"placebo tripwire: funding/OI dims {perp_dims} are ACTIVE on {n_active} entries — "
            "MICRO_DIMS under-shuffles; extend the Cell-5 shuffle-dim set before evaluating"
        )


def block_time_permute(
    x: np.ndarray,
    segment_id: np.ndarray,
    *,
    micro_dims: tuple[int, ...] = MICRO_DIMS,
    seed: int = 0,
) -> np.ndarray:
    """Surrogate: within each segment, permute the microstructure block as a unit across time."""
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    cols = list(micro_dims)
    for si, (s0, s1) in enumerate(segment_bounds(segment_id)):
        rng = np.random.default_rng((seed, si))  # per-segment, seed-pinned
        perm = rng.permutation(s1 - s0)
        out[s0:s1, cols] = x[s0:s1, :][perm][:, cols]
    return out


def phase_randomize(
    x: np.ndarray,
    segment_id: np.ndarray,
    *,
    micro_dims: tuple[int, ...] = MICRO_DIMS,
    seed: int = 0,
) -> np.ndarray:
    """Surrogate: per channel per segment, randomize Fourier phase preserving the power spectrum."""
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    for si, (s0, s1) in enumerate(segment_bounds(segment_id)):
        n = s1 - s0
        if n < 4:
            continue
        rng = np.random.default_rng((seed, si))
        for d in micro_dims:
            f = np.fft.rfft(x[s0:s1, d])
            ph = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, f.shape[0]))
            ph[0] = 1.0  # keep DC real
            if n % 2 == 0:
                ph[-1] = 1.0  # keep Nyquist real
            out[s0:s1, d] = np.fft.irfft(np.abs(f) * ph, n=n)
    return out


def placebo_verdict(ir_cell2: float, ir_cell4: float, ir_cell5: float) -> dict[str, object]:
    """Δ_micro vs Δ_placebo (§8.C.4). The microstructure leg holds only if Δ_micro exceeds
    Δ_placebo by a DSR-surviving margin (the margin test is applied at M6)."""
    d_micro = ir_cell4 - ir_cell2
    d_placebo = ir_cell5 - ir_cell2
    return {
        "delta_micro": d_micro,
        "delta_placebo": d_placebo,
        "micro_exceeds_placebo": bool(d_micro > d_placebo),
        "note": "raw comparison only — DSR-margin test applied at M6 on the 5 trained cells",
    }


__all__ = [
    "MICRO_DIMS",
    "PERP_DIMS",
    "assert_perp_dims_masked",
    "block_time_permute",
    "phase_randomize",
    "placebo_verdict",
]
