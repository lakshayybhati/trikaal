"""Multi-symbol universe training loader — the fold-aligned draw (m6_design §4, §6 item 4).

The audit finding this module closes: the old Stage-1/Stage-2 samplers split train/val as a 90/10
index tail per symbol, which OVERLAPS the eval fold plan's forward blocks — training windows could
reach into the region the harness later scores. Here the training draw is derived from the SAME
purged walk-forward geometry the eval uses:

* the train/eval boundary is a **calendar timestamp** shared by every symbol (the 70 % point of
  the observation window) — a per-symbol index split would misalign symbols in calendar time and
  leak the cross-symbol common factor;
* a training window ``[s, s+L)`` is valid iff its LAST bar closes **strictly before
  boundary − E bars** (leading embargo, E = 120 = the M5 fold design) AND it lies within one
  segment — a window crossing the train/eval boundary is structurally impossible (KAT'd);
* symbol sampling ∝ ``n_windows^α`` (α = 0.5, m6_design §4) — the thin-by-history coins are
  down-weighted by their usable-window count, not dropped;
* the sampler carries an explicit cursor (RNG state + draw count) for resumable checkpointing.

Normalization: the lake's features are **causally self-normalized per symbol at build time**
(streaming EWMA z-scores — bars ≤ t only); there is NO train-fitted statistic anywhere in this
path, so per-fold "frozen stats" holds structurally: nothing can be re-fit on eval data.

Cell-5 integration: ``arm`` = "micro_shuffled" applies :func:`trikaal.train.arms.shuffle_micro`
(per-(seed, symbol), segment-wise) BEFORE arm selection — the same function the eval driver calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trikaal.constants import N_FEATURES
from trikaal.data.universe import date_to_ms
from trikaal.train.arms import (
    ARM_MICRO,
    ARM_MICRO_CONSTANT,
    ARM_MICRO_SHUFFLED,
    constant_micro,
    select_arm,
    shuffle_micro,
)

BAR_MS = 60_000
EMBARGO_BARS_DEFAULT = 120  # E = H_max + L_corr (folds.py — the M5 fold design)


def calendar_boundary_ms(window_start: str, window_end: str, train_frac: float = 0.7) -> int:
    """The GLOBAL train/eval boundary: the ``train_frac`` point of the calendar window.

    Calendar-time (not per-symbol index) so every symbol's train region ends at the same instant —
    the cross-symbol common factor cannot leak from one symbol's eval region into another's train.
    Deterministic and data-independent (a pure function of the window config)."""
    lo, hi = date_to_ms(window_start), date_to_ms(window_end)
    return int(lo + train_frac * (hi - lo))


def eval_block_bounds_ms(
    window_start: str, window_end: str, *, train_frac: float = 0.7, k: int = 6
) -> list[tuple[int, int]]:
    """K contiguous forward eval blocks over [boundary, window_end) in calendar ms (M5 fold K=6)."""
    b = calendar_boundary_ms(window_start, window_end, train_frac)
    hi = date_to_ms(window_end)
    edges = [b + ((hi - b) * i) // k for i in range(k)] + [hi]
    return [(edges[i], edges[i + 1]) for i in range(k)]


def fold_valid_starts(
    ts: np.ndarray,
    segment_id: np.ndarray,
    *,
    seq_len: int,
    boundary_ms: int,
    embargo_bars: int = EMBARGO_BARS_DEFAULT,
    stride: int = 1,
) -> np.ndarray:
    """Start indices of every length-L window that is segment-contained AND fold-legal.

    Fold-legal: the window's last bar opens strictly before ``boundary_ms − embargo_bars`` bars —
    the leading embargo of the purged walk-forward plan. A window that would cross (or even
    approach within E bars of) the train/eval boundary CANNOT be emitted from here."""
    ts = np.asarray(ts, dtype=np.int64)
    seg = np.asarray(segment_id, dtype=np.int64)
    n = ts.shape[0]
    cutoff = boundary_ms - embargo_bars * BAR_MS
    starts: list[int] = []
    i = 0
    while i + seq_len <= n:
        last = i + seq_len - 1
        if ts[last] >= cutoff:
            break  # ts is monotone — nothing further can be legal
        if seg[i] == seg[last]:
            starts.append(i)
            i += stride
        else:
            i += 1
    return np.asarray(starts, dtype=np.int64)


@dataclass
class SymbolWindows:
    """One symbol's loaded arrays + its fold-legal training window starts."""

    symbol: str
    x: np.ndarray  # [T, F_arm] float32 (arm-selected; Cell-5 shuffle already applied)
    mask: np.ndarray  # [T, F_arm] uint8
    segment_id: np.ndarray  # [T]
    ts: np.ndarray  # [T] int64
    starts: np.ndarray  # fold-legal window starts

    @property
    def n_windows(self) -> int:
        return int(self.starts.size)


def load_symbol_windows(
    con,
    symbol: str,
    *,
    arm: str = ARM_MICRO,
    seq_len: int = 128,
    boundary_ms: int,
    embargo_bars: int = EMBARGO_BARS_DEFAULT,
    seed: int = 0,
    frequency: str = "1m",
) -> SymbolWindows:
    """Load one symbol from the lake (DuckDB ``bars`` view) → arm-transformed, fold-legal windows.

    Order of operations matters: the Cell-5 shuffle runs on the FULL 16-dim layout (it needs the
    micro column positions + the funding/OI tripwire), THEN the arm selects its column subset."""
    xcols = ", ".join(f"x_{i}" for i in range(N_FEATURES))
    mcols = ", ".join(f"m_{i}" for i in range(N_FEATURES))
    t = con.execute(
        f"SELECT bar_open_ms, segment_id, {xcols}, {mcols} "
        "FROM bars WHERE symbol = ? AND frequency = ? ORDER BY bar_open_ms",
        [symbol, frequency],
    ).to_arrow_table()
    n = t.num_rows
    x = np.empty((n, N_FEATURES), dtype=np.float32)
    m = np.empty((n, N_FEATURES), dtype=np.uint8)
    for i in range(N_FEATURES):
        x[:, i] = t.column(f"x_{i}").to_numpy()
        m[:, i] = t.column(f"m_{i}").to_numpy()
    ts = t.column("bar_open_ms").to_numpy().astype(np.int64)
    seg = t.column("segment_id").to_numpy().astype(np.int64)
    return build_symbol_windows(
        symbol,
        x,
        m,
        seg,
        ts,
        arm=arm,
        seq_len=seq_len,
        boundary_ms=boundary_ms,
        embargo_bars=embargo_bars,
        seed=seed,
    )


def build_symbol_windows(
    symbol: str,
    x: np.ndarray,
    mask: np.ndarray,
    segment_id: np.ndarray,
    ts: np.ndarray,
    *,
    arm: str = ARM_MICRO,
    seq_len: int = 128,
    boundary_ms: int,
    embargo_bars: int = EMBARGO_BARS_DEFAULT,
    seed: int = 0,
) -> SymbolWindows:
    """Pure-array core of :func:`load_symbol_windows` (unit-testable without a lake)."""
    if arm == ARM_MICRO_SHUFFLED:
        x, mask = shuffle_micro(x, mask, segment_id, symbol=symbol, seed=seed)
        x = x.astype(np.float32)
    # THE CELL-6 ARM WAS A NO-OP AND THAT IS A SILENT WRONG NUMBER, not a crash.
    # `arm_feature_idx` returns the SAME 16 dims for micro, micro_shuffled and micro_constant, so
    # this dispatch is the ONLY thing that distinguishes them. Without this branch, `micro_constant`
    # column-selects the full micro vector and returns it untouched: cell 6 trains clean, scores
    # clean, and IS CELL 4 UNDER A DIFFERENT NAME — ΔIR(4−6) comes out ≈ 0 and reads as "the
    # capacity handicap is negligible". The bracket's lower end would have been fabricated.
    # It belongs HERE, beside the shuffle, for the same reason the shuffle does: the transform
    # needs the full [T,16] layout and must run BEFORE arm selection. Applied after windowing it
    # would index the wrong axis of an [N, seq_len, F] array.
    if arm == ARM_MICRO_CONSTANT:
        x, mask = constant_micro(x, mask)
        x = x.astype(np.float32)
    x_arm, m_arm = select_arm(np.asarray(x, dtype=np.float32), mask, arm)
    starts = fold_valid_starts(
        ts, segment_id, seq_len=seq_len, boundary_ms=boundary_ms, embargo_bars=embargo_bars
    )
    return SymbolWindows(
        symbol=symbol,
        x=x_arm,
        mask=m_arm,
        segment_id=np.asarray(segment_id, dtype=np.int64),
        ts=np.asarray(ts, dtype=np.int64),
        starts=starts,
    )


class MultiSymbolWindowSampler:
    """Two-stage draw over N symbols: symbol ∝ n_windows^α, then a uniform fold-legal window.

    Deterministic under ``seed``; the cursor (``state()``/``load_state()``) captures the RNG state
    + draw count so a resumed run continues the exact draw sequence (§6 item 9's sampler-cursor
    requirement). Per-symbol draw counts are tracked for the smoke evidence (thin-coin presence).
    """

    def __init__(self, per_symbol: list[SymbolWindows], *, alpha: float = 0.5, seed: int = 0):
        if not per_symbol:
            raise ValueError("no symbols")
        empty = [sw.symbol for sw in per_symbol if sw.n_windows == 0]
        if empty:
            raise ValueError(f"symbols with zero fold-legal windows: {empty}")
        self.per_symbol = sorted(per_symbol, key=lambda s: s.symbol)  # deterministic order
        self.alpha = float(alpha)
        self.seed = int(seed)
        counts = np.array([sw.n_windows for sw in self.per_symbol], dtype=np.float64)
        w = counts**self.alpha
        self.weights = w / w.sum()
        self._rng = np.random.default_rng(seed)
        self.n_drawn = 0
        self.drawn_by_symbol = dict.fromkeys((sw.symbol for sw in self.per_symbol), 0)

    def state(self) -> dict:
        """The resumable cursor: RNG bit-generator state + draw counters."""
        return {
            "bit_generator": self._rng.bit_generator.state,
            "n_drawn": self.n_drawn,
            "drawn_by_symbol": dict(self.drawn_by_symbol),
        }

    def load_state(self, state: dict) -> None:
        self._rng.bit_generator.state = state["bit_generator"]
        self.n_drawn = int(state["n_drawn"])
        self.drawn_by_symbol = dict(state["drawn_by_symbol"])

    def sample_batch(
        self, batch_size: int, seq_len: int
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Draw ``batch_size`` windows → ``(x[B,L,F], mask[B,L,F], symbols)``."""
        sym_ids = self._rng.choice(len(self.per_symbol), size=batch_size, p=self.weights)
        xs, ms, names = [], [], []
        for si in sym_ids:
            sw = self.per_symbol[int(si)]
            s = int(sw.starts[self._rng.integers(0, sw.n_windows)])
            xs.append(sw.x[s : s + seq_len])
            ms.append(sw.mask[s : s + seq_len])
            names.append(sw.symbol)
            self.drawn_by_symbol[sw.symbol] += 1
        self.n_drawn += batch_size
        return np.stack(xs), np.stack(ms), names


def connect_lake(root: str | Path = "processed/universe_bars"):
    """DuckDB connection with the ``bars`` view over the compacted universe lake."""
    from trikaal.data.lake import connect

    return connect(Path(root))


def train_region_liquidity_stats(
    con, boundary_ms: int, *, lookback_days: int = 180
) -> dict[str, tuple[float, int]]:
    """Per-symbol TRAIN-REGION liquidity proxy: ``{symbol: (active_minute_share, n_bars_train)}``.

    CAUSAL BY CONSTRUCTION: the query reads ``bar_open_ms < boundary_ms`` only, and the causality
    receipt is asserted (max timestamp touched < boundary). The proxy is the **active-minute
    share over a calendar-COMMON window** — the train region's final ``lookback_days`` days,
    denominator = the window's full calendar minute count, numerator = the symbol's present,
    non-stale, non-zero-volume bars in it (``is_stale`` is strictly trailing → causal; QA-only
    in the TRAINING path, legitimately read here on the eval/cost side; ``m_5 = 0`` excludes
    zero-volume minutes and, at in-window segment heads, warm-up bars — a data-gap penalty):

    * a calendar-common window is REQUIRED for era fairness — the full-train-region usable
      share ranked 2023 listings above 2021 majors (weeks of pristine modern coverage beat
      years containing archive-era gaps): a data-era artifact, not liquidity;
    * absent minutes count as non-traded (not listed = not tradable), so late listings and
      pre-window delistings rank low — economically correct;
    * zero-volume minutes are the top-cluster discriminator (measured: 155 symbols tie on
      stale-free coverage alone; only 8 tie once no-trade minutes count, and the resulting
      order is the recognizable liquidity order — XRP/SOL/ADA/BNB/... — not the alphabet);
    * raw ADV is NOT reconstructible in-lake (features are causally z-scored per symbol);
      this is the strongest liquidity ordering the lake itself supports. Tier assignment
      from these stats lives in ``eval/costs.py`` (majors pinned per spec §8.B.2).

    Symbols with zero train-region bars do not appear; the decile builder assigns them the
    conservative "tail" via ``(0.0, 0)`` defaults."""
    lo = int(boundary_ms) - int(lookback_days) * 86_400_000
    window_minutes = int(lookback_days) * 1440
    rows = con.execute(
        "SELECT symbol, count(*) AS n_bars, "
        "sum(CASE WHEN bar_open_ms >= ? AND is_stale = 0 AND m_5 = 0 THEN 1 ELSE 0 END) "
        "AS active_win, "
        "max(bar_open_ms) AS max_ms "
        "FROM bars WHERE bar_open_ms < ? GROUP BY symbol ORDER BY symbol",
        [lo, int(boundary_ms)],
    ).fetchall()
    out: dict[str, tuple[float, int]] = {}
    for symbol, n_bars, active_win, max_ms in rows:
        if int(max_ms) >= int(boundary_ms):  # the causality receipt — structurally impossible
            raise AssertionError(f"{symbol}: train-region query touched ts {max_ms} >= boundary")
        out[str(symbol)] = (float(int(active_win or 0) / window_minutes), int(n_bars))
    return out


__all__ = [
    "EMBARGO_BARS_DEFAULT",
    "MultiSymbolWindowSampler",
    "SymbolWindows",
    "build_symbol_windows",
    "calendar_boundary_ms",
    "connect_lake",
    "eval_block_bounds_ms",
    "fold_valid_starts",
    "load_symbol_windows",
    "train_region_liquidity_stats",
]
