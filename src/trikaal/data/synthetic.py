"""Seeded synthetic raw stream — the slice's stand-in for Binance ingest (§2.1 deferred).

Emits a per-bar OHLCV + microstructure stream (already reduced from synthetic trades, with
a per-bar trade-size pool retained for ``trade_size_dispersion`` / ``large_trade_share``),
plus funding/OI event series with their own effective timestamps. Deliberately injects the
four high-risk anomaly classes the causal-safety sampler must target:

  * **gaps** (missing grid timestamps → segment boundary),
  * **bad ticks** (a single implausible wick excursion → strictly-causal clip),
  * **structural breaks** (a ≥50% single-bar move → strictly-causal segment split),
  * **stale runs** (a flat, zero-volume constant-price run → trailing ``is_stale`` flag).

Funding/OI event timestamps are placed mid-bar (``bar_open + BAR_MS//2``) to avoid the
inclusive ``≤ close(t)`` boundary coincidence that would otherwise create an off-by-one.

Reduction note (scope): the aggTrades→bar reduction (§2.1/§2.2) is causal *by construction*
— every trade maps to its own bar by its own timestamp — so the leakage-prone surface lives
entirely in the per-bar feature transform exercised here. The fixture emits the post-reduction
per-bar table (with size pools) rather than raw aggTrade rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trikaal.data.config import BAR_MS

EPOCH_MS: int = 1_600_000_000_000  # arbitrary fixed UTC anchor (2020-09-13), deterministic


@dataclass
class AnomalyIndex:
    """Ground-truth anomaly bar indices (into the *final* gapped stream) for stratified sampling."""

    bad_tick: list[int] = field(default_factory=list)
    structural_break: list[int] = field(default_factory=list)
    segment_start: list[int] = field(default_factory=list)  # first bar of each segment
    stale_run: list[int] = field(default_factory=list)  # all bars inside detected stale runs


@dataclass
class RawStream:
    """Immutable-ish raw per-bar stream + funding/OI events. All prices/volumes float64."""

    bar_open_ms: np.ndarray  # [T] int64, strictly increasing (gaps allowed)
    open: np.ndarray  # [T] f64
    high: np.ndarray  # [T] f64
    low: np.ndarray  # [T] f64
    close: np.ndarray  # [T] f64
    v_kline: np.ndarray  # [T] f64 (canonical kline base volume)
    quote_volume: np.ndarray  # [T] f64
    v_buy: np.ndarray  # [T] f64 (taker buy base volume, aggTrades)
    v_sell: np.ndarray  # [T] f64
    n_buy: np.ndarray  # [T] int64
    n_sell: np.ndarray  # [T] int64
    sizes: list[np.ndarray]  # [T] ragged per-bar trade size pools (f64)
    is_perp: bool
    funding_ts: np.ndarray  # [F] int64 effective timestamps
    funding_val: np.ndarray  # [F] f64
    oi_ts: np.ndarray  # [O] int64
    oi_val: np.ndarray  # [O] f64

    def __len__(self) -> int:
        return int(self.bar_open_ms.shape[0])

    @property
    def v_agg(self) -> np.ndarray:
        return self.v_buy + self.v_sell

    def close_ms(self, k: int) -> int:
        """Effective timestamp (close) of bar ``k`` = bar_open + one bar width."""
        return int(self.bar_open_ms[k]) + BAR_MS

    def truncate_to_bar(self, k: int) -> RawStream:
        """Keep bars 0..k and funding/OI events with effective ts ≤ close(k)."""
        ts_max = self.close_ms(k)
        fmask = self.funding_ts <= ts_max
        omask = self.oi_ts <= ts_max
        return RawStream(
            bar_open_ms=self.bar_open_ms[: k + 1].copy(),
            open=self.open[: k + 1].copy(),
            high=self.high[: k + 1].copy(),
            low=self.low[: k + 1].copy(),
            close=self.close[: k + 1].copy(),
            v_kline=self.v_kline[: k + 1].copy(),
            quote_volume=self.quote_volume[: k + 1].copy(),
            v_buy=self.v_buy[: k + 1].copy(),
            v_sell=self.v_sell[: k + 1].copy(),
            n_buy=self.n_buy[: k + 1].copy(),
            n_sell=self.n_sell[: k + 1].copy(),
            sizes=[s.copy() for s in self.sizes[: k + 1]],
            is_perp=self.is_perp,
            funding_ts=self.funding_ts[fmask].copy(),
            funding_val=self.funding_val[fmask].copy(),
            oi_ts=self.oi_ts[omask].copy(),
            oi_val=self.oi_val[omask].copy(),
        )

    def perturb_after_bar(self, k: int, rng: np.random.Generator, mode: str = "noise") -> RawStream:
        """Arbitrarily corrupt all raw data strictly after bar ``k`` (eff ts > close(k)).

        ``mode``:
          * ``"noise"``    — replace future OHLCV/sizes/events with random values.
          * ``"delete"``   — drop future bars/events (equivalent to truncation).
          * ``"revert"``   — fabricate a strong mean-reversion in bars k+1..k+3 (the exact
                             pattern a forward-revert bad-tick rule would illegally read).
          * ``"signflip"`` — flip the sign of every future close-to-close move.
        """
        if mode == "delete":
            return self.truncate_to_bar(k)

        s = self.copy()
        T = len(s)
        fut = np.arange(k + 1, T)
        ts_max = self.close_ms(k)
        if fut.size:
            if mode == "noise":
                base = float(s.close[k])
                s.close[fut] = base * np.exp(rng.normal(0, 0.5, fut.size))
                s.open[fut] = base * np.exp(rng.normal(0, 0.5, fut.size))
                s.high[fut] = np.maximum(s.open[fut], s.close[fut]) * (1 + rng.random(fut.size))
                s.low[fut] = np.minimum(s.open[fut], s.close[fut]) * (1 - 0.5 * rng.random(fut.size))
                s.v_kline[fut] = rng.random(fut.size) * 1e6
                s.quote_volume[fut] = s.v_kline[fut] * base
                s.v_buy[fut] = rng.random(fut.size) * 1e6
                s.v_sell[fut] = rng.random(fut.size) * 1e6
                s.n_buy[fut] = rng.integers(0, 500, fut.size)
                s.n_sell[fut] = rng.integers(0, 500, fut.size)
                for j in fut:
                    s.sizes[j] = rng.random(int(s.n_buy[j] + s.n_sell[j]) + 1) * 100.0
            elif mode == "revert":
                anchor = float(s.close[max(0, k - 1)])
                for off, j in enumerate(fut[:3]):
                    s.close[j] = anchor  # snap back to pre-bar-k price
                    s.open[j] = anchor
                    s.high[j] = anchor * 1.001
                    s.low[j] = anchor * 0.999
            elif mode == "signflip":
                rets = np.diff(np.log(s.close[k:]))
                new_close = [float(s.close[k])]
                for r in rets:
                    new_close.append(new_close[-1] * np.exp(-r))
                s.close[k:] = np.array(new_close)
                s.open[fut] = s.close[fut - 1]
                s.high[fut] = np.maximum(s.open[fut], s.close[fut]) * 1.001
                s.low[fut] = np.minimum(s.open[fut], s.close[fut]) * 0.999
            else:
                raise ValueError(f"unknown perturb mode: {mode}")

        # corrupt future funding/OI event *values* (timestamps stay > close(k), so still future)
        ffut = s.funding_ts > ts_max
        ofut = s.oi_ts > ts_max
        if ffut.any():
            s.funding_val[ffut] = rng.normal(0, 0.01, int(ffut.sum()))
        if ofut.any():
            s.oi_val[ofut] = rng.random(int(ofut.sum())) * 1e9
        return s

    def copy(self) -> RawStream:
        return RawStream(
            bar_open_ms=self.bar_open_ms.copy(),
            open=self.open.copy(),
            high=self.high.copy(),
            low=self.low.copy(),
            close=self.close.copy(),
            v_kline=self.v_kline.copy(),
            quote_volume=self.quote_volume.copy(),
            v_buy=self.v_buy.copy(),
            v_sell=self.v_sell.copy(),
            n_buy=self.n_buy.copy(),
            n_sell=self.n_sell.copy(),
            sizes=[s.copy() for s in self.sizes],
            is_perp=self.is_perp,
            funding_ts=self.funding_ts.copy(),
            funding_val=self.funding_val.copy(),
            oi_ts=self.oi_ts.copy(),
            oi_val=self.oi_val.copy(),
        )


def make_synthetic_stream(
    seed: int = 1337,
    n_bars: int = 3000,
    *,
    is_perp: bool = True,
    inject_anomalies: bool = True,
) -> tuple[RawStream, AnomalyIndex]:
    """Generate a deterministic synthetic raw stream and its ground-truth anomaly indices.

    The stream is built on a contiguous 1m grid first (so anomaly indices are easy to place),
    then gaps are carved out by *removing* bars — which is what creates real grid holes →
    segment boundaries. Returned indices are into the final (post-removal) stream.
    """
    rng = np.random.default_rng(seed)
    n0 = n_bars

    # ---- event plan (positions on the contiguous grid) ------------------------------------
    struct_pos: dict[int, float] = {}
    stale_plan: list[tuple[int, int]] = []
    bad_tick_pos: list[int] = []
    gap_spans: list[tuple[int, int]] = []
    if inject_anomalies and n0 >= 600:
        # positions as fractions of the stream so the fixture scales (small fixtures for the
        # causal-safety gate; larger ones for model-overfit gates)
        def at(frac: float) -> int:
            return int(frac * n0)

        struct_pos = {at(0.23): 1.8, at(0.63): 1.8}  # ≥50 % single-bar jumps
        stale_plan = [(at(0.35), 40), (at(0.87), 25)]  # (start, length) flat zero-vol runs
        bad_tick_pos = [at(0.15), at(0.43), at(0.80)]  # single implausible wick spikes
        gap_spans = [(at(0.50), at(0.50) + 30)]  # 31-bar grid hole → segment boundary
    stale_bar = np.zeros(n0, dtype=bool)
    for s, length in stale_plan:
        stale_bar[s : s + length] = True

    # ---- single forward pass: build the close path with all level events in position order
    rets = rng.normal(0.0, 0.0012, n0)  # ~0.12 %/min baseline vol
    hi_wick = np.abs(rng.normal(0, 0.0008, n0))
    lo_wick = np.abs(rng.normal(0, 0.0008, n0))
    close = np.empty(n0, dtype=np.float64)
    open_ = np.empty(n0, dtype=np.float64)
    high = np.empty(n0, dtype=np.float64)
    low = np.empty(n0, dtype=np.float64)
    close[0] = 30_000.0
    open_[0] = close[0] * np.exp(-rets[0])
    stale_level = 0.0
    for i in range(n0):
        if i > 0:
            if stale_bar[i]:
                if not stale_bar[i - 1]:
                    stale_level = close[i - 1]  # plateau holds the pre-run level
                close[i] = stale_level
            elif i in struct_pos:
                close[i] = close[i - 1] * struct_pos[i]
            else:
                close[i] = close[i - 1] * np.exp(rets[i])  # resumes from prior close (incl. plateau)
            open_[i] = close[i - 1]
        if stale_bar[i]:
            open_[i] = high[i] = low[i] = close[i]
        else:
            high[i] = max(open_[i], close[i]) * np.exp(hi_wick[i])
            low[i] = min(open_[i], close[i]) * np.exp(-lo_wick[i])
    for bt in bad_tick_pos:
        high[bt] = max(open_[bt], close[bt]) * 1.30  # +30 % wick, ≫ trailing MAD

    # ---- volume / microstructure ----------------------------------------------------------
    v_kline = rng.lognormal(mean=2.0, sigma=0.6, size=n0)
    mid = 0.5 * (open_ + close)
    quote_volume = v_kline * mid
    buy_frac = np.clip(rng.beta(5, 5, n0), 0.02, 0.98)
    v_buy = v_kline * buy_frac
    v_sell = v_kline * (1.0 - buy_frac)
    n_trades = rng.integers(8, 40, n0)
    n_buy = np.minimum(n_trades, rng.binomial(n_trades, buy_frac)).astype(np.int64)
    n_sell = (n_trades - n_buy).astype(np.int64)

    sizes: list[np.ndarray] = []
    for i in range(n0):
        nt = int(n_trades[i])
        pool = rng.lognormal(mean=np.log(max(v_kline[i] / max(nt, 1), 1e-6)), sigma=0.7, size=nt)
        sizes.append(pool.astype(np.float64))

    bar_open_ms = (EPOCH_MS + np.arange(n0, dtype=np.int64) * BAR_MS).astype(np.int64)

    # ---- funding / OI events (perp only), placed MID-bar to dodge the close boundary -------
    if is_perp:
        f_bar_idx = np.arange(0, n0, 480)  # every 8h
        funding_ts = (bar_open_ms[f_bar_idx] + BAR_MS // 2).astype(np.int64)
        funding_val = rng.normal(0.0001, 0.00005, f_bar_idx.size).astype(np.float64)
        o_bar_idx = np.arange(0, n0, 5)  # 5m snapshots
        oi_ts = (bar_open_ms[o_bar_idx] + BAR_MS // 2).astype(np.int64)
        oi_val = (1e6 * np.exp(np.cumsum(rng.normal(0, 0.01, o_bar_idx.size)))).astype(np.float64)
    else:
        funding_ts = np.zeros(0, dtype=np.int64)
        funding_val = np.zeros(0, dtype=np.float64)
        oi_ts = np.zeros(0, dtype=np.int64)
        oi_val = np.zeros(0, dtype=np.float64)

    # ---- stale bars carry zero volume / no trades (price already flat from the forward pass)
    v_kline[stale_bar] = 0.0
    quote_volume[stale_bar] = 0.0
    v_buy[stale_bar] = 0.0
    v_sell[stale_bar] = 0.0
    n_buy[stale_bar] = 0
    n_sell[stale_bar] = 0
    for i in np.nonzero(stale_bar)[0]:
        sizes[i] = np.zeros(0, dtype=np.float64)

    anom_bad_tick = list(bad_tick_pos)
    anom_struct = sorted(struct_pos)
    stale_spans = [(s, s + length - 1) for (s, length) in stale_plan]

    # ---- carve gaps by removing bars; remap all indices to the final stream ----------------
    keep = np.ones(n0, dtype=bool)
    for g0, g1 in gap_spans:
        keep[g0 : g1 + 1] = False
    old_to_new = np.full(n0, -1, dtype=np.int64)
    old_to_new[keep] = np.arange(int(keep.sum()), dtype=np.int64)

    def remap(idx: int) -> int:
        return int(old_to_new[idx])

    stream = RawStream(
        bar_open_ms=bar_open_ms[keep],
        open=open_[keep],
        high=high[keep],
        low=low[keep],
        close=close[keep],
        v_kline=v_kline[keep],
        quote_volume=quote_volume[keep],
        v_buy=v_buy[keep],
        v_sell=v_sell[keep],
        n_buy=n_buy[keep],
        n_sell=n_sell[keep],
        sizes=[sizes[i] for i in range(n0) if keep[i]],
        is_perp=is_perp,
        funding_ts=funding_ts,
        funding_val=funding_val,
        oi_ts=oi_ts,
        oi_val=oi_val,
    )

    anom = AnomalyIndex(
        bad_tick=[remap(b) for b in anom_bad_tick if keep[b]],
        structural_break=[remap(s) for s in anom_struct if keep[s]],
        segment_start=[0]  # first bar of stream is always a segment start
        + [remap(g1 + 1) for (_, g1) in gap_spans if g1 + 1 < n0 and keep[g1 + 1]]
        + [remap(s) for s in anom_struct if keep[s]],  # structural breaks also start segments
        stale_run=[remap(j) for (a, b) in stale_spans for j in range(a, b + 1) if keep[j]],
    )
    anom.segment_start = sorted(set(i for i in anom.segment_start if i >= 0))
    return stream, anom
