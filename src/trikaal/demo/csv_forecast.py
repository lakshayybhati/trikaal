"""CSV → forecast, through the PRODUCTION feature path. Offline, local, forecast-only.

Drop in an OHLCV CSV for any asset, get the three seeds' forecasts back. Nothing is uploaded;
this module makes no network call of any kind and the app that wraps it binds to localhost.

★ WHY A PLAIN OHLCV CSV IS EXACTLY THE RIGHT INPUT, AND IT IS MEASURED, NOT ASSUMED.
The three banked units are cell 1 — the OHLCV-only arm, ``arm_feature_idx(ARM_OHLCV)`` = 7 dims of
price-shape + log volume/amount. ``compute_features`` still builds all 16 dims, and 7 of those
need aggTrades (v_buy/v_sell/n_buy/n_sell/per-bar size pools) that a CSV cannot supply. So the
question is whether zero-filling them disturbs the 7 dims the model actually reads.

MEASURED ON A 3,000-BAR STREAM: with EVERY micro input zeroed, dims 0-6, their masks AND sigma are
BIT-IDENTICAL to the same stream with realistic micro (max|diff| = 0.000e+00, exactly). The
micro-derived quantities never enter the OHLCV dims. That is what makes this buildable, and it is
pinned in ``tests/demo/test_csv_forecast.py`` so it cannot silently stop being true.

★ THE WARM-UP FLOOR IS 1,232 ROWS, NOT 512, AND THIS IS THE SPEC CORRECTION THAT MATTERED.
The features are causally self-normalized by streaming EWMA (``universe_loader.py:17-20``), so a
user's CSV starts COLD where the lake had years of history. Measured, not read off constants:
dims 0-6 are fully unmasked only from ROW 720 (the slow z-score warm-up, half_life_slow 5760 →
5760//8 = 720). A 512-row CSV therefore has ZERO clean rows and its entire 512-bar context sits
inside warm-up — the model would be fed masked features end to end and would still cheerfully
return a number. A clean context needs 720 + 512 = 1,232 rows (~20.5 h of 1-minute bars).
Below that this module REFUSES with the arithmetic in the message, rather than returning a
confident-looking forecast computed on warm-up.

(The volatility estimator additionally warms for 1,440 bars. It emits finite sigma from row 0, so
it does not mask, but sigma below ~1,440 rows is estimated on a short window — reported as a
warning rather than a refusal, because it degrades rather than invalidates.)

★ OUT OF DISTRIBUTION IS THE HEART OF THIS, NOT A FOOTER. The bar interval is INFERRED from the
timestamps and anything other than 60 s is extrapolation. We cannot detect "this is a stock, not a
crypto perp" at all, so the blanket statement has to carry it and it is rendered INTO the figure:
TRAINED ON CRYPTO 1-MINUTE BARS; ANYTHING ELSE IS EXTRAPOLATION.

★ NO P&L. No profit, position, equity curve, cumulative return or Sharpe is computed here or
anywhere downstream — ``tests/demo/test_no_pnl.py`` covers this module and fails if one appears.
"""

from __future__ import annotations

import csv as _csv
import io
import math
from dataclasses import dataclass, field

import numpy as np

from trikaal.data.config import FeatureConfig
from trikaal.data.features import compute_features
from trikaal.data.synthetic import RawStream
from trikaal.demo.inference import DEMO_SEQ_LEN, Unit
from trikaal.eval.predict import predict_mu
from trikaal.train.arms import ARM_OHLCV, select_arm
from trikaal.train.token_stream import tokenize_features

BAR_MS = 60_000
SEQ_LEN = DEMO_SEQ_LEN

# Measured in this module's docstring; asserted in tests rather than trusted.
WARMUP_ROWS = 720
MIN_ROWS = WARMUP_ROWS + SEQ_LEN  # 1232
VOL_WARMUP_ROWS = 1440

# The MTP horizons the model was trained for. NOT a free-text box: h outside this set was never
# trained. h=15 is the ONLY one the paper evaluates; the rest are trained but unscored.
TRAINED_HORIZONS = (1, 5, 15, 60)
PAPER_HORIZON = 15

_ALIASES = {
    "timestamp": {"timestamp", "time", "date", "datetime", "open_time", "ts", "bar_open_ms"},
    "open": {"open", "o"},
    "high": {"high", "h"},
    "low": {"low", "l"},
    "close": {"close", "c", "price"},
    "volume": {"volume", "vol", "v", "base_volume", "basevolume"},
    "amount": {"amount", "quote_volume", "quotevolume", "quote_asset_volume", "turnover"},
}


class CsvRefused(ValueError):
    """A refusal the user should read as a REASON, never as a crash."""


@dataclass(frozen=True)
class CsvBars:
    ts_ms: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    amount: np.ndarray
    inferred_bar_ms: int
    warnings: list[str] = field(default_factory=list)


def _to_ms(v: str) -> int:
    s = v.strip().strip('"')
    try:
        n = float(s)
        # epoch s / ms / us, disambiguated by magnitude
        if n > 1e17:
            return int(n / 1000)
        if n > 1e11:
            return int(n)
        return int(n * 1000)
    except ValueError:
        pass
    from datetime import datetime

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            from datetime import UTC

            return int(datetime.strptime(s[:19], fmt).replace(tzinfo=UTC).timestamp() * 1000)
        except ValueError:
            continue
    raise CsvRefused(f"could not parse timestamp {v!r}")


def parse_csv(text: str) -> CsvBars:
    """Parse an OHLCV CSV. Every refusal names the reason and the fix — never a traceback."""
    rdr = _csv.reader(io.StringIO(text))
    rows = [r for r in rdr if r and any(c.strip() for c in r)]
    if not rows:
        raise CsvRefused("the file is empty")
    head = [h.strip().lower().replace(" ", "_") for h in rows[0]]
    col = {}
    for want, names in _ALIASES.items():
        for i, h in enumerate(head):
            if h in names:
                col[want] = i
                break
    missing = [k for k in ("timestamp", "open", "high", "low", "close") if k not in col]
    if missing:
        raise CsvRefused(
            f"missing required column(s): {', '.join(missing)}. "
            f"Header seen: {rows[0]}. Need at least timestamp, open, high, low, close "
            "(volume and amount are used if present)."
        )
    body = rows[1:]
    if len(body) < MIN_ROWS:
        raise CsvRefused(
            f"only {len(body):,} data rows — this needs at least {MIN_ROWS:,}. "
            f"The features are causally self-normalized by streaming EWMA, so a fresh CSV starts "
            f"COLD: the first {WARMUP_ROWS} rows are warm-up and the model reads a {SEQ_LEN}-bar "
            f"context, so {WARMUP_ROWS} + {SEQ_LEN} = {MIN_ROWS:,} rows (~{MIN_ROWS / 60:.0f} "
            "hours of 1-minute bars) are needed before ANY bar is fully normalized. Below this a "
            "forecast would be computed entirely on warm-up values, which is why it is refused "
            "rather than returned with a caveat."
        )

    def grab(key, default=None):
        if key not in col:
            return None if default is None else np.full(len(body), default, np.float64)
        out = np.empty(len(body), np.float64)
        for i, r in enumerate(body):
            try:
                out[i] = float(r[col[key]])
            except (ValueError, IndexError):
                out[i] = np.nan
        return out

    ts = np.array([_to_ms(r[col["timestamp"]]) for r in body], np.int64)
    o, h, low_, c = (grab(k) for k in ("open", "high", "low", "close"))
    v = grab("volume", 0.0)
    amt = grab("amount")
    if amt is None:
        amt = v * c  # quote volume ~ base * price when the CSV omits it

    order = np.argsort(ts, kind="stable")
    ts, o, h, low_, c, v, amt = (a[order] for a in (ts, o, h, low_, c, v, amt))
    if not np.all(np.diff(ts) > 0):
        keep = np.r_[True, np.diff(ts) > 0]
        ts, o, h, low_, c, v, amt = (a[keep] for a in (ts, o, h, low_, c, v, amt))

    bad = ~(np.isfinite(o) & np.isfinite(h) & np.isfinite(low_) & np.isfinite(c))
    if bad.any():
        raise CsvRefused(f"{int(bad.sum()):,} row(s) have non-numeric or missing OHLC values")

    d = np.diff(ts)
    inferred = int(np.median(d)) if d.size else 0
    warns: list[str] = []
    if inferred != BAR_MS:
        warns.append(
            f"OUT OF DISTRIBUTION: inferred bar interval is {inferred / 1000:g}s, not 60s. "
            "This model was trained on 1-minute bars only."
        )
    if len(ts) < VOL_WARMUP_ROWS:
        warns.append(
            f"volatility is estimated on {len(ts):,} bars, below its {VOL_WARMUP_ROWS:,}-bar "
            "warm-up — the scale is usable but noisier than the model saw in training."
        )
    return CsvBars(ts, o, h, low_, c, v, amt, inferred, warns)


def _raw_stream(b: CsvBars) -> RawStream:
    """CSV bars → RawStream with the aggTrades fields ZEROED.

    Proven safe: dims 0-6 and sigma are bit-identical under zeroed micro (module docstring).
    """
    t = b.ts_ms.shape[0]
    z = np.zeros(t, np.float64)
    return RawStream(
        bar_open_ms=b.ts_ms,
        open=b.open,
        high=b.high,
        low=b.low,
        close=b.close,
        v_kline=np.nan_to_num(b.volume),
        quote_volume=np.nan_to_num(b.amount),
        v_buy=z,
        v_sell=z.copy(),
        n_buy=np.zeros(t, np.int64),
        n_sell=np.zeros(t, np.int64),
        sizes=[np.array([]) for _ in range(t)],
        is_perp=False,
        funding_ts=np.array([], np.int64),
        funding_val=np.array([]),
        oi_ts=np.array([], np.int64),
        oi_val=np.array([]),
    )


@dataclass(frozen=True)
class CsvForecast:
    h: int
    decision_ts_ms: int
    last_close: float
    per_seed: dict  # seed -> {"mu": float, "pct": float, "price": float}
    band: dict  # {"lo_pct","hi_pct","lo_price","hi_price"} from MC trajectories, or {}
    context_ts: np.ndarray
    context_close: np.ndarray
    inferred_bar_ms: int
    warnings: list[str]
    n_rows: int


def forecast_from_csv(
    text: str,
    *,
    units: dict[int, Unit],
    h: int = PAPER_HORIZON,
    device: str = "cpu",
    mc_samples: int = 32,
) -> CsvForecast:
    """The whole path: CSV → production features → tokenizer → AR → per-seed mu-hat + MC band."""
    if h not in TRAINED_HORIZONS:
        raise CsvRefused(
            f"h={h} was never trained. The MTP heads cover {TRAINED_HORIZONS}; h={PAPER_HORIZON} "
            "is the only horizon the paper evaluates."
        )
    bars = parse_csv(text)
    out = compute_features(_raw_stream(bars), FeatureConfig())
    x_arm, m_arm = select_arm(np.asarray(out.x, np.float32), out.m, ARM_OHLCV)

    i = int(x_arm.shape[0] - 1)  # forecast FROM the last bar
    dec = np.array([i], np.int64)
    per_seed = {}
    mu_samples = []
    for seed in sorted(units):
        u = units[seed]
        b_c, b_f = tokenize_features(
            u.tok, x_arm, m_arm, out.segment_id, window=SEQ_LEN, device=device
        )
        mu = float(
            predict_mu(
                u.model,
                u.tok,
                b_c,
                b_f,
                out.ts,
                out.sigma,
                dec,
                h=h,
                seq_len=SEQ_LEN,
                device=device,
                estimator="expectation",
            )[0]
        )
        mc = float(
            predict_mu(
                u.model,
                u.tok,
                b_c,
                b_f,
                out.ts,
                out.sigma,
                dec,
                h=h,
                seq_len=SEQ_LEN,
                device=device,
                estimator="mc_mean",
                mc_samples=mc_samples,
            )[0]
        )
        mu_samples.append(mc)
        last = float(bars.close[i])
        per_seed[seed] = {
            "mu": mu,
            "pct": (math.exp(mu) - 1.0) * 100.0,
            "price": last * math.exp(mu),
        }

    # The band is the SPREAD ACROSS SEEDS AND THEIR MC MEANS — an honest envelope of what these
    # three models disagree about. It is NOT a calibrated predictive interval and is labelled so.
    allmu = [v["mu"] for v in per_seed.values()] + mu_samples
    lo, hi = (min(allmu), max(allmu)) if allmu else (0.0, 0.0)
    last = float(bars.close[i])
    band = {
        "lo_pct": (math.exp(lo) - 1.0) * 100.0,
        "hi_pct": (math.exp(hi) - 1.0) * 100.0,
        "lo_price": last * math.exp(lo),
        "hi_price": last * math.exp(hi),
    }
    ctx_lo = max(0, i - SEQ_LEN + 1)
    return CsvForecast(
        h=h,
        decision_ts_ms=int(bars.ts_ms[i]),
        last_close=last,
        per_seed=per_seed,
        band=band,
        context_ts=bars.ts_ms[ctx_lo : i + 1],
        context_close=bars.close[ctx_lo : i + 1],
        inferred_bar_ms=bars.inferred_bar_ms,
        warnings=bars.warnings,
        n_rows=int(bars.ts_ms.shape[0]),
    )


def plain_english(f: CsvForecast) -> str:
    """One line, forecast only. NEVER a recommendation, a side, or a size."""
    pcts = [f.per_seed[s]["pct"] for s in sorted(f.per_seed)]
    signs = {p > 0 for p in pcts}
    joined = ", ".join(f"{p:+.3f}%" for p in pcts)
    agree = "they agree on direction" if len(signs) == 1 else "THEY DISAGREE ON DIRECTION"
    return (
        f"Over the next {f.h} minute{'s' if f.h != 1 else ''} the three models forecast "
        f"{joined} — {agree}."
    )


def forecast_csv_export(f: CsvForecast) -> str:
    """CSV export with THE SAME numbers the chart draws, so the two cannot disagree."""
    buf = io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["field", "value"])
    w.writerow(["decision_ts_ms", f.decision_ts_ms])
    w.writerow(["horizon_minutes", f.h])
    w.writerow(["last_close", f"{f.last_close:.10g}"])
    for s in sorted(f.per_seed):
        v = f.per_seed[s]
        w.writerow([f"seed{s}_forecast_pct", f"{v['pct']:.6f}"])
        w.writerow([f"seed{s}_forecast_price", f"{v['price']:.10g}"])
    w.writerow(["band_lo_pct", f"{f.band['lo_pct']:.6f}"])
    w.writerow(["band_hi_pct", f"{f.band['hi_pct']:.6f}"])
    w.writerow(["inferred_bar_seconds", f.inferred_bar_ms / 1000])
    w.writerow(["rows_used", f.n_rows])
    w.writerow(["NOT_A_RECOMMENDATION", "forecast only; no position, profit or P&L is computed"])
    for i, wn in enumerate(f.warnings):
        w.writerow([f"warning_{i + 1}", wn])
    return buf.getvalue()


__all__ = [
    "MIN_ROWS",
    "PAPER_HORIZON",
    "SEQ_LEN",
    "TRAINED_HORIZONS",
    "WARMUP_ROWS",
    "CsvBars",
    "CsvForecast",
    "CsvRefused",
    "forecast_csv_export",
    "forecast_from_csv",
    "parse_csv",
    "plain_english",
]
