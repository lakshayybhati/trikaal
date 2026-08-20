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

★ WHY THE FLOOR IS 1,232 AND NOT 1,952 — THE SIGMA QUESTION, DECIDED EXPLICITLY.
VOL_WARMUP_ROWS is 1,440, above MIN_ROWS, so a CSV between them clears the floor while the
volatility flag still reads "warm". That is deliberate, and the reasoning is here rather than
inherited:

  1. MECHANICALLY, sigma CANNOT CHANGE THE SIGN. predict_mu ends with ``cum * sig_t`` — sigma is a
     strictly positive multiplicative scale applied last. Direction comes entirely from the token
     rollout. But this page reports a MAGNITUDE, and a magnitude is exactly what a scale distorts,
     so that argument alone does not settle it.

  2. MEASURED, IT IS NOT DISTORTED AT ALL. sigma at the final bar, truncated to 1,232 rows versus
     a 5,000-row fully-warm reference, agrees to a ratio of 1.0000 — 0.0% error on the displayed
     percentage — under constant vol, a 10x calm->storm shift, a 10x storm->calm shift, and a 5x
     slow drift.

  3. WHY: the sigma EWMA uses vol_lambda = 0.97, i.e. HALF-LIFE 22.8 BARS and effective memory
     ~33 bars. At 1,232 rows it has FORTY TIMES its own memory of burn-in. ``VOL_WARMUP_ROWS`` is
     ``max(n_warm_vol=30, half_life_fast=1440)`` — it inherits the Z-SCORE fast half-life, not
     sigma's. It is a conservative TRAINING-EXCLUSION flag, not a convergence threshold.

  ★ MY FIRST VERSION WARNED "the scale is usable but noisier than the model saw in training".
  THAT WAS AN UNMEASURED CLAIM AND MY OWN MEASUREMENT CONTRADICTS IT. The warning now states what
  is true — a conservative flag, measured error 0.0% — rather than implying a degradation that
  does not exist. A false caveat is not a safe caveat; it teaches a reader to discount a number
  that is fine, and the next person to inherit it has no way to tell.

  (The FIRST measurement of this was itself degenerate and I caught it: a constant-volatility
  fixture makes an EWMA converge to the same value at every truncation, so it reported 0.0% for
  free. The regime-shift fixtures above are what can actually discriminate.)

★ OUT OF DISTRIBUTION IS THE HEART OF THIS, NOT A FOOTER. The bar interval is INFERRED from the
timestamps and anything other than 60 s is extrapolation. We cannot detect "this is a stock, not a
crypto perp" at all, so the blanket statement has to carry it and it is rendered INTO the figure:
TRAINED ON CRYPTO 1-MINUTE BARS; ANYTHING ELSE IS EXTRAPOLATION.

★ WHY THE FIGURE SHOWS BANDS AND A MEDIAN RATHER THAN INDIVIDUAL SAMPLED TRAJECTORIES — AND WHY
RE-MEASURING ON BTC DOES NOT OVERTURN IT. Each per-step value is a sampled token put through the
tokenizer decoder, and the RETURN channel is the worst-reconstructed of the seven OHLCV dims
(rank 7/7, 6/7, 6/7 by seed). On the banked basis — ONE window of 512 bars from 1INCHUSDT — its
MAE is 0.5501 z against a per-step decode sd of 1.1102 z, an artifact fraction of 0.50. On an
8x512 BTCUSDT basis the same quantity is 0.109 z, a fraction of 0.10. FIVE TIMES SMALLER.

★★ THE 0.50 IS THE BINDING CASE PRECISELY BECAUSE IT IS THE WORSE ONE. THIS DASHBOARD ACCEPTS
ANY CSV THE USER UPLOADS. Thin, illiquid, alt-class inputs are squarely in scope — 1INCHUSDT is
not an unlucky sample, it is a REPRESENTATIVE user input. A figure that is honest on BTC and
misleading on a thin alt is not honest; it is honest on the sample we happened to test. The
design must survive the WORST admissible input, not the best, and bands survive it: quantization
shifts every sample alike so the band's WIDTH and the MEDIAN hold, while an individual strand's
texture is codebook granularity rendered as market structure.

DO NOT "FIX" THIS BY RE-MEASURING ON A LIQUID SYMBOL AND RESTORING STRANDS. The BTC number is not
a correction of the alt number; they are two points on a range whose upper end is what an open
upload box guarantees you will eventually be handed.

★ NO P&L. No profit, position, equity curve, cumulative return or Sharpe is computed here or
anywhere downstream — ``tests/demo/test_no_pnl.py`` covers this module and fails if one appears.
"""

from __future__ import annotations

import csv as _csv
import io
import math
from collections.abc import Callable
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

# The MTP horizons the model was TRAINED for — a fact about the checkpoint, kept because the
# figure and the export both cite it.
TRAINED_HORIZONS = (1, 5, 15, 60)

# ★ ONE MC SEED FOR THE WHOLE DEMO, NAMED, NOT INHERITED — AND THIS WAS A REAL DEFECT.
# The dashboard used to call ``predict_mu(estimator="mc_mean", mc_samples=32)`` WITHOUT passing
# ``mc_seed``, so that scalar came from production's ``MC_DEFAULT_SEED``, while the percentile
# bands drawn beside it came from ``mc_paths``' own default of 20260704. The number that widened
# the seed band and the fan the user was looking at were therefore TWO DIFFERENT ROLLOUTS of the
# same model — measured +0.0158% against +0.0403% on the same BTCUSDT bars. Neither is "wrong"
# (the estimator is unbiased at any seed); what was wrong is that they disagreed by construction
# and nothing could notice. This is the shared-input lesson in its other form: PIN THE SOURCE
# EXPLICITLY RATHER THAN INHERITING A LIBRARY DEFAULT, because a default is exactly what two
# call sites silently disagree about.
MC_SEED = 20260704

# ★ WHAT THE UI MAY OFFER. h=1 IS DELIBERATELY ABSENT AND MUST NOT BE RESTORED FROM
# TRAINED_HORIZONS. mu-hat is defined over [t+1, t+h] — the entry is at C_{t+1} — so the rollout
# EXCLUDES k=1 (`if k >= 2` in predict._rollout_greedy / _rollout_mc_mean). At h=1 there are no
# steps left to accumulate and the forecast is EXACTLY 0.000000% for every seed, by construction.
# A user who picks "1 min" and sees a flat zero concludes either that the tool is broken or that
# the model is genuinely neutral at one minute. NEITHER IS TRUE, and neither a label nor a caveat
# fixes an option that structurally cannot produce a forecast — so it is not offered, and
# `forecast_from_csv` refuses it at the API too.
#
# CONSEQUENCE, STATED ON THE FIGURE: the "four trained anchors" are really THREE usable points
# {5, 15, 60}, and at the paper horizon h=15 only TWO are in range, {5, 15}. The anchor feature is
# thinner than the design assumed and the figure must not imply four.
SELECTABLE_HORIZONS = (5, 15, 60)
PAPER_HORIZON = 15

# ─────────────────────────────────────────────────────────────────────────────────────────────
# PROGRESS — the stage lines the dashboard's loading screen shows, emitted BY THE PIPELINE ITSELF
# ─────────────────────────────────────────────────────────────────────────────────────────────
# ★ WHY THIS LIVES IN THE PRODUCTION FUNCTION AND NOT IN THE UI. A loading screen that lists
# stages the UI *believes* are happening is an animation, and an animation cannot go wrong in a
# way anyone notices — it would keep showing "COMPUTING CAUSAL FEATURES" long after that call had
# been moved, removed or reordered. The lines are therefore emitted from the exact statements that
# do the work, carrying values read off the real objects (row counts, array shapes, recomputed
# hashes, per-seed mu-hat), so the screen is a TRACE, not a story about one. It is the same reason
# ``mc_paths`` is gated bit-for-bit against production rather than trusted to mirror it.
#
# ``state`` is "run" when a stage STARTS (itself a real event), "ok" when it finishes with its
# value, and "warn" for a stage that COMPLETED but whose measured value is out of distribution —
# the inferred bar interval is the one that matters, and it must not be able to finish quietly
# green. The caller renders "fail" from the refusal it catches. The callback is invoked directly
# and is NOT wrapped in a try/except: a progress sink that swallows its own exceptions is a check
# that cannot fail, and this project has paid for that shape four times.
Progress = Callable[[str, str, str], None]


def _no_progress(stage: str, detail: str, state: str) -> None:
    """Default sink: the pipeline is uninstrumented unless a caller asks for the trace."""


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
    """Epoch number or ISO-ish string -> epoch milliseconds.

    TWO DEFECTS FIXED HERE, BOTH SILENT.

    (1) ``n > 1e17`` used to return ``n / 1000``, which turns NANOSECONDS into MICROSECONDS —
    off by a factor of 1,000, with no error. ``DatetimeIndex.astype("int64")`` yields nanoseconds
    and is how most Python users produce an integer timestamp column, so this was the common path.
    Magnitudes are now separated explicitly: ~1.7e9 s, ~1.7e12 ms, ~1.7e15 us, ~1.7e18 ns.

    (2) The string path did ``strptime(s[:19], fmt)`` and then stamped UTC — truncating any
    timezone offset off the end of the string and pretending it was not there. ``...T22:13:20Z``
    and ``...T22:13:20-05:00`` parsed to the SAME instant, five hours apart from the truth.
    ``fromisoformat`` is tried first and honours the offset; a naive stamp is still read as UTC,
    which is now stated rather than assumed.
    """
    s = v.strip().strip('"').lstrip("\ufeff")
    try:
        n = float(s)
    except ValueError:
        pass
    else:
        if n > 1e17:  # nanoseconds
            return int(n / 1e6)
        if n > 1e14:  # microseconds
            return int(n / 1e3)
        if n > 1e11:  # milliseconds
            return int(n)
        return int(n * 1000)  # seconds

    from datetime import UTC, datetime

    iso = s.replace("Z", "+00:00") if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(iso)
        return int((dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp() * 1000)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return int(datetime.strptime(s, fmt).replace(tzinfo=UTC).timestamp() * 1000)
        except ValueError:
            continue
    raise CsvRefused(
        f"could not parse timestamp {v!r}. Accepted: epoch seconds/ms/us/ns, or an ISO-8601 "
        "string such as 2023-11-14T22:13:20Z or 2023-11-14 22:13:20 (naive is read as UTC)."
    )


def _rows_named(flags: np.ndarray, order: np.ndarray, limit: int = 5) -> str:
    """Name the offending file lines: a refusal that says WHERE is actionable, a count is not.

    ``order`` maps sorted position back to body position; +2 converts a body index to a 1-based
    file line, accounting for the header.
    """
    idx = np.nonzero(flags)[0]
    if idx.size == 0:
        return ""
    lines = sorted(int(order[i]) + 2 for i in idx[:limit])
    more = f" (+{idx.size - limit:,} more)" if idx.size > limit else ""
    label = "lines " if len(lines) > 1 else "line "
    return " First at file " + label + ", ".join(map(str, lines)) + more + "."


def parse_csv(text: str, *, progress: Progress = _no_progress) -> CsvBars:
    """Parse an OHLCV CSV. Every refusal names the reason and the fix — never a traceback."""
    progress("PARSING CSV", "", "run")
    rdr = _csv.reader(io.StringIO(text))
    rows = [r for r in rdr if r and any(c.strip() for c in r)]
    if not rows:
        raise CsvRefused("the file is empty")
    # ``lstrip("\ufeff")``: Excel writes a UTF-8 BOM by default, which glued itself to the first
    # header cell so ``timestamp`` read as ``\ufefftimestamp`` and the file was refused for a
    # missing column it plainly had.
    head = [h.strip().lstrip("\ufeff").lower().replace(" ", "_") for h in rows[0]]
    col = {}
    for want, names in _ALIASES.items():
        # ★ THE CANONICAL NAME WINS, WHEREVER IT SITS. This used to walk the headers in FILE order
        # and take the first cell matching any alias, so a sheet with both `price` and `close`
        # resolved `close` to whichever came first — silently forecasting a different column.
        # Aliases are now tried in priority order (canonical first) across all headers.
        for name in [want, *sorted(names - {want})]:
            if name in head:
                col[want] = head.index(name)
                break
    missing = [k for k in ("timestamp", "open", "high", "low", "close") if k not in col]
    if missing:
        raise CsvRefused(
            f"missing required column(s): {', '.join(missing)}. "
            f"Header seen: {rows[0]}. Need at least timestamp, open, high, low, close "
            "(volume and amount are used if present)."
        )
    body = rows[1:]
    progress(
        "PARSING CSV",
        f"{len(body):,} data rows, columns mapped: {', '.join(sorted(col))}",
        "ok",
    )

    progress("VALIDATING LENGTH", "", "run")
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

    progress(
        "VALIDATING LENGTH",
        f"{len(body):,} >= {MIN_ROWS:,} ({WARMUP_ROWS} warm-up + {SEQ_LEN} context)",
        "ok",
    )
    progress("READING VALUES", "", "run")

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
    n_before = int(ts.shape[0])
    if not np.all(np.diff(ts) > 0):
        keep = np.r_[True, np.diff(ts) > 0]
        ts, o, h, low_, c, v, amt = (a[keep] for a in (ts, o, h, low_, c, v, amt))
    dropped = n_before - int(ts.shape[0])
    # ★ RE-CHECKED. The floor above ran on ``len(body)`` — before this dedup, and the count was
    # never looked at again. A 1,400-row file with 700 rows duplicated cleared the 1,232 floor and
    # then forecast from 700 bars, i.e. below the warm-up the floor exists to guarantee.
    if int(ts.shape[0]) < MIN_ROWS:
        raise CsvRefused(
            f"after dropping {dropped:,} duplicate or out-of-order timestamp(s), only "
            f"{int(ts.shape[0]):,} distinct bars remain — this needs at least {MIN_ROWS:,}. "
            f"The file had {n_before:,} rows, so the shortfall is duplicates rather than length: "
            "check for a repeated export or a concatenation of overlapping windows."
        )

    bad = ~(np.isfinite(o) & np.isfinite(h) & np.isfinite(low_) & np.isfinite(c))
    if bad.any():
        raise CsvRefused(
            f"{int(bad.sum()):,} row(s) have non-numeric or missing OHLC values"
            + _rows_named(bad, order)
        )

    # ── (6) prices must be positive. A forecast price of -93.79 was rendered from a file with a
    # negative close, because nothing ever looked. Log-returns of a non-positive price are not a
    # number, and the tripwire in compute_features is downstream of the damage.
    nonpos = ~((o > 0) & (h > 0) & (low_ > 0) & (c > 0))
    if nonpos.any():
        raise CsvRefused(
            f"{int(nonpos.sum()):,} row(s) have a non-positive open/high/low/close. Prices must "
            "be > 0 — the features are built from log returns, which are undefined at or below "
            "zero." + _rows_named(nonpos, order)
        )

    # ── (7) high/low must bracket open and close. Swapped columns are otherwise undetectable:
    # every downstream number is computed happily from an inverted bar.
    eps = 1e-9
    inconsistent = (h < low_ - eps) | (h < np.maximum(o, c) - eps) | (low_ > np.minimum(o, c) + eps)
    if inconsistent.any():
        raise CsvRefused(
            f"{int(inconsistent.sum()):,} row(s) are not a valid bar: high must be >= low and >= "
            "max(open, close), and low must be <= min(open, close). The usual cause is a swapped "
            "high/low column." + _rows_named(inconsistent, order)
        )
    progress(
        "READING VALUES",
        f"{int(ts.shape[0]):,} bars sorted by timestamp, "
        + (f"{dropped:,} duplicate/out-of-order dropped" if dropped else "no duplicates"),
        "ok",
    )

    progress("INFERRING BAR INTERVAL", "", "run")
    d = np.diff(ts)
    inferred = int(np.median(d)) if d.size else 0
    progress(
        "INFERRING BAR INTERVAL",
        f"{inferred / 1000:g}s "
        + ("- matches training" if inferred == BAR_MS else "- NOT 60s: OUT OF DISTRIBUTION"),
        "ok" if inferred == BAR_MS else "warn",
    )
    warns: list[str] = []

    # ── (8) a series with no variation. MEASURED: a dead-flat 1,400-bar file produced a confident
    # +0.092% call whose ENTIRE 80% band sat above zero, narrower than any real input tested —
    # because a constant series has zero realized volatility, so sigma collapses to its floor and
    # every z-score is a ratio of ~0 to ~0. There is no forecast to make from a constant.
    if float(np.max(c) - np.min(c)) == 0.0:
        raise CsvRefused(
            f"the close price is constant at {float(c[0]):g} for all {len(c):,} bars. There is no "
            "return series to forecast: realized volatility is exactly zero, so every normalized "
            "feature is 0/0 and the model's output would be an artefact of the epsilon floors, "
            "not a prediction. This is refused rather than returned with a caveat because it "
            "produces a NARROWER, more confident band than real data does."
        )
    n_distinct = int(np.unique(c).size)
    if n_distinct < max(10, len(c) // 500):
        warns.append(
            f"NEAR-CONSTANT INPUT: only {n_distinct:,} distinct close prices across "
            f"{len(c):,} bars. Confidence bands narrow as variation falls, so a tight band here "
            "reflects the input, not conviction."
        )

    # ── (5) volume is not optional information. Two of the seven model-visible dims are
    # log_volume and log_amount; when volume is absent, unparseable or zero they are FABRICATED
    # from zeros with no notice. MEASURED: the corrupted file produced a MORE confident call
    # (+0.333%, whole band above zero) than the clean one (+0.079%, band straddling zero).
    vol_finite = np.isfinite(v)
    if "volume" not in col:
        warns.append(
            "NO VOLUME COLUMN: 2 of the 7 model-visible dimensions (log_volume, log_amount) are "
            "FABRICATED as zeros. The model was trained with real volume; treat the forecast as "
            "running on a 5-dimensional subset of its input, and note that this can make the "
            "output MORE confident rather than less."
        )
    elif not vol_finite.all():
        warns.append(
            f"UNPARSEABLE VOLUME on {int((~vol_finite).sum()):,} of {len(v):,} rows — those bars "
            "feed zeros into log_volume and log_amount, 2 of the 7 model-visible dimensions. The "
            "forecast is computed on fabricated values for them."
        )
    elif float(np.max(v)) == 0.0:
        warns.append(
            "VOLUME IS ZERO ON EVERY BAR: log_volume and log_amount — 2 of the 7 model-visible "
            "dimensions — carry no information. This is not equivalent to a quiet market; it is "
            "equivalent to deleting two inputs, and it can narrow the band rather than widen it."
        )

    if inferred != BAR_MS:
        warns.append(
            f"OUT OF DISTRIBUTION: inferred bar interval is {inferred / 1000:g}s, not 60s. "
            "This model was trained on 1-minute bars only."
        )
    if len(ts) < VOL_WARMUP_ROWS:
        warns.append(
            f"FYI, not a problem: {len(ts):,} bars is below the {VOL_WARMUP_ROWS:,}-bar "
            "volatility warm-up FLAG, but that flag inherits the z-score half-life, not "
            "sigma's. Sigma's EWMA has a 22.8-bar half-life and is fully converged here — "
            "measured error on the forecast magnitude at this length is 0.0%."
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
    # seed -> {percentile: [h] cumulative log-return path}. SAMPLE percentiles of the model's own
    # sampled trajectories — never a confidence interval, and deliberately NOT individual paths.
    quantiles: dict = field(default_factory=dict)
    n_mc_samples: int = 0
    # seed -> {h_anchor: mu}. The four TRAINED MTP horizons; between them is interpolation.
    mtp_anchors: dict = field(default_factory=dict)
    # ★ THE POOLED PREDICTIVE DISTRIBUTION — all n_seeds x n_mc_paths trajectories together.
    # {percentile: [h] cumulative log-return path}, taken POINTWISE at each step.
    #
    # THIS IS NOT THE CROSS-SEED AVERAGE THE PROJECT REFUSES. A mean of the three seed scalars
    # would blend three separate models into one confident number and delete the disagreement.
    # This is the MEDIAN OF THE MODEL'S OWN PREDICTIVE DISTRIBUTION MARGINALISED OVER THE
    # ENSEMBLE: every sampled path from every seed goes into one sample, and the quantiles of
    # THAT are read. It carries which-model and which-path uncertainty in one statistic and it
    # has a name. The per-seed paths are still drawn separately and are never collapsed.
    pooled: dict = field(default_factory=dict)
    n_pooled: int = 0


def forecast_from_csv(
    text: str,
    *,
    units: dict[int, Unit],
    h: int = PAPER_HORIZON,
    device: str = "cpu",
    mc_samples: int = 32,
    n_mc_paths: int = 48,
    progress: Progress = _no_progress,
) -> CsvForecast:
    """The whole path: CSV → production features → tokenizer → AR → per-seed mu-hat + MC band."""
    if h == 1:
        raise CsvRefused(
            "h=1 cannot produce a forecast. mu-hat is defined over [t+1, t+h] with the entry at "
            "C_{t+1}, so the rollout excludes k=1 and there is nothing left to accumulate — the "
            f"answer is EXACTLY 0.000000% for every seed, by construction. Use one of "
            f"{SELECTABLE_HORIZONS}; h={PAPER_HORIZON} is the only horizon the paper evaluates."
        )
    if h not in SELECTABLE_HORIZONS:
        raise CsvRefused(
            f"h={h} is not offered. The MTP heads were trained for {TRAINED_HORIZONS}, of which "
            f"{SELECTABLE_HORIZONS} can produce a forecast (h=1 is structurally zero); "
            f"h={PAPER_HORIZON} is the only horizon the paper evaluates."
        )
    bars = parse_csv(text, progress=progress)

    progress("COMPUTING CAUSAL FEATURES", "", "run")
    out = compute_features(_raw_stream(bars), FeatureConfig())
    x_arm, m_arm = select_arm(np.asarray(out.x, np.float32), out.m, ARM_OHLCV)
    progress(
        "COMPUTING CAUSAL FEATURES",
        f"{x_arm.shape[1]} dims x {x_arm.shape[0]:,} bars, streaming EWMA over bars <= t",
        "ok",
    )

    i = int(x_arm.shape[0] - 1)  # forecast FROM the last bar
    dec = np.array([i], np.int64)
    per_seed = {}
    mu_samples = []
    quantiles: dict = {}
    anchors: dict = {}
    all_paths: list = []
    for seed in sorted(units):
        u = units[seed]
        progress("TOKENIZING", f"seed {seed}", "run")
        b_c, b_f = tokenize_features(
            u.tok, x_arm, m_arm, out.segment_id, window=SEQ_LEN, device=device
        )
        progress(
            "TOKENIZING",
            f"seed {seed}: {b_c.shape[0]:,} bars -> {b_c.shape[0]:,} coarse + "
            f"{b_f.shape[0]:,} fine ids",
            "ok",
        )
        progress("FORECASTING", f"seed {seed}", "run")
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
        progress(
            "FORECASTING",
            f"seed {seed}: expectation {(math.exp(mu) - 1.0) * 100.0:+.4f}%",
            "ok",
        )
        progress("SAMPLING TRAJECTORIES", f"seed {seed}", "run")
        paths = mc_paths(
            u,
            x_arm,
            m_arm,
            out.segment_id,
            out.ts,
            out.sigma,
            i,
            h=h,
            n_samples=n_mc_paths,
            device=device,
        )
        # ★ THE mc_mean VALUE IS NOW DERIVED FROM THESE PATHS, NOT ROLLED OUT A SECOND TIME.
        # It is the SAME rollout: `mc_paths` mirrors `_rollout_mc_mean` step for step and the
        # generator stream is prefix-identical, so the first `mc_samples` endpoints ARE the
        # samples production would have drawn. Taking the prefix (rather than switching the band
        # to 48) is what keeps every displayed number exactly where it was.
        mc = mc_mean_from_paths(paths, float(out.sigma[i]), mc_samples)
        mu_samples.append(mc)
        progress(
            "SAMPLING TRAJECTORIES",
            f"seed {seed}: {paths.shape[0]} paths x {paths.shape[1]} steps; "
            f"mc_mean@{mc_samples} {(math.exp(mc) - 1.0) * 100.0:+.4f}% derived from the first "
            f"{mc_samples} (bit-identical to production, gated)",
            "ok",
        )
        progress("COMPUTING QUANTILE BANDS", f"seed {seed}", "run")
        all_paths.append(paths)
        quantiles[seed] = path_quantiles(paths)
        progress(
            "COMPUTING QUANTILE BANDS",
            f"seed {seed}: " + "/".join(f"p{q}" for q in sorted(quantiles[seed])),
            "ok",
        )
        progress("MTP ANCHORS", f"seed {seed}", "run")
        # ★ ha == h IS THE CALL THAT ALREADY PRODUCED `mu` — same estimator, same h, same context,
        # same everything — so it is reused rather than rolled out again. Not an approximation:
        # identical arguments to a deterministic function. Pinned by
        # `test_the_anchor_at_the_selected_horizon_IS_the_headline_mu`.
        anchors[seed] = {
            int(ha): mu
            if int(ha) == h
            else float(
                predict_mu(
                    u.model,
                    u.tok,
                    b_c,
                    b_f,
                    out.ts,
                    out.sigma,
                    dec,
                    h=int(ha),
                    seq_len=SEQ_LEN,
                    device=device,
                    estimator="expectation",
                )[0]
            )
            for ha in SELECTABLE_HORIZONS
            if int(ha) <= h
        }
        progress(
            "MTP ANCHORS",
            f"seed {seed}: {len(anchors[seed])} of 3 usable trained horizons in range "
            f"({', '.join(str(a) + 'm' for a in sorted(anchors[seed]))})",
            "ok",
        )
        last = float(bars.close[i])
        # ★★ ONE ESTIMATOR ON THE PAGE, AND IT IS THE ONE THE PICTURE IS DRAWN FROM.
        # `mu`/`pct` are the MEDIAN OF THIS SEED'S OWN 48 SAMPLED TRAJECTORIES at the horizon —
        # the same paths that produce the bands, the dashed median line and the pooled headline.
        #
        # THEY USED TO BE THE `expectation` SCALAR, AND THAT PUT TWO ESTIMATORS ON ONE PAGE THAT
        # DISAGREE ON SIGN. Measured on the operator's own BTCUSDT file: expectation was 2-of-3
        # POSITIVE (+0.0113 / -0.0036 / +0.0173) while the trajectories were 2-of-3 NEGATIVE, with
        # seed 4 flipping outright. The pooled headline came from the trajectories and the per-seed
        # labels beside it came from the expectation, so a reader saw a negative headline over two
        # positive seeds and concluded the arithmetic was broken. It was not; the page was quoting
        # two different estimators in one visual register.
        #
        # This is the seed-mismatch defect in a new costume and it takes the same fix: A NUMBER
        # SUMMARISING A PICTURE MUST COME FROM THAT PICTURE. The expectation survives as a NAMED
        # SECONDARY under its own key, exported and labelled as a different estimator, never
        # rendered beside the headline.
        med_mu = float(quantiles[seed][50][h - 1])
        per_seed[seed] = {
            "mu": med_mu,
            "pct": (math.exp(med_mu) - 1.0) * 100.0,
            "price": last * math.exp(med_mu),
            "mu_expectation": mu,
            "pct_expectation": (math.exp(mu) - 1.0) * 100.0,
            "mc_mean_pct": (math.exp(mc) - 1.0) * 100.0,
        }

    # ── POOL EVERY TRAJECTORY FROM EVERY SEED INTO ONE SAMPLE, then read its quantiles
    # POINTWISE at each step. Pointwise matters and is labelled on the figure: the resulting path
    # is a locus of per-step medians, NOT one of the sampled trajectories, and no single run of
    # the model looks like it.
    pooled: dict = {}
    n_pooled = 0
    if all_paths:
        stacked = np.concatenate(all_paths, axis=0)
        n_pooled = int(stacked.shape[0])
        pooled = path_quantiles(stacked)
        progress(
            "POOLING TRAJECTORIES",
            f"{n_pooled} paths ({len(all_paths)} seeds x {all_paths[0].shape[0]}) -> pointwise "
            "p10/p25/p50/p75/p90 of the ensemble's predictive distribution",
            "ok",
        )

    # The band is the SPREAD ACROSS SEEDS AND THEIR MC MEANS — an honest envelope of what these
    # three models disagree about. It is NOT a calibrated predictive interval and is labelled so.
    # ONE ESTIMATOR HERE TOO: the envelope of what the three models disagree about, measured on
    # the same trajectory medians the page displays. It used to mix expectation scalars with
    # mc_mean scalars, which is two estimators in one number.
    allmu = [v["mu"] for v in per_seed.values()]
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
        quantiles=quantiles,
        n_mc_samples=n_mc_paths,
        mtp_anchors=anchors,
        pooled=pooled,
        n_pooled=n_pooled,
    )


DISPLAY_DP = 3  # the precision the sentence and the legend actually render
_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

# Kept equal to ``render.ROUND_TRIP_FEE_PCT`` by ``test_the_round_trip_cost_is_one_number``: two
# copies of a headline economic constant drifting apart is exactly the class of defect this
# project keeps paying for.
ROUND_TRIP_PCT = 0.10


@dataclass(frozen=True)
class Headline:
    """The one number and the three things that may never be separated from it."""

    ok: bool  # False when the pooled sample is degenerate; `note` then says how
    pct: float
    lo_pct: float
    hi_pct: float
    n_paths: int
    value: str  # the number as rendered, or the named degenerate state
    spread: str
    agreement: str
    note: str = ""


def _direction_counts(f: CsvForecast) -> tuple[int, int, int]:
    """(n, n_up, n_down) over the DISPLAYED per-seed values — one implementation, two callers.

    Shared with ``plain_english`` deliberately: two copies of "how many point up" would be two
    chances to disagree with each other on the same page, which is the class of defect the pooled
    seed constant was just fixed for.
    """
    shown = [round(f.per_seed[s]["pct"], DISPLAY_DP) for s in sorted(f.per_seed)]
    return len(shown), sum(1 for x in shown if x > 0), sum(1 for x in shown if x < 0)


def headline(f: CsvForecast) -> Headline:
    """The POOLED MEDIAN at the horizon, with its spread and the agreement state.

    ★ WHY THIS STATISTIC AND NOT A MEAN OF THE THREE SEEDS. A mean of three scalars blends three
    separate models into one confident number and deletes the disagreement that IS this project's
    finding. This is the median of the ensemble's own predictive distribution — every sampled path
    from every seed pooled into one sample — so which-model and which-path uncertainty are carried
    by one named statistic rather than averaged away.

    ★ THE TWO COMPANIONS ARE NOT DECORATION AND ARE RETURNED TOGETHER. This is now the most
    quotable object on the page, so the context-stripping rule binds hardest here: a central
    number without its SPREAD is the failure mode, and the AGREEMENT STATE must sit BESIDE it —
    because a pooled median is perfectly capable of looking decisive while the three models it
    pools point in different directions.

    (A fee-ratio companion shipped here briefly and was REMOVED by operator decision. The fee
    facts live in the page's "what this is not" panel. ``plain_english`` still carries a fee
    clause of its own; that is a different surface and was not part of the removal.)

    ★ IT CARRIES ITS OWN DEGENERACY GUARD. The agreement sentence already has one, but this is a
    DIFFERENT STATISTIC ON DIFFERENT INPUTS — 144 trajectories rather than 3 scalars — and a
    collapsed pooled sample would print a confident zero for free. Guarded before the value is
    read, per the degeneracy rule.
    """
    n, n_up, n_down = _direction_counts(f)
    if n_up == n and n > 0:
        agreement = f"all {_WORD.get(n, str(n))} models point the same way"
    elif n_down == n and n > 0:
        agreement = f"all {_WORD.get(n, str(n))} models point the same way"
    else:
        agreement = "THE THREE MODELS DISAGREE ON DIRECTION"

    if not f.pooled or f.n_pooled == 0:
        return Headline(
            ok=False,
            pct=0.0,
            lo_pct=0.0,
            hi_pct=0.0,
            n_paths=0,
            value="NO POOLED FORECAST",
            spread="",
            agreement=agreement,
            note="no sampled trajectories were produced, so there is no pooled distribution.",
        )

    k = f.h - 1
    end = {q: (math.exp(f.pooled[q][k]) - 1.0) * 100.0 for q in sorted(f.pooled)}
    pct, lo, hi = end[50], end[10], end[90]

    # ── DEGENERACY, BEFORE THE VALUE IS READ ─────────────────────────────────────────────────
    if hi == lo:
        return Headline(
            ok=False,
            pct=pct,
            lo_pct=lo,
            hi_pct=hi,
            n_paths=f.n_pooled,
            value="DEGENERATE",
            spread=f"all {f.n_pooled} pooled paths land on the same value",
            agreement=agreement,
            note=(
                f"every one of the {f.n_pooled} sampled paths ended identically, which is a "
                "collapsed sample rather than a confident forecast."
            ),
        )
    if round(pct, DISPLAY_DP) == 0.0:
        return Headline(
            ok=False,
            pct=pct,
            lo_pct=lo,
            hi_pct=hi,
            n_paths=f.n_pooled,
            value="NO DIRECTIONAL SIGNAL",
            spread=f"p10 to p90 of the same {f.n_pooled} paths: {lo:+.{DISPLAY_DP}f}% to "
            f"{hi:+.{DISPLAY_DP}f}%",
            agreement=agreement,
            note=(
                f"the pooled median rounds to zero at {DISPLAY_DP} decimal places, so there is no "
                "direction to report — this is an absent signal, not a neutral call."
            ),
        )

    return Headline(
        ok=True,
        pct=pct,
        lo_pct=lo,
        hi_pct=hi,
        n_paths=f.n_pooled,
        value=f"{pct:+.{DISPLAY_DP}f}%",
        spread=f"p10 to p90 of the same {f.n_pooled} paths: {lo:+.{DISPLAY_DP}f}% to "
        f"{hi:+.{DISPLAY_DP}f}%",
        agreement=agreement,
    )


def plain_english(f: CsvForecast) -> str:
    """One line, forecast only. NEVER a recommendation, a side, or a size.

    ★ THE AGREEMENT CLAUSE CARRIES A DEGENERACY GUARD, AND IT MUST. The degeneracy rule
    (docs/ENGINEERING.md, from C-1): "A PERFECT AGREEMENT SCORE IS AS SUSPICIOUS AS A FAILING
    ONE. Any
    agreement statistic must carry a degeneracy check on its own inputs BEFORE its value is read."
    A COLLAPSED INPUT SCORES PERFECT AGREEMENT FOR FREE.

    It did exactly that here, in the one artifact a non-technical reader actually looks at. At
    h=1 all three mu-hats are structurally 0.000000% (the rollout excludes k=1), and the first
    version emitted "they agree on direction" — THE MOST CONFIDENT SENTENCE THIS DASHBOARD CAN
    PRODUCE, generated by an arm of the model with nothing to accumulate, about three models we
    have spent the whole project showing DISAGREE.

    Removing h=1 hides that instance; it does not fix the statistic. Three defects were present
    and all three are guarded here, on the DISPLAYED values because those are what a reader sees:

      1. ALL-ZERO COLLAPSE -> was "they agree". Now named: no directional signal.
      2. INVISIBLE DISAGREEMENT -> (+0.0001, -0.0002, +0.00005) renders as three zeros and
         claimed "THEY DISAGREE" from sign differences the reader cannot see. The mirror defect,
         and it was NOT found by the h=1 report.
      3. EXACT ZERO COUNTED AS NEGATIVE -> `p > 0` folded 0.0 into the bearish bucket, so
         (-0.20, -0.10, 0.0) claimed unanimity. A flat forecast is not a bearish one.
    """
    pcts = [f.per_seed[s]["pct"] for s in sorted(f.per_seed)]
    mus = [f.per_seed[s]["mu"] for s in sorted(f.per_seed)]
    shown = [round(p, DISPLAY_DP) for p in pcts]
    joined = ", ".join(f"{p:+.{DISPLAY_DP}f}%" for p in pcts)
    lead = f"Over the next {f.h} minute{'s' if f.h != 1 else ''} the three models forecast {joined}"

    # ── DEGENERACY CHECKS, BEFORE the agreement value is read ────────────────────────────────
    if all(s == 0.0 for s in shown):
        return (
            f"{lead} — ALL THREE ROUND TO ZERO at this precision: NO DIRECTIONAL SIGNAL. "
            "This is not agreement between the models."
        )
    if len(set(mus)) == 1:
        return (
            f"{lead} — all three are IDENTICAL, which is degenerate rather than a consensus. "
            "Treat this as no independent signal."
        )

    n_flat = sum(1 for s in shown if s == 0.0)
    if n_flat:
        return (
            f"{lead} — {n_flat} of {len(shown)} round to zero (no direction), so there is no "
            "unanimous call here."
        )

    # ── THE COLLECTIVE STATE, IN WORDS. ★ THERE IS DELIBERATELY NO MEAN, MEDIAN OR WEIGHTED
    # AVERAGE ACROSS SEEDS HERE, AND THERE MUST NOT BE. A single blended number is the most
    # quotable thing this page could produce, and it would be quoted alone — at which point it
    # reads as one confident small call, when the actual finding is that three versions of the
    # same model disagree about direction. Averaging them DELETES the result. That is the
    # context-stripping rule aimed at the number most likely to travel without its context.
    n = len(shown)
    n_up = sum(1 for s in shown if s > 0)
    n_down = n - n_up
    if n_up == n:
        state = f"ALL {_WORD.get(n, str(n)).upper()} POINT UP"
    elif n_down == n:
        state = f"ALL {_WORD.get(n, str(n)).upper()} POINT DOWN"
    else:
        up, down = (n_up, "up"), (n_down, "down")
        big, small = (up, down) if n_up >= n_down else (down, up)
        few = _WORD.get(small[0], str(small[0])).upper()
        verb = "POINT" if small[0] > 1 else "POINTS"
        state = (
            f"{_WORD.get(big[0], str(big[0])).upper()} OF {_WORD.get(n, str(n)).upper()} "
            f"POINT {big[1].upper()}, {few} {verb} {small[1].upper()}"
        )

    if n_up == n or n_down == n:
        verdict = (
            f"{state}, between {min(pcts):+.{DISPLAY_DP}f}% and {max(pcts):+.{DISPLAY_DP}f}%. "
            "That is the closest this model comes to a directional call."
        )
    else:
        verdict = (
            f"{state}. THE MODELS DISAGREE ON DIRECTION, which is the usual outcome for this "
            "model and means there is no directional call here."
        )

    # ★ THE COMPARISON THAT MAKES IT LAND: how far apart the models are, against how big the move
    # they are forecasting is. For these models the first is routinely larger than the second,
    # and a reader who sees that once understands what they are looking at faster than any
    # amount of prose about seed variance.
    spread = max(pcts) - min(pcts)
    biggest = max(abs(x) for x in pcts)
    ratio = spread / biggest if biggest > 0 else float("inf")
    gap = (
        f"They disagree by {spread:.{DISPLAY_DP}f} percentage points, {ratio:.1f}x the largest of "
        f"the {_WORD.get(n, str(n))}"
        + (
            ": the gap between the models is wider than the move they are forecasting."
            if ratio >= 1.0
            else "."
        )
    )
    fee = (
        f" All {_WORD.get(n, str(n))} are smaller than the ~{ROUND_TRIP_PCT:.2f}% a round trip "
        "costs."
        if biggest < ROUND_TRIP_PCT
        else f" The largest is {biggest:.{DISPLAY_DP}f}%, against the ~{ROUND_TRIP_PCT:.2f}% a "
        "round trip costs."
    )
    return f"{lead} — {verdict} {gap}{fee}"


def forecast_csv_export(f: CsvForecast) -> str:
    """CSV export with THE SAME numbers the chart draws, so the two cannot disagree."""
    buf = io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["field", "value"])
    w.writerow(["decision_ts_ms", f.decision_ts_ms])
    w.writerow(["horizon_minutes", f.h])
    w.writerow(["last_close", f"{f.last_close:.10g}"])
    w.writerow(
        [
            "# PRIMARY ESTIMATOR = the MEDIAN of that seed's own sampled trajectories at h.",
            "This is what the chart, the legend and the headline all show.",
        ]
    )
    for s in sorted(f.per_seed):
        v = f.per_seed[s]
        w.writerow([f"seed{s}_median_pct", f"{v['pct']:.6f}"])
        w.writerow([f"seed{s}_median_price", f"{v['price']:.10g}"])
    if f.pooled:
        pooled_pct = (math.exp(f.pooled[50][f.h - 1]) - 1.0) * 100.0
        w.writerow(["pooled_median_pct", f"{pooled_pct:.6f}"])
        w.writerow(["pooled_n_paths", f.n_pooled])
        for q in (10, 90):
            v = (math.exp(f.pooled[q][f.h - 1]) - 1.0) * 100.0
            w.writerow([f"pooled_p{q}_pct", f"{v:.6f}"])
    # ★ SECONDARY ESTIMATORS, NAMED AS SUCH AND NEVER SHOWN BESIDE THE HEADLINE. On the operator's
    # own file these DISAGREE ON MAJORITY SIGN with the primary, and seed 4 flips outright. That
    # is a real property of the model, not a bug, and it is exported so it can be inspected —
    # but a page that renders two estimators in one visual register reads as broken arithmetic.
    w.writerow([])
    w.writerow(["# SECONDARY ESTIMATORS — different quantities, NOT what the chart draws."])
    for s in sorted(f.per_seed):
        v = f.per_seed[s]
        w.writerow([f"seed{s}_expectation_pct", f"{v['pct_expectation']:.6f}"])
        w.writerow([f"seed{s}_mc_mean_pct", f"{v['mc_mean_pct']:.6f}"])
    w.writerow(["band_lo_pct", f"{f.band['lo_pct']:.6f}"])
    w.writerow(["band_hi_pct", f"{f.band['hi_pct']:.6f}"])
    w.writerow(["inferred_bar_seconds", f.inferred_bar_ms / 1000])
    w.writerow(["rows_used", f.n_rows])
    w.writerow(["NOT_A_RECOMMENDATION", "forecast only; no position, profit or P&L is computed"])
    for i, wn in enumerate(f.warnings):
        w.writerow([f"warning_{i + 1}", wn])

    # ── PER-STEP QUANTILES: the same numbers the chart draws, so the two cannot disagree ──────
    if f.quantiles:
        w.writerow([])
        w.writerow(["# per-step SAMPLE percentiles of the model's own sampled trajectories."])
        w.writerow(["# NOT confidence intervals. mu-hat is measured 25-446x OVER-DISPERSED"])
        w.writerow(["# relative to the returns it forecasts, so this fan is far WIDER than"])
        w.writerow(["# reality. Individual trajectories are deliberately NOT exported: the"])
        w.writerow(["# return channel is the worst-reconstructed of the 7 OHLCV dims, so a"])
        w.writerow(["# single path's per-step shape is substantially codebook granularity."])
        qs = sorted(next(iter(f.quantiles.values())))
        w.writerow(["seed", "step_minutes"] + [f"p{q}_pct" for q in qs])
        for seed in sorted(f.quantiles):
            qd = f.quantiles[seed]
            for k in range(f.h):
                row = [seed, k + 1]
                row += [f"{(math.exp(qd[q][k]) - 1.0) * 100.0:.6f}" for q in qs]
                w.writerow(row)
    if f.mtp_anchors:
        w.writerow([])
        w.writerow(["# MTP anchors — horizons the heads were actually TRAINED for."])
        w.writerow(["# Values between anchors on the chart are INTERPOLATION."])
        w.writerow(["seed", "trained_horizon_minutes", "forecast_pct"])
        for seed in sorted(f.mtp_anchors):
            for ha in sorted(f.mtp_anchors[seed]):
                mu = f.mtp_anchors[seed][ha]
                w.writerow([seed, ha, f"{(math.exp(mu) - 1.0) * 100.0:.6f}"])
    return buf.getvalue()


__all__ = [
    "DISPLAY_DP",
    "MC_SEED",
    "MIN_ROWS",
    "PAPER_HORIZON",
    "ROUND_TRIP_PCT",
    "SELECTABLE_HORIZONS",
    "SEQ_LEN",
    "TRAINED_HORIZONS",
    "WARMUP_ROWS",
    "CsvBars",
    "CsvForecast",
    "CsvRefused",
    "Headline",
    "Progress",
    "forecast_csv_export",
    "forecast_from_csv",
    "headline",
    "mc_mean_from_paths",
    "parse_csv",
    "plain_english",
]


# ─────────────────────────────────────────────────────────────────────────────────────────────
# PROBABILISTIC PATHS — the MC rollout, keeping what production throws away
# ─────────────────────────────────────────────────────────────────────────────────────────────
# ``predict._rollout_mc_mean`` samples full autoregressive chains and then AVERAGES them to one
# scalar. The per-step cumulative values it discards are genuine sampled trajectories. This keeps
# them. It is a re-implementation of that loop and therefore a drift risk, so it is gated: the
# MEAN of these paths' endpoints must equal ``predict_mu(estimator="mc_mean")`` BIT-FOR-BIT at the
# same seed and sample count (``test_mc_paths_endpoint_matches_production``). If the two ever
# diverge, this function is showing a different rollout from the scored one.
#
# ★ READ THE BAND, NOT THE WIGGLE. Each per-step value is ``decode_tokens(...)[:, 0, 0]`` — a
# SAMPLED TOKEN PUT THROUGH THE TOKENIZER DECODER. The return channel is the WORST-reconstructed
# of the seven OHLCV dims (rank 7/7, 6/7, 6/7 by seed) and the codebook cannot resolve it finer
# than ~0.55 z against a per-step decode sd of 1.11 z. An individual strand's texture is therefore
# substantially codebook granularity rendered as market structure — which is why this module
# exposes QUANTILES and a MEDIAN and deliberately does NOT expose individual trajectories for
# plotting. Quantization affects every sample alike, so the band's WIDTH and the MEDIAN survive it;
# a single strand's shape does not.


def mc_paths(
    unit: Unit,
    x_arm: np.ndarray,
    m_arm: np.ndarray,
    segment_id: np.ndarray,
    ts: np.ndarray,
    sigma: np.ndarray,
    decision: int,
    *,
    h: int,
    n_samples: int = 64,
    seed: int = MC_SEED,
    device: str = "cpu",
) -> np.ndarray:
    """``[n_samples, h]`` cumulative raw log-return paths from sampled AR chains.

    Mirrors ``predict._rollout_mc_mean`` step for step, including the k=1 exclusion (mu-hat covers
    [t+1, t+h], so the entry bar contributes nothing) and the final ``* sigma_t`` vol rescale.
    """
    import torch
    import torch.nn.functional as F

    from trikaal.data.config import BAR_MS as _BAR_MS
    from trikaal.eval.predict import _fill_context

    model, tok = unit.model, unit.tok
    dev = torch.device(device)
    model.eval()
    tok.eval()

    b_c, b_f = tokenize_features(tok, x_arm, m_arm, segment_id, window=SEQ_LEN, device=device)
    d = np.array([decision], np.int64)
    offs = np.arange(-SEQ_LEN + 1, 1)
    idx = d[:, None] + offs[None, :]
    cc = torch.from_numpy(b_c[idx].astype(np.int64)).to(dev)
    cf = torch.from_numpy(b_f[idx].astype(np.int64)).to(dev)
    cts = torch.from_numpy(ts[idx].astype(np.int64)).to(dev)
    ts_dec = cts[:, -1:]
    sig_t = float(sigma[decision])

    gen = torch.Generator(device="cpu").manual_seed(seed)
    out_paths = np.zeros((n_samples, h), dtype=np.float64)
    with torch.no_grad():
        # ★ THE CONTEXT PREFILL IS IDENTICAL FOR EVERY SAMPLE, SO IT IS COMPUTED ONCE.
        # ``_fill_context`` issues SEQ_LEN single-token ``model.step`` calls over context that is
        # fixed across the whole loop — 512 forward passes per sample, repeated n_samples times,
        # to rebuild byte-for-byte the same KV cache. It dominated the dashboard's wall clock.
        #
        # A SHALLOW DICT COPY IS SUFFICIENT AND IS THE WHOLE TRICK. ``AttentionBlock.step`` grows
        # the cache with ``cache["k"] = torch.cat([cache["k"], k], dim=2)`` — that REBINDS the dict
        # entry to a NEW tensor and never writes the old one in place (attention.py:124-127). So
        # each sample needs its own dict, not its own tensors, and the base tensors survive
        # untouched. ``base_out`` is read-only until the first ``model.step`` reassigns it.
        #
        # ★ MEMOIZING A DETERMINISTIC PURE COMPUTATION MUST BE BIT-IDENTICAL, AND THAT IS THE
        # ACCEPTANCE CRITERION, NOT A HOPE:
        # ``test_mc_paths_endpoint_matches_production_bit_for_bit``
        # compares this against the UNCHANGED production ``predict_mu(estimator="mc_mean")``. If a
        # single bit moves, the optimisation changed the model and it reverts — no tolerance is
        # added and no argument is made about whether the difference is acceptable.
        base_caches, base_out = _fill_context(model, cc, cf, cts, SEQ_LEN)
        for s in range(n_samples):
            caches = [{"k": c["k"], "v": c["v"]} for c in base_caches]
            out = base_out
            cum = torch.zeros(1, dtype=torch.float32, device=dev)
            for k in range(1, h + 1):
                pc_p = F.softmax(out.logits_c.float(), dim=-1).reshape(1, -1).cpu()
                pc = torch.multinomial(pc_p, 1, generator=gen).reshape(1, 1).to(dev)
                logits_f = model.fine_logits(out.h_final, pc)
                pf_p = F.softmax(logits_f.float(), dim=-1).reshape(1, -1).cpu()
                pf = torch.multinomial(pf_p, 1, generator=gen).reshape(1, 1).to(dev)
                if k >= 2:
                    cum = cum + tok.decode_tokens(pc, pf)[:, 0, 0]
                out_paths[s, k - 1] = float(cum.item()) * sig_t
                if k < h:
                    out = model.step(pc, pf, ts_dec + k * _BAR_MS, caches, SEQ_LEN - 1 + k)
    return out_paths


def mc_mean_from_paths(paths: np.ndarray, sig_t: float, n: int) -> float:
    """``predict_mu(estimator="mc_mean", mc_samples=n)``, DERIVED from an existing rollout.

    ★ WHY THIS EXISTS. The dashboard used to roll the same Monte-Carlo chain TWICE per seed: once
    inside production's ``_rollout_mc_mean`` for a single scalar, and again in ``mc_paths`` to keep
    the per-step values. The second rollout is a superset of the first, so the scalar is a pure
    function of the paths — measured at 63.7 s per seed of duplicated work at n=48.

    ★ THE PREFIX IS THE POINT. ``paths`` may contain MORE samples than ``n``. Both rollouts drive
    one ``torch.Generator`` seeded identically and consume exactly ``2*h`` draws per sample, so
    sample *i* is the same sample in a 32-run and in a 48-run — the streams are prefix-identical.
    Taking ``paths[:n]`` therefore reproduces production at ITS sample count, which is what lets
    the bands widen to 48 samples while every displayed number stays exactly where it was.
    That prefix property is an assumption this function rests on, so it is gated in its own right
    (``test_a_longer_rollout_contains_the_shorter_one_as_a_prefix``).

    ★ PRODUCTION'S REDUCTION ORDER IS REPRODUCED EXACTLY, NOT APPROXIMATED. ``_rollout_mc_mean``
    accumulates the UNSCALED float32 ``cum`` in float64, divides by n, and applies sigma ONCE at
    the very end; ``mc_paths`` scales EVERY step by sigma so callers get raw log-returns.
    Mathematically identical, numerically ONE ULP apart (measured 1.94e-16). Undoing the scale in
    float32 and re-applying it last makes it exact — and "exact" is the acceptance criterion here,
    not "close", because a tolerance would hide a genuinely different rollout.
    """
    if paths.shape[0] < n:
        raise ValueError(
            f"need at least {n} sampled paths to derive mc_mean@{n}, got {paths.shape[0]}"
        )
    acc = 0.0
    for v in paths[:n, -1]:
        acc = acc + float(np.float32(v / sig_t))
    return (acc / n) * sig_t


def path_quantiles(paths: np.ndarray, qs=(10, 25, 50, 75, 90)) -> dict:
    """SAMPLE PERCENTILES of the sampled paths — never a confidence interval.

    These describe the spread of the MODEL'S OWN samples. Our measured mu-hat over-dispersion is
    25-446x relative to the returns being forecast, so this fan is far WIDER than reality. It is
    the model's uncertainty, and that uncertainty is badly mis-scaled.
    """
    return {int(q): np.percentile(paths, q, axis=0) for q in qs}
