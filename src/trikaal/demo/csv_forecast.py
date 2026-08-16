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


def forecast_from_csv(
    text: str,
    *,
    units: dict[int, Unit],
    h: int = PAPER_HORIZON,
    device: str = "cpu",
    mc_samples: int = 32,
    n_mc_paths: int = 48,
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
    bars = parse_csv(text)
    out = compute_features(_raw_stream(bars), FeatureConfig())
    x_arm, m_arm = select_arm(np.asarray(out.x, np.float32), out.m, ARM_OHLCV)

    i = int(x_arm.shape[0] - 1)  # forecast FROM the last bar
    dec = np.array([i], np.int64)
    per_seed = {}
    mu_samples = []
    quantiles: dict = {}
    anchors: dict = {}
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
        quantiles[seed] = path_quantiles(paths)
        anchors[seed] = {
            int(ha): float(
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
        quantiles=quantiles,
        n_mc_samples=n_mc_paths,
        mtp_anchors=anchors,
    )


DISPLAY_DP = 3  # the precision the sentence and the legend actually render


def plain_english(f: CsvForecast) -> str:
    """One line, forecast only. NEVER a recommendation, a side, or a size.

    ★ THE AGREEMENT CLAUSE CARRIES A DEGENERACY GUARD, AND IT MUST. The degeneracy rule
    (CLAUDE.md, from C-1): "A PERFECT AGREEMENT SCORE IS AS SUSPICIOUS AS A FAILING ONE. Any
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

    pos = [s > 0 for s in shown if s != 0.0]
    n_flat = sum(1 for s in shown if s == 0.0)
    if n_flat:
        return (
            f"{lead} — {n_flat} of 3 round to zero (no direction), so there is no unanimous "
            "call here."
        )
    verdict = "they agree on direction" if len(set(pos)) == 1 else "THEY DISAGREE ON DIRECTION"
    return f"{lead} — {verdict}."


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
    "MIN_ROWS",
    "PAPER_HORIZON",
    "SELECTABLE_HORIZONS",
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
    seed: int = 20260704,
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
        for s in range(n_samples):
            _caches, out = _fill_context(model, cc, cf, cts, SEQ_LEN)
            caches = _caches
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


def path_quantiles(paths: np.ndarray, qs=(10, 25, 50, 75, 90)) -> dict:
    """SAMPLE PERCENTILES of the sampled paths — never a confidence interval.

    These describe the spread of the MODEL'S OWN samples. Our measured mu-hat over-dispersion is
    25-446x relative to the returns being forecast, so this fan is far WIDER than reality. It is
    the model's uncertainty, and that uncertainty is badly mis-scaled.
    """
    return {int(q): np.percentile(paths, q, axis=0) for q in qs}
