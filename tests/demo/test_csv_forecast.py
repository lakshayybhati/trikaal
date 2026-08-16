"""CSV dashboard — the two premises it stands on, and the boundary it must not corrupt.

Nothing here mocks the thing under test. The forecast path is exercised end to end with the real
tokenizer and predictor when they are present, and skipped (not faked) when they are not.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest

from trikaal.data.config import FeatureConfig
from trikaal.data.features import compute_features
from trikaal.data.synthetic import RawStream
from trikaal.demo import csv_forecast as cf
from trikaal.train.arms import ARM_OHLCV, arm_feature_idx


def _bars(t: int, *, bar_ms: int = 60_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    ms = 1_700_000_000_000 + np.arange(t) * bar_ms
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, t)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + abs(rng.normal(0, 3e-4, t)))
    lo = np.minimum(o, c) * (1 - abs(rng.normal(0, 3e-4, t)))
    v = np.abs(rng.lognormal(3, 1, t))
    return ms, o, hi, lo, c, v, v * c


def _stream(ms, o, hi, lo, c, v, amt, *, micro: bool, seed: int = 1) -> RawStream:
    t = len(ms)
    if micro:
        rng = np.random.default_rng(seed)
        vb = v * rng.uniform(0.3, 0.7, t)
        vs, nb, ns = v - vb, rng.integers(1, 50, t), rng.integers(1, 50, t)
        sizes = [rng.lognormal(0, 1, 5) for _ in range(t)]
    else:
        vb = np.zeros(t)
        vs, nb, ns = np.zeros(t), np.zeros(t, np.int64), np.zeros(t, np.int64)
        sizes = [np.array([]) for _ in range(t)]
    return RawStream(
        bar_open_ms=ms,
        open=o,
        high=hi,
        low=lo,
        close=c,
        v_kline=v,
        quote_volume=amt,
        v_buy=vb,
        v_sell=vs,
        n_buy=nb,
        n_sell=ns,
        sizes=sizes,
        is_perp=False,
        funding_ts=np.array([], np.int64),
        funding_val=np.array([]),
        oi_ts=np.array([], np.int64),
        oi_val=np.array([]),
    )


def _csv_text(ms, o, hi, lo, c, v, amt) -> str:
    buf = io.StringIO()
    buf.write("timestamp,open,high,low,close,volume,amount\n")
    for i in range(len(ms)):
        # ★ repr(), NOT %.10g. float64 needs 17 significant digits to round-trip; %.10g loses
        # ~1e-8 relative and the acceptance gate CAUGHT IT — the two routes disagreed at the 9th
        # hex digit. That was a defect in this serializer, not in the product, and isolating it is
        # the point: the gate must test the CSV BOUNDARY LOGIC, not decimal truncation.
        buf.write(
            ",".join(
                [str(ms[i])] + [repr(float(x)) for x in (o[i], hi[i], lo[i], c[i], v[i], amt[i])]
            )
            + "\n"
        )
    return buf.getvalue()


# ── PREMISE 1: a plain OHLCV CSV is sufficient ────────────────────────────────────────────────
def test_ohlcv_dims_are_INVARIANT_to_the_micro_inputs_a_csv_cannot_supply():
    """THE WHOLE BUILD RESTS ON THIS. If it ever stops holding, the dashboard is unsound.

    compute_features builds all 16 dims and 7 of them need aggTrades a CSV cannot provide. Zeroing
    those must not disturb the 7 dims cell 1 actually reads.
    """
    b = _bars(3000)
    cfg = FeatureConfig()
    with_micro = compute_features(_stream(*b, micro=True), cfg)
    without = compute_features(_stream(*b, micro=False), cfg)
    idx = list(arm_feature_idx(ARM_OHLCV))
    assert idx == [0, 1, 2, 3, 4, 5, 6]
    assert np.array_equal(np.nan_to_num(with_micro.x[:, idx]), np.nan_to_num(without.x[:, idx])), (
        "OHLCV dims changed when micro was zeroed — the CSV path would NOT be the scored path"
    )
    assert np.array_equal(with_micro.m[:, idx], without.m[:, idx]), "masks diverged"
    assert np.array_equal(with_micro.sigma, without.sigma), "sigma diverged"


def test_the_invariance_check_can_fail():
    """FIXTURE DISCRIMINATION — perturbing a dim the CSV DOES supply must break the comparison."""
    ms, o, hi, lo, c, v, amt = _bars(1500)
    cfg = FeatureConfig()
    a = compute_features(_stream(ms, o, hi, lo, c, v, amt, micro=False), cfg)
    b = compute_features(_stream(ms, o, hi, lo, c * 1.01, v, amt, micro=False), cfg)
    idx = list(arm_feature_idx(ARM_OHLCV))
    assert not np.array_equal(np.nan_to_num(a.x[:, idx]), np.nan_to_num(b.x[:, idx]))


# ── PREMISE 2: the warm-up floor is real ──────────────────────────────────────────────────────
def test_warmup_floor_is_measured_not_asserted():
    """The 1,232-row floor must match what compute_features ACTUALLY produces."""
    b = _bars(2600)
    out = compute_features(_stream(*b, micro=False), FeatureConfig())
    idx = list(arm_feature_idx(ARM_OHLCV))
    clean = out.m[:, idx].sum(1) == 0
    first_clean = int(np.argmax(clean)) if clean.any() else -1
    assert first_clean == cf.WARMUP_ROWS, (
        f"first fully-unmasked row is {first_clean}, but WARMUP_ROWS says {cf.WARMUP_ROWS}"
    )
    assert cf.MIN_ROWS == cf.WARMUP_ROWS + cf.SEQ_LEN == 1232
    # and at the supervisor's originally-proposed 512 floor, NOTHING is clean
    assert not clean[:512].any(), "512 rows would be entirely warm-up — the floor must exceed it"


# ── REFUSALS: reasons, never tracebacks ───────────────────────────────────────────────────────
def test_refuses_short_csv_with_the_arithmetic_in_the_message():
    ms, o, hi, lo, c, v, amt = _bars(600)
    with pytest.raises(cf.CsvRefused) as ei:
        cf.parse_csv(_csv_text(ms, o, hi, lo, c, v, amt))
    m = str(ei.value)
    assert "1,232" in m and "720" in m and "512" in m, "the refusal must show its arithmetic"


def test_refuses_missing_columns_and_names_them():
    with pytest.raises(cf.CsvRefused, match="close"):
        cf.parse_csv("timestamp,open,high,low\n1,2,3,4\n")


def test_refuses_untrained_horizon():
    with pytest.raises(cf.CsvRefused, match="never trained"):
        cf.forecast_from_csv("", units={}, h=7)


def test_flags_non_60s_bars_as_out_of_distribution():
    b = _bars(1400, bar_ms=300_000)  # 5-minute bars
    parsed = cf.parse_csv(_csv_text(*b))
    assert parsed.inferred_bar_ms == 300_000
    assert any("OUT OF DISTRIBUTION" in w for w in parsed.warnings)


def test_accepts_60s_bars_without_the_ood_warning():
    parsed = cf.parse_csv(_csv_text(*_bars(1400)))
    assert parsed.inferred_bar_ms == 60_000
    assert not any("OUT OF DISTRIBUTION" in w for w in parsed.warnings)


# ── the plain-English line must never recommend ───────────────────────────────────────────────
def test_plain_english_reports_and_never_recommends():
    f = cf.CsvForecast(
        h=15,
        decision_ts_ms=0,
        last_close=100.0,
        per_seed={
            0: {"mu": 0.0003, "pct": 0.03, "price": 100.03},
            2: {"mu": -0.0001, "pct": -0.01, "price": 99.99},
            4: {"mu": 0.0011, "pct": 0.11, "price": 100.11},
        },
        band={"lo_pct": -0.01, "hi_pct": 0.11, "lo_price": 99.99, "hi_price": 100.11},
        context_ts=np.array([0]),
        context_close=np.array([100.0]),
        inferred_bar_ms=60_000,
        warnings=[],
        n_rows=1300,
    )
    line = cf.plain_english(f)
    assert "DISAGREE" in line and "+0.030%" in line
    for banned in ("buy", "sell", "should", "recommend", "position", "profit", "target price"):
        assert banned not in line.lower(), f"the read must never contain {banned!r}"


def test_export_columns_match_the_chart_numbers():
    f = cf.CsvForecast(
        h=15,
        decision_ts_ms=123,
        last_close=100.0,
        per_seed={0: {"mu": 0.001, "pct": 0.1, "price": 100.1}},
        band={"lo_pct": 0.0, "hi_pct": 0.2, "lo_price": 100.0, "hi_price": 100.2},
        context_ts=np.array([0]),
        context_close=np.array([100.0]),
        inferred_bar_ms=60_000,
        warnings=["w"],
        n_rows=1300,
    )
    text = cf.forecast_csv_export(f)
    assert "seed0_forecast_pct,0.100000" in text
    assert "NOT_A_RECOMMENDATION" in text
    assert "warning_1,w" in text


# ── ACCEPTANCE: the CSV boundary must not corrupt anything ────────────────────────────────────
@pytest.mark.slow
def test_csv_path_reproduces_the_direct_feature_path_bit_for_bit():
    """★ THE ACCEPTANCE GATE. Same bars, two routes, one answer.

    NOT circular: the dashboard route goes through CSV serialization, text parsing, timestamp
    inference, column aliasing, sort/dedup and RawStream reconstruction — all NEW code. The
    reference route hands the identical bars straight to compute_features. If any of that new
    surface corrupts a value, mu-hat differs. Bit-for-bit on float64, no tolerance.
    """
    from trikaal.demo.inference import available_seeds, load_unit

    seeds = available_seeds()
    if not seeds:
        pytest.skip("banked units not present")
    units = {seeds[0]: load_unit(seeds[0])}

    b = _bars(1400)
    got = cf.forecast_from_csv(_csv_text(*b), units=units, h=15, mc_samples=2)

    # reference: identical bars, straight into the production feature path
    from trikaal.eval.predict import predict_mu
    from trikaal.train.arms import select_arm
    from trikaal.train.token_stream import tokenize_features

    out = compute_features(_stream(*b, micro=False), FeatureConfig())
    x_arm, m_arm = select_arm(np.asarray(out.x, np.float32), out.m, ARM_OHLCV)
    u = units[seeds[0]]
    b_c, b_f = tokenize_features(u.tok, x_arm, m_arm, out.segment_id, window=cf.SEQ_LEN)
    ref = float(
        predict_mu(
            u.model,
            u.tok,
            b_c,
            b_f,
            out.ts,
            out.sigma,
            np.array([x_arm.shape[0] - 1], np.int64),
            h=15,
            seq_len=cf.SEQ_LEN,
            estimator="expectation",
        )[0]
    )

    assert got.per_seed[seeds[0]]["mu"].hex() == ref.hex(), (
        "the CSV boundary changed the forecast — the dashboard is not showing the scored model"
    )
    assert math.isfinite(got.per_seed[seeds[0]]["pct"])


def test_sigma_is_converged_at_the_floor_and_the_fixture_can_discriminate():
    """★ THE SIGMA DECISION, PINNED. The 1,232 floor is safe for the MAGNITUDE, not just the sign.

    predict_mu ends with ``cum * sig_t``, so sigma scales the reported percentage linearly. The
    floor is only defensible if sigma is converged there — asserted, not argued.

    ★ FIXTURE DISCRIMINATION IS THE WHOLE TEST. A CONSTANT-volatility series makes an EWMA
    converge to the same value at every truncation, so it reports 0.0% error for free — my first
    measurement did exactly that and was worthless. This uses a 10x regime shift and FIRST proves
    the series actually moves sigma, so a genuine warm-up error could not hide.
    """
    rng = np.random.default_rng(11)
    tt = 6000
    ms = 1_700_000_000_000 + np.arange(tt) * 60_000
    vol = np.where(np.arange(tt) > tt - 800, 0.01, 0.001)  # calm -> 10x storm
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1, tt) * vol))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * 1.0003
    lo = np.minimum(o, c) * 0.9997
    v = np.abs(rng.lognormal(3, 1, tt))
    cfg = FeatureConfig()

    def sigma_at_end(n: int) -> float:
        sl = slice(tt - n, tt)
        st = _stream(ms[sl], o[sl], hi[sl], lo[sl], c[sl], v[sl], (v * c)[sl], micro=False)
        return float(compute_features(st, cfg).sigma[-1])

    full = compute_features(_stream(ms, o, hi, lo, c, v, v * c, micro=False), cfg).sigma
    # the fixture MUST move sigma, or the convergence check below is vacuous
    assert full.max() / full.min() > 3.0, "fixture does not exercise sigma — cannot discriminate"

    ref = float(full[-1])
    at_floor = sigma_at_end(cf.MIN_ROWS)
    assert abs(at_floor / ref - 1.0) < 0.01, (
        f"sigma at the {cf.MIN_ROWS}-row floor is {at_floor / ref:.4f}x the fully-warm value; "
        "the floor would distort the reported magnitude"
    )
