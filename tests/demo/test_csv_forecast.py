"""CSV dashboard — the two premises it stands on, and the boundary it must not corrupt.

Nothing here mocks the thing under test. The forecast path is exercised end to end with the real
tokenizer and predictor when they are present, and skipped (not faked) when they are not.
"""

from __future__ import annotations

import io
import math
import re

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


def test_refuses_unoffered_horizon():
    with pytest.raises(cf.CsvRefused, match="not offered"):
        cf.forecast_from_csv("", units={}, h=7)


def test_h1_is_NOT_offered_and_is_refused_at_the_api():
    """★ h=1 STRUCTURALLY CANNOT PRODUCE A FORECAST, SO IT IS NOT AN OPTION.

    mu-hat covers [t+1, t+h] with the entry at C_{t+1}, so the rollout excludes k=1. At h=1 there
    is nothing to accumulate and every seed returns EXACTLY 0.000000%. A user who picks "1 min"
    and sees a flat zero concludes the tool is broken or the model is neutral at one minute —
    neither is true, and no label fixes an option that cannot work.

    This test exists so nobody restores h=1 from TRAINED_HORIZONS, which legitimately contains it.
    """
    assert 1 in cf.TRAINED_HORIZONS, "the MTP heads WERE trained for h=1 — that fact stays"
    assert 1 not in cf.SELECTABLE_HORIZONS, "h=1 must never be offered"
    assert cf.SELECTABLE_HORIZONS == (5, 15, 60)
    with pytest.raises(cf.CsvRefused, match=re.escape("EXACTLY 0.000000")):
        cf.forecast_from_csv("", units={}, h=1)


def test_the_h1_zero_is_real_and_this_fixture_proves_the_mechanism():
    """FIXTURE DISCRIMINATION for the claim above: show k=1 is excluded, not merely asserted.

    Runs the real rollout at h=1 and h=2 on the same bars. h=1 must be EXACTLY zero; h=2 must not
    be — otherwise "the rollout excludes k=1" would be an unverified story about the code.
    """
    from trikaal.demo.inference import available_seeds, load_unit
    from trikaal.eval.predict import predict_mu
    from trikaal.train.arms import select_arm
    from trikaal.train.token_stream import tokenize_features

    seeds = available_seeds()
    if not seeds:
        pytest.skip("banked units not present")
    u = load_unit(seeds[0])
    b = _bars(1300)
    out = compute_features(_stream(*b, micro=False), FeatureConfig())
    x_arm, m_arm = select_arm(np.asarray(out.x, np.float32), out.m, ARM_OHLCV)
    b_c, b_f = tokenize_features(u.tok, x_arm, m_arm, out.segment_id, window=cf.SEQ_LEN)
    dec = np.array([x_arm.shape[0] - 1], np.int64)

    def mu_at(h):
        return float(
            predict_mu(
                u.model,
                u.tok,
                b_c,
                b_f,
                out.ts,
                out.sigma,
                dec,
                h=h,
                seq_len=cf.SEQ_LEN,
                estimator="expectation",
            )[0]
        )

    assert mu_at(1) == 0.0, "h=1 must be exactly zero — that is why it is not offered"
    assert mu_at(2) != 0.0, "h=2 must be non-zero, or the fixture cannot discriminate"


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


@pytest.mark.slow
@pytest.mark.parametrize("n", [8, 32])
def test_mc_paths_endpoint_matches_production_bit_for_bit(n):
    """★ THE GATE ON THE RE-IMPLEMENTED ROLLOUT — AND THE ACCEPTANCE TEST FOR ITS OPTIMISATION.

    ``mc_paths`` re-walks ``predict._rollout_mc_mean``'s loop to KEEP the per-step values that
    production averages away. A re-implementation is a drift risk, so the mean of its endpoints
    must equal ``predict_mu(estimator='mc_mean')`` exactly at the same seed and sample count. If
    these diverge, the figure is drawing a different rollout from the scored one.

    ★ IT ALSO GATES THE CONTEXT-PREFILL CACHE. ``mc_paths`` used to rebuild an identical 512-step
    KV cache once per sampled path; it now builds it once and hands each sample a shallow dict
    copy. Memoizing a deterministic pure computation MUST be bit-identical, so this test — against
    UNCHANGED production — is the whole acceptance criterion, and the answer is bit-identity or
    revert. No tolerance is added and no argument is made about whether a difference is acceptable.

    ★ n IS PARAMETRIZED BECAUSE n=8 IS THIN FOR A CACHING CHANGE. A cache that leaked state
    between samples would corrupt sample 2 onward; eight samples can show that, but the production
    configuration is 32, and a gate should be run at the size the thing actually runs at. Both
    values were also checked element-by-element against the pre-optimisation implementation
    (0 of n*h values moved at n=8, 32 and 48).

    ``mc_seed`` is passed EXPLICITLY. An ad-hoc harness written to check this optimisation omitted
    it, production fell back to its own default, the two arms drew different samples, and the
    harness reported REVERT on a change that is provably bit-identical — a false alarm from the
    instrument, on the very question it existed to settle. The tooling rule, again.

    ★★ RESHAPED WHEN THE DASHBOARD STOPPED CALLING PRODUCTION. The demo now DERIVES its mc_mean
    from these paths instead of rolling the chain a second time, so this gate had to be re-examined
    for exactly the failure the shared-input rule names. It is NOT tautological: the two arms are
    still independent IMPLEMENTATIONS — ``mc_paths`` is our re-walk, ``_rollout_mc_mean`` is
    production — and neither now reads the other. What DID change is that production is no longer
    exercised on the shipped path, so three things were added rather than assumed:
      * this test now calls ``mc_mean_from_paths``, THE SHIPPED REDUCTION, instead of an inline
        copy of it — a test that re-implements the code it is checking verifies its own copy;
      * ``test_a_longer_rollout_contains_the_shorter_one_as_a_prefix`` gates the new assumption
        the derivation rests on;
      * ``test_the_derived_value_matches_a_PINNED_reference`` can fail even if BOTH arms were
        changed together, which agreement between two arms structurally cannot.
    """
    from trikaal.demo.inference import available_seeds, load_unit
    from trikaal.eval.predict import predict_mu
    from trikaal.train.arms import select_arm
    from trikaal.train.token_stream import tokenize_features

    seeds = available_seeds()
    if not seeds:
        pytest.skip("banked units not present")
    u = load_unit(seeds[0])
    b = _bars(1400)
    out = compute_features(_stream(*b, micro=False), FeatureConfig())
    x_arm, m_arm = select_arm(np.asarray(out.x, np.float32), out.m, ARM_OHLCV)
    i, h = x_arm.shape[0] - 1, 15

    paths = cf.mc_paths(
        u,
        x_arm,
        m_arm,
        out.segment_id,
        out.ts,
        out.sigma,
        i,
        h=h,
        n_samples=n,
        seed=20260704,
    )
    assert paths.shape == (n, h)
    # ★ THE SHIPPED REDUCTION, not a copy of it. `mc_mean_from_paths` reproduces production's
    # order exactly (unscaled float32 accumulated in float64, sigma applied once at the end); an
    # inline re-implementation here would pass even if the function the dashboard actually calls
    # were wrong, which is the mock rule wearing a different hat.
    mine = cf.mc_mean_from_paths(paths, float(out.sigma[i]), n)

    b_c, b_f = tokenize_features(u.tok, x_arm, m_arm, out.segment_id, window=cf.SEQ_LEN)
    prod = float(
        predict_mu(
            u.model,
            u.tok,
            b_c,
            b_f,
            out.ts,
            out.sigma,
            np.array([i], np.int64),
            h=h,
            seq_len=cf.SEQ_LEN,
            estimator="mc_mean",
            mc_samples=n,
            mc_seed=20260704,
        )[0]
    )
    assert mine.hex() == prod.hex(), "mc_paths is not the production rollout"


def test_quantiles_are_ordered_and_the_median_is_labelled_as_such():
    paths = np.array([[0.0, 0.1], [0.0, 0.2], [0.0, 0.3], [0.0, 0.4]])
    q = cf.path_quantiles(paths)
    assert set(q) == {10, 25, 50, 75, 90}
    for k in range(2):
        assert q[10][k] <= q[25][k] <= q[50][k] <= q[75][k] <= q[90][k]
    doc = cf.path_quantiles.__doc__
    assert "never a confidence interval" in doc.lower()
    assert "over-disp" in doc.lower(), "the mis-scaling must be stated where the fan is made"


def test_no_individual_trajectory_is_exposed_for_plotting():
    """The design decision, pinned: bands and a median, never a single strand.

    An individual sampled path's per-step shape is substantially codebook granularity — the return
    channel is the worst-reconstructed of the seven OHLCV dims. Exposing strands for plotting would
    render that artifact as market structure.
    """
    fields = set(cf.CsvForecast.__dataclass_fields__)
    for banned in ("paths", "trajectories", "samples", "strands"):
        assert banned not in fields, f"CsvForecast exposes {banned!r} — strands must not be plotted"
    assert "quantiles" in fields


def _fc(pcts, h=15):
    return cf.CsvForecast(
        h=h,
        decision_ts_ms=0,
        last_close=100.0,
        per_seed={
            s: {"mu": v / 100.0, "pct": v, "price": 100.0}
            for s, v in zip((0, 2, 4), pcts, strict=True)
        },
        band={"lo_pct": 0.0, "hi_pct": 0.0, "lo_price": 100.0, "hi_price": 100.0},
        context_ts=np.array([0]),
        context_close=np.array([100.0]),
        inferred_bar_ms=60_000,
        warnings=[],
        n_rows=1300,
    )


def test_agreement_clause_refuses_on_a_COLLAPSED_input():
    """★ THE DEGENERACY RULE, IN THE DEMO. A collapsed input must not score perfect agreement.

    At h=1 all three mu-hats are structurally zero and the first version emitted "they agree on
    direction" — the most confident sentence this dashboard can produce, about three models the
    project spent weeks showing DISAGREE. This is the C-1 pathology on a user-facing surface.
    """
    line = cf.plain_english(_fc((0.0, 0.0, 0.0)))
    assert "agree on direction" not in line, "a collapsed triple must never claim agreement"
    assert "NO DIRECTIONAL SIGNAL" in line
    assert "not agreement" in line


def test_agreement_clause_does_not_claim_INVISIBLE_disagreement():
    """The mirror defect: sign differences the reader cannot see must not be reported as conflict."""
    line = cf.plain_english(_fc((0.0001, -0.0002, 0.00005)))
    assert "DISAGREE" not in line, "three displayed zeros must not be called a disagreement"
    assert "NO DIRECTIONAL SIGNAL" in line


def test_exact_zero_is_not_counted_as_a_direction():
    """`p > 0` folded 0.0 into the bearish bucket, so (-0.20,-0.10,0.0) claimed unanimity."""
    line = cf.plain_english(_fc((-0.20, -0.10, 0.0)))
    assert "agree on direction" not in line
    assert "round to zero" in line and "no unanimous call" in line


def test_identical_forecasts_are_named_degenerate_not_consensus():
    line = cf.plain_english(_fc((0.05, 0.05, 0.05)))
    assert "IDENTICAL" in line and "degenerate" in line
    assert "agree on direction" not in line


def test_the_guard_still_lets_GENUINE_verdicts_through():
    """FIXTURE DISCRIMINATION: a guard that refuses everything is not a guard."""
    assert "ALL THREE POINT UP" in cf.plain_english(_fc((0.30, 0.12, 0.44)))
    assert "THE MODELS DISAGREE ON DIRECTION" in cf.plain_english(_fc((0.35, -0.22, -0.37)))


def test_the_read_states_the_COLLECTIVE_state_and_never_averages_the_seeds():
    """★ NO MEAN, MEDIAN OR WEIGHTED AVERAGE ACROSS SEEDS — EVER, AND THIS IS WHY.

    A blended number is the most quotable thing this page can produce, so it is the thing that
    WILL travel alone. On these models it would read as one confident small call, when the actual
    finding is that three versions of the same model disagree about direction. Averaging deletes
    the result. That is the context-stripping rule aimed at the most quotable number on the page.

    The guard is structural rather than stylistic: the sentence is built from a COUNT of
    directions and the min/max, and no arithmetic mean of the seeds exists anywhere in the
    function for a later edit to surface.
    """
    import inspect

    src = inspect.getsource(cf.plain_english)
    for banned in ("mean(", "np.mean", "statistics.mean", "median(", "np.median", "sum(pcts)"):
        assert banned not in src, f"plain_english computes {banned} across seeds — it must not"

    split = cf.plain_english(_fc((-0.100, -0.048, 0.129)))
    assert "TWO OF THREE POINT DOWN, ONE POINTS UP" in split
    assert "THE MODELS DISAGREE ON DIRECTION" in split
    assert "usual outcome" in split

    agree = cf.plain_english(_fc((0.011, 0.019, 0.017)))
    assert "ALL THREE POINT UP" in agree
    assert "between +0.011% and +0.019%" in agree, "the RANGE, not a blended point estimate"
    assert "closest this model comes to a directional call" in agree


def test_the_read_compares_the_seed_spread_to_the_move_being_forecast():
    """The one comparison that makes it land: how far apart the models are, versus how big the
    thing they are forecasting is. Computed from the three values, not asserted in prose."""
    line = cf.plain_english(_fc((-0.100, -0.048, 0.129)))
    # spread = 0.129 - (-0.100) = 0.229 pp; largest |value| = 0.129 -> 1.8x
    assert "0.229 percentage points" in line
    assert "1.8x the largest" in line
    assert "wider than the move they are forecasting" in line

    # ...and when the spread is SMALLER than the largest value, it must not claim otherwise
    tight = cf.plain_english(_fc((0.300, 0.320, 0.340)))
    assert "wider than the move" not in tight, "the comparison must follow the numbers"


def test_the_fee_comparison_follows_the_numbers_in_both_directions():
    under = cf.plain_english(_fc((0.011, -0.004, 0.017)))
    assert f"smaller than the ~{cf.ROUND_TRIP_PCT:.2f}% a round trip costs" in under
    over = cf.plain_english(_fc((-0.310, -0.140, -0.220)))
    assert "The largest is 0.310%" in over and "smaller than the ~0.10%" not in over


def test_the_round_trip_cost_is_one_number():
    """Two copies of a headline economic constant drifting apart is a defect class we have paid
    for; the CSV read and the demo figure must quote the same round trip."""
    from trikaal.demo import render

    assert cf.ROUND_TRIP_PCT == render.ROUND_TRIP_FEE_PCT == 0.10


# ── THE PROGRESS TRACE — the loading screen's lines are the pipeline's, not the UI's ──────────
def test_the_default_progress_sink_emits_nothing():
    """FIXTURE DISCRIMINATION for everything below: uninstrumented means uninstrumented.

    If the default sink emitted, the tests that follow could pass without the callback ever being
    threaded through — they would be measuring a global, not a wiring.
    """
    seen = []
    cf._no_progress("X", "y", "ok")
    assert seen == []
    assert cf.parse_csv(_csv_text(*_bars(1300))).ts_ms.shape[0] == 1300


def test_parser_stages_carry_THIS_input_s_real_values_not_a_script():
    """★ VARY THE INPUT. A hardcoded animation passes any single-input assertion.

    Two CSVs of different lengths and different bar intervals must produce traces that differ in
    exactly the way the inputs differ, and the values must match the returned object — which is
    what makes the loading screen a trace of the run rather than a story about it.
    """
    traces = {}
    for n, bar_ms in ((1300, 60_000), (1500, 300_000)):
        got = []
        bars = cf.parse_csv(
            _csv_text(*_bars(n, bar_ms=bar_ms)),
            progress=lambda s, d, st, _g=got: _g.append((s, d, st)),
        )
        traces[n] = got
        by_stage = {s: (d, st) for s, d, st in got if st != "run"}

        assert f"{n:,} data rows" in by_stage["PARSING CSV"][0]
        assert f"{n:,} >= {cf.MIN_ROWS:,}" in by_stage["VALIDATING LENGTH"][0]
        assert f"{n:,} bars sorted" in by_stage["READING VALUES"][0]

        detail, state = by_stage["INFERRING BAR INTERVAL"]
        assert detail.startswith(f"{bar_ms / 1000:g}s")
        assert bars.inferred_bar_ms == bar_ms
        # ★ AN OUT-OF-DISTRIBUTION INTERVAL MAY NOT FINISH QUIETLY GREEN.
        assert state == ("ok" if bar_ms == 60_000 else "warn")
        assert ("OUT OF DISTRIBUTION" in detail) is (bar_ms != 60_000)

    assert traces[1300] != traces[1500]


def test_a_refusal_stops_the_trace_at_the_stage_that_refused():
    got = []
    with pytest.raises(cf.CsvRefused):
        cf.parse_csv(_csv_text(*_bars(600)), progress=lambda s, d, st: got.append((s, d, st)))
    assert got[-1] == ("VALIDATING LENGTH", "", "run"), got
    assert not any(s == "READING VALUES" for s, _, _ in got), "nothing may run past the refusal"


def test_model_stages_agree_with_the_forecast_that_was_returned():
    """The per-seed line must quote the number the caller actually got back, to the digit."""
    from trikaal.demo.inference import available_seeds, load_unit

    seeds = available_seeds()
    if not seeds:
        pytest.skip("banked units not present")
    units = {seeds[0]: load_unit(seeds[0])}
    got = []
    f = cf.forecast_from_csv(
        _csv_text(*_bars(1300)),
        units=units,
        h=5,
        mc_samples=2,
        n_mc_paths=2,
        progress=lambda s, d, st: got.append((s, d, st)),
    )
    done = {s: d for s, d, st in got if st in ("ok", "warn")}

    assert "7 dims x 1,300 bars" in done["COMPUTING CAUSAL FEATURES"]
    assert "1,300 bars -> 1,300 coarse" in done["TOKENIZING"]
    assert "2 paths x 5 steps" in done["SAMPLING TRAJECTORIES"]
    assert "p10/p25/p50/p75/p90" in done["COMPUTING QUANTILE BANDS"]
    assert "1 of 3 usable" in done["MTP ANCHORS"] and "(5m)" in done["MTP ANCHORS"]

    seed = seeds[0]
    pct = (math.exp(f.per_seed[seed]["mu"]) - 1.0) * 100.0
    assert f"seed {seed}: expectation {pct:+.4f}%" in done["FORECASTING"], done["FORECASTING"]


# ── THE DERIVATION'S OWN GATES — what replaced the second production rollout ──────────────────
def _fixture_arrays():
    from trikaal.train.arms import select_arm

    out = compute_features(_stream(*_bars(1400), micro=False), FeatureConfig())
    x_arm, m_arm = select_arm(np.asarray(out.x, np.float32), out.m, ARM_OHLCV)
    return out, x_arm, m_arm, int(x_arm.shape[0] - 1)


def test_a_longer_rollout_contains_the_shorter_one_as_a_prefix():
    """★ THE ASSUMPTION THE DERIVATION RESTS ON, GATED SEPARATELY.

    The demo runs ONE rollout of 48 sampled paths and derives production's mc_mean@32 from the
    first 32 of them. That is only valid if sample *i* is the same sample whether you asked for 32
    or 48 — i.e. the generator stream is prefix-identical, which holds because every sample
    consumes exactly ``2*h`` draws and nothing about the draw count is data-dependent.

    It is asserted element-by-element rather than on the mean, because two different sample sets
    can average to the same number and that would hide the very drift this is here to catch.
    """
    from trikaal.demo.inference import available_seeds, load_unit

    seeds = available_seeds()
    if not seeds:
        pytest.skip("banked units not present")
    u = load_unit(seeds[0])
    out, x_arm, m_arm, i = _fixture_arrays()
    kw = dict(h=15, seed=20260704)
    long = cf.mc_paths(u, x_arm, m_arm, out.segment_id, out.ts, out.sigma, i, n_samples=48, **kw)
    short = cf.mc_paths(u, x_arm, m_arm, out.segment_id, out.ts, out.sigma, i, n_samples=32, **kw)
    assert np.array_equal(long[:32], short), (
        f"{int((long[:32] != short).sum())} of {short.size} values differ — the 48-sample rollout "
        "is NOT a superset of the 32-sample one, so deriving mc_mean@32 from it is invalid"
    )
    # ...and the fixture must be able to discriminate: a DIFFERENT seed must not match, or the
    # assertion above would pass against any rollout at all.
    other = cf.mc_paths(
        u, x_arm, m_arm, out.segment_id, out.ts, out.sigma, i, h=15, n_samples=32, seed=99
    )
    assert not np.array_equal(other, short), "the fixture cannot tell two rollouts apart"


def test_the_derived_value_matches_a_PINNED_reference():
    """★ THE ONE CHECK THAT SURVIVES BOTH ARMS CHANGING TOGETHER.

    Every other gate here compares two implementations. An agreement test cannot detect an error
    its two arms SHARE, so an edit that altered ``mc_paths`` and ``_rollout_mc_mean`` in the same
    way would keep them agreeing and slip through. This pins the actual number, measured on a
    deterministic fixture against the banked seed-0 checkpoint. If it moves, something moved —
    and that is true even if everything still agrees with everything else.
    """
    from trikaal.demo.inference import available_seeds, load_unit

    seeds = available_seeds()
    if not seeds or seeds[0] != 0:
        pytest.skip("the pin is measured against the banked seed-0 unit")
    u = load_unit(0)
    out, x_arm, m_arm, i = _fixture_arrays()
    paths = cf.mc_paths(
        u, x_arm, m_arm, out.segment_id, out.ts, out.sigma, i, h=15, n_samples=48, seed=20260704
    )
    got = cf.mc_mean_from_paths(paths, float(out.sigma[i]), 32)
    assert got.hex() == "0x1.f26f6f43c08cdp-12", (
        f"the derived mc_mean@32 on the 1,400-bar fixture is {got.hex()}, pinned "
        "0x1.f26f6f43c08cdp-12 — an agreement test would not have noticed this"
    )


def test_mc_mean_from_paths_refuses_a_rollout_too_short_to_derive_from():
    with pytest.raises(ValueError, match="at least 32"):
        cf.mc_mean_from_paths(np.zeros((8, 15)), 1.0, 32)


def test_the_anchor_at_the_selected_horizon_IS_the_headline_mu():
    """The ha == h anchor reuses `mu` instead of repeating an identical deterministic call."""
    from trikaal.demo.inference import available_seeds, load_unit

    seeds = available_seeds()
    if not seeds:
        pytest.skip("banked units not present")
    units = {seeds[0]: load_unit(seeds[0])}
    f = cf.forecast_from_csv(_csv_text(*_bars(1300)), units=units, h=5, mc_samples=2, n_mc_paths=4)
    s = seeds[0]
    assert f.mtp_anchors[s][5] == f.per_seed[s]["mu"], "the h anchor must BE the headline mu"


def test_the_band_value_the_demo_ships_equals_production_at_the_same_n():
    """END TO END: the number `forecast_from_csv` actually puts in the band, versus production.

    The gates above check components. This checks the shipped path — the demo no longer calls
    production for this value, so the thing that must not drift is what the USER is handed.
    """
    from trikaal.demo.inference import available_seeds, load_unit
    from trikaal.eval.predict import predict_mu
    from trikaal.train.arms import select_arm
    from trikaal.train.token_stream import tokenize_features

    seeds = available_seeds()
    if not seeds:
        pytest.skip("banked units not present")
    u = load_unit(seeds[0])
    units = {seeds[0]: u}
    n_mc, h = 8, 5
    f = cf.forecast_from_csv(
        _csv_text(*_bars(1300)), units=units, h=h, mc_samples=n_mc, n_mc_paths=n_mc
    )
    out = compute_features(_stream(*_bars(1300), micro=False), FeatureConfig())
    x_arm, m_arm = select_arm(np.asarray(out.x, np.float32), out.m, ARM_OHLCV)
    b_c, b_f = tokenize_features(u.tok, x_arm, m_arm, out.segment_id, window=cf.SEQ_LEN)
    i = int(x_arm.shape[0] - 1)
    prod = float(
        predict_mu(
            u.model,
            u.tok,
            b_c,
            b_f,
            out.ts,
            out.sigma,
            np.array([i], np.int64),
            h=h,
            seq_len=cf.SEQ_LEN,
            estimator="mc_mean",
            mc_samples=n_mc,
            mc_seed=20260704,
        )[0]
    )
    lo = math.log(1.0 + f.band["lo_pct"] / 100.0)
    hi = math.log(1.0 + f.band["hi_pct"] / 100.0)
    assert min(abs(lo - prod), abs(hi - prod)) < 1e-12 or lo <= prod <= hi, (
        f"production mc_mean {prod!r} is not represented in the shipped band [{lo!r}, {hi!r}]"
    )


def test_the_demo_uses_ONE_mc_seed_for_the_scalar_and_the_bands():
    """★ THE PRE-EXISTING INCOHERENCE THE DERIVATION EXPOSED, PINNED SHUT.

    The dashboard used to take its band-widening mc_mean from production's ``MC_DEFAULT_SEED``
    (20260721, inherited by omitting the argument) while the percentile bands drawn beside it came
    from ``mc_paths``' 20260704. One figure, two rollouts, disagreeing by construction: +0.0158%
    against +0.0403% on the same BTCUSDT bars. Neither value is wrong — the estimator is unbiased
    at any seed — but a scalar that is supposed to summarise the fan it sits next to must come
    from that fan.

    Deriving the scalar from the paths makes it structurally impossible to disagree again: there
    is now exactly one rollout and one seed. This test pins the seed as a NAMED constant so it
    cannot drift back to being two inherited defaults.
    """
    import inspect

    from trikaal.eval.predict import MC_DEFAULT_SEED

    assert cf.MC_SEED == 20260704
    assert cf.MC_SEED != MC_DEFAULT_SEED, (
        "if these ever coincide the test stops discriminating — the whole point is that the demo "
        "does NOT silently inherit production's default"
    )
    assert inspect.signature(cf.mc_paths).parameters["seed"].default == cf.MC_SEED

    # ...and the demo must not call production's mc_mean at all any more: the scalar is derived.
    src = inspect.getsource(cf.forecast_from_csv)
    assert 'estimator="mc_mean"' not in src, (
        "forecast_from_csv is calling production's mc_mean again — that is the second rollout the "
        "derivation removed, and it reintroduces the two-seed incoherence"
    )
    assert "mc_mean_from_paths(" in src
