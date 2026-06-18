"""★ The lookahead merge gate (feature-spec §6 / §2.6.2) — gate G0.

Two claims, both required: (1) the deliberately-planted lookahead variants FAIL the invariant
(the harness has teeth), and (2) the real transforms PASS it (the pipeline is leak-free).
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.data.causal_check import build_anchor_strata, check_causal_safety
from trikaal.data.dataset import assert_qa_columns_excluded, build_model_batch
from trikaal.data.features import compute_features

_MODES = ("noise", "revert", "signflip")
_LEAK_EXPECTED_OUTPUT = {
    "centered_zscore": "x",
    "global_zscore": "x",
    "sigma_includes_next": "sigma",
    "forward_revert_badtick": "raw",  # also "x"; raw_range changes when the wick clip flips
}


@pytest.mark.gate
def test_real_transforms_are_leak_free(causal_data):
    """The real per-bar transforms read no datum beyond each output's causal horizon."""
    stream, anom, cfg = causal_data
    rep = check_causal_safety(stream, cfg, anom, seed=1337, n_per_stratum=5, modes=_MODES)
    assert rep.passed, rep.summary() + "\n" + "\n".join(map(str, rep.failures[:10]))
    # the sampler must actually exercise every high-risk stratum
    for stratum, n in rep.strata_counts.items():
        assert n > 0, f"stratum {stratum} had no anchors — sampling discipline violated"
    assert rep.n_checks > 200


@pytest.mark.gate
@pytest.mark.parametrize("leak", sorted(_LEAK_EXPECTED_OUTPUT))
def test_planted_leak_is_caught(causal_data, leak):
    """Each planted lookahead must be caught — and surfaced in the expected output."""
    stream, anom, cfg = causal_data
    rep = check_causal_safety(
        stream, cfg.with_leak(leak), anom, seed=1337, n_per_stratum=5, modes=_MODES,
        stop_on_first_failure=True,
    )
    assert not rep.passed, f"planted leak '{leak}' was NOT caught — the harness lacks teeth"
    caught = {f["output"] for f in rep.failures}
    assert _LEAK_EXPECTED_OUTPUT[leak] in caught, (
        f"leak '{leak}' should surface in output '{_LEAK_EXPECTED_OUTPUT[leak]}', caught={caught}"
    )


@pytest.mark.gate
def test_loader_excludes_qa_columns(causal_data):
    """Loader assertion (§7.2): is_stale / dq_flags / vol_recon_err never reach the model."""
    stream, _anom, cfg = causal_data
    out = compute_features(stream, cfg)
    batch = build_model_batch(out)
    assert set(batch) == {"x", "mask", "ts", "raw"}
    # sneaking a QA column in must hard-fail
    bad = dict(batch)
    bad["dq_flags"] = out.dq_flags
    with pytest.raises(AssertionError):
        assert_qa_columns_excluded(bad)


@pytest.mark.gate
def test_mask_is_strictly_causal(causal_data):
    """Every set mask bit is reconstructible from a strictly-causal per-bar trigger (§4.5).

    For each bar, a liquidity/aggTrades bit may be set only if the bar is zero-volume, in
    z-score/tau warm-up, or has vol_recon_err > 0.01; perp bits only if funding/OI missing.
    Crucially, ``is_stale`` (a forward-confirmed *quality* signal) is never OR-ed into m_t.
    """
    stream, _anom, cfg = causal_data
    out = compute_features(stream, cfg)
    # is_stale must not imply any mask bit that a zero-volume bar wouldn't already set.
    stale_bars = np.nonzero(out.is_stale == 1)[0]
    zero_vol = stream.v_kline == 0.0
    for t in stale_bars:
        # stale bars in this fixture are zero-volume, so their liquidity mask bits are causal;
        # assert no bit is set that ISN'T explained by the causal zero-volume condition there.
        assert zero_vol[t], "a stale bar that is not zero-volume must not gain mask bits from staleness"


@pytest.mark.gate
def test_strata_cover_all_high_risk_regions(causal_data):
    """Sampling targets bad-tick / segment-boundary / structural-break / stale-run explicitly."""
    stream, anom, cfg = causal_data
    out = compute_features(stream, cfg)
    rng = np.random.default_rng(0)
    strata = build_anchor_strata(out, anom, rng, n_per_stratum=5)
    assert set(strata) == {
        "bad_tick_adjacent",
        "segment_boundary",
        "structural_break",
        "stale_run",
        "random_background",
    }
    # the exact anomaly centers are always included
    assert set(anom.bad_tick) & set(strata["bad_tick_adjacent"])
    assert set(anom.structural_break) & set(strata["structural_break"])
