"""★ The lookahead merge gate (feature-spec §6 / §2.6.2) — gate G0.

Three claims, all required:
  (1) the EXHAUSTIVE truncation sweep covers every bar and the real transforms pass it
      (coverage is the gating criterion — not a sparse stochastic sample);
  (2) a LOCALIZED single-bar leak that a sparse anchor sampler misses is caught by the
      exhaustive sweep (both a divisor leak and a target_valid survivorship leak);
  (3) the global planted-lookahead variants fail and the loader excludes QA-only columns.
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.data.causal_check import (
    build_anchor_strata,
    check_causal_safety,
    exhaustive_truncation_sweep,
)
from trikaal.data.dataset import assert_qa_columns_excluded, build_model_batch
from trikaal.data.features import compute_features

_MODES = ("noise", "revert", "signflip")
_LEAK_EXPECTED_OUTPUT = {
    "centered_zscore": "x",
    "global_zscore": "x",
    "sigma_includes_next": "sigma",
    "forward_revert_badtick": "raw",
}


def _non_anchor_bar(out, anom):
    """A mid-segment, post-warm-up bar with a valid target that the sparse sampler does NOT draw."""
    strata = build_anchor_strata(out, anom, np.random.default_rng(1337), 5)
    anchors = {t for v in strata.values() for t in v}
    for b in range(80, len(out) - 5):
        if b not in anchors and bool(out.target_valid[b]):
            return b
    raise AssertionError("no suitable non-anchor bar found")


@pytest.mark.gate
@pytest.mark.slow
def test_real_transforms_pass_exhaustive_sweep(causal_data):
    """The merge gate: truncate at EVERY bar; every output is byte-identical. 100% coverage."""
    stream, _anom, cfg = causal_data
    rep = exhaustive_truncation_sweep(stream, cfg)
    assert rep.passed, rep.summary() + "\n" + "\n".join(map(str, rep.failures[:10]))
    assert rep.coverage == 1.0, f"gate must cover every bar, got {rep.coverage:.3f}"
    assert rep.n_checks > 1000


@pytest.mark.gate
def test_real_transforms_pass_perturbation_probe(causal_data):
    """Adversarial perturbation probe on the high-risk strata (added defense beyond truncation)."""
    stream, anom, cfg = causal_data
    rep = check_causal_safety(stream, cfg, anom, seed=1337, n_per_stratum=5, modes=_MODES)
    assert rep.passed, rep.summary() + "\n" + "\n".join(map(str, rep.failures[:10]))
    for stratum, n in rep.strata_counts.items():
        assert n > 0, f"stratum {stratum} had no anchors — sampling discipline violated"


@pytest.mark.gate
def test_localized_divisor_leak_caught_by_exhaustive_but_missed_by_sampler(causal_data):
    """A single-bar σ_t look-ahead at a non-anchor bar: sparse sampler misses, exhaustive catches."""
    stream, anom, cfg = causal_data
    out = compute_features(stream, cfg)
    lb = _non_anchor_bar(out, anom)
    leaked = cfg.with_localized_leak(lb, kind="sigma")

    sampler = check_causal_safety(stream, leaked, anom, seed=1337, n_per_stratum=5)
    assert sampler.passed, "precondition: the sparse sampler must MISS this localized leak"

    sweep = exhaustive_truncation_sweep(stream, leaked, stop_on_first_failure=True)
    assert not sweep.passed, "the exhaustive sweep must CATCH the localized divisor leak"
    assert "sigma" in {f["output"] for f in sweep.failures}


@pytest.mark.gate
def test_localized_survivorship_leak_in_target_valid_is_caught(causal_data):
    """target_valid is now a checked output: a future-conditioned sample drop is caught."""
    stream, anom, cfg = causal_data
    out = compute_features(stream, cfg)
    lb = _non_anchor_bar(out, anom)
    leaked = cfg.with_localized_leak(lb, kind="validity")

    sampler = check_causal_safety(stream, leaked, anom, seed=1337, n_per_stratum=5)
    assert sampler.passed, "precondition: the sparse sampler must MISS this localized leak"

    sweep = exhaustive_truncation_sweep(stream, leaked, stop_on_first_failure=True)
    assert not sweep.passed, "the exhaustive sweep must CATCH the survivorship leak in target_valid"
    assert "target_valid" in {f["output"] for f in sweep.failures}


@pytest.mark.gate
@pytest.mark.parametrize("leak", sorted(_LEAK_EXPECTED_OUTPUT))
def test_global_planted_leak_is_caught(causal_data, leak):
    """Each global planted lookahead is caught — surfaced in the expected output."""
    stream, anom, cfg = causal_data
    rep = check_causal_safety(
        stream,
        cfg.with_leak(leak),
        anom,
        seed=1337,
        n_per_stratum=5,
        modes=_MODES,
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
    bad = dict(batch)
    bad["dq_flags"] = out.dq_flags
    with pytest.raises(AssertionError):
        assert_qa_columns_excluded(bad)


@pytest.mark.gate
def test_mask_is_strictly_causal(causal_data):
    """is_stale (a forward-confirmed quality signal) is never OR-ed into the model mask (§4.5)."""
    stream, _anom, cfg = causal_data
    out = compute_features(stream, cfg)
    zero_vol = stream.v_kline == 0.0
    for t in np.nonzero(out.is_stale == 1)[0]:
        assert zero_vol[t], (
            "a stale bar that is not zero-volume must not gain mask bits from staleness"
        )


@pytest.mark.gate
def test_strata_cover_all_high_risk_regions(causal_data):
    """Sampling targets bad-tick / segment-boundary / structural-break / stale-run explicitly."""
    stream, anom, cfg = causal_data
    out = compute_features(stream, cfg)
    strata = build_anchor_strata(out, anom, np.random.default_rng(0), n_per_stratum=5)
    assert set(strata) == {
        "bad_tick_adjacent",
        "segment_boundary",
        "structural_break",
        "stale_run",
        "random_background",
    }
    assert set(anom.bad_tick) & set(strata["bad_tick_adjacent"])
    assert set(anom.structural_break) & set(strata["structural_break"])
