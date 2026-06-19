"""Gate-A causal re-validation (§8.A.5 / §8.E.5): the pipeline-purity exhaustive sweep is re-run as
a merge gate on the harness — leak-free on a Q4-representative sample, and it still has teeth.
"""

from __future__ import annotations

import pytest

from trikaal.data.causal_check import exhaustive_truncation_sweep
from trikaal.data.config import synthetic_test_config
from trikaal.data.synthetic import make_synthetic_stream


@pytest.mark.gate
def test_gate_a_causal_sweep_leakfree_on_q4_sample():
    """A Q4-representative slice (bad-tick / segment-boundary / break / stale regions) is leak-free
    under the exhaustive truncation sweep — 100% bar coverage, the harness merge gate."""
    stream, _ = make_synthetic_stream(seed=44, n_bars=420, inject_anomalies=True)
    cfg = synthetic_test_config(half_life_fast=48, half_life_slow=96, n_warm=12, w_tau=20)
    rep = exhaustive_truncation_sweep(stream, cfg, stride=1)
    assert rep.passed, f"Gate-A causal sweep FAILED on Q4 sample: {rep.failures[:3]}"
    assert rep.coverage_bars == rep.total_bars  # 100% coverage is the gating criterion


@pytest.mark.gate
def test_gate_a_sweep_still_has_teeth():
    """A planted lookahead must be caught — the gate is not vacuously passing."""
    stream, _ = make_synthetic_stream(seed=44, n_bars=300, inject_anomalies=True)
    cfg = synthetic_test_config(half_life_fast=48, half_life_slow=96, n_warm=12, w_tau=20)
    leaked = exhaustive_truncation_sweep(stream, cfg.with_leak("global_zscore"), stride=1)
    assert not leaked.passed, "planted lookahead slipped past the Gate-A sweep"
