"""The full eval machine runs end-to-end without error on a tiny synthetic lake (smoke).

This proves the wiring (folds → rollout → strategy → cost → net-IR → DSR/PBO → placebo →
diagnostics → Q3/Q4 code path) executes; the numbers are meaningless. The individual primitives
are validated by their own known-answer tests.
"""

from __future__ import annotations

import numpy as np

from trikaal.data.config import BAR_MS
from trikaal.eval.harness import HarnessResult, run_harness
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.utils.seeding import set_determinism


def _synthetic_lake(n=2600):
    rng = np.random.default_rng(0)
    return {
        "b_c": rng.integers(0, 891, n),
        "b_f": rng.integers(0, 1225, n),
        "ts": (1_600_000_000_000 + np.arange(n) * BAR_MS).astype(np.int64),
        "segment_id": np.zeros(n, dtype=np.int64),
        "sigma": np.full(n, 0.01),
        "raw_ret_close": (0.001 * rng.standard_normal(n)).astype(np.float64),
        "x": rng.standard_normal((n, 16)).astype(np.float32),
        "mask": _mask_perp(np.zeros((n, 16), dtype=np.uint8)),
    }


def _mask_perp(mask: np.ndarray) -> np.ndarray:
    # the real build masks funding/OI everywhere (§1.2); the placebo tripwire enforces it
    mask[:, 13:16] = 1
    return mask


def test_run_harness_end_to_end_smoke():
    set_determinism(0)
    model = TrikaalAR(n_layers=1, d_model=32, d_ff=64, n_heads=2, mtp_depths=0, max_len=64).eval()
    tok = TokenizerAE(d_model=32, n_layers=1, n_heads=2, d_ff=64).eval()
    res = run_harness(_synthetic_lake(), model, tok, h=5, seq_len=16, cap=40, device="cpu", seed=0)
    assert isinstance(res, HarnessResult)
    assert res.quadrants_run == ["Q1", "Q2", "Q3", "Q4"]  # all quadrants' code paths ran
    assert set(res.cost_stress) == {0.001, 0.002, 0.003}  # cost-stress curve produced
    assert "micro_exceeds_placebo" in res.placebo  # placebo machinery ran
    assert {"rank_ic", "ic", "vol_mae", "picp80"} <= set(res.diagnostics)  # diagnostics ran
    # IR / DSR / break-even are finite floats or NaN (meaningless values, but well-typed)
    for v in (res.ir_headline, res.ir_modeled_cost, res.dsr, res.break_even, res.kappa_chosen):
        assert isinstance(float(v), float)
    # §6 item 10d: the full per-κ VAL IR curve is persisted (κ*-knife-edge auditable), and κ* is
    # its argmax; activity is the traded fraction of the headline calendar grid
    from trikaal.eval.harness import KAPPAS

    assert set(res.val_ir_by_kappa) == set(KAPPAS)
    finite = {k: v for k, v in res.val_ir_by_kappa.items() if np.isfinite(v)}
    if finite:
        assert res.kappa_chosen == max(finite, key=finite.get)
    assert 0.0 <= res.activity <= 1.0 or np.isnan(res.activity)
