"""The §7 v1.4 STANDING M6-time micro-legibility gate — mechanics KATs (gate-2 ruling).

The gate itself runs inside ``run_cell`` (post-Stage-1, pre-Stage-2, hard stop) on the real
training stream; these tests exercise its machinery on tiny shells: the failure path MUST
raise (an untrained tokenizer's ids are illegible), the pass path returns the receipt, and
masked-everywhere dims are skipped rather than trivially passed."""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.data.universe_loader import build_symbol_windows
from trikaal.train.cells import CELLS, build_cell_tokenizer
from trikaal.train.gates import MICRO_DIMS, micro_legibility_gate
from trikaal.train.token_stream import tokenize_features

TINY = dict(d_model=32, n_layers=1, n_heads=2, d_ff=64, max_len=64)
N = 30_000


def _fixture():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((N, 16)).astype(np.float32)
    mask = np.zeros((N, 16), dtype=np.uint8)
    mask[:, 13:16] = 1
    seg = np.zeros(N, dtype=np.int64)
    ts = (1_704_067_200_000 + np.arange(N) * 60_000).astype(np.int64)
    sw = build_symbol_windows(
        "SYNAUSDT", x, mask, seg, ts, arm="micro", seq_len=32, boundary_ms=ts[-1] + 1, seed=0
    )
    tok = build_cell_tokenizer(CELLS[3], **TINY).eval()
    b_c, b_f = tokenize_features(tok, sw.x, sw.mask, sw.segment_id, window=32, device="cpu")
    return tok, [sw], {"SYNAUSDT": (b_c, b_f)}


def test_gate_hard_stops_on_illegible_ids():
    """An UNTRAINED tiny tokenizer's ids carry ~chance legibility — the gate must raise,
    and the message must carry the pre-authorized-fallback pointer."""
    tok, per_symbol, tokens = _fixture()
    with pytest.raises(RuntimeError, match="MICRO LEGIBILITY GATE FAILED"):
        micro_legibility_gate(tok, per_symbol, tokens, min_acc=0.9, run_name="kat")


def test_gate_receipt_shape_on_pass():
    """min_acc=0 always passes: the receipt must cover every unmasked micro dim with
    sign_acc + base rate, and report pass=True."""
    tok, per_symbol, tokens = _fixture()
    receipt = micro_legibility_gate(tok, per_symbol, tokens, min_acc=0.0, run_name="kat")
    assert receipt["pass"] is True
    for dim in MICRO_DIMS:
        rec = receipt["per_dim"][str(dim)]
        assert "sign_acc" in rec and 0.0 <= rec["sign_acc"] <= 1.0
        assert 0.0 <= rec["base_rate_positive"] <= 1.0


def test_gate_skips_masked_everywhere_dims():
    """A micro dim masked on every bar is SKIPPED (recorded), never trivially passed."""
    tok, per_symbol, tokens = _fixture()
    sw = per_symbol[0]
    mask = sw.mask.copy()
    mask[:, 8] = 1  # dim 8 masked everywhere
    sw2 = type(sw)(
        symbol=sw.symbol,
        x=sw.x,
        mask=mask,
        segment_id=sw.segment_id,
        ts=sw.ts,
        starts=sw.starts,
    )
    receipt = micro_legibility_gate(tok, [sw2], tokens, min_acc=0.0, run_name="kat")
    assert "skipped" in receipt["per_dim"]["8"]


# ------------------------------------------------- §7 v1.5 item E: the λ calibration slice
def test_lambda_calibration_slice_is_carved_from_the_END_of_the_TRAIN_region():
    """It must sit INSIDE the train region and strictly BEFORE the train/eval boundary — never in
    a scored block and never in eval block 0, which already carries κ* selection AND every
    clause-5 DSR trial entry. Inert unless the gate fires."""
    from trikaal.train.gates import (
        LAMBDA_CALIBRATION_SLICE_FRAC,
        lambda_calibration_boundary_ms,
    )

    train_start, train_end = 1_000_000, 11_000_000  # the train/eval boundary is train_end
    b = lambda_calibration_boundary_ms(train_start, train_end)
    assert train_start < b < train_end  # strictly INSIDE the train region
    assert b == train_end - int((train_end - train_start) * LAMBDA_CALIBRATION_SLICE_FRAC)
    frac = (train_end - b) / (train_end - train_start)
    assert abs(frac - LAMBDA_CALIBRATION_SLICE_FRAC) < 1e-12  # the LAST frac of train


def test_lambda_calibration_slice_rejects_degenerate_inputs():
    from trikaal.train.gates import lambda_calibration_boundary_ms

    with pytest.raises(ValueError, match="slice fraction"):
        lambda_calibration_boundary_ms(0, 100, frac=0.0)
    with pytest.raises(ValueError, match="empty train region"):
        lambda_calibration_boundary_ms(100, 100)
