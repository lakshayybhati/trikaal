"""KAT: the §7 v1.4.5 h-sweep mirror reproduces ``xsection.score_cell`` exactly.

``scripts/m6_h_sweep.py`` cannot call ``score_cell`` directly — score_cell tokenizes the full
42M-bar stream on every call and the sweep needs the token stream hoisted out of the h-loop (25
calls, and three 14M-bar feature matrices do not fit in RAM at once). The mirror is therefore a
line-for-line copy of score_cell's decision path with the tokens passed in.

A copy is only trustworthy if it is PROVEN equal to the original, so this KAT runs both on the
same tiny fixture and asserts the load-bearing outputs agree BIT-FOR-BIT: κ*, the headline IR at
κ*, the per-κ VAL IR curve, the decision count, and every μ̂ diagnostic the degeneracy guard
reads (mean / std / frac_negative / per-κ decision-activity / activity at κ*). If someone edits
score_cell and forgets the mirror, this fails here rather than silently in a receipt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from trikaal.data.config import BAR_MS
from trikaal.eval.xsection import SymbolEval, XSectionConfig, score_cell
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import ARM_MICRO, select_arm
from trikaal.train.token_stream import tokenize_features
from trikaal.utils.seeding import set_determinism

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from m6_h_sweep import score_pretokenized  # noqa: E402

T0 = 1_704_067_200_000  # 2024-01-01 UTC in ms
SEQ = 16


def _sym(symbol: str, n: int = 4320, seed: int = 0) -> SymbolEval:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 16)).astype(np.float32)
    m = np.zeros((n, 16), dtype=np.uint8)
    m[:, 13:16] = 1
    return SymbolEval(
        symbol=symbol,
        x=x,
        mask=m,
        segment_id=np.zeros(n, dtype=np.int64),
        ts=(T0 + np.arange(n) * BAR_MS).astype(np.int64),
        sigma=np.full(n, 0.01),
        raw_ret_close=(0.001 * rng.standard_normal(n)).astype(np.float64),
    )


def _models():
    set_determinism(0)
    kw = dict(d_model=16, n_layers=1, n_heads=2, d_ff=32, max_len=64)
    tok = TokenizerAE(**kw).eval()
    model = TrikaalAR(
        n_layers=1,
        d_model=16,
        d_ff=32,
        n_heads=2,
        mtp_depths=0,
        max_len=64,
        ffn_dropout=0.0,
        resid_dropout=0.0,
        attn_dropout=0.0,
        token_dropout=0.0,
    ).eval()
    return model, tok


@pytest.mark.parametrize("h", [2, 5])
def test_mirror_matches_score_cell_bitwise(h):
    model, tok = _models()
    symbols = [_sym("AAAUSDT", seed=0), _sym("BBBUSDT", seed=1)]
    cfg = XSectionConfig(
        window_start="2024-01-01",
        window_end="2024-01-04",
        train_frac=0.7,
        h=h,
        seq_len=SEQ,
        cap_per_symbol=40,
        device="cpu",
        seed=0,
    )
    ref = score_cell("cell4", model, tok, ARM_MICRO, symbols, cfg)

    prepared = {}
    for se in symbols:  # the hoisted tokenization the sweep does once per checkpoint
        x_arm, m_arm = select_arm(np.asarray(se.x, np.float32), se.mask, ARM_MICRO)
        b_c, b_f = tokenize_features(
            tok, x_arm, m_arm, se.segment_id, window=cfg.seq_len, device="cpu"
        )
        prepared[se.symbol] = {"b_c": b_c, "b_f": b_f, "se": se}
    got = score_pretokenized(model, tok, prepared, cfg)

    assert got["kappa_star"] == float(ref.kappa_chosen)
    assert got["ir_at_kstar"] == float(ref.ir_headline)
    assert got["n_decisions"] == int(ref.n_decisions)
    assert got["val_ir_by_kappa"] == {f"{k:g}": float(v) for k, v in ref.val_ir_by_kappa.items()}
    for field in ("mean", "std", "frac_negative", "n", "activity_decisions"):
        assert got["mu_diag"][field] == ref.mu_diag[field], field
    assert (
        got["mu_diag"]["activity_decisions_by_kappa"] == ref.mu_diag["activity_decisions_by_kappa"]
    )


def test_sparse_tokenization_is_bit_identical_on_needed_bars():
    """``tokenize_needed`` must equal ``tokenize_features`` everywhere it writes.

    This is the load-bearing fidelity claim of the sweep's memory optimization: because
    ``tokenize_features`` encodes NON-OVERLAPPING aligned chunks, bar t's token depends on exactly
    one chunk, so skipping other chunks cannot perturb it. If that ever stops being true (an
    overlapping or context-carrying tokenizer), this fails loudly instead of quietly shifting
    every diagnostic in the receipt.
    """
    from m6_h_sweep import needed_bar_mask, tokenize_needed

    _model, tok = _models()
    se = _sym("AAAUSDT", seed=3)
    cfg = XSectionConfig(
        window_start="2024-01-01",
        window_end="2024-01-04",
        train_frac=0.7,
        h=1,
        seq_len=SEQ,
        cap_per_symbol=None,
        device="cpu",
        seed=0,
    )
    x_arm, m_arm = select_arm(np.asarray(se.x, np.float32), se.mask, ARM_MICRO)
    full_c, full_f = tokenize_features(tok, x_arm, m_arm, se.segment_id, window=SEQ, device="cpu")
    needed = needed_bar_mask(cfg, se.ts)
    assert needed.any() and not needed.all(), "the mask must actually be sparse to test anything"
    got_c, got_f = tokenize_needed(
        tok, x_arm, m_arm, se.segment_id, needed, window=SEQ, device="cpu"
    )

    written = got_c >= 0
    assert written[needed].all(), "every needed bar must have been tokenized"
    np.testing.assert_array_equal(got_c[written], full_c[written])
    np.testing.assert_array_equal(got_f[written], full_f[written])
    assert (got_c[~written] == -1).all(), "untokenized bars stay -1 so a stray read is loud"
    assert written.sum() < got_c.size, "the sparse path must skip real work"


def test_mirror_reads_the_same_frac_neg_band_as_the_guard():
    """The sweep's upgrade test must key on the guard's band, never a private copy."""
    import m6_h_sweep as hs

    from trikaal.eval.verdict import FRAC_NEG_BAND

    assert hs.FRAC_NEG_BAND is FRAC_NEG_BAND
    assert hs.BAND_DRIFT_LIMIT == 0.15  # pre-declared, not tunable after the fact
    assert hs.HORIZONS == (1, 2, 3, 5, 15)
