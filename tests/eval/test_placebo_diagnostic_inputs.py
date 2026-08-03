"""§7 v1.6.25 (RE-AUDIT R12e) — THE TWO REQUIRED DISCLOSURES DESCRIBED A TENSOR CELL 5 NEVER SAW.

``score_cell`` applies the Cell-5 block permutation to LOCAL variables, tokenizes from them, and
then — until this fix — rebuilt the diagnostic window from ``se.x``, the RAW feature matrix. So on
the placebo arm, and only there, ``ohlcv_recon`` and ``decode_agreement`` measured a tokenizer
trained on shuffled micro against UNSHUFFLED micro: out-of-distribution input, not "the capacity
handicap". Both are REQUIRED disclosures (§7 v1.6 C-12 M1 and v1.6.11 C-1), and the C-12
disclosure's whole argument is about what Cell 5 spends its bits on — so it was wrong in exactly
the arm it exists to characterise. Every other arm was unaffected, which is why nothing looked odd.

THE ASSERTION. ``tokenize_features`` goes through ``tok.encode_tokens``; both diagnostics go
through ``tok.latent``. Spying on ``latent`` therefore captures the diagnostics' inputs and
nothing else, and the contract is exact: **on the placebo arm the UNSHUFFLED window must never
reach the tokenizer at all.**
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.data.config import BAR_MS
from trikaal.eval.xsection import SymbolEval, XSectionConfig, score_cell
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import ARM_MICRO, ARM_MICRO_SHUFFLED, select_arm, shuffle_micro
from trikaal.utils.seeding import set_determinism

WINDOW = dict(window_start="2024-01-01", window_end="2024-01-04", train_frac=0.7)
T0 = 1_704_067_200_000


def _cfg(**kw) -> XSectionConfig:
    return XSectionConfig(h=5, seq_len=16, cap_per_symbol=8, device="cpu", **WINDOW, **kw)


def _sym(symbol: str = "AAAUSDT", n: int = 4320, seed: int = 0) -> SymbolEval:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 16)).astype(np.float32)
    m = np.zeros((n, 16), dtype=np.uint8)
    m[:, 13:16] = 1  # funding/OI are masked everywhere (the v1 lake's structural absence)
    return SymbolEval(
        symbol=symbol,
        x=x,
        mask=m,
        segment_id=np.zeros(n, dtype=np.int64),
        ts=(T0 + np.arange(n) * BAR_MS).astype(np.int64),
        sigma=np.full(n, 0.01),
        raw_ret_close=(0.001 * rng.standard_normal(n)).astype(np.float64),
    )


def _spy(tok: TokenizerAE, seen: list[np.ndarray]) -> None:
    """Record every tensor handed to ``latent`` — i.e. every diagnostic input."""
    orig = tok.latent

    def latent(x, m, *a, **kw):
        seen.append(x.detach().cpu().numpy().copy())
        return orig(x, m, *a, **kw)

    tok.latent = latent  # type: ignore[method-assign]


def _ar() -> TrikaalAR:
    return TrikaalAR(
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


def _windows(se: SymbolEval, cfg: XSectionConfig):
    """(what the placebo arm SHOULD be fed, what it was fed BEFORE the fix)."""
    sx, sm = shuffle_micro(se.x, se.mask, se.segment_id, symbol=se.symbol, seed=cfg.seed)
    shuffled, _ = select_arm(np.asarray(sx, np.float32), sm, ARM_MICRO_SHUFFLED)
    raw, _ = select_arm(np.asarray(se.x, np.float32), se.mask, ARM_MICRO_SHUFFLED)
    return shuffled[: cfg.seq_len], raw[: cfg.seq_len]


# ------------------------------------------------------------------ DISCRIMINATION FIRST
def test_the_fixture_can_tell_the_two_windows_apart():
    """If the permutation left the first window alone, every assertion below would pass against
    the broken code as well — the fixture-discrimination rule."""
    cfg = _cfg()
    shuffled, raw = _windows(_sym(), cfg)
    assert shuffled.shape == raw.shape
    assert not np.allclose(shuffled, raw), "the shuffle does not touch the first window"
    # ...and it must differ in the MICRO dims only: the OHLCV columns are byte-identical across
    # arms, which is the premise the C-12 like-for-like comparison rests on.
    n_ohlcv = 7
    np.testing.assert_array_equal(shuffled[:, :n_ohlcv], raw[:, :n_ohlcv])
    assert not np.allclose(shuffled[:, n_ohlcv:], raw[:, n_ohlcv:])


# ------------------------------------------------------------------ the regression
@pytest.mark.slow
def test_placebo_diagnostics_are_fed_the_SHUFFLED_window(tmp_path):
    set_determinism(0)
    cfg = _cfg()
    se = _sym()
    shuffled, raw = _windows(se, cfg)
    tok = TokenizerAE(d_model=16, n_layers=1, n_heads=2, d_ff=32, max_len=64).eval()
    seen: list[np.ndarray] = []
    _spy(tok, seen)

    score_cell("cell5", _ar(), tok, ARM_MICRO_SHUFFLED, [se], cfg)

    assert seen, "no diagnostic ran — the spy proves nothing"
    fed = [a[0] if a.ndim == 3 else a for a in seen]
    assert any(np.array_equal(f, shuffled) for f in fed), (
        "the placebo diagnostics were never given the shuffled window they are supposed to "
        "characterise"
    )
    assert not any(np.array_equal(f, raw) for f in fed), (
        "the UNSHUFFLED window reached the tokenizer on the placebo arm — this is R12e: the "
        "required C-12/C-1 disclosures would describe input cell 5 was never trained or scored on"
    )


@pytest.mark.slow
def test_the_unshuffled_arm_is_unchanged_by_the_fix(tmp_path):
    """The other arms fed `select_arm(se.x, ...)` before and after, so their numbers must not
    move — the fix is scoped to the placebo, and saying so requires checking it."""
    set_determinism(0)
    cfg = _cfg()
    se = _sym()
    expected, _ = select_arm(np.asarray(se.x, np.float32), se.mask, ARM_MICRO)
    tok = TokenizerAE(d_model=16, n_layers=1, n_heads=2, d_ff=32, max_len=64).eval()
    seen: list[np.ndarray] = []
    _spy(tok, seen)

    score_cell("cell4", _ar(), tok, ARM_MICRO, [se], cfg)

    fed = [a[0] if a.ndim == 3 else a for a in seen]
    assert any(np.array_equal(f, expected[: cfg.seq_len]) for f in fed)
