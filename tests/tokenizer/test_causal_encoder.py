"""CI flip-KAT — invariant 2 extended to TOKEN CONTENT (prereg §7 v1.3, M6-scoped).

The finding this gates: with a bidirectional encoder over fixed tokenization chunks, the
token for bar t depends on bars AFTER t, and ``predict_mu`` conditions on those tokens as
eval INPUTS (measured on the real lake: 41.8 %/28.3 % coarse/fine past-token flips under
future-bar-only mutation — ``runs_manifest/m6_token_causality_probe.json``). The adjudicated
remedy is ``encoder_causal=True`` for every M6 cell; THIS test is the teeth: mutate only
future bars within a chunk → assert ZERO past-bar token flips for both quantizers, and
assert the bidirectional control DOES flip (anti-vacuity — a vacuously-passing mask test
would gate nothing). Structural property of the attention mask, so untrained shells suffice
and the test stays in the fast suite. Scope: the M6 CELL tokenizers (``build_cell_tokenizer``);
the historical M3 fixture is deliberately not gated (it reloads its own stored config and
feeds the anchored M5 instrument, not the M6 decision path)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.cells import CELLS, build_cell_tokenizer

TINY = dict(d_model=32, n_layers=2, n_heads=2, d_ff=64, max_len=64)
WINDOW, MUTATE_FROM, N_CHUNKS = 32, 20, 6


def _flip_counts(tok: TokenizerAE, n_features: int) -> tuple[int, int]:
    """Total past-bar (coarse, fine) token flips when only bars ≥ MUTATE_FROM are replaced."""
    tok.eval()
    rng = np.random.default_rng(0)
    mut = np.random.default_rng(99)
    flips_c = flips_f = 0
    for _ in range(N_CHUNKS):
        x = torch.from_numpy(rng.standard_normal((1, WINDOW, n_features)).astype(np.float32))
        m = torch.zeros(1, WINDOW, n_features)
        if n_features == 16:
            m[:, :, 13:16] = 1.0  # the lake's perp-dims-masked convention
        x2 = x.clone()
        x2[0, MUTATE_FROM:, :] = torch.from_numpy(
            mut.standard_normal((WINDOW - MUTATE_FROM, n_features)).astype(np.float32)
        )
        with torch.no_grad():
            c0, f0 = tok.encode_tokens(x, m)
            c1, f1 = tok.encode_tokens(x2, m)
        flips_c += int((c0[0, :MUTATE_FROM] != c1[0, :MUTATE_FROM]).sum())
        flips_f += int((f0[0, :MUTATE_FROM] != f1[0, :MUTATE_FROM]).sum())
    return flips_c, flips_f


@pytest.mark.parametrize("spec", [CELLS[2], CELLS[3]], ids=["bsq_cell3", "fsq_cell4"])
def test_cell_tokenizers_have_zero_future_dependence(spec):
    """Both quantizer arms, built THROUGH the cell constructor: mutating only future bars
    must change ZERO past-bar tokens, coarse and fine — the §7 v1.3 structural guarantee."""
    tok = build_cell_tokenizer(spec, **TINY)
    assert tok.get_config()["encoder_causal"] is True  # the wiring itself
    flips_c, flips_f = _flip_counts(tok, n_features=16)
    assert flips_c == 0 and flips_f == 0, (
        f"CAUSAL LEAK: past-bar tokens changed under future-only mutation "
        f"(coarse {flips_c}, fine {flips_f}) — invariant 2 binds token content"
    )


def test_bidirectional_control_does_flip():
    """Anti-vacuity: the same probe against an explicitly bidirectional encoder MUST show
    flips (untrained bidirectional measured ~23 % at discovery) — proving the zero above is
    the mask working, not a test that cannot fail."""
    tok = TokenizerAE(encoder_causal=False, **TINY).eval()
    flips_c, flips_f = _flip_counts(tok, n_features=16)
    assert flips_c + flips_f > 0, "the control should flip — the probe has lost its teeth"


def test_bidirectional_cell_is_not_constructible():
    """A caller trying to sneak encoder_causal=False into a cell is a hard error (§7 v1.3)."""
    with pytest.raises(ValueError, match="causal-encoder ONLY"):
        build_cell_tokenizer(CELLS[3], encoder_causal=False, **TINY)
