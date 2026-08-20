"""A STAGE-1 DEFAULT ON A STAGE-2 FUNCTION, WHERE THE DOCS AGREED WITH THE SIGNATURE AND BOTH
DISAGREED WITH THE SHIPPED ARTIFACT.

``tokenize_features(window=128)`` and ``XSectionConfig.seq_len = 128`` carried the M3-era Stage-1
TOKENIZER training length. The shipped tokenizer's own ``max_len`` is **512**, the money surface is
pinned at ``conformance.PINNED_MONEY_SEQ_LEN = 512``, and every call site in the repository passes
512 — so the default could only ever be taken by a caller who did not know it was there, and
taking it silently ran at a QUARTER of the shipped context. Nothing errors: the token ids are
valid, the stream content-hashes cleanly, and the numbers are simply built on less context than
the model was trained for.

That is the hardest shape to catch, because the signature and the docstring corroborated each
other. The only thing that disagreed was the checkpoint.

``window`` is now REQUIRED — there is no correct default, so there is none — and the eval config
defaults to the pinned 512. The literal in ``xsection`` is spelled out rather than imported
because ``conformance`` imports ``xsection``; this file is what stops the two drifting.
"""

from __future__ import annotations

import inspect

import pytest

from trikaal.eval.conformance import PINNED_MONEY_SEQ_LEN
from trikaal.eval.xsection import XSectionConfig
from trikaal.train.token_stream import tokenize_features

STAGE1_LEGACY = 128


def test_the_eval_config_defaults_to_the_pinned_money_seq_len() -> None:
    assert XSectionConfig().seq_len == PINNED_MONEY_SEQ_LEN == 512


def test_the_eval_config_no_longer_defaults_to_the_stage1_length() -> None:
    assert XSectionConfig().seq_len != STAGE1_LEGACY, (
        "seq_len is back to the Stage-1 tokenizer length on a Stage-2 config"
    )


def test_tokenize_features_has_no_window_default_at_all() -> None:
    """Required, not merely corrected: a wrong default and a stale-but-plausible default fail the
    same way, so the parameter is made an explicit decision."""
    param = inspect.signature(tokenize_features).parameters["window"]
    assert param.default is inspect.Parameter.empty, (
        f"window has a default again ({param.default!r}) — there is no safe value for it"
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_omitting_the_window_raises_rather_than_guessing() -> None:
    with pytest.raises(TypeError, match="window"):
        tokenize_features(None, None, None, None)  # type: ignore[arg-type]


def test_the_shipped_tokenizer_length_is_what_the_default_now_matches() -> None:
    """Binds the constant to the ARTIFACT, not to another constant. Skips only where the shipped
    checkpoint is absent (it lives under gitignored runs_cloud/), and says so rather than passing
    quietly — the value it would assert is stated in the message either way."""
    import json
    from pathlib import Path

    rel = Path(__file__).resolve().parents[1] / "runs_manifest/m6_weights_release.json"
    man = json.loads(rel.read_text())
    unit = Path(__file__).resolve().parents[1] / man["units"]["0"]["source_path"] / "tokenizer.pt"
    if not unit.is_file():
        pytest.skip(f"shipped tokenizer absent at {unit}; the asserted value is max_len == 512")
    import torch

    cfg = torch.load(unit, map_location="cpu", weights_only=False)["config"]
    assert cfg["max_len"] == PINNED_MONEY_SEQ_LEN == XSectionConfig().seq_len
