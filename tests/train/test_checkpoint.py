"""Content-hashed checkpoint save/load roundtrip (§7.7) — tokenizer + predictor.

Uses ``model.get_config()`` (the complete constructor kwargs) so a behavior-only field like the
tokenizer's ``encoder_causal`` is captured and the config-bound hash catches any reload mismatch.
"""

from __future__ import annotations

import pytest
import torch

from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.checkpoint import load_checkpoint, save_checkpoint
from trikaal.utils.seeding import set_determinism


def test_tokenizer_checkpoint_roundtrip(tmp_path):
    set_determinism(0)
    tok = TokenizerAE(d_model=32, n_layers=1, n_heads=2, d_ff=64).eval()
    h = save_checkpoint(tmp_path / "tok.pt", tok, tok.get_config())

    loaded = load_checkpoint(tmp_path / "tok.pt", TokenizerAE, expect_hash=h)
    assert loaded.hash == h
    x = torch.randn(2, 16, 16)
    mask = torch.zeros(2, 16, 16)
    with torch.no_grad():
        a = tok(x, mask)["x_hat"]
        b = loaded.model.eval()(x, mask)["x_hat"]
    assert torch.equal(a, b), "reloaded tokenizer is not bit-identical"


def test_predictor_checkpoint_roundtrip_and_wrong_hash(tmp_path):
    set_determinism(0)
    model = TrikaalAR(n_layers=1, d_model=32, d_ff=64, n_heads=2, mtp_depths=0, max_len=32).eval()
    h = save_checkpoint(tmp_path / "ar.pt", model, model.get_config())

    loaded = load_checkpoint(tmp_path / "ar.pt", TrikaalAR)
    assert loaded.hash == h
    b_c = torch.randint(0, model.v_c, (1, 12))
    b_f = torch.randint(0, model.v_f, (1, 12))
    ts = (1_600_000_000_000 + torch.arange(12).reshape(1, 12) * 60_000).long()
    with torch.no_grad():
        assert torch.equal(model(b_c, b_f, ts).logits_c, loaded.model.eval()(b_c, b_f, ts).logits_c)

    with pytest.raises(ValueError, match="not the expected artifact"):
        load_checkpoint(tmp_path / "ar.pt", TrikaalAR, expect_hash="sha256:deadbeef")


def test_config_bound_hash_catches_behavior_only_field(tmp_path):
    """A behavior-only field (encoder_causal) changes no tensor; the config-bound hash must still
    catch a config that doesn't match what was saved (the silent-reload-mismatch guard)."""
    set_determinism(0)
    tok = TokenizerAE(d_model=32, n_layers=1, n_heads=2, d_ff=64, encoder_causal=True).eval()
    save_checkpoint(tmp_path / "tok.pt", tok, tok.get_config())

    # tamper the stored config to flip the behavior-only field, then attempt to load
    ckpt = torch.load(tmp_path / "tok.pt", weights_only=False)
    assert ckpt["config"]["encoder_causal"] is True  # get_config captured it
    ckpt["config"]["encoder_causal"] = False
    torch.save(ckpt, tmp_path / "tampered.pt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(tmp_path / "tampered.pt", TokenizerAE)
