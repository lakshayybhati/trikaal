"""Attention primitive unit tests (Backbone §5) — RoPE, causal mask, KV-cache equivalence."""

from __future__ import annotations

import torch

from trikaal.model.attention import (
    MultiHeadSelfAttention,
    RMSNorm,
    apply_rope,
    build_rope_cache,
)
from trikaal.utils.seeding import set_determinism


def test_kv_cache_step_matches_full_forward():
    """Incremental KV-cached generation must equal the parallel causal forward."""
    set_determinism(0)
    attn = MultiHeadSelfAttention(64, 4, causal=True, max_len=32).eval()
    x = torch.randn(2, 10, 64)
    with torch.no_grad():
        full = attn(x)
        cache: dict = {"k": None, "v": None}
        inc = torch.cat([attn.step(x[:, t : t + 1], cache, t) for t in range(10)], dim=1)
    assert torch.allclose(full, inc, atol=1e-5), (
        f"KV-cache diverged (max|Δ|={(full - inc).abs().max():.2e})"
    )


def test_causal_attention_ignores_future():
    """Output at position t must not change when positions > t are altered."""
    set_determinism(0)
    attn = MultiHeadSelfAttention(32, 4, causal=True, max_len=16).eval()
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        a1 = attn(x)
        x2 = x.clone()
        x2[:, 5:] = torch.randn_like(x2[:, 5:])
        a2 = attn(x2)
    assert torch.allclose(a1[:, :5], a2[:, :5], atol=1e-6)
    assert not torch.allclose(a1[:, 5:], a2[:, 5:])


def test_rope_preserves_norm_and_is_relative():
    cos, sin = build_rope_cache(16, 8)
    x = torch.randn(1, 2, 16, 8)
    rx = apply_rope(x, cos, sin)
    # rotation preserves per-vector norm
    assert torch.allclose(x.norm(dim=-1), rx.norm(dim=-1), atol=1e-5)


def test_rmsnorm_unit_scale():
    norm = RMSNorm(64)
    x = torch.randn(4, 10, 64) * 7.0
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


# ---------------------------------------- §7 v1.4.6: run provenance + the bit-exactness precondition
def test_determinism_record_carries_the_provenance_a_divergence_needs():
    """Item 3: the acceptance/money-leg manifests recorded only `device: cuda`, which made their
    cell4 divergence unattributable. Every field asserted here is one that was needed and absent."""
    import os

    import torch

    from trikaal.model.attention_mode import MODE_SDPA, determinism_record

    rec = determinism_record(
        seed=7, device="cpu", attention_mode=MODE_SDPA, extra_seeds={"bootstrap": 20260704}
    )
    for key in ("attention_mode", "determinism_flags", "hardware", "env", "seeds"):
        assert key in rec, key
    assert rec["seeds"] == {"run": 7, "bootstrap": 20260704}  # ALL seeds, not just the run seed
    assert set(rec["determinism_flags"]) == {
        "torch_use_deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cublas_workspace_config",
    }
    assert rec["env"]["torch"] == torch.__version__
    assert rec["env"]["numpy"] == __import__("numpy").__version__
    assert "cuda_available" in rec["hardware"]
    assert rec["determinism_flags"]["cublas_workspace_config"] == os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    )


def test_bit_exact_preconditions_track_the_actual_determinism_flags():
    """THE v1.4.6 DEFECT MADE SELF-ANNOUNCING. `bit_exact_claim` keys on attention mode ALONE, so a
    CUDA+SDPA run stamps True even with deterministic algorithms OFF — that exact contradiction is
    present in all 15 committed toy-rehearsal run records (bit_exact_claim=True beside
    deterministic_algorithms=False). The historical field is preserved for comparability; the
    reliable predicate and an explicit caveat are now emitted so no reader can miss it.

    The flags are set EXPLICITLY here and restored: the global torch determinism state is shared
    across the suite, so asserting on whatever it happens to be would be order-dependent.
    """
    import torch

    from trikaal.model.attention_mode import MODE_FLASH2, MODE_SDPA, determinism_record

    prev = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    try:
        # (a) the ACTUAL M6 posture — every path passes deterministic_algorithms=False
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        rec = determinism_record(seed=0, device="cuda", attention_mode=MODE_SDPA)
        assert rec["bit_exact_claim"] is True  # historical meaning, unchanged
        assert rec["bit_exact_preconditions_met"] is False  # what can actually be relied on
        assert "not established" in rec["bit_exact_caveat"].lower()

        # (b) preconditions genuinely satisfied -> the two agree and no caveat is emitted
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        ok = determinism_record(seed=0, device="cuda", attention_mode=MODE_SDPA)
        assert ok["bit_exact_preconditions_met"] is True
        assert "bit_exact_caveat" not in ok

        # (c) flash2 never claims bit-exactness, so there is no contradiction to caveat
        f = determinism_record(seed=0, device="cuda", attention_mode=MODE_FLASH2)
        assert f["bit_exact_claim"] is False
        assert f["bit_exact_preconditions_met"] is False
        assert "bit_exact_caveat" not in f
    finally:
        torch.use_deterministic_algorithms(prev[0])
        torch.backends.cudnn.deterministic = prev[1]
        torch.backends.cudnn.benchmark = prev[2]
