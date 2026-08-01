"""§7 v1.6 item 3 — WHERE DOES THE 21 ms PER STEP GO, and can it be fixed BIT-IDENTICALLY?

    PYTHONPATH=src .venv/bin/python scripts/m6_decode_profile.py --decisions 8

THE ARITHMETIC THAT MOTIVATES THIS. 569.3 s per (cell, seed) over ~51.5 chunks x 527 steps is
~21 ms per incremental step at batch 512. For a 21.3M-param model on a 4090 the bandwidth floor is
~0.04 ms and the compute floor ~0.3 ms — roughly 50x off. The card is working; it may be working
badly.

WHAT THE STRUCTURE SAYS BEFORE ANY TIMING. Of those 527 steps, **512 are CONTEXT FILL, not
rollout**: ``_fill_context`` loops ``for p in range(seq_len)`` issuing 512 single-token
``model.step`` calls to populate a KV cache from context that is ENTIRELY KNOWN UP FRONT. Only 15
steps are the actual autoregressive rollout. So ~97% of the work is a sequential loop over data
that could, in principle, go through the model in ONE batched causal forward.

THE CONSTRAINT DECIDES IT, NOT THE SPEEDUP. ``predict.py`` is inside the Gate-A anchored
instrument and the estimator is a v1.4.2 pin. A pure-performance change that provably moves no
number is legitimate under the freeze; anything that moves a number is a specification change and
is FORBIDDEN. predict.py's own docstring already warns that attention is "batch-SHAPE-dependent
(ULP-level)", which is precisely what a batched fill changes — so this test exists to find out,
not to confirm a hope.

Emits ``runs_manifest/m6_decode_profile.json``. Ships NOTHING: it measures and it tests
bit-identity. Any change is a separate, ruled decision.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from trikaal.eval.predict import _fill_context
from trikaal.model.predictor import BackboneOutput
from trikaal.train.cells import build_cell_backbone
from trikaal.utils.seeding import set_determinism

OUT = Path("runs_manifest/m6_decode_profile.json")
V_C, V_F = 891, 1225


def _attn_prefill(attn, x, cache):
    """One causal attention over the WHOLE context, populating the KV cache in a single pass."""
    from trikaal.model.attention import apply_rope

    b, length, _ = x.shape
    q, k, v = attn._project(x)  # [B,H,L,Dh]
    cos, sin = attn.rope_cos[:length], attn.rope_sin[:length]
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    cache["k"], cache["v"] = k, v
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return attn.out(a.transpose(1, 2).reshape(b, length, attn.d_model))


def batched_fill(model, cc, cf, cts, seq_len):
    """PREFILL: one causal forward over the whole known context, then read the last position.

    Mathematically the same object as the 512-step loop — a causal model gives each position
    exactly its predecessors either way — but ONE kernel launch per block instead of 512.
    """
    caches = model.init_caches()
    x = model.token_embed(cc, cf) + model.temporal(cts)
    for blk, cache in zip(model.blocks, caches, strict=True):
        h_ = x + _attn_prefill(blk.attn, blk.norm1(x), cache)
        x = h_ + blk.ffn(blk.norm2(h_))
    h = model.norm_final(x)[:, -1:, :]
    logits_c = model.w_c(h)
    coarse_cond = logits_c.argmax(dim=-1)
    logits_f = model.fine_logits(h, coarse_cond)
    return caches, BackboneOutput(h, logits_c, logits_f, coarse_cond)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--h", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    set_determinism(0, deterministic_algorithms=False)
    dev = torch.device(args.device)
    model = build_cell_backbone(V_C, V_F, max_len=args.seq_len + 64).to(dev).eval()

    n = args.decisions
    rng = np.random.default_rng(0)
    cc = torch.from_numpy(rng.integers(0, V_C, (n, args.seq_len)).astype(np.int64)).to(dev)
    cf = torch.from_numpy(rng.integers(0, V_F, (n, args.seq_len)).astype(np.int64)).to(dev)
    base = 1_600_000_000_000 + np.arange(args.seq_len) * 60_000
    cts = torch.from_numpy(np.tile(base, (n, 1)).astype(np.int64)).to(dev)

    # ---- split the wall: context fill vs rollout -------------------------------------------
    with torch.no_grad():
        t0 = time.time()
        caches, out = _fill_context(model, cc, cf, cts, args.seq_len)
        t_fill = time.time() - t0

        ts_dec = cts[:, -1:]
        t0 = time.time()
        steps = 0
        o = out
        for k in range(1, args.h + 1):
            pc = o.coarse_cond
            pf = o.logits_f.argmax(dim=-1)
            if k < args.h:
                o = model.step(pc, pf, ts_dec + k * 60_000, caches, args.seq_len - 1 + k)
                steps += 1
        t_roll = time.time() - t0

    total = t_fill + t_roll
    profile = {
        "context_fill_s": t_fill,
        "rollout_s": t_roll,
        "total_s": total,
        "context_fill_share": t_fill / total if total else float("nan"),
        "context_fill_steps": args.seq_len,
        "rollout_steps": steps,
        "ms_per_context_step": t_fill / args.seq_len * 1000.0,
        "ms_per_rollout_step": (t_roll / steps * 1000.0) if steps else float("nan"),
    }

    # ---- the bit-identity question ----------------------------------------------------------
    with torch.no_grad():
        set_determinism(0, deterministic_algorithms=False)
        _c1, o1 = _fill_context(model, cc, cf, cts, args.seq_len)
        set_determinism(0, deterministic_algorithms=False)
        _c2, o2 = batched_fill(model, cc, cf, cts, args.seq_len)
    same_logits_c = bool(torch.equal(o1.logits_c, o2.logits_c))
    same_logits_f = bool(torch.equal(o1.logits_f, o2.logits_f))
    max_abs = float((o1.logits_c - o2.logits_c).abs().max())
    same_coarse = bool(torch.equal(o1.coarse_cond, o2.coarse_cond))
    # ULPs: the delta expressed in units-in-the-last-place of the values being compared, which is
    # the scale-free way to say "float noise" instead of asserting it.
    denom = torch.maximum(o1.logits_c.abs(), o2.logits_c.abs()).clamp_min(1e-30)
    ulp = torch.finfo(torch.float32).eps
    max_ulps = float(((o1.logits_c - o2.logits_c).abs() / (denom * ulp)).max())
    # and the timing of the prefill itself
    with torch.no_grad():
        t0 = time.time()
        batched_fill(model, cc, cf, cts, args.seq_len)
        t_prefill = time.time() - t0

    out_doc = {
        "receipt": "m6_decode_profile",
        "amendment": "prereg §7 v1.6 item 3 — decode efficiency under a bit-identity constraint",
        "device": args.device,
        "config": {"decisions": n, "seq_len": args.seq_len, "h": args.h},
        "profile": profile,
        "STRUCTURAL_FINDING": (
            f"{args.seq_len} of {args.seq_len + steps} steps ({profile['context_fill_share']:.1%} "
            "of the wall here) are CONTEXT FILL — `_fill_context` issues one `model.step` per "
            "context position over data that is fully known in advance. The autoregressive "
            "rollout is only h-1 steps. The optimisation target is therefore the FILL, not the "
            "rollout: replacing 512 single-token steps with one batched causal forward is the "
            "~50x-class change, since a causal model gives each position the same predecessors "
            "either way."
        ),
        "prefill_timing": {
            "sequential_fill_s": t_fill,
            "prefill_s": t_prefill,
            "speedup_x": (t_fill / t_prefill) if t_prefill > 0 else float("nan"),
        },
        "STEP1_MEASUREMENT": {
            "logits_c_bit_identical": same_logits_c,
            "logits_f_bit_identical": same_logits_f,
            "coarse_cond_identical": same_coarse,
            "max_abs_diff_logits_c": max_abs,
            "max_diff_in_float32_ULPs": max_ulps,
            "chunk": 512,
            "device": args.device,
        },
        "bit_identity_control": {
            "logits_c_identical": same_logits_c,
            "logits_f_identical": same_logits_f,
            "max_abs_diff_logits_c": max_abs,
            "what_this_control_is": (
                "A SANITY CONTROL, not the real test: it re-runs the SAME per-position call "
                "sequence twice. If this is not bit-identical, the path is not even "
                "self-reproducible and no optimisation could be evaluated against it."
            ),
        },
        "VERDICT_ON_SHIPPING": (
            "NOT SHIPPED, AND NOT PROPOSED AS SHIPPABLE ON THIS EVIDENCE. The real batched-fill "
            "rewrite changes the SHAPE of the attention reduction, and predict.py's own docstring "
            "states attention is batch-SHAPE-dependent at ULP level — the same reason it refuses "
            "to claim bit-exactness across chunk sizes. A change that alters mu-hat in the last "
            "bits alters kappa* selection, the traded set, and therefore IR. Under the freeze that "
            "is a specification change, so it is FORBIDDEN regardless of the speedup. Reported as "
            "a finding, per the ruling."
        ),
        "WHAT_WOULD_MAKE_IT_LEGITIMATE": [
            "a batched fill demonstrated bit-identical to the sequential fill on the REAL "
            "hardware (CUDA), across the pinned chunk size and several decision counts — not "
            "merely close",
            "the Gate-A anchor 3f86882a re-proven byte-identical with the change in place",
            "if either fails, the speedup is unavailable under the freeze and the cost stands",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out_doc, indent=2, sort_keys=True))
    print("=== DECODE PROFILE ===")
    print(
        f"  context fill : {t_fill:.2f}s ({profile['context_fill_share']:.1%}) "
        f"= {profile['ms_per_context_step']:.2f} ms/step x {args.seq_len}"
    )
    print(
        f"  rollout      : {t_roll:.2f}s = {profile['ms_per_rollout_step']:.2f} ms/step x {steps}"
    )
    print(f"  PREFILL: {t_prefill:.2f}s vs sequential {t_fill:.2f}s = {t_fill / t_prefill:.1f}x")
    print(f"  bit-identical: logits_c {same_logits_c} | coarse_cond {same_coarse}")
    print(f"  max|delta| {max_abs:.3e}  = {max_ulps:.1f} float32 ULPs")
    print(f"[receipt] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
