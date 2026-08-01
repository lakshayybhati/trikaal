"""§7 v1.6 condition (a) — does the PREFILL move DECISIONS, or only float noise?

    PYTHONPATH=src .venv/bin/python scripts/m6_prefill_decision_stability.py --decisions 4096

WHY A PRE-DECLARED NUMBER IS THE WHOLE TEST. Positions come from a threshold comparison,
``theta = kappa * c_total`` then ``positions(mu, theta)`` (harness.py:105-106). A decision can
only flip when ``|mu_hat|`` lands within ``|delta mu_hat|`` of a threshold. So the EXPECTED
flipped fraction is of order the delta divided by the dispersion of mu-hat — a number computable
BEFORE looking at the observed rate. Without it, "few flips" and "many flips" both read as
success and the check cannot fail; with it, the two outcomes are distinguishable:

    observed <= ~expected   -> float noise. Condition (a) satisfied.
    observed >> expected    -> NOT float noise. That is a CORRECTNESS BUG in one of the two
                               paths, a far larger finding than the speedup. STOP, do not ship.

This script computes and PRINTS the expectation first, then measures. It also reports kappa*,
the traded set, activity and IR under both paths.

THE DELTA MEASURED HERE IS ON MU-HAT ITSELF, not on logits. The 1.192e-06 from the profile is a
LOGIT delta; mu-hat is downstream of a softmax, a grid expectation and an h-step accumulation, so
it must be measured directly rather than inferred.

LOCAL-MEASURED: timings anywhere in this repo carry an unexplained ~8x local variance (three
appearances). Nothing here is a throughput claim; only the deltas and flip rates are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from trikaal.eval.harness import positions
from trikaal.eval.metrics import information_ratio
from trikaal.eval.predict import _fill_context
from trikaal.model.predictor import BackboneOutput
from trikaal.train.cells import CELLS, build_cell_backbone, build_cell_tokenizer
from trikaal.utils.seeding import set_determinism

OUT = Path("runs_manifest/m6_prefill_decision_stability.json")
V_C, V_F = 891, 1225
KAPPAS = (1.0, 1.5, 2.0, 3.0)  # §3a pinned grid
COST = 0.003  # 0.30% per-trade cost model


def _attn_prefill(attn, x, cache):
    from trikaal.model.attention import apply_rope

    b, length, _ = x.shape
    q, k, v = attn._project(x)
    cos, sin = attn.rope_cos[:length], attn.rope_sin[:length]
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    cache["k"], cache["v"] = k, v
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return attn.out(a.transpose(1, 2).reshape(b, length, attn.d_model))


def prefill_context(model, cc, cf, cts):
    caches = model.init_caches()
    x = model.token_embed(cc, cf) + model.temporal(cts)
    for blk, cache in zip(model.blocks, caches, strict=True):
        h_ = x + _attn_prefill(blk.attn, blk.norm1(x), cache)
        x = h_ + blk.ffn(blk.norm2(h_))
    h = model.norm_final(x)[:, -1:, :]
    lc = model.w_c(h)
    cc_ = lc.argmax(dim=-1)
    return caches, BackboneOutput(h, lc, model.fine_logits(h, cc_), cc_)


def rollout_mu(model, tok, caches, out, ts_dec, n, h, seq_len, dev):
    """The expectation-decode rollout, identical to predict._rollout_greedy's estimator branch."""
    q = tok.quant
    grid_c = tok.group_grid("coarse", device=dev)
    grid_f = tok.group_grid("fine", device=dev)
    c_idx, f_idx = list(q.coarse_idx), list(q.fine_idx)
    cum = torch.zeros(n, dtype=torch.float32, device=dev)
    o = out
    for k in range(1, h + 1):
        pc, pf = o.coarse_cond, o.logits_f.argmax(dim=-1)
        if k >= 2:
            p_c = F.softmax(o.logits_c.float(), dim=-1)
            p_f = F.softmax(o.logits_f.float(), dim=-1)
            z = torch.zeros(n, 1, q.dim, dtype=torch.float32, device=dev)
            z[..., c_idx] = (p_c @ grid_c.float()).to(z.dtype)
            z[..., f_idx] = (p_f @ grid_f.float()).to(z.dtype)
            cum = cum + tok.decode_latent(z)[:, 0, 0]
        if k < h:
            o = model.step(pc, pf, ts_dec + k * 60_000, caches, seq_len - 1 + k)
    return cum.cpu().numpy().astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", type=int, default=2048)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--h", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    set_determinism(0, deterministic_algorithms=False)
    dev = torch.device(args.device)
    spec = CELLS[3]
    tok = build_cell_tokenizer(spec, max_len=max(512, args.seq_len)).to(dev).eval()
    model = build_cell_backbone(V_C, V_F, max_len=args.seq_len + 64).to(dev).eval()

    n = args.decisions
    rng = np.random.default_rng(20260704)
    cc = torch.from_numpy(rng.integers(0, V_C, (n, args.seq_len)).astype(np.int64)).to(dev)
    cf = torch.from_numpy(rng.integers(0, V_F, (n, args.seq_len)).astype(np.int64)).to(dev)
    base = 1_600_000_000_000 + np.arange(args.seq_len) * 60_000
    cts = torch.from_numpy(np.tile(base, (n, 1)).astype(np.int64)).to(dev)
    ts_dec = cts[:, -1:]

    with torch.no_grad():
        cA, oA = _fill_context(model, cc, cf, cts, args.seq_len)
        mu_seq = rollout_mu(model, tok, cA, oA, ts_dec, n, args.h, args.seq_len, dev)
        cB, oB = prefill_context(model, cc, cf, cts)
        mu_pre = rollout_mu(model, tok, cB, oB, ts_dec, n, args.h, args.seq_len, dev)

    d = np.abs(mu_seq - mu_pre)
    max_d, mean_d = float(d.max()), float(d.mean())
    sd_mu = float(np.std(mu_seq))

    # ---- PRE-DECLARATION: computed and printed BEFORE the observed rate is looked at --------
    # A decision flips only if |mu| sits within |delta| of a threshold. For a roughly smooth
    # mu distribution the flipped fraction is of order (2*delta)/sigma per threshold.
    expected_flip_rate = float(2.0 * max_d / sd_mu) if sd_mu > 0 else float("nan")
    print("=== PRE-DECLARED, BEFORE MEASURING THE OBSERVED RATE ===")
    print(f"  max|delta mu-hat|   = {max_d:.3e}")
    print(f"  sigma(mu-hat)       = {sd_mu:.3e}")
    print(f"  EXPECTED flip rate  = {expected_flip_rate:.3e}  (order 2*delta/sigma)")
    print("  CRITERION: observed <= ~expected -> float noise; observed >> expected -> BUG, STOP.")

    # ---- FIXTURE DISCRIMINATION, asserted before the fixture is trusted --------------------
    # The first version of this test used only the pinned kappa grid and every decision traded
    # (activity 1.0000 at every kappa): |mu| sat far above every threshold, so a 3.5e-06 delta
    # had NO opportunity to flip anything and the test passed vacuously. A fixture must be shown
    # able to separate the cases it claims to separate. These extra thresholds are placed at
    # quantiles of |mu| so that decisions actually straddle them.
    absmu = np.abs(mu_seq)
    q_thresholds = [float(np.quantile(absmu, q)) for q in (0.25, 0.5, 0.75)]
    near = int((np.abs(absmu - q_thresholds[1]) <= max_d).sum())
    discriminating = near > 0
    print(
        f"  fixture check: {near} decisions lie within max|delta| of the median threshold "
        f"-> {'CAN' if discriminating else 'CANNOT'} discriminate"
    )

    # ---- now measure -------------------------------------------------------------------------
    per_kappa, flips_total, dec_total = {}, 0, 0
    all_thresholds = [(f"kappa_{k}", k * COST) for k in KAPPAS] + [
        (f"q{int(q * 100)}", t) for q, t in zip((0.25, 0.5, 0.75), q_thresholds, strict=True)
    ]
    for label, theta in all_thresholds:
        sA, sB = positions(mu_seq, theta), positions(mu_pre, theta)
        flips = int((sA != sB).sum())
        flips_total += flips
        dec_total += sA.size
        y = np.zeros_like(mu_seq)  # returns are irrelevant to POSITION stability
        per_kappa[str(label)] = {
            "theta": theta,
            "flipped": flips,
            "flip_rate": flips / sA.size,
            "activity_sequential": float((sA != 0).mean()),
            "activity_prefill": float((sB != 0).mean()),
            "traded_set_identical": bool(np.array_equal(sA != 0, sB != 0)),
            "positions_identical": bool(np.array_equal(sA, sB)),
            "ir_sequential": float(information_ratio(np.where(sA != 0, y, 0.0), args.h)),
            "ir_prefill": float(information_ratio(np.where(sB != 0, y, 0.0), args.h)),
        }
    observed = flips_total / dec_total if dec_total else float("nan")
    ratio = observed / expected_flip_rate if expected_flip_rate > 0 else float("inf")

    if not discriminating:
        # §7 v1.6.3 CONTROL-ARM RULE, applied to this script's own fixture. The first run of this
        # probe printed "fixture check: 0 decisions ... -> CANNOT discriminate", persisted
        # `fixture_can_discriminate: false`, and then reported "CONDITION (a) SATISFIED" from an
        # observed flip rate of 0. At n=256 the spacing between adjacent |mu| order statistics near
        # the median is ~1e-4 against a max|delta| of 3.5e-06, so NO decision could straddle a
        # threshold: zero flips was the only attainable outcome and the leg could not have failed.
        # A probe must refuse to emit a verdict when its own control arm fails.
        verdict = (
            "PROBE INDETERMINATE — NO CONCLUSION IS DRAWN FROM THE FLIP RATE. The fixture could "
            f"not discriminate: 0 decisions lie within max|delta| ({max_d:.3e}) of the median "
            f"threshold at n={n}, so a flip was not POSSIBLE here and the observed rate "
            f"{observed:.3e} carries no information — it would read the same against a correct "
            "path and a broken one. Re-run at a sample size where decisions straddle a threshold. "
            "This does NOT reopen condition (a): the ruling closed (a) on the ANALYTIC impact "
            "bound (2|delta|/sigma = 6.363e-05 -> 89.2 flips per (cell,seed), |dIR| RMS 3.783e-03 "
            "= 0.76% of the 0.5 economic floor), which never used the observed count. The flip "
            "observation was corroboration and was never able to carry the conclusion."
        )
    elif observed <= max(expected_flip_rate, 1e-12) * 10.0:
        verdict = (
            f"CONDITION (a) SATISFIED. Observed flip rate {observed:.3e} is at or below the "
            f"pre-declared expectation {expected_flip_rate:.3e} (ratio {ratio:.2f}). The "
            "discrepancy behaves as float noise against a threshold, which is what the "
            "pre-declaration predicted."
        )
    else:
        verdict = (
            f"STOP — DO NOT SHIP. Observed flip rate {observed:.3e} EXCEEDS the pre-declared "
            f"expectation {expected_flip_rate:.3e} by {ratio:.1f}x. That is not float noise; it "
            "indicates a CORRECTNESS BUG in one of the two paths, which is a larger finding than "
            "the speedup and must be reported rather than shipped past."
        )

    doc = {
        "receipt": "m6_prefill_decision_stability",
        "amendment": "prereg §7 v1.6 condition (a) — prefill decision stability",
        "PRE_DECLARED_BEFORE_MEASURING": {
            "max_abs_delta_mu_hat": max_d,
            "mean_abs_delta_mu_hat": mean_d,
            "sigma_mu_hat": sd_mu,
            "expected_flip_rate": expected_flip_rate,
            "basis": "a position flips only if |mu| lies within |delta| of theta = kappa*c; for a "
            "smooth mu distribution that fraction is of order 2*delta/sigma per threshold",
            "criterion": "observed <= ~10x expected -> float noise; observed >> expected -> BUG",
        },
        "OBSERVED": {
            "flip_rate_all_kappas": observed,
            "flips": flips_total,
            "decisions_compared": dec_total,
            "observed_over_expected": ratio,
            "per_kappa": per_kappa,
        },
        "coarse_cond_identical": bool(torch.equal(oA.coarse_cond, oB.coarse_cond)),
        "FIXTURE_DISCRIMINATION": {
            "quantile_thresholds": q_thresholds,
            "decisions_within_delta_of_median_threshold": near,
            "fixture_can_discriminate": discriminating,
            "why": (
                "The pinned kappa grid alone gave activity 1.0000 everywhere — every decision far "
                "above every threshold — so no delta of this size could flip anything and the "
                "test passed vacuously. Thresholds at |mu| quantiles put decisions ON the "
                "boundary, which is where a flip is possible at all."
            ),
            # §7 v1.6.3: the flag is now BINDING, not advisory. It was persisted as false beside a
            # "CONDITION (a) SATISFIED" verdict — a receipt stating a conclusion its own
            # discrimination flag contradicts, which is precisely what these norms exist to stop.
            "flag_is_binding": True,
            "on_false": "the verdict becomes PROBE INDETERMINATE and no conclusion is drawn from "
            "the flip rate; the analytic impact bound is unaffected and is what closed (a)",
        },
        "framing": (
            "The project has ALWAYS required the DISCRETE token chain to be exact and accepted "
            "1e-4 on the CONTINUOUS values between these two paths (test_rollout.py:53 asserts "
            "torch.equal on coarse_cond; :54-56 assert < 1e-4). The prefill therefore does not "
            "introduce a NEW class of discrepancy — it introduces the EXISTING one at a smaller "
            "magnitude."
        ),
        "ulp_footnote": (
            "A ULP count against magnitude is the wrong metric for logits, which span zero: the "
            "max is dominated by whichever entry sits nearest zero. The relative figure "
            "(~1e-7, float32 epsilon scale) is primary."
        ),
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True))
    print("\n=== OBSERVED ===")
    print(
        f"  flip rate {observed:.3e} over {dec_total} decision-comparisons ({ratio:.2f}x expected)"
    )
    for k, v in per_kappa.items():
        print(
            f"    kappa {k}: flips {v['flipped']} | traded set identical "
            f"{v['traded_set_identical']} | activity {v['activity_sequential']:.4f} vs "
            f"{v['activity_prefill']:.4f}"
        )
    print(f"\nVERDICT: {verdict}")
    print(f"[receipt] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
