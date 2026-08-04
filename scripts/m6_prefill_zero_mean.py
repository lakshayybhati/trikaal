"""§7 v1.6 condition (a), THE ATTACHED CONDITION — is the prefill perturbation ZERO-MEAN?

    PYTHONPATH=src .venv/bin/python scripts/m6_prefill_zero_mean.py --decisions 4096

WHY THIS IS NOT A FORMALITY. The impact bound closed condition (a) on an RMS yardstick: |dIR|
3.783e-03 rather than the worst case 3.592e-02. That factor is exactly sqrt(89.2449) = 9.447 — the
flip count PER (cell, seed) — and it is bought entirely by assuming flips carry INDEPENDENT signs.
A perturbation with a preferred direction makes them accumulate LINEARLY and the RMS figure is the
wrong number. So the conditional attached to the ruling is load-bearing.

THE OPERATIVE STATISTIC IS THE FLIP DIRECTION, AND IT IS EXACT. A decision flips at threshold
theta iff theta lies between |mu_seq| and |mu_pre|. WHICH WAY it flips is fixed by the sign of
(|mu_pre| - |mu_seq|) alone, independent of where theta sits:

    |mu_pre| > |mu_seq|  ->  the prefill TRADES where sequential was flat   (ENTER)
    |mu_pre| < |mu_seq|  ->  the prefill is FLAT where sequential traded    (EXIT)

So direction is measurable on EVERY decision, not only the rare ones that straddle a threshold.
That matters, because the flip RATE is not.

TWO DEFECTS FOUND WHILE WRITING THIS, BOTH DISCLOSED RATHER THAN QUIETLY FIXED.

(1) THE RECEIPT THIS CONDITION IS ATTACHED TO HAD A VACUOUS LEG. The n=256
``m6_prefill_decision_stability`` receipt recorded ``fixture_can_discriminate: false`` —
``decisions_within_delta_of_median_threshold`` was 0 — and then reported "CONDITION (a)
SATISFIED" from an observed flip rate of 0. At n=256 the spacing between adjacent |mu| order
statistics near the median is ~1e-4 against a max|delta| of 3.5e-06: no decision COULD straddle a
threshold, so zero flips was the only attainable outcome. That leg could not have failed. It is
re-run here at a size where flips are possible and the count is reported whatever it is. The impact
bound is untouched by this — it is analytic in 2|delta|/sigma and never used the observed count.

(2) MY OWN FIRST CRITERION WAS WRONG BY 5x, AND ITS CONTROL CAUGHT IT. The first draft set the
allowed imbalance at 1/sqrt(2231) using the flip count POOLED over all 25 (cell, seed) units. But
IR is computed PER UNIT, so the count that enters each IR is 89.2449 and the correct half-width is
1/sqrt(89.2449) = 0.10585. The error surfaced because the sign-randomised control FAILED at n=256:
the criterion was finer than the test's own resolution. That is the control-arm rule doing its job
— the run emitted PROBE INVALID and no conclusion about the real perturbation.

PRE-DECLARED CRITERIA, computed and printed before any observed value is looked at:

  Z1 (DIRECTION — what the RMS yardstick needs). The systematic drift is |2f-1|*N per unit and the
      random drift is sqrt(N); RMS is the right yardstick only while systematic <= random, i.e.
      |2f-1| <= 1/sqrt(89.2449) = 0.10585. Evaluated at the 95% UPPER CONFIDENCE BOUND on |2f-1|,
      not the point estimate, so a wide interval fails rather than passes.
  Z2 (MEAN). |mean(delta)| <= 3 * SE(delta): the signed mean is not distinguishable from zero.

Either failing is a REPORT, not a workaround: condition (a) reopens.

CONTROLS, asserted before the verdict is trusted (fixture-discrimination norm), and chosen so the
fixture must demonstrate sensitivity AT the decision boundary rather than only against an
implausibly large bias:
  * a full systematic shift          -> must FAIL both criteria
  * a planted imbalance at 1.5x the allowed half-width -> must FAIL Z1
  * 20 sign-randomised replicates of identical magnitude -> at least 19 must PASS Z1
A check that cannot fail is not a check.

LOCAL-MEASURED: no timing is claimed here. Only deltas, signs and counts are used, and those do not
carry the repo's ~8x local timing variance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from trikaal.eval.harness import positions
from trikaal.eval.predict import _fill_context, _fill_context_prefill
from trikaal.train.cells import CELLS, build_cell_backbone, build_cell_tokenizer
from trikaal.utils.seeding import set_determinism

OUT = Path("runs_manifest/m6_prefill_zero_mean.json")
V_C, V_F = 891, 1225
KAPPAS = (1.0, 1.5, 2.0, 3.0)  # §3a pinned grid
COST = 0.003  # 0.30% per-trade cost model

# ---- from runs_manifest/m6_prefill_impact_bound.json, reconstructed and verified ---------------
N_FLIPS_PER_CELL_SEED = 89.2448928  # the count that enters ONE IR — the relevant N
DIR_WORST_ALIGNED = 0.035921772750351424  # |dIR| with every flip aligned, i.e. |2f-1| = 1
DIR_RMS_INDEPENDENT = 0.003783  # |dIR| with independent signs
ECONOMIC_FLOOR = 0.5
FIXTURE_SEED = 20260704  # identical to m6_prefill_decision_stability, so the fixtures nest
Z95 = 1.959963984540054


def _two_sided_p(z: float) -> float:
    """Normal two-sided tail. Self-written: eval statistics are not imported (CLAUDE.md)."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def rollout_mu(model, tok, caches, out, ts_dec, n, h, seq_len, dev):
    """The expectation-decode rollout — the estimator branch of ``predict._rollout_greedy``."""
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


def direction_stats(mu_seq: np.ndarray, mu_pre: np.ndarray) -> dict:
    """Signed-mean and enter/exit balance. Direction is fixed by sign(|pre| - |seq|), exactly."""
    delta = mu_pre - mu_seq
    n = delta.size
    mean_d = float(delta.mean())
    std_d = float(delta.std(ddof=1)) if n > 1 else float("nan")
    se_d = std_d / math.sqrt(n) if n > 1 else float("nan")
    t_d = mean_d / se_d if se_d > 0 else float("nan")

    d_abs = np.abs(mu_pre) - np.abs(mu_seq)
    n_enter = int((d_abs > 0).sum())
    n_exit = int((d_abs < 0).sum())
    n_tie = int((d_abs == 0).sum())
    n_dir = n_enter + n_exit
    f = n_enter / n_dir if n_dir else float("nan")
    imbalance = abs(2 * f - 1) if math.isfinite(f) else float("nan")
    se_imb = 1.0 / math.sqrt(n_dir) if n_dir else float("nan")  # SE of (2f-1) at f~0.5
    # 95% UPPER bound on the true |2f-1|, clipped at 0 — a wide interval must fail, not pass
    upper = imbalance + Z95 * se_imb if math.isfinite(imbalance) else float("nan")
    z_bal = (n_enter - n_dir / 2.0) / (0.5 * math.sqrt(n_dir)) if n_dir else float("nan")

    w_enter = float(np.clip(d_abs, 0, None).sum())
    w_exit = float(np.clip(-d_abs, 0, None).sum())

    return {
        "n_decisions": n,
        "signed_mean_delta_mu_hat": mean_d,
        "std_delta_mu_hat": std_d,
        "se_mean_delta": se_d,
        "t_stat_mean_vs_zero": t_d,
        "p_two_sided_mean": _two_sided_p(t_d) if math.isfinite(t_d) else float("nan"),
        "mean_abs_delta_mu_hat": float(np.abs(delta).mean()),
        "max_abs_delta_mu_hat": float(np.abs(delta).max()),
        "frac_delta_positive": float((delta > 0).mean()),
        "n_enter": n_enter,
        "n_exit": n_exit,
        "n_tie": n_tie,
        "enter_fraction_f": f,
        "imbalance_abs_2f_minus_1": imbalance,
        "se_imbalance": se_imb,
        "imbalance_upper95": upper,
        "z_balance_vs_half": z_bal,
        "p_two_sided_balance": _two_sided_p(z_bal) if math.isfinite(z_bal) else float("nan"),
        "width_weighted_enter_fraction": (
            w_enter / (w_enter + w_exit) if (w_enter + w_exit) > 0 else float("nan")
        ),
    }


def verdict_for(st: dict, half_width: float) -> dict:
    z1 = bool(math.isfinite(st["imbalance_upper95"]) and st["imbalance_upper95"] <= half_width)
    z2 = bool(math.isfinite(st["t_stat_mean_vs_zero"]) and abs(st["t_stat_mean_vs_zero"]) <= 3.0)
    return {"Z1_direction_balanced": z1, "Z2_mean_indistinguishable_from_zero": z2}


def plant_imbalance(mu_seq: np.ndarray, delta: np.ndarray, g: float, seed: int) -> np.ndarray:
    """Perturbation of the SAME magnitude as the real one but with a planted enter-fraction."""
    r = np.random.default_rng(seed)
    s = np.where(r.random(mu_seq.size) < (1.0 + g) / 2.0, 1.0, -1.0)
    return mu_seq + np.abs(delta) * s * np.sign(mu_seq)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=256, help="decisions per forward; memory only")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--h", type=int, default=15)
    ap.add_argument("--thresholds", type=int, default=200, help="quantile thresholds for the rate")
    ap.add_argument("--replicates", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    half_width = 1.0 / math.sqrt(N_FLIPS_PER_CELL_SEED)
    print("=== PRE-DECLARED, BEFORE ANY OBSERVED VALUE IS LOOKED AT ===")
    print(
        f"  Z1 DIRECTION : upper95(|2f-1|) <= 1/sqrt({N_FLIPS_PER_CELL_SEED:.4f}) "
        f"= {half_width:.5f}"
    )
    print("               N is the flip count PER (cell,seed) because IR is computed per unit;")
    print("               the pooled 2231 would be a 5x-too-strict criterion (first draft, fixed).")
    print("  Z2 MEAN      : |mean(delta)| <= 3 * SE(delta)  i.e. |t| <= 3")
    print("  EITHER FAILING -> STOP AND REPORT. Condition (a) reopens; no workaround.")
    print(
        f"  resolution at n={args.decisions}: SE(|2f-1|) = {1 / math.sqrt(args.decisions):.5f}, "
        f"so a balanced perturbation clears the bound with margin "
        f"{half_width / (Z95 / math.sqrt(args.decisions)):.2f}x"
    )

    set_determinism(0, deterministic_algorithms=False)
    dev = torch.device(args.device)
    spec = CELLS[3]
    tok = build_cell_tokenizer(spec, max_len=max(512, args.seq_len)).to(dev).eval()
    model = build_cell_backbone(V_C, V_F, max_len=args.seq_len + 64).to(dev).eval()

    n = args.decisions
    # Drawn in ONE call with the stability script's seed, then sliced: the first 256 decisions are
    # bit-for-bit the earlier fixture, so this run NESTS it rather than replacing it.
    rng = np.random.default_rng(FIXTURE_SEED)
    cc_all = rng.integers(0, V_C, (n, args.seq_len)).astype(np.int64)
    cf_all = rng.integers(0, V_F, (n, args.seq_len)).astype(np.int64)
    base = 1_600_000_000_000 + np.arange(args.seq_len) * 60_000
    cts_all = np.tile(base, (n, 1)).astype(np.int64)

    mu_seq = np.empty(n, dtype=np.float64)
    mu_pre = np.empty(n, dtype=np.float64)
    with torch.no_grad():
        for lo in range(0, n, args.chunk):
            hi = min(lo + args.chunk, n)
            m = hi - lo
            cc = torch.from_numpy(cc_all[lo:hi]).to(dev)
            cf = torch.from_numpy(cf_all[lo:hi]).to(dev)
            cts = torch.from_numpy(cts_all[lo:hi]).to(dev)
            ts_dec = cts[:, -1:]
            cA, oA = _fill_context(model, cc, cf, cts, args.seq_len)
            mu_seq[lo:hi] = rollout_mu(model, tok, cA, oA, ts_dec, m, args.h, args.seq_len, dev)
            cB, oB = _fill_context_prefill(model, cc, cf, cts, args.seq_len)
            mu_pre[lo:hi] = rollout_mu(model, tok, cB, oB, ts_dec, m, args.h, args.seq_len, dev)
            print(f"  [{hi}/{n}] chunk done", flush=True)

    st = direction_stats(mu_seq, mu_pre)
    v = verdict_for(st, half_width)
    delta = mu_pre - mu_seq

    # ---- CONTROLS: show the fixture separates the cases BEFORE its verdict is trusted ----------
    st_full = direction_stats(mu_seq, mu_seq + st["mean_abs_delta_mu_hat"])
    v_full = verdict_for(st_full, half_width)
    g_edge = 1.5 * half_width
    st_edge = direction_stats(mu_seq, plant_imbalance(mu_seq, delta, g_edge, 11))
    v_edge = verdict_for(st_edge, half_width)
    reps = [
        verdict_for(
            direction_stats(mu_seq, plant_imbalance(mu_seq, delta, 0.0, 100 + i)), half_width
        )["Z1_direction_balanced"]
        for i in range(args.replicates)
    ]
    n_pass = int(sum(reps))
    controls_ok = bool(
        not v_full["Z1_direction_balanced"]
        and not v_full["Z2_mean_indistinguishable_from_zero"]
        and not v_edge["Z1_direction_balanced"]
        and n_pass >= args.replicates - 1
    )

    # ---- the flip RATE, re-measured NON-VACUOUSLY (the leg that was vacuous at n=256) ---------
    absmu = np.abs(mu_seq)
    qs = np.linspace(0.005, 0.995, args.thresholds)
    thetas = [(f"q{q:.4f}", float(np.quantile(absmu, q))) for q in qs]
    thetas += [(f"kappa_{k}", k * COST) for k in KAPPAS]
    max_d = st["max_abs_delta_mu_hat"]
    near = int(sum(int((np.abs(absmu - t) <= max_d).sum()) for _, t in thetas))
    flips_total = dec_total = flip_enter = flip_exit = 0
    per_threshold = {}
    for label, theta in thetas:
        sA, sB = positions(mu_seq, theta), positions(mu_pre, theta)
        k = int((sA != sB).sum())
        flips_total += k
        dec_total += sA.size
        if k:
            flip_enter += int(((sA == 0) & (sB != 0)).sum())
            flip_exit += int(((sA != 0) & (sB == 0)).sum())
            per_threshold[label] = {"theta": theta, "flipped": k}
    observed = flips_total / dec_total if dec_total else float("nan")
    sd_mu = float(np.std(mu_seq))
    expected = 2.0 * max_d / sd_mu if sd_mu > 0 else float("nan")
    n_fd = flip_enter + flip_exit

    # ---- propagate the measured bound to the quantity that actually matters -------------------
    sys_dir = st["imbalance_upper95"] * DIR_WORST_ALIGNED  # |2f-1| = 1 IS the aligned worst case
    propagated = {
        "systematic_dIR_at_upper95": sys_dir,
        "random_dIR_rms": DIR_RMS_INDEPENDENT,
        "systematic_over_random": sys_dir / DIR_RMS_INDEPENDENT,
        "systematic_vs_economic_floor": sys_dir / ECONOMIC_FLOOR,
        "reading": (
            "the systematic component is bounded by upper95(|2f-1|) x the fully-aligned worst "
            "case, since |2f-1| = 1 IS that worst case. Below 1.0 on systematic_over_random the "
            "random term dominates and the RMS figure is the correct yardstick to record."
        ),
    }

    if not controls_ok:
        verdict = (
            f"PROBE INVALID — the controls did not behave (full-shift {v_full}, edge-plant "
            f"{v_edge}, sign-randomised {n_pass}/{args.replicates} passed). A statistic that has "
            "not been shown able to separate the cases it claims to separate concludes NOTHING "
            "about the real perturbation."
        )
    elif v["Z1_direction_balanced"] and v["Z2_mean_indistinguishable_from_zero"]:
        verdict = (
            f"ZERO-MEAN CONDITION MET. Enter/exit split f = {st['enter_fraction_f']:.4f}; the 95% "
            f"upper bound on |2f-1| is {st['imbalance_upper95']:.5f}, inside the allowed "
            f"{half_width:.5f}. Signed mean {st['signed_mean_delta_mu_hat']:.3e} sits at t = "
            f"{st['t_stat_mean_vs_zero']:.2f}, within 3 SE of zero. Flips carry no preferred "
            f"direction, so the systematic |dIR| is bounded by {sys_dir:.3e} = "
            f"{sys_dir / DIR_RMS_INDEPENDENT:.2f}x the random term — the RMS yardstick recorded in "
            "the impact bound is the correct one."
        )
    else:
        failed = [k for k, ok in v.items() if not ok]
        verdict = (
            f"STOP AND REPORT — SYSTEMATIC BIAS. Failed: {failed}. f = "
            f"{st['enter_fraction_f']:.4f}, upper95(|2f-1|) = {st['imbalance_upper95']:.5f} vs "
            f"allowed {half_width:.5f}; signed mean {st['signed_mean_delta_mu_hat']:.3e} at t = "
            f"{st['t_stat_mean_vs_zero']:.2f}. The perturbation has a preferred direction, so "
            "flips accumulate LINEARLY rather than as sqrt(N) and the RMS yardstick understates "
            "the impact. Condition (a) reopens; reported, not worked around."
        )

    doc = {
        "receipt": "m6_prefill_zero_mean",
        "amendment": "prereg §7 v1.6 — the zero-mean condition attached to condition (a)'s closure",
        "PRE_DECLARED_BEFORE_MEASURING": {
            "Z1_direction": {
                "criterion": "upper95(|2f-1|) <= 1/sqrt(N_flips_per_cell_seed)",
                "N_flips_per_cell_seed": N_FLIPS_PER_CELL_SEED,
                "half_width": half_width,
                "why": "systematic drift is |2f-1|*N per unit, random drift sqrt(N); RMS is the "
                "right yardstick only while systematic <= random. Verified against the impact "
                "bound: worst/rms = 9.496 = sqrt(89.2449).",
                "corrected_first_draft": "the first draft used 1/sqrt(2231) — the flip count "
                "POOLED across all 25 units — which is 5x too strict, because IR is computed PER "
                "(cell,seed). Caught by the sign-randomised control failing at n=256, which the "
                "control-arm rule turned into PROBE INVALID rather than a conclusion.",
            },
            "Z2_mean": {"criterion": "|t| <= 3 on mean(delta) vs zero"},
            "on_failure": "STOP AND REPORT — condition (a) reopens. No workaround.",
            "resolution_se_imbalance": 1.0 / math.sqrt(n),
        },
        "config": {
            "decisions": n,
            "chunk": args.chunk,
            "seq_len": args.seq_len,
            "h": args.h,
            "cell": "cell4 fsq_micro",
            "fixture_seed": FIXTURE_SEED,
            "fixture_nests_n256_receipt": True,
            "prefill_impl": "trikaal.eval.predict._fill_context_prefill — the SHIPPED function, "
            "not a local copy, so this measures what the run would actually execute",
        },
        "OBSERVED": st,
        "criteria_met": v,
        "PROPAGATED_TO_dIR": propagated,
        "CONTROLS": {
            "controls_ok": controls_ok,
            "full_systematic_shift": {"stats": st_full, "criteria_met": v_full},
            "planted_imbalance_at_1p5x_half_width": {
                "planted_g": g_edge,
                "stats": st_edge,
                "criteria_met": v_edge,
            },
            "sign_randomised_replicates": {
                "n": args.replicates,
                "n_passing_Z1": n_pass,
                "required": args.replicates - 1,
            },
            "requirement": "full-shift FAILS both; edge-plant FAILS Z1 (sensitivity AT the "
            "boundary, not merely against an implausible bias); >= 19/20 sign-randomised PASS Z1",
        },
        "FLIP_RATE_REMEASURED": {
            "why": "the n=256 receipt recorded fixture_can_discriminate=false and still reported "
            "CONDITION (a) SATISFIED from 0 flips. At that size no decision could straddle a "
            "threshold, so 0 was the only attainable outcome — a check that could not fail.",
            "n_thresholds": len(thetas),
            "decisions_within_max_delta_of_some_threshold": near,
            "fixture_can_discriminate": bool(near > 0),
            "observed_flip_rate": observed,
            "expected_flip_rate_conservative": expected,
            "observed_over_expected": (observed / expected) if expected > 0 else float("inf"),
            "flips": flips_total,
            "comparisons": dec_total,
            "flip_enter": flip_enter,
            "flip_exit": flip_exit,
            "observed_flip_enter_fraction": (flip_enter / n_fd) if n_fd else float("nan"),
            "per_threshold_nonzero": per_threshold,
        },
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True))

    print("\n=== CONTROLS ===")
    print(f"  full systematic shift -> {v_full}  (must be False/False)")
    print(f"  edge plant g={g_edge:.5f}   -> {v_edge}  (Z1 must be False)")
    print(
        f"  sign-randomised       -> {n_pass}/{args.replicates} pass Z1 (need "
        f">= {args.replicates - 1})"
    )
    print(f"  controls_ok = {controls_ok}")
    print("\n=== OBSERVED ===")
    print(f"  signed mean(delta) = {st['signed_mean_delta_mu_hat']:.6e}")
    print(
        f"  SE                 = {st['se_mean_delta']:.6e}   t = {st['t_stat_mean_vs_zero']:.3f}"
        f"   p = {st['p_two_sided_mean']:.3f}"
    )
    print(f"  mean|delta|        = {st['mean_abs_delta_mu_hat']:.6e}")
    print(f"  enter {st['n_enter']} / exit {st['n_exit']} / tie {st['n_tie']}")
    print(
        f"  f = {st['enter_fraction_f']:.4f}  |2f-1| = {st['imbalance_abs_2f_minus_1']:.5f}  "
        f"upper95 = {st['imbalance_upper95']:.5f}  (allowed {half_width:.5f})  "
        f"z = {st['z_balance_vs_half']:.2f}"
    )
    print(
        f"  => systematic |dIR| <= {sys_dir:.3e} = "
        f"{sys_dir / DIR_RMS_INDEPENDENT:.2f}x the random RMS term"
    )
    print(
        f"\n  flip rate {observed:.3e} over {dec_total} comparisons "
        f"({(observed / expected) if expected > 0 else float('nan'):.2f}x expected), "
        f"flips {flips_total} (enter {flip_enter} / exit {flip_exit})"
    )
    print(f"  fixture discriminates: {near > 0} ({near} within max|delta| of a threshold)")
    print(f"\nVERDICT: {verdict}")
    print(f"[receipt] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
