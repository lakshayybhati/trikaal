"""M6 TOKEN-SPACE CONTROL, rung (i) — the NOISY MONOTONE MAP plant (amended).

    PYTHONPATH=src python3 scripts/m6_token_control.py --step0             # local, $0 (gates)
    PYTHONPATH=src python3 scripts/m6_token_control.py --run --device cuda # the box run (~$1)
    PYTHONPATH=src python3 scripts/m6_token_control.py --toy               # cpu machinery check

Supervisor rulings, in order:
- Parity-bijection plant SHELVED as rung (ii) (external audit: functional-form confound —
  categorical needle; reserved for the findability-cliff map; can never ground an H-A claim).
- Translation plant RETIRED (Step-0 receipt accepted: the cell4 coarse digit marginals are
  99.2-99.8% of max entropy, and a translation on near-uniform ordinal alphabets is
  information-poor by physics — exhaustive search topped out at 0.2769 nats < the 0.7-1.1
  band, making the original (a) val clause unreachable; spec flaw, not an implementation one).
- ADOPTED (this file): the NOISY MONOTONE MAP — at EVERY position t the target digit of c_t
  is REDRAWN from a discretized concentrated distribution (Gaussian on the level grid) whose
  CENTER is a monotone function of the source level of c_{t-2} (centers sweeping the target
  alphabet) and whose width sigma_plant is TUNED so the exactly-computed planted information
  lands in 0.7-1.1 nats. Closed form and MC plug-in must agree; the target-marginal KL vs
  the carrier and the unplanted-control Spearman are receipts. If the KL cannot be kept
  small at in-band information -> fall back to option-2 stacking (multiple disjoint pairs
  summing into the band) and report which path was taken. Never bend a clause to fit.

FUNCTIONAL-FORM RATIONALE (recorded verbatim in the manifest): rules at matched mutual
information differ in SGD findability; rung (i) matches the real signal's dense/graded shape;
rung (ii) (categorical needle) is reserved for the findability-cliff map and can never ground
an H-A claim.

STEP 0 (local, $0, gates the spend):
  (A) id-visibility probe (stands; Receipt-2 numbers are STANDING mechanistic H-T evidence
      and are restated verbatim in the run manifest): logistic on sign(s_t) + small MLP
      regressing s_t from the digit decomposition of (c_id, f_id) at bar t, on the
      regenerated v6 PLANTED stream through the v6 planted-arm tokenizer.
  (B) plant construction receipts on the carrier: digit marginals; the (source, target)
      pair; sigma_plant tuned so exact info lands in-band (closed form + MC agree); the
      target-marginal KL and entropy-shift receipts; oracle probe ceilings; unplanted
      control ~0. Gate: info in-band AND KL acceptable (KL <= 0.05 and |dH| <= 0.05,
      both well inside the 0.1 H0-comparability tolerance) — else STOP at $0.

THE RUN (box, ~$1): carrier = regenerated v6 noise stream (42M bars, same generator seeds),
tokenized once with the FROZEN v6 noise-arm cell4 tokenizer; plant applied densely (channel
and sigma loaded VERBATIM from the committed Step-0 artifact; redraw RNG seeds recorded);
canonical d512 backbone; orchestrator-real schedule; fixed 20k steps; probes every 500 =
teacher-forced Spearman(predicted target-digit level, source level at t-2) + val + train.
H0 = the v6 noise trajectory final val 13.5979 (REUSED; runs_manifest/
m6_canary_v6_stage1_manifest.json, runs.noise.final_val_nll). Spike > 1 nat -> halve-lr-once
restart. Sub-epoch + gap checks in the decision block. Repo-only transfer; destroy after.

BRANCHES (exhaustive; (a) RE-PRINCIPLED per the amendment):
  (a) DETECT — teacher-forced Spearman >= 0.3 on >= 2 consecutive evals AND final val <=
      H0 - 0.5 x I_planted (with I in-band that is >= 0.35 nats — cleanly above the 0.1
      H0-comparability tolerance) -> the AR learns dense token-space conditionals -> v6's
      finding localizes to the tokenizer->AR INTERFACE (H-T). STOP; interface re-spec
      adjudication is the supervisor's.
  (b) CLEAN FAIL — schedule complete, probe corr < 0.05 at every eval, |val - H0| < 0.1,
      sub-epoch/gap checks hold -> the STRONG form of H-A: the AR could not learn a dense,
      graded, task-shaped lag-2 dependency presented in its native ordinal digit structure.
      STOP; pre-named next = the scaling probe (d256/d128, reduced vocab, AND rung-(ii)
      parity at the scale where (i)-style rules ARE learned, to map the findability cliff).
  (c) clearly still descending at cap -> ONE 20k extension.
  (d) spike > 1 nat -> halve-lr-once restart, log.
  (e) anything else -> INCONCLUSIVE, report, zero further spend.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from trikaal.constants import N_FEATURES
from trikaal.data.universe_loader import build_symbol_windows
from trikaal.model.predictor import TrikaalAR  # noqa: F401  (checkpoint reload type)
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.cells import build_cell_backbone
from trikaal.train.checkpoint import load_checkpoint, save_checkpoint
from trikaal.train.gates import cosine_warmup_lr
from trikaal.train.token_stream import tokenize_features
from trikaal.utils.seeding import set_determinism

# ---- carrier geometry (v6, verbatim — H0 comparability rests on this) ----------------------
SYMBOLS = ("SYNAUSDT", "SYNBUSDT", "SYNCUSDT")
BATCH = 32
SEQ_LEN = 32
SIGMA = 0.01
SIGNAL_LAG = 2
T0_MS = 1_704_067_200_000
HOLD_FRAC = 0.65
SUB_EPOCH_FACTOR = 1.3
BARS_TOTAL_DEFAULT = 42_000_000
C_SIGNAL = 3.0  # v6 planted rule (used ONLY to regenerate the visibility-probe stream)

STAGE2_PEAK_LR = 3e-4
WARMUP_FRAC = 0.1

# ---- the FSQ coarse digit structure (cell4 tokenizer, verified from the checkpoint) ---------
COARSE_LEVELS = (11, 9, 9)  # big-endian: id = (d0*9 + d1)*9 + d2
LAG = 2

# ---- H0 (reused, per the ruling: no new noise run) ------------------------------------------
H0_VAL = 13.5979
H0_SOURCE = (
    "runs_manifest/m6_canary_v6_stage1_manifest.json :: runs.noise.final_val_nll "
    "(v6 noise-arm cell4, identical carrier construction/budget/schedule/val protocol)"
)

# ---- branch thresholds ----------------------------------------------------------------------
DETECT_CORR = 0.3
DETECT_VAL_FRAC = 0.5  # (a) re-principled: final val <= H0 - 0.5 x I_planted
FLAT_CORR = 0.05
H0_TOL = 0.1
MEMO_GAP = 0.5
DESC_NATS = 0.05

INFO_BAND = (0.7, 1.1)  # target planted information, nats (v6-comparable)
KL_MAX = 0.05  # acceptable target-marginal KL(planted || carrier), nats
DH_MAX = 0.05  # acceptable |H(planted marginal) - H(carrier marginal)|, nats

RATIONALE = (
    "rules at matched mutual information differ in SGD findability; rung (i) matches the "
    "real signal's dense/graded shape; rung (ii) (categorical needle) is reserved for the "
    "findability-cliff map and can never ground an H-A claim"
)

# Receipt 2 (Step-0 of 2026-07-20, supervisor-accepted STANDING mechanistic H-T evidence;
# restated VERBATIM in the run manifest per the amendment).
ID_VISIBILITY_RECEIPT = {
    "logistic_sign_accuracy": 0.5135,
    "logistic_chance": 0.5,
    "mlp_state_pearson_corr": 0.0142,
    "stream": "regenerated v6 PLANTED, symbol 0, first 300,000 bars",
    "meaning": "the v6 feature-space state is nearly invisible in single-bar (c_id, f_id) "
    "while v6 dim-9 WINDOW-decode recon was non-degenerate — the state is smeared across "
    "the window's ids (causal encoder writes state_t into later tokens), not present in "
    "bar t's own id: standing mechanistic H-T evidence",
    "source": "runs_manifest/m6_token_control_step0.json (commit e891d8c)",
}

TOY_BB_KW = dict(
    n_layers=2,
    d_model=48,
    d_ff=96,
    n_heads=2,
    mtp_depths=0,
    ffn_dropout=0.0,
    resid_dropout=0.0,
    attn_dropout=0.0,
    token_dropout=0.0,
)

DEFAULT_NOISE_TOK = "runs_cloud/canary_v6/m6_canary_v6/noise/cell4_fsq_micro_seed0/tokenizer.pt"
DEFAULT_PLANTED_TOK = "runs_cloud/canary_v6/m6_canary_v6/planted/cell4_fsq_micro_seed0/tokenizer.pt"
STEP0_ARTIFACT = "runs_manifest/m6_token_control_step0.json"


def synth_symbol(rng: np.random.Generator, *, planted: bool, n_bars: int) -> dict:
    """v6's generator VERBATIM (H0 comparability). The carrier uses planted=False."""
    state = rng.standard_normal(n_bars)
    eps = rng.standard_normal(n_bars)
    r = SIGMA * eps
    if planted:
        r[SIGNAL_LAG:] = r[SIGNAL_LAG:] + C_SIGNAL * SIGMA * state[:-SIGNAL_LAG]
    x = rng.standard_normal((n_bars, N_FEATURES)).astype(np.float32)
    x[:, 0] = (r / SIGMA).astype(np.float32)
    x[:, 9] = (state if planted else rng.standard_normal(n_bars)).astype(np.float32)
    m = np.zeros((n_bars, N_FEATURES), dtype=np.uint8)
    m[:, 13:16] = 1
    return {
        "x": x,
        "mask": m,
        "segment_id": np.zeros(n_bars, dtype=np.int64),
        "ts": (T0_MS + np.arange(n_bars) * 60_000).astype(np.int64),
        "sigma": np.full(n_bars, SIGMA),
        "raw_ret_close": r.astype(np.float64),
    }


def arm_rng(arm_name: str, i: int) -> np.random.Generator:
    return np.random.default_rng((zlib.crc32(arm_name.encode()), i))


# ---- digit arithmetic (verified against fsq.mixed_radix_flatten big-endian) -----------------
def coarse_digits(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return c // 81, (c // 9) % 9, c % 9


def coarse_assemble(d0: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    return (d0 * 9 + d1) * 9 + d2


def fine_digits(f: np.ndarray) -> tuple[np.ndarray, ...]:
    return f // 175, (f // 25) % 7, (f // 5) % 5, f % 5


# ---- the noisy monotone map -----------------------------------------------------------------
def noisy_map_channel(L_s: int, L_t: int, sigma_plant: float) -> np.ndarray:
    """Channel matrix D[s, v'] = P(target level v' | source level s): a discretized Gaussian
    on the target level grid, centered at the monotone sweep mu(s) = (L_t-1) * s/(L_s-1)."""
    v = np.arange(L_t, dtype=np.float64)
    d = np.empty((L_s, L_t))
    for s in range(L_s):
        mu = (L_t - 1) * s / (L_s - 1)
        w = np.exp(-0.5 * ((v - mu) / sigma_plant) ** 2)
        d[s] = w / w.sum()
    return d


def channel_info(p: np.ndarray, d: np.ndarray, q: np.ndarray) -> dict:
    """Exact planted information + marginal receipts for redraw channel ``d`` under source
    marginal ``p``, vs the carrier target marginal ``q``."""
    m = p @ d
    with np.errstate(divide="ignore", invalid="ignore"):
        info = float(np.sum(p[:, None] * d * np.where(d > 0, np.log(d / m[None, :]), 0.0)))
        kl = float(np.sum(np.where(m > 0, m * np.log(m / q), 0.0)))
    h_m = float(-(m[m > 0] * np.log(m[m > 0])).sum())
    h_q = float(-(q[q > 0] * np.log(q[q > 0])).sum())
    return {
        "info_nats": info,
        "target_marginal_kl_planted_vs_carrier": kl,
        "entropy_planted_target_marginal": h_m,
        "entropy_carrier_target_marginal": h_q,
        "entropy_shift": h_m - h_q,
        "planted_target_marginal": m.tolist(),
    }


def tune_sigma(p: np.ndarray, q: np.ndarray, L_s: int, L_t: int, target_info: float) -> float:
    """Binary-search sigma_plant so the exact info hits ``target_info`` (info is monotone
    decreasing in sigma)."""
    lo, hi = 0.05, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        i_mid = channel_info(p, noisy_map_channel(L_s, L_t, mid), q)["info_nats"]
        if i_mid > target_info:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def choose_noisy_plant(marginals: list[np.ndarray], target_info: float) -> dict:
    """All 6 ordered coarse pairs at the sigma hitting target_info; pick the min-KL pair."""
    best = None
    for src, tgt in permutations(range(3), 2):
        p, q = marginals[src], marginals[tgt]
        l_s, l_t = COARSE_LEVELS[src], COARSE_LEVELS[tgt]
        sig = tune_sigma(p, q, l_s, l_t, target_info)
        d = noisy_map_channel(l_s, l_t, sig)
        rec = channel_info(p, d, q)
        cand = {
            "form": "noisy monotone map: target digit REDRAWN ~ discretized Gaussian on the "
            "level grid, center mu(s) = (L_t-1)*s/(L_s-1) (monotone sweep), width sigma_plant",
            "source_digit": src,
            "target_digit": tgt,
            "source_levels": l_s,
            "target_levels": l_t,
            "sigma_plant": round(sig, 6),
            "channel_matrix": [[round(float(x), 8) for x in row] for row in d],
            **rec,
        }
        if (
            best is None
            or cand["target_marginal_kl_planted_vs_carrier"]
            < best["target_marginal_kl_planted_vs_carrier"]
        ):
            best = cand
    return best


def apply_noisy_plant(
    b_c: np.ndarray, d: np.ndarray, src: int, tgt: int, rng: np.random.Generator
) -> np.ndarray:
    """Dense redraw: target digit of c_t ~ channel row of source level at t-2. The source
    digit is never modified, so a single pass is self-consistent. First LAG positions keep
    the carrier digit (no source exists)."""
    digs = list(coarse_digits(b_c))
    src_lvl = digs[src][:-LAG]
    u = rng.random(src_lvl.size)
    cdf = np.cumsum(d, axis=1)
    new_tgt = digs[tgt].copy()
    drawn = np.empty(src_lvl.size, dtype=np.int64)
    for s in range(d.shape[0]):
        m = src_lvl == s
        if m.any():
            drawn[m] = np.searchsorted(cdf[s], u[m], side="right")
    new_tgt[LAG:] = np.clip(drawn, 0, d.shape[1] - 1)
    digs[tgt] = new_tgt
    return coarse_assemble(*digs)


# ---- Spearman (numpy only; average ranks over ties) -----------------------------------------
def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(a.size, dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(np.asarray(a)), _rank(np.asarray(b))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def tokenize_stream(tok, d: dict, *, device: str, batch_windows: int = 512):
    return tokenize_features(
        tok,
        d["x"],
        d["mask"],
        d["segment_id"],
        window=SEQ_LEN,
        batch_windows=batch_windows,
        device=device,
    )


def sub_epoch_arithmetic(steps: int, bars_total: int) -> dict:
    n_per = bars_total // len(SYMBOLS)
    visits = steps * BATCH * SEQ_LEN
    required = int(np.ceil(SUB_EPOCH_FACTOR * visits))
    train_bars = int(HOLD_FRAC * n_per) * len(SYMBOLS)
    return {
        "rule": "train-region bars >= 1.3 x (steps x batch x seq_len) bar-visits",
        "steps": steps,
        "batch": BATCH,
        "seq_len": SEQ_LEN,
        "bar_visits": visits,
        "required_train_bars": required,
        "bars_total": n_per * len(SYMBOLS),
        "n_per_symbol": n_per,
        "hold_frac": HOLD_FRAC,
        "train_bars": train_bars,
        "epochs_equivalent_train_region": round(visits / train_bars, 4),
        "holds": bool(train_bars >= required),
    }


def mc_plugin_info(src_lvl: np.ndarray, tgt_lvl: np.ndarray, l_s: int, l_t: int) -> float:
    joint = np.zeros((l_s, l_t))
    np.add.at(joint, (src_lvl, tgt_lvl), 1.0)
    joint /= joint.sum()
    ps, pt = joint.sum(1), joint.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum(np.where(joint > 0, joint * np.log(joint / np.outer(ps, pt)), 0.0)))


# =============================================================================== STEP 0 ======
def step0(args) -> int:
    t0 = time.time()
    dev = "cpu"
    out: dict = {
        "artifact": "m6_token_control_step0",
        "plant_family": "noisy monotone map (amended; translation RETIRED, parity SHELVED "
        "as rung (ii))",
        "rationale_verbatim": RATIONALE,
    }

    # ---- (B) plant construction on the carrier ---------------------------------------------
    print(f"[step0] carrier sample: noise symbol 0, first {args.carrier_sample:,} bars", flush=True)
    tok_n = load_checkpoint(args.noise_tok, TokenizerAE).model.to(dev).eval()
    assert tok_n.v_c == 891 and list(tok_n.get_config()["levels"]) == [11, 9, 9, 7, 7, 5, 5]
    d_noise = synth_symbol(arm_rng("noise", 0), planted=False, n_bars=args.carrier_sample)
    b_c, _b_f = tokenize_stream(tok_n, d_noise, device=dev, batch_windows=256)
    digs = coarse_digits(b_c)
    marginals = [
        np.bincount(digs[i], minlength=COARSE_LEVELS[i]).astype(np.float64) for i in range(3)
    ]
    marginals = [m / m.sum() for m in marginals]
    ent = [float(-(m[m > 0] * np.log(m[m > 0])).sum()) for m in marginals]
    out["carrier"] = {
        "sample_bars": int(b_c.size),
        "generator": "np.random.default_rng((zlib.crc32(b'noise'), 0)) — v6 verbatim",
        "coarse_digit_marginals": [m.tolist() for m in marginals],
        "coarse_digit_entropies_nats": ent,
        "tokenizer": args.noise_tok,
    }
    print(f"[step0] digit entropies (nats): {[round(e, 3) for e in ent]}", flush=True)

    plant = choose_noisy_plant(marginals, args.target_info)
    out["plant"] = plant
    print(
        f"[step0] chosen plant: src {plant['source_digit']} -> tgt {plant['target_digit']}, "
        f"sigma {plant['sigma_plant']}, EXACT info {plant['info_nats']:.4f} nats, "
        f"KL {plant['target_marginal_kl_planted_vs_carrier']:.5f}, "
        f"dH {plant['entropy_shift']:+.5f}",
        flush=True,
    )

    # MC plug-in check + oracle ceilings on the actually-planted sample
    d_mat = np.array(plant["channel_matrix"])
    plant_rng = np.random.default_rng((0x9A47, args.seed, 0))
    b_cp = apply_noisy_plant(b_c, d_mat, plant["source_digit"], plant["target_digit"], plant_rng)
    src_lvl = coarse_digits(b_c)[plant["source_digit"]][:-LAG]
    tgt_lvl = coarse_digits(b_cp)[plant["target_digit"]][LAG:]
    mi_mc = mc_plugin_info(src_lvl, tgt_lvl, plant["source_levels"], plant["target_levels"])
    oracle_pred = np.argmax(d_mat, axis=1)[src_lvl]
    out["plant_checks"] = {
        "mc_plugin_info_nats": mi_mc,
        "closed_form_vs_mc_abs_diff": abs(mi_mc - plant["info_nats"]),
        "oracle_spearman_raw_src_vs_planted_tgt": spearman(src_lvl, tgt_lvl),
        "oracle_spearman_argmax_predictor": spearman(oracle_pred, tgt_lvl),
        "unplanted_control_spearman": spearman(
            coarse_digits(b_c)[plant["source_digit"]][:-LAG],
            coarse_digits(b_c)[plant["target_digit"]][LAG:],
        ),
        "plant_rng": "np.random.default_rng((0x9A47, seed, symbol_index))",
    }
    print(f"[step0] plant checks: {out['plant_checks']}", flush=True)

    # ---- (A) id-visibility probe (recomputed; identical seeds -> identical numbers) --------
    print(f"[step0] visibility probe: planted symbol 0, first {args.vis_sample:,} bars", flush=True)
    tok_p = load_checkpoint(args.planted_tok, TokenizerAE).model.to(dev).eval()
    d_pl = synth_symbol(arm_rng("planted", 0), planted=True, n_bars=args.vis_sample)
    pc, pf = tokenize_stream(tok_p, d_pl, device=dev, batch_windows=256)
    cd, fd = coarse_digits(pc), fine_digits(pf)
    levels_all = [*COARSE_LEVELS, 7, 7, 5, 5]
    cols = list(cd) + list(fd)
    feats = np.zeros((pc.size, sum(levels_all)), dtype=np.float32)
    off = 0
    for col, lv in zip(cols, levels_all, strict=True):
        feats[np.arange(pc.size), off + col] = 1.0
        off += lv
    s = d_pl["x"][:, 9].astype(np.float32)
    n_tr = int(0.8 * pc.size)
    xt = torch.from_numpy(feats[:n_tr])
    xv = torch.from_numpy(feats[n_tr:])
    st = torch.from_numpy(s[:n_tr])
    sv = torch.from_numpy(s[n_tr:])
    torch.manual_seed(0)

    logit = torch.nn.Linear(feats.shape[1], 1)
    opt = torch.optim.Adam(logit.parameters(), lr=1e-2)
    yt = (st > 0).float()
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit(xt).squeeze(-1), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float(((logit(xv).squeeze(-1) > 0) == (sv > 0)).float().mean())

    mlp = torch.nn.Sequential(
        torch.nn.Linear(feats.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for _ in range(400):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(mlp(xt).squeeze(-1), st)
        loss.backward()
        opt.step()
    with torch.no_grad():
        shat = mlp(xv).squeeze(-1).numpy()
    vis_corr = float(np.corrcoef(shat, sv.numpy())[0, 1])
    out["id_visibility_probe"] = {
        "stream": "regenerated v6 PLANTED, symbol 0",
        "tokenizer": args.planted_tok,
        "sample_bars": int(pc.size),
        "train_test_split": [n_tr, int(pc.size - n_tr)],
        "logistic_sign_accuracy": acc,
        "logistic_chance": 0.5,
        "mlp_state_pearson_corr": vis_corr,
        "note": "how visible the v6 feature-space state s_t is in (c_id, f_id) at bar t; "
        "supervisor-accepted STANDING mechanistic H-T evidence",
    }
    print(
        f"[step0] visibility: logistic sign-acc {acc:.4f} (chance 0.5), MLP corr {vis_corr:.4f}",
        flush=True,
    )

    lo, hi = INFO_BAND
    in_band = lo <= plant["info_nats"] <= hi
    kl_ok = (
        plant["target_marginal_kl_planted_vs_carrier"] <= KL_MAX
        and abs(plant["entropy_shift"]) <= DH_MAX
    )
    mc_ok = out["plant_checks"]["closed_form_vs_mc_abs_diff"] < 0.02
    ceiling_ok = out["plant_checks"]["oracle_spearman_argmax_predictor"] >= DETECT_CORR + 0.1
    out["gates"] = {
        "info_band": list(INFO_BAND),
        "info_in_band": bool(in_band),
        "kl_acceptable": bool(kl_ok),
        "kl_max": KL_MAX,
        "dh_max": DH_MAX,
        "closed_form_mc_agree": bool(mc_ok),
        "oracle_ceiling_clears_detect_threshold": bool(ceiling_ok),
        "stacking_fallback_needed": bool(in_band and not kl_ok),
        "PROCEED": bool(in_band and kl_ok and mc_ok and ceiling_ok),
    }
    out["wall_s"] = round(time.time() - t0, 1)
    Path(args.step0_artifact).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[step0] gates: {out['gates']}", flush=True)
    print(f"[step0] artifact -> {args.step0_artifact}", flush=True)
    return 0 if out["gates"]["PROCEED"] else 1


# =============================================================================== THE RUN =====
def run(args) -> int:
    dev = args.device
    set_determinism(args.seed, deterministic_algorithms=False)
    step0_art = json.loads(Path(args.step0_artifact).read_text())
    if "noisy monotone map" not in step0_art.get("plant_family", ""):
        raise SystemExit(
            "step0 artifact is not the amended noisy-monotone-map plant — rerun --step0"
        )
    if not step0_art["gates"]["PROCEED"]:
        raise SystemExit("step0 gate is BLOCKED — the run is not authorized")
    plant = step0_art["plant"]
    d_mat = np.array(plant["channel_matrix"])
    src_dig, tgt_dig = plant["source_digit"], plant["target_digit"]
    sub_epoch = sub_epoch_arithmetic(args.steps, args.bars_total)
    print(f"[run] sub-epoch arithmetic: {sub_epoch}", flush=True)
    if not args.toy and not sub_epoch["holds"]:
        raise SystemExit("sub-epoch rule violated — raise --bars-total")
    print(
        f"[run] plant (from committed step0): src {src_dig} -> tgt {tgt_dig}, "
        f"sigma {plant['sigma_plant']}, step0 exact info {plant['info_nats']:.4f} nats",
        flush=True,
    )

    tok = load_checkpoint(args.noise_tok, TokenizerAE).model.to(dev).eval()
    assert tok.get_config()["encoder_causal"] is True
    n_per = args.bars_total // len(SYMBOLS)
    hold_bar = int(HOLD_FRAC * n_per)
    hold_ms = T0_MS + hold_bar * 60_000

    streams = {}
    marg_src = np.zeros(COARSE_LEVELS[src_dig])
    marg_tgt_carrier = np.zeros(COARSE_LEVELS[tgt_dig])
    mc_src_all, mc_tgt_all = [], []
    ts0 = time.time()
    for i, sym in enumerate(SYMBOLS):
        d = synth_symbol(arm_rng("noise", i), planted=False, n_bars=n_per)
        sw = build_symbol_windows(
            sym,
            d["x"],
            d["mask"],
            d["segment_id"],
            d["ts"],
            arm="micro",
            seq_len=SEQ_LEN,
            boundary_ms=hold_ms,
            seed=args.seed,
        )
        b_c, b_f = tokenize_stream(
            tok,
            {"x": sw.x, "mask": sw.mask, "segment_id": sw.segment_id},
            device=dev,
            batch_windows=512,
        )
        digs = coarse_digits(b_c)
        marg_src += np.bincount(digs[src_dig], minlength=COARSE_LEVELS[src_dig])
        marg_tgt_carrier += np.bincount(digs[tgt_dig], minlength=COARSE_LEVELS[tgt_dig])
        plant_rng = np.random.default_rng((0x9A47, args.seed, i))
        b_cp = apply_noisy_plant(b_c, d_mat, src_dig, tgt_dig, plant_rng)
        mc_src_all.append(digs[src_dig][:-LAG])
        mc_tgt_all.append(coarse_digits(b_cp)[tgt_dig][LAG:])
        streams[sym] = {"b_c": b_cp, "b_f": b_f, "ts": sw.ts, "starts": sw.starts}
        del d, sw, b_c
        print(
            f"[run] {sym}: {n_per:,} bars generated+tokenized+planted "
            f"({time.time() - ts0:.0f}s cum)",
            flush=True,
        )
    p_full = marg_src / marg_src.sum()
    q_full = marg_tgt_carrier / marg_tgt_carrier.sum()
    full_rec = channel_info(p_full, d_mat, q_full)
    mi_mc_full = mc_plugin_info(
        np.concatenate(mc_src_all),
        np.concatenate(mc_tgt_all),
        plant["source_levels"],
        plant["target_levels"],
    )
    i_planted = full_rec["info_nats"]
    detect_val_drop = round(DETECT_VAL_FRAC * i_planted, 4)
    print(
        f"[run] FULL-STREAM exact info {i_planted:.4f} nats (MC {mi_mc_full:.4f}), "
        f"KL {full_rec['target_marginal_kl_planted_vs_carrier']:.5f}; "
        f"branch (a) val clause: final val <= {H0_VAL} - {detect_val_drop}",
        flush=True,
    )

    # ---- val + probe fixtures (held-out slice, v6 protocol) --------------------------------
    val_starts = np.arange(hold_bar, hold_bar + 96 * SEQ_LEN, SEQ_LEN)
    val = [
        (
            torch.from_numpy(np.stack([streams[s]["b_c"][i : i + SEQ_LEN] for i in val_starts])),
            torch.from_numpy(np.stack([streams[s]["b_f"][i : i + SEQ_LEN] for i in val_starts])),
            torch.from_numpy(np.stack([streams[s]["ts"][i : i + SEQ_LEN] for i in val_starts])),
        )
        for s in SYMBOLS
    ]
    s0 = streams[SYMBOLS[0]]
    lo = hold_bar + SEQ_LEN
    probe_starts = np.arange(lo, lo + 120 * SEQ_LEN, SEQ_LEN)
    pb_c = torch.from_numpy(np.stack([s0["b_c"][s : s + SEQ_LEN] for s in probe_starts]))
    pb_f = torch.from_numpy(np.stack([s0["b_f"][s : s + SEQ_LEN] for s in probe_starts]))
    pb_ts = torch.from_numpy(np.stack([s0["ts"][s : s + SEQ_LEN] for s in probe_starts]))
    probe_src = np.stack([coarse_digits(s0["b_c"][s : s + SEQ_LEN])[src_dig] for s in probe_starts])

    def probe(model) -> float:
        """Teacher-forced Spearman(predicted target-digit level at t, source level at t-2)."""
        model.eval()
        with torch.no_grad():
            o = model(pb_c.to(dev), pb_f.to(dev), pb_ts.to(dev))
            pred_c = o.logits_c.argmax(-1).cpu().numpy()  # position j predicts token j+1
        pred_tgt = coarse_digits(pred_c)[tgt_dig]
        preds = pred_tgt[:, 1:-1].ravel()  # predictions for rel positions 2..SEQ_LEN-1
        srcs = probe_src[:, 0:-2].ravel()  # source levels at rel positions 0..SEQ_LEN-3
        return spearman(preds, srcs)

    rng = np.random.default_rng(args.seed)
    n_starts = np.array([streams[s]["starts"].size for s in SYMBOLS], dtype=np.float64)
    weights = n_starts**0.5 / (n_starts**0.5).sum()

    def batch():
        sym_ids = rng.choice(len(SYMBOLS), size=BATCH, p=weights)
        bc_l, bf_l, ts_l = [], [], []
        for si in sym_ids:
            st = streams[SYMBOLS[int(si)]]
            s = int(st["starts"][rng.integers(0, st["starts"].size)])
            bc_l.append(st["b_c"][s : s + SEQ_LEN])
            bf_l.append(st["b_f"][s : s + SEQ_LEN])
            ts_l.append(st["ts"][s : s + SEQ_LEN])
        return (
            torch.from_numpy(np.stack(bc_l).astype(np.int64)),
            torch.from_numpy(np.stack(bf_l).astype(np.int64)),
            torch.from_numpy(np.stack(ts_l).astype(np.int64)),
        )

    bb_kw = {} if not args.toy else dict(TOY_BB_KW)
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)

    def _ar_attempt(peak_lr: float, *, abort_on_spike: bool) -> dict:
        set_determinism(args.seed, deterministic_algorithms=False)
        model = build_cell_backbone(
            tok.v_c,
            tok.v_f,
            expect_base_params=21_301_248 if not args.toy else None,
            max_len=SEQ_LEN + 64,
            **bb_kw,
        ).to(dev)
        opt = torch.optim.AdamW(
            model.parameters(), lr=peak_lr, weight_decay=0.01, betas=(0.9, 0.95)
        )
        warmup = int(WARMUP_FRAC * args.steps)

        def val_loss() -> float:
            model.eval()
            tot = 0.0
            with torch.no_grad():
                for bc, bf, ts in val:
                    tot += float(model.compute_loss(bc.to(dev), bf.to(dev), ts.to(dev))["nll_main"])
            return tot / len(val)

        best = float("inf")
        trajectory, spikes = [], []
        acc_tot, acc_main, n_acc = 0.0, 0.0, 0
        t0 = time.time()
        for step in range(1, args.steps + 1):
            lr_now = cosine_warmup_lr(step, peak_lr, warmup, args.steps)
            for g in opt.param_groups:
                g["lr"] = lr_now
            bc, bf, ts = batch()
            model.train()
            opt.zero_grad(set_to_none=True)
            outl = model.compute_loss(bc.to(dev), bf.to(dev), ts.to(dev))
            loss = outl["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if not torch.isfinite(loss):
                raise RuntimeError(f"stage-2 loss non-finite at step {step}")
            acc_tot += float(loss.detach())
            acc_main += float(outl["nll_main"].detach())
            n_acc += 1
            if step % args.eval_every == 0:
                vl = val_loss()
                entry = {
                    "step": step,
                    "lr": round(lr_now, 8),
                    "train_nll": round(acc_tot / max(1, n_acc), 4),
                    "train_nll_main": round(acc_main / max(1, n_acc), 4),
                    "val_nll": round(vl, 4),
                    "probe_spearman": round(probe(model), 4),
                }
                acc_tot, acc_main, n_acc = 0.0, 0.0, 0
                trajectory.append(entry)
                print(f"    [token_control] {entry}", flush=True)
                if vl > best + args.spike_nats:
                    spikes.append({"step": step, "val_nll": round(vl, 4), "best": round(best, 4)})
                    print(f"    [token_control] VAL SPIKE at step {step}", flush=True)
                    if abort_on_spike:
                        return {
                            "peak_lr": peak_lr,
                            "aborted_by_spike": True,
                            "spikes": spikes,
                            "trajectory": trajectory,
                            "wall_s": round(time.time() - t0, 1),
                        }
                best = min(best, vl)
        save_checkpoint(run_dir / "predictor.pt", model, model.get_config())
        return {
            "peak_lr": peak_lr,
            "aborted_by_spike": False,
            "spikes": spikes,
            "steps": args.steps,
            "final_val_nll": trajectory[-1]["val_nll"] if trajectory else float("nan"),
            "best_val_nll": round(best, 4),
            "trajectory": trajectory,
            "wall_s": round(time.time() - t0, 1),
        }

    attempts = [_ar_attempt(STAGE2_PEAK_LR, abort_on_spike=True)]
    if attempts[0]["aborted_by_spike"]:
        print(
            f"[run] RESTART (branch d): peak lr {STAGE2_PEAK_LR} -> {STAGE2_PEAK_LR / 2}",
            flush=True,
        )
        attempts.append(_ar_attempt(STAGE2_PEAK_LR / 2, abort_on_spike=False))
    fin = attempts[-1]
    tr = fin["trajectory"]

    # ---- branches ---------------------------------------------------------------------------
    corr = [e["probe_spearman"] for e in tr]
    cross_step = None
    for i in range(len(corr) - 1):
        if corr[i] >= DETECT_CORR and corr[i + 1] >= DETECT_CORR:
            cross_step = tr[i]["step"]
            break
    fv = fin["final_val_nll"]
    gap = round(tr[-1]["val_nll"] - tr[-1]["train_nll_main"], 4) if tr else float("nan")
    schedule_complete = not fin["aborted_by_spike"]
    a = bool(cross_step is not None and fv <= H0_VAL - detect_val_drop)
    flat = bool(corr) and all(c < FLAT_CORR for c in corr)
    b = bool(
        schedule_complete
        and flat
        and abs(fv - H0_VAL) < H0_TOL
        and sub_epoch["holds"]
        and gap < MEMO_GAP
    )
    back = max(1, 5000 // args.eval_every)
    c = bool(
        len(tr) > back
        and tr[-1]["val_nll"] < tr[-1 - back]["val_nll"] - DESC_NATS
        and not a
        and not b
    )
    branch = (
        "a_detect_interface"
        if a
        else "b_clean_fail_strong_HA"
        if b
        else "c_extension"
        if c
        else "e_inconclusive"
    )
    decision = {
        "branch": branch,
        "H0_val": H0_VAL,
        "H0_source": H0_SOURCE,
        "final_val_nll": fv,
        "final_val_minus_H0": round(fv - H0_VAL, 4),
        "i_planted_full_stream": i_planted,
        "detect_val_clause": f"final val <= H0 - 0.5 x I_planted = {H0_VAL} - {detect_val_drop}",
        "probe_cross_step": cross_step,
        "probe_max": max(corr) if corr else None,
        "probe_final": corr[-1] if corr else None,
        "probe_flat_every_eval": flat,
        "schedule_complete": schedule_complete,
        "final_val_train_main_gap": gap,
        "sub_epoch_holds": sub_epoch["holds"],
        "lr_halved": len(attempts) > 1,
        "thresholds": {
            "detect_corr": DETECT_CORR,
            "detect_val_frac_of_info": DETECT_VAL_FRAC,
            "detect_val_drop_nats": detect_val_drop,
            "flat_corr": FLAT_CORR,
            "h0_tol": H0_TOL,
            "memo_gap": MEMO_GAP,
            "desc_nats": DESC_NATS,
        },
    }
    print(f"[run] decision: {json.dumps(decision)}", flush=True)

    manifest = {
        "control": "m6_token_space_control_rung_i_noisy_map",
        "rationale_verbatim": RATIONALE,
        "rung_ii_status": "SHELVED (parity bijection — categorical needle; reserved for the "
        "findability-cliff map; can never ground an H-A claim)",
        "translation_plant_status": "RETIRED (Step-0 receipt: 99%+ max-entropy digit "
        "marginals cap translation at 0.2769 nats — info-poor by physics)",
        "id_visibility_receipt_verbatim": ID_VISIBILITY_RECEIPT,
        "plant_from_step0": plant,
        "step0_artifact": args.step0_artifact,
        "full_stream_receipts": {
            "exact_info_nats": i_planted,
            "mc_plugin_info_nats": mi_mc_full,
            "closed_form_vs_mc_abs_diff": abs(mi_mc_full - i_planted),
            "target_marginal_kl_planted_vs_carrier": full_rec[
                "target_marginal_kl_planted_vs_carrier"
            ],
            "entropy_shift": full_rec["entropy_shift"],
            "source_marginal": p_full.tolist(),
            "carrier_target_marginal": q_full.tolist(),
        },
        "sub_epoch": sub_epoch,
        "attempts": attempts,
        "trajectory": tr,
        "decision_inputs": decision,
        "branch_taken": branch,
        "branch_semantics": {
            "a_detect_interface": "AR learns dense token-space conditionals -> v6 finding "
            "localizes to the tokenizer->AR INTERFACE (H-T); interface re-spec adjudication "
            "is the supervisor's",
            "b_clean_fail_strong_HA": "STRONG H-A: the AR could not learn a dense, graded, "
            "task-shaped lag-2 dependency in its native ordinal digit structure; pre-named "
            "next = the scaling probe (d256/d128, reduced vocab, AND rung-(ii) parity at the "
            "scale where (i)-style rules ARE learned, to map the findability cliff)",
            "c_extension": "ONE 20k extension",
            "d_spike": "halve-lr-once restart (automatic, recorded as lr_halved)",
            "e_inconclusive": "report, zero further spend, await ruling",
        },
        "recipe": {
            "carrier": "regenerated v6 noise stream (same crc32 seeds), tokenized by the "
            "frozen v6 noise-arm cell4 tokenizer",
            "noise_tokenizer": args.noise_tok,
            "plant_rng": "np.random.default_rng((0x9A47, seed, symbol_index))",
            "bars_total": args.bars_total,
            "n_per_symbol": n_per,
            "hold_frac": HOLD_FRAC,
            "lag": LAG,
            "coarse_levels": list(COARSE_LEVELS),
            "budget_steps": args.steps,
            "eval_every": args.eval_every,
            "schedule": {
                "fn": "cosine_warmup_lr",
                "peak_lr_stage2": STAGE2_PEAK_LR,
                "warmup_frac": WARMUP_FRAC,
                "floor_frac": 0.1,
            },
            "spike_nats": args.spike_nats,
            "seed": args.seed,
            "device": dev,
            "toy": bool(args.toy),
        },
    }
    out = run_dir / "token_control_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[run] manifest -> {out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--step0", action="store_true", help="local $0 stage: visibility probe + plant receipts"
    )
    ap.add_argument("--run", action="store_true", help="the box run")
    ap.add_argument("--toy", action="store_true", help="cpu machinery check of --run (tiny)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--bars-total", type=int, default=BARS_TOTAL_DEFAULT)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--spike-nats", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--carrier-sample", type=int, default=1_000_000)
    ap.add_argument("--vis-sample", type=int, default=300_000)
    ap.add_argument(
        "--target-info",
        type=float,
        default=0.9,
        help="sigma is tuned so exact info hits this (band midpoint)",
    )
    ap.add_argument("--noise-tok", default=DEFAULT_NOISE_TOK)
    ap.add_argument("--planted-tok", default=DEFAULT_PLANTED_TOK)
    ap.add_argument("--step0-artifact", default=STEP0_ARTIFACT)
    ap.add_argument("--out", default="runs/m6_token_control")
    args = ap.parse_args()
    if args.toy:
        args.run = True
        args.device = "cpu"
        args.steps = min(args.steps, 600)
        args.eval_every = min(args.eval_every, 100)
        args.bars_total = min(args.bars_total, 300_000)
    if args.step0:
        return step0(args)
    if args.run:
        return run(args)
    ap.error("pass --step0, --run, or --toy")
    return 2


if __name__ == "__main__":
    sys.exit(main())
