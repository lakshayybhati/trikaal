"""M6 INTERFACE RE-SPEC design pass (prereg §7 v1.4) — local, $0, receipts for every gate.

    PYTHONPATH=src python3 scripts/m6_interface_respec.py

Trains the OLD layout (contextual fine — constructed directly; the cell surface now refuses
it) and the NEW layout (pointwise fine, via build_cell_tokenizer) side by side on the SAME
regenerated v6 planted sample, then measures:

  gate 2 — PER-BAR LEGIBILITY (the measured defect, now permanent): the standing
           legibility probe (scripts/m6_canary.py, the same instrument the canary runs
           before any AR spend) on both tokenizers' ids; NEW must reach logistic
           sign-accuracy >= 0.9 (OLD reproduced ~0.51 = chance at discovery).
  gate 3 — RECONSTRUCTION SANITY: window recon MAE + per-dim MAE vs the predict-the-mean
           baseline for all 16 dims, old vs new; dim 9 non-degenerate required on NEW;
           regressions reported.
  gate 4 — G-PARITY under the new layout: assert_cells_parity() at canonical shell dims
           (bpt + params), plus old-vs-new parameter accounting.

(Gate 1, the extended flip-KAT with anti-vacuity, is CI: tests/tokenizer/
test_causal_encoder.py — run in the fast suite; its result is recorded here by reference.)

Writes runs_manifest/m6_interface_respec_design_pass.json.
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m6_canary import filler_calibration_receipt, legibility_probe, synth_symbol

from trikaal.constants import N_FEATURES
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.cells import CELLS, assert_cells_parity, build_cell_tokenizer
from trikaal.train.gates import cosine_warmup_lr
from trikaal.train.token_stream import tokenize_features
from trikaal.utils.seeding import set_determinism

N_BARS = 400_000
SEQ_LEN = 32
BATCH = 32
STAGE1_STEPS = 3000
STAGE1_PEAK_LR = 1e-3
WARMUP_FRAC = 0.1
SIGMA = 0.01
C_SIGNAL = 3.0
SIGNAL_LAG = 2
T0_MS = 1_704_067_200_000
LEGIBILITY_MIN = 0.9
ARTIFACT = "runs_manifest/m6_interface_respec_design_pass.json"


def synth_planted(n_bars: int) -> dict:
    """The CALIBRATED v6 planted fixture (gate-2 rider) — via the canary's shared generator."""
    rng = np.random.default_rng((zlib.crc32(b"planted"), 0))
    return synth_symbol(rng, planted=True, n_bars=n_bars)


def train_tokenizer(tok: TokenizerAE, d: dict, *, seed: int = 0) -> float:
    """Stage-1 reconstruction training, canary-identical schedule; returns final loss."""
    set_determinism(seed, deterministic_algorithms=False)
    opt = torch.optim.AdamW(tok.parameters(), lr=STAGE1_PEAK_LR)
    warmup = int(WARMUP_FRAC * STAGE1_STEPS)
    rng = np.random.default_rng(seed)
    n_train = int(0.8 * N_BARS)
    last = float("nan")
    for step in range(1, STAGE1_STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_warmup_lr(step, STAGE1_PEAK_LR, warmup, STAGE1_STEPS)
        starts = rng.integers(0, n_train - SEQ_LEN, size=BATCH)
        xb = np.stack([d["x"][s : s + SEQ_LEN] for s in starts])
        mb = np.stack([d["mask"][s : s + SEQ_LEN] for s in starts]).astype(np.float32)
        tok.train()
        opt.zero_grad(set_to_none=True)
        loss = tok(torch.from_numpy(xb), torch.from_numpy(mb))["loss"]
        loss.backward()
        opt.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f"stage-1 loss non-finite at step {step}")
        last = float(loss.detach())
        if step % 500 == 0:
            print(f"    step {step}: loss {last:.4f}", flush=True)
    tok.eval()
    return last


def per_dim_recon(tok: TokenizerAE, d: dict) -> dict:
    """Held-out window recon: overall MAE + per-dim MAE vs predict-the-mean baseline."""
    n_train = int(0.8 * N_BARS)
    n_win = 256
    starts = np.arange(n_train, n_train + n_win * SEQ_LEN, SEQ_LEN)
    x = torch.from_numpy(np.stack([d["x"][s : s + SEQ_LEN] for s in starts]))
    m = torch.from_numpy(np.stack([d["mask"][s : s + SEQ_LEN] for s in starts]).astype(np.float32))
    with torch.no_grad():
        cidx, fidx = tok.encode_tokens(x, m)
        x_hat = tok.decode_tokens(cidx, fidx)
    per_dim = {}
    xh = x_hat.float().numpy()
    xt = x.numpy()
    for dim in range(N_FEATURES):
        if dim >= 13:  # perp dims masked everywhere in this stream
            continue
        truth = xt[:, :, dim].ravel().astype(np.float64)
        rec = xh[:, :, dim].ravel().astype(np.float64)
        mae = float(np.abs(rec - truth).mean())
        baseline = float(np.abs(truth - truth.mean()).mean())
        per_dim[str(dim)] = {
            "recon_mae": round(mae, 5),
            "mean_baseline_mae": round(baseline, 5),
            "non_degenerate": bool(mae < 0.9 * baseline),
        }
    overall = float(np.abs(xh[:, :, :13] - xt[:, :, :13]).mean())
    return {"overall_mae_unmasked_dims": round(overall, 5), "per_dim": per_dim}


def nonlinear_diagnostics(tok: TokenizerAE, d: dict, b_c: np.ndarray, b_f: np.ndarray) -> dict:
    """What the ids carry beyond linear readability: MLP sign-acc on the digit one-hots,
    the point-decoder's own dim-9 recon corr (fine code ALONE, where present), and the
    fine-digit entropies (collapse check). These quantify the gap between the logistic
    gate and the information actually present per-bar."""
    from m6_canary import _digit_onehots

    n = min(150_000, b_c.size)
    feats = _digit_onehots(tok, b_c[:n], b_f[:n])
    s = d["x"][:n, 9].astype(np.float32)
    n_tr = int(0.8 * n)
    xt = torch.from_numpy(feats[:n_tr])
    xv = torch.from_numpy(feats[n_tr:])
    yt = torch.from_numpy((s[:n_tr] > 0).astype(np.float32))
    torch.manual_seed(0)
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
        loss = torch.nn.functional.binary_cross_entropy_with_logits(mlp(xt).squeeze(-1), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        mlp_acc = float(((mlp(xv).squeeze(-1) > 0).numpy() == (s[n_tr:] > 0)).mean())

    out = {"mlp_sign_acc_on_digits": round(mlp_acc, 4)}
    # fine-digit entropies (collapse check)
    q = tok.quant
    fine_group = list(q.fine_idx)
    radix = tok._group_radix(fine_group)
    rem = b_f[:n].copy()
    ents = []
    for pos in reversed(range(len(fine_group))):
        dig = rem % radix[pos]
        rem = rem // radix[pos]
        p = np.bincount(dig, minlength=radix[pos]).astype(np.float64)
        p /= p.sum()
        ents.append(round(float(-(p[p > 0] * np.log(p[p > 0])).sum()), 4))
    out["fine_digit_entropies_nats"] = list(reversed(ents))
    out["fine_digit_max_entropies_nats"] = [round(float(np.log(r)), 4) for r in radix]
    # the fine code ALONE: point-decoder recon corr for dim 9 (nonlinear ceiling per-bar)
    if getattr(tok, "fine_pointwise", False):
        n_win = 512
        starts = np.arange(0, n_win * SEQ_LEN, SEQ_LEN)
        x = torch.from_numpy(np.stack([d["x"][s0 : s0 + SEQ_LEN] for s0 in starts]))
        m = torch.from_numpy(
            np.stack([d["mask"][s0 : s0 + SEQ_LEN] for s0 in starts]).astype(np.float32)
        )
        with torch.no_grad():
            z = tok.latent(x, m)
            z_hat, _codes, _c, _f = tok.quant(z)
            xp = tok.point_decoder(z_hat[..., list(tok.quant.fine_idx)])
        xpn = xp.numpy().astype(np.float64)
        xtn = x.numpy().astype(np.float64)
        corrs = [
            round(float(np.corrcoef(xpn[:, :, dd].ravel(), xtn[:, :, dd].ravel())[0, 1]), 4)
            for dd in range(13)
        ]
        out["point_decoder_per_dim_corrs"] = corrs
        out["point_decoder_dim9_corr"] = corrs[9]
        rec9, tru9 = xpn[:, :, 9].ravel(), xtn[:, :, 9].ravel()
        out["point_decoder_sign_acc"] = round(float(((rec9 > 0) == (tru9 > 0)).mean()), 4)
    return out


def gate2_single(
    lam: float, seed: int, out_json: str, beta: float = 1.0, cell_path: bool = False
) -> int:
    """One (lambda, seed) λ-search trial (§7 v1.4.1 calibration): train the weighted
    pointwise-fine tokenizer on the calibrated fixture, measure state-dim legibility +
    per-dim point-decoder corrs. Direct construction (the cell surface pins λ only AFTER
    the calibration selects it)."""
    dev = "cpu"
    d = synth_planted(N_BARS)
    canonical = dict(d_model=256, n_layers=3, n_heads=4, d_ff=512, max_len=512)
    # Seed BEFORE construction: model init must be seed-determined, not ambient-RNG
    # (disclosed defect: the first calibration campaign seeded only inside train_tokenizer,
    # AFTER init, so its triplets carried unseeded-init variation and were unreproducible).
    set_determinism(seed, deterministic_algorithms=False)
    if cell_path:
        # FORMAL cell-path re-run (final adjudication item 4): construct through the REAL
        # pinned cell constructor — lambda comes from the pin, never from the CLI.
        tok = build_cell_tokenizer(CELLS[3], **canonical).to(dev)
    else:
        tok = TokenizerAE(
            quantizer="fsq",
            n_features=16,
            encoder_causal=True,
            fine_pointwise=True,
            micro_point_weight=lam,
            point_loss_coef=beta,
            **canonical,
        ).to(dev)
    loss = train_tokenizer(tok, d, seed=seed)
    b_c, b_f = tokenize_features(
        tok, d["x"], d["mask"], d["segment_id"], window=SEQ_LEN, batch_windows=256, device=dev
    )
    leg = legibility_probe(tok, d["x"][:, 9], b_c, b_f)
    diag = nonlinear_diagnostics(tok, d, b_c, b_f)
    recon = per_dim_recon(tok, d)
    rec = {
        "lambda": tok.get_config()["micro_point_weight"],
        "beta_point_loss_coef": beta,
        "constructed_via": "build_cell_tokenizer (pinned path)" if cell_path else "direct",
        "seed": seed,
        "stage1_final_loss": round(loss, 4),
        "legibility_logistic_sign_acc": round(leg, 4),
        "point_decoder_per_dim_corrs": diag.get("point_decoder_per_dim_corrs"),
        "recon_per_dim": recon["per_dim"],
        "overall_recon_mae": recon["overall_mae_unmasked_dims"],
    }
    Path(out_json).write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"[gate2] lam={lam} seed={seed}: legibility {leg:.4f} -> {out_json}", flush=True)
    return 0


def main() -> int:
    t0 = time.time()
    dev = "cpu"
    print(f"[respec] regenerating v6 planted sample ({N_BARS:,} bars, symbol-0 seeds)…")
    d = synth_planted(N_BARS)

    canonical = dict(d_model=256, n_layers=3, n_heads=4, d_ff=512, max_len=512)
    out: dict = {
        "artifact": "m6_interface_respec_design_pass",
        "prereg": "§7 v1.4 + gate-2 calibration rider",
        "filler_calibration": filler_calibration_receipt(),
    }
    print(f"[respec] filler calibration: {json.dumps(out['filler_calibration'])}", flush=True)

    def measure(tok, seed_note: str) -> dict:
        b_c, b_f = tokenize_features(
            tok, d["x"], d["mask"], d["segment_id"], window=SEQ_LEN, batch_windows=256, device=dev
        )
        leg = legibility_probe(tok, d["x"][:, 9], b_c, b_f)
        recon = per_dim_recon(tok, d)
        diag = nonlinear_diagnostics(tok, d, b_c, b_f)
        print(
            f"[respec] {seed_note}: legibility {leg:.4f}, dim9 recon {recon['per_dim']['9']}, "
            f"diag {json.dumps(diag)}",
            flush=True,
        )
        return {
            "legibility_logistic_sign_acc": round(leg, 4),
            "recon": recon,
            "nonlinear_diagnostics": diag,
        }

    results: dict = {}
    # old contextual-fine contrast (one seed — the defect receipt)
    print(f"[respec] training old_contextual_fine (canonical dims, {STAGE1_STEPS} steps)…")
    tok = TokenizerAE(
        quantizer="fsq", n_features=16, encoder_causal=True, fine_pointwise=False, **canonical
    ).to(dev)
    old_loss = train_tokenizer(tok, d, seed=0)
    results["old_contextual_fine"] = {
        "config": tok.get_config(),
        "n_params": sum(p.numel() for p in tok.parameters()),
        "stage1_final_loss": round(old_loss, 4),
        **measure(tok, "old_contextual_fine seed0"),
    }

    # NEW layout: the rider's 3-seed protocol — ALL THREE must clear (the lottery must be
    # gone, not re-rolled until won)
    seed_runs = {}
    for seed in (0, 1, 2):
        print(f"[respec] training new_pointwise_fine seed {seed} ({STAGE1_STEPS} steps)…")
        tok = build_cell_tokenizer(CELLS[3], **canonical).to(dev)
        loss = train_tokenizer(tok, d, seed=seed)
        rec = measure(tok, f"new_pointwise_fine seed{seed}")
        pd_corrs = rec["nonlinear_diagnostics"].get("point_decoder_per_dim_corrs", [])
        seed_runs[str(seed)] = {
            "stage1_final_loss": round(loss, 4),
            **rec,
            "channel_receipt_dims_ge_090": int(sum(1 for c in pd_corrs if c >= 0.9)),
        }
    results["new_pointwise_fine_3seed"] = {
        "config": tok.get_config(),
        "n_params": sum(p.numel() for p in tok.parameters()),
        "seeds": seed_runs,
    }
    out["tokenizers"] = results

    print("[respec] gate 4: G-parity under the new layout (canonical shells)…")
    parity = assert_cells_parity(**canonical)
    out["g_parity_new_layout"] = parity
    print(f"[respec] parity: {parity}")

    seeds = results["new_pointwise_fine_3seed"]["seeds"]
    old = results["old_contextual_fine"]
    legs = {s: v["legibility_logistic_sign_acc"] for s, v in seeds.items()}
    out["gates"] = {
        "gate1_flip_kat": "CI — tests/tokenizer/test_causal_encoder.py "
        "(test_cell_fine_tokens_are_per_bar both quantizers + contextual-fine anti-vacuity "
        "+ not-constructible); run via the fast suite, result recorded in the report",
        "gate2_legibility": {
            "threshold": LEGIBILITY_MIN,
            "protocol": "3 seeds, ALL must clear (rider: the lottery must be gone, "
            "not re-rolled until won)",
            "new_by_seed": legs,
            "old": old["legibility_logistic_sign_acc"],
            "old_at_discovery": 0.5135,
            "pass": bool(all(v >= LEGIBILITY_MIN for v in legs.values())),
        },
        "channel_receipt_non_gating": {
            "dims_ge_090_by_seed": {s: v["channel_receipt_dims_ge_090"] for s, v in seeds.items()},
            "expectation": ">= 4 dims at point-decoder corr >= 0.9 (evidence, not a gate)",
        },
        "gate3_recon": {
            "new_dim9_non_degenerate_by_seed": {
                s: v["recon"]["per_dim"]["9"]["non_degenerate"] for s, v in seeds.items()
            },
            "old_overall_mae": old["recon"]["overall_mae_unmasked_dims"],
            "pass": bool(all(v["recon"]["per_dim"]["9"]["non_degenerate"] for v in seeds.values())),
        },
        "gate4_parity": {
            "d_bpt": parity["d_bpt"],
            "param_frac": parity["param_frac"],
            "pass": True,  # assert_cells_parity raises on violation
        },
    }
    out["gates"]["PROCEED"] = bool(
        out["gates"]["gate2_legibility"]["pass"] and out["gates"]["gate3_recon"]["pass"]
    )
    out["wall_s"] = round(time.time() - t0, 1)
    Path(ARTIFACT).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[respec] gates: {json.dumps(out['gates'], indent=1)}")
    print(f"[respec] artifact -> {ARTIFACT}")
    return 0 if out["gates"]["PROCEED"] else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gate2-only", action="store_true")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--cell-path", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="")
    a = ap.parse_args()
    if a.gate2_only:
        sys.exit(gate2_single(a.lam, a.seed, a.out_json, beta=a.beta, cell_path=a.cell_path))
    sys.exit(main())
