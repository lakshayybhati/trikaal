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
from m6_canary import legibility_probe

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
    """The v6 planted generator, verbatim (symbol 0 seeds)."""
    rng = np.random.default_rng((zlib.crc32(b"planted"), 0))
    state = rng.standard_normal(n_bars)
    eps = rng.standard_normal(n_bars)
    r = SIGMA * eps
    r[SIGNAL_LAG:] = r[SIGNAL_LAG:] + C_SIGNAL * SIGMA * state[:-SIGNAL_LAG]
    x = rng.standard_normal((n_bars, N_FEATURES)).astype(np.float32)
    x[:, 0] = (r / SIGMA).astype(np.float32)
    x[:, 9] = state.astype(np.float32)
    m = np.zeros((n_bars, N_FEATURES), dtype=np.uint8)
    m[:, 13:16] = 1
    return {"x": x, "mask": m, "segment_id": np.zeros(n_bars, dtype=np.int64)}


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
        rec9 = xp[:, :, 9].reshape(-1).numpy().astype(np.float64)
        tru9 = x[:, :, 9].reshape(-1).numpy().astype(np.float64)
        out["point_decoder_dim9_corr"] = round(float(np.corrcoef(rec9, tru9)[0, 1]), 4)
        out["point_decoder_sign_acc"] = round(float(((rec9 > 0) == (tru9 > 0)).mean()), 4)
    return out


def main() -> int:
    t0 = time.time()
    dev = "cpu"
    print(f"[respec] regenerating v6 planted sample ({N_BARS:,} bars, symbol-0 seeds)…")
    d = synth_planted(N_BARS)

    canonical = dict(d_model=256, n_layers=3, n_heads=4, d_ff=512, max_len=512)
    out: dict = {"artifact": "m6_interface_respec_design_pass", "prereg": "§7 v1.4"}

    results = {}
    for name, make in (
        (
            "old_contextual_fine",
            lambda: TokenizerAE(
                quantizer="fsq",
                n_features=16,
                encoder_causal=True,
                fine_pointwise=False,
                **canonical,
            ),
        ),
        ("new_pointwise_fine", lambda: build_cell_tokenizer(CELLS[3], **canonical)),
    ):
        print(f"[respec] training {name} (canonical dims, {STAGE1_STEPS} steps, cpu)…")
        tok = make().to(dev)
        n_params = sum(p.numel() for p in tok.parameters())
        final_loss = train_tokenizer(tok, d)
        b_c, b_f = tokenize_features(
            tok,
            d["x"],
            d["mask"],
            d["segment_id"],
            window=SEQ_LEN,
            batch_windows=256,
            device=dev,
        )
        leg = legibility_probe(tok, d["x"][:, 9], b_c, b_f)
        recon = per_dim_recon(tok, d)
        diag = nonlinear_diagnostics(tok, d, b_c, b_f)
        results[name] = {
            "config": tok.get_config(),
            "n_params": n_params,
            "stage1_final_loss": round(final_loss, 4),
            "legibility_logistic_sign_acc": round(leg, 4),
            "recon": recon,
            "nonlinear_diagnostics": diag,
        }
        print(
            f"[respec] {name}: legibility {leg:.4f}, overall recon MAE "
            f"{recon['overall_mae_unmasked_dims']}, dim9 {recon['per_dim']['9']}",
            flush=True,
        )
    out["tokenizers"] = results

    print("[respec] gate 4: G-parity under the new layout (canonical shells)…")
    parity = assert_cells_parity(**canonical)
    out["g_parity_new_layout"] = parity
    print(f"[respec] parity: {parity}")

    new = results["new_pointwise_fine"]
    old = results["old_contextual_fine"]
    out["gates"] = {
        "gate1_flip_kat": "CI — tests/tokenizer/test_causal_encoder.py "
        "(test_cell_fine_tokens_are_per_bar both quantizers + contextual-fine anti-vacuity "
        "+ not-constructible); run via the fast suite, result recorded in the report",
        "gate2_legibility": {
            "threshold": LEGIBILITY_MIN,
            "new": new["legibility_logistic_sign_acc"],
            "old": old["legibility_logistic_sign_acc"],
            "old_at_discovery": 0.5135,
            "pass": bool(new["legibility_logistic_sign_acc"] >= LEGIBILITY_MIN),
        },
        "gate3_recon": {
            "new_dim9_non_degenerate": new["recon"]["per_dim"]["9"]["non_degenerate"],
            "overall_mae_new_vs_old": [
                new["recon"]["overall_mae_unmasked_dims"],
                old["recon"]["overall_mae_unmasked_dims"],
            ],
            "pass": bool(new["recon"]["per_dim"]["9"]["non_degenerate"]),
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
    sys.exit(main())
