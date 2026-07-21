"""μ̂ estimator comparison (acceptance adjudication item 1, $0 local) — MODE vs MEAN.

    PYTHONPATH=src python3 scripts/m6_mu_estimator_comparison.py

Named hypothesis (supervisor): ``predict_mu``'s greedy-argmax rollout is a MODE estimator;
μ̂ is pre-registered as a conditional MEAN; under skew the mode is biased — the acceptance
fixture made it constant-sign (mu mean −0.097 vs std 0.036 at h=15 → every decision short).

Three estimators, identical rollout scaffolding (context fill, k=1 exclusion, σ_t scaling):
  (a) argmax        — the current greedy chain + greedy decode (the mode);
  (b) expectation   — greedy chain, but each step decodes the EXPECTED latent
                      E[z] = softmax(logits_c) @ grid_c ⊕ softmax(logits_f|cond) @ grid_f
                      through the decoder (mean-decode; chain stays modal);
  (c) mc mean       — n_samples=32 sampled chains (coarse ~ multinomial via step(sample=True),
                      fine ~ multinomial | sampled coarse), decode each, average (pinned seed).

Surfaces: the acceptance fixture (planted cell4 stage-1 checkpoints, eval-region slice) AND
a real-lake BTCUSDT slice through the anchored M3 model. Receipts: per-method mu mean/std,
corr(mu, fwdret) (exact label window: bars t+2..t+h), and (fixture) corr(mu, state).

Writes runs_manifest/m6_mu_estimator_comparison.json.
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m6_canary import SEQ_LEN as FIX_SEQ_LEN
from m6_canary import synth_symbol

from trikaal.data.config import BAR_MS
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.checkpoint import load_checkpoint
from trikaal.train.token_stream import tokenize_features

MC_SAMPLES = 32
MC_SEED = 20260721
ARTIFACT = "runs_manifest/m6_mu_estimator_comparison.json"


def group_grid_table(tok: TokenizerAE, group: list[int], v: int, device) -> torch.Tensor:
    """[v, len(group)] dequantized latent values for every id of a subtoken group —
    exactly decode_tokens' unflatten + dequantize, vectorized over the whole vocabulary."""
    q = tok.quant
    radix = tok._group_radix(list(group))
    ids = torch.arange(v, dtype=torch.long)
    codes = torch.zeros(v, len(group), dtype=torch.long)
    rem = ids.clone()
    for pos in reversed(range(len(group))):
        codes[:, pos] = rem % radix[pos]
        rem = rem // radix[pos]
    if tok.quantizer == "bsq":
        z = (2.0 * codes.to(torch.float32) - 1.0) / (float(q.dim) ** 0.5)
    else:
        levels = q.levels_t[list(group)].cpu()
        half = (levels - 1.0) / 2.0
        z = codes.to(levels.dtype) - half
    return z.to(device)


@torch.no_grad()
def mu_three_ways(
    model, tok, b_c, b_f, ts, sigma, decisions, *, h, seq_len, device, mc_samples=MC_SAMPLES
) -> dict:
    """All three estimators over one decision batch, sharing the context fill."""
    dev = torch.device(device)
    d = np.asarray(decisions, dtype=np.int64)
    n = d.shape[0]
    offs = np.arange(-seq_len + 1, 1)
    idx = d[:, None] + offs[None, :]
    cc = torch.from_numpy(b_c[idx].astype(np.int64)).to(dev)
    cf = torch.from_numpy(b_f[idx].astype(np.int64)).to(dev)
    cts = torch.from_numpy(ts[idx].astype(np.int64)).to(dev)
    sig_t = sigma[d].astype(np.float64)
    q = tok.quant
    t_c = group_grid_table(tok, list(q.coarse_idx), tok.v_c, dev)
    t_f = group_grid_table(tok, list(q.fine_idx), tok.v_f, dev)

    def fill_context():
        caches = model.init_caches()
        out = None
        for p in range(seq_len):
            out = model.step(cc[:, p : p + 1], cf[:, p : p + 1], cts[:, p : p + 1], caches, p)
        return caches, out

    ts_dec = cts[:, -1:]

    # ---- (a) argmax (the current estimator) + (b) expectation decode on the same chain ----
    caches, out = fill_context()
    cum_a = torch.zeros(n, dtype=torch.float32, device=dev)
    cum_b = torch.zeros(n, dtype=torch.float32, device=dev)
    for k in range(1, h + 1):
        pc = out.coarse_cond
        pf = out.logits_f.argmax(dim=-1)
        if k >= 2:
            x_hat = tok.decode_tokens(pc, pf)
            cum_a = cum_a + x_hat[:, 0, 0]
            p_c = F.softmax(out.logits_c.float(), dim=-1)  # [n,1,v_c]
            p_f = F.softmax(out.logits_f.float(), dim=-1)
            z = torch.zeros(n, 1, q.dim, dtype=t_c.dtype, device=dev)
            z[..., list(q.coarse_idx)] = (p_c @ t_c.to(p_c.dtype)).to(z.dtype)
            z[..., list(q.fine_idx)] = (p_f @ t_f.to(p_f.dtype)).to(z.dtype)
            cum_b = cum_b + tok.decode_latent(z)[:, 0, 0]
        if k < h:
            out = model.step(pc, pf, ts_dec + k * BAR_MS, caches, seq_len - 1 + k)
    mu_a = cum_a.cpu().numpy().astype(np.float64) * sig_t
    mu_b = cum_b.cpu().numpy().astype(np.float64) * sig_t

    # ---- (c) small-MC mean: sampled chains, pinned seed --------------------------------
    torch.manual_seed(MC_SEED)
    acc = torch.zeros(n, dtype=torch.float32, device=dev)  # mps lacks f64; /32 in f64 on cpu
    for _s in range(mc_samples):
        caches, out = fill_context()
        cum = torch.zeros(n, dtype=torch.float32, device=dev)
        for k in range(1, h + 1):
            probs_c = F.softmax(out.logits_c.float(), dim=-1).reshape(n, -1)
            pc = torch.multinomial(probs_c, 1).reshape(n, 1)
            logits_f = model.fine_logits(out.h_final, pc)
            probs_f = F.softmax(logits_f.float(), dim=-1).reshape(n, -1)
            pf = torch.multinomial(probs_f, 1).reshape(n, 1)
            if k >= 2:
                cum = cum + tok.decode_tokens(pc, pf)[:, 0, 0]
            if k < h:
                out = model.step(pc, pf, ts_dec + k * BAR_MS, caches, seq_len - 1 + k)
        acc = acc + cum
    mu_c = (acc.cpu().numpy().astype(np.float64) / mc_samples) * sig_t

    return {"argmax": mu_a, "expectation": mu_b, "mc_mean": mu_c}


def stats(mu: np.ndarray, fwd: np.ndarray, state: np.ndarray | None) -> dict:
    out = {
        "mu_mean": round(float(mu.mean()), 6),
        "mu_std": round(float(mu.std()), 6),
        "frac_negative": round(float((mu < 0).mean()), 4),
        "corr_mu_fwdret": round(float(np.corrcoef(mu, fwd)[0, 1]), 4),
    }
    if state is not None:
        out["corr_mu_state"] = round(float(np.corrcoef(mu, state)[0, 1]), 4)
    return out


def main() -> int:
    t0 = time.time()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out: dict = {
        "artifact": "m6_mu_estimator_comparison",
        "hypothesis": "greedy-argmax rollout = MODE estimator; mu-hat is pre-registered as "
        "the conditional MEAN; under skew the mode is biased",
        "mc": {"n_samples": MC_SAMPLES, "seed": MC_SEED},
        "device": dev,
    }

    # ---- surface 1: the acceptance fixture --------------------------------------------------
    print("[cmp] fixture: regenerating planted eval slice + loading acceptance checkpoints…")
    rng = np.random.default_rng((zlib.crc32(b"planted"), 0))
    d = synth_symbol(rng, planted=True, n_bars=14_000_000)
    lo, n_ctx = 9_200_000, 120_000
    tok = (
        load_checkpoint(
            "runs_cloud/acceptance/m6_acceptance/planted/cell4_fsq_micro_seed0/tokenizer.pt",
            TokenizerAE,
            map_location=dev,
        )
        .model.to(dev)
        .eval()
    )
    mod = (
        load_checkpoint(
            "runs_cloud/acceptance/m6_acceptance/planted/cell4_fsq_micro_seed0/predictor.pt",
            TrikaalAR,
            map_location=dev,
        )
        .model.to(dev)
        .eval()
    )
    b_c, b_f = tokenize_features(
        tok,
        d["x"][lo : lo + n_ctx],
        d["mask"][lo : lo + n_ctx],
        d["segment_id"][lo : lo + n_ctx],
        window=FIX_SEQ_LEN,
        batch_windows=256,
        device=dev,
    )
    dec = np.arange(FIX_SEQ_LEN, FIX_SEQ_LEN + 1500 * 64, 64, dtype=np.int64)
    res_fix: dict = {}
    for h in (2, 15):
        mus = mu_three_ways(
            mod,
            tok,
            b_c,
            b_f,
            d["ts"][lo : lo + n_ctx],
            d["sigma"][lo : lo + n_ctx],
            dec,
            h=h,
            seq_len=FIX_SEQ_LEN,
            device=dev,
        )
        st = d["x"][lo : lo + n_ctx][dec, 9]
        fwd = np.array([d["raw_ret_close"][lo + t + 2 : lo + t + 1 + h].sum() for t in dec])
        res_fix[f"h{h}"] = {m: stats(mu, fwd, st) for m, mu in mus.items()}
        print(f"[cmp] fixture h={h}: " + json.dumps(res_fix[f"h{h}"]))
    out["fixture_planted_cell4"] = res_fix

    # ---- surface 2: real-lake BTCUSDT through the anchored M3 model -------------------------
    print("[cmp] real lake: BTCUSDT through the anchored M3 checkpoints…")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from m5_eval_harness import RUN_DIR, load_lake

    lake = load_lake()
    tok_l = load_checkpoint(RUN_DIR / "tokenizer.pt", TokenizerAE, map_location=dev)
    pred_l = load_checkpoint(RUN_DIR / "predictor.pt", TrikaalAR, map_location=dev)
    tok_r = tok_l.model.to(dev).eval()
    mod_r = pred_l.model.to(dev).eval()
    seq_r = 128
    b_cr, b_fr = tokenize_features(
        tok_r,
        lake["x"],
        lake["mask"],
        lake["segment_id"],
        window=seq_r,
        batch_windows=256,
        device=dev,
    )
    n_bars = lake["x"].shape[0]
    lo_r = int(0.75 * n_bars)
    dec_r = np.arange(lo_r, min(n_bars - 70, lo_r + 600 * 60), 60, dtype=np.int64)
    res_real: dict = {}
    for h in (15,):
        mus = mu_three_ways(
            mod_r,
            tok_r,
            b_cr,
            b_fr,
            lake["ts"],
            lake["sigma"],
            dec_r,
            h=h,
            seq_len=seq_r,
            device=dev,
        )
        fwd = np.array([lake["raw_ret_close"][t + 2 : t + 1 + h].sum() for t in dec_r])
        res_real[f"h{h}"] = {m: stats(mu, fwd, None) for m, mu in mus.items()}
        print(f"[cmp] BTCUSDT h={h}: " + json.dumps(res_real[f"h{h}"]))
    out["real_btcusdt_m3_model"] = res_real

    fix15 = res_fix["h15"]
    out["verdict"] = {
        "drift_removed_by_expectation": bool(
            abs(fix15["expectation"]["mu_mean"])
            < 2 * fix15["expectation"]["mu_std"] / np.sqrt(len(dec))
            or abs(fix15["expectation"]["mu_mean"]) < 0.2 * abs(fix15["argmax"]["mu_mean"])
        ),
        "drift_removed_by_mc": bool(
            abs(fix15["mc_mean"]["mu_mean"]) < 0.2 * abs(fix15["argmax"]["mu_mean"])
        ),
    }
    out["wall_s"] = round(time.time() - t0, 1)
    Path(ARTIFACT).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[cmp] verdict: {out['verdict']}")
    print(f"[cmp] artifact -> {ARTIFACT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
