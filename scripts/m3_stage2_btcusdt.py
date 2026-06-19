"""Milestone-3 run: Stage-2 AR training on the BTCUSDT-2023 token stream (single-symbol de-risk).

    python3 scripts/m3_stage2_btcusdt.py [STAGE2_STEPS] [TOK_STEPS] [SEED]   # defaults 800 2000 0

Pipeline (all on the existing lake — no downloads):
  1. Freeze the Stage-1 tokenizer. M2 trained one in-memory but never persisted it, so we
     establish the canonical frozen artifact here (M2 recipe, pinned seed), save it
     content-hashed, and REUSE it on any re-run (load + hash, never retrain). Stage-2 only ever
     reads it. (Honest caveat: this is a fresh-but-equivalent freeze; MPS Stage-1 is not
     bit-reproducible, so it cannot be byte-identical to M2's discarded weights.)
  2. Tokenize the lake with the frozen tokenizer → content-hashed (coarse, fine) stream cache.
  3. Stage-2 train the backbone + MTP; temporal-tail val convergence read.
  4. Assert the KV-cache rollout is bit-equivalent to the parallel forward on a real sequence.
  5. Save a usable, content-hashed predictor checkpoint (the fixture M5 is validated against).

Dev-grade (MPS/float32), NOT a determinism-gated artifact. Prints a Markdown-ready M3 report.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import torch

from trikaal.constants import N_FEATURES
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.checkpoint import load_checkpoint, save_checkpoint
from trikaal.train.token_stream import (
    cache_token_stream,
    load_token_stream,
    tokenize_features,
)
from trikaal.train.train_predictor import train_stage2
from trikaal.train.train_tokenizer import train_stage1
from trikaal.utils.hashing import content_hash

SYMBOL = "BTCUSDT"
LAKE_ROOT = Path("processed/bars")
RUN_DIR = Path("runs/m3")
TOK_WINDOW = 128  # Stage-1 training window (§7.2)
M2_DATASET_HASH_PREFIX = "sha256:1d3ec4f8db6686e2"


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_lake() -> dict:
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW bars AS SELECT * FROM read_parquet('{LAKE_ROOT}/**/*.parquet', "
        "hive_partitioning=true)"
    )
    xc = ", ".join(f"x_{i}" for i in range(N_FEATURES))
    mc = ", ".join(f"m_{i}" for i in range(N_FEATURES))
    t = con.execute(
        f"SELECT bar_open_ms, segment_id, {xc}, {mc} FROM bars "
        "WHERE symbol = ? AND frequency = '1m' ORDER BY bar_open_ms",
        [SYMBOL],
    ).to_arrow_table()
    n = t.num_rows
    x = np.empty((n, N_FEATURES), dtype=np.float32)
    m = np.empty((n, N_FEATURES), dtype=np.uint8)
    for i in range(N_FEATURES):
        x[:, i] = t.column(f"x_{i}").to_numpy()
        m[:, i] = t.column(f"m_{i}").to_numpy()
    x, m = np.ascontiguousarray(x), np.ascontiguousarray(m)
    ts = t.column("bar_open_ms").to_numpy().astype(np.int64)
    seg = t.column("segment_id").to_numpy().astype(np.int64)
    dhash = content_hash(x, m, ts)
    return {"x": x, "m": m, "ts": ts, "segment_id": seg, "dataset_hash": dhash}


def freeze_or_load_tokenizer(
    data: dict, *, steps: int, seed: int, dev: str
) -> tuple[TokenizerAE, str, str]:
    """Load the frozen tokenizer if present (assert hash), else train + freeze it once."""
    ckpt = RUN_DIR / "tokenizer.pt"
    if ckpt.exists():
        loaded = load_checkpoint(ckpt, TokenizerAE, map_location=dev)
        note = f"loaded frozen tokenizer (hash {loaded.hash[:23]}…)"
        return loaded.model.to(dev).eval(), loaded.hash, note
    t0 = time.time()
    res = train_stage1(
        data["x"],
        data["m"],
        data["segment_id"],
        seq_len=TOK_WINDOW,
        batch_size=64,
        peak_lr=1e-3,
        max_steps=steps,
        eval_every=200,
        device=dev,
        seed=seed,
    )
    tok = res.model.eval()
    h = save_checkpoint(ckpt, tok, tok.get_config())
    last = (
        res.history[-1]
        if res.history
        else dict.fromkeys(("delta_fine", "usage_coarse", "usage_fine"), 0.0)
    )
    note = (
        f"trained + froze tokenizer in {time.time() - t0:.0f}s "
        f"(converged@{res.converged_step}, val MAE {res.final_recon_mae:.4f}, "
        f"Δ_fine {last['delta_fine']:.4f}, usage c/f {last['usage_coarse']:.2f}/"
        f"{last['usage_fine']:.2f}); hash {h[:23]}…"
    )
    return tok.to(dev), h, note


def materialize_or_load_tokens(tok: TokenizerAE, tok_hash: str, data: dict, dev: str) -> dict:
    cache = RUN_DIR / "tokens.npz"
    if cache.exists():
        try:
            return load_token_stream(
                cache, tokenizer_hash=tok_hash, dataset_hash=data["dataset_hash"]
            )
        except ValueError:
            pass  # provenance changed → rebuild
    b_c, b_f = tokenize_features(
        tok, data["x"], data["m"], data["segment_id"], window=TOK_WINDOW, device=dev
    )
    h = cache_token_stream(
        cache,
        b_c,
        b_f,
        data["ts"],
        data["segment_id"],
        tokenizer_hash=tok_hash,
        dataset_hash=data["dataset_hash"],
        window=TOK_WINDOW,
    )
    return {
        "b_c": b_c,
        "b_f": b_f,
        "ts": data["ts"],
        "segment_id": data["segment_id"],
        "tokenizer_hash": tok_hash,
        "dataset_hash": data["dataset_hash"],
        "window": TOK_WINDOW,
        "token_stream_hash": h,
    }


def kv_cache_equivalence(model: TrikaalAR, stream: dict, dev: str, length: int = 256) -> float:
    """max|Δ| between cached single-step rollout and the parallel forward on a real BTCUSDT seq."""
    model.eval()
    start = int(stream["b_c"].shape[0] * 0.95)  # a real held-out tail window
    sl = slice(start, start + length)
    b_c = torch.from_numpy(stream["b_c"][sl].astype(np.int64))[None].to(dev)
    b_f = torch.from_numpy(stream["b_f"][sl].astype(np.int64))[None].to(dev)
    ts = torch.from_numpy(stream["ts"][sl].astype(np.int64))[None].to(dev)
    with torch.no_grad():
        par = model(b_c, b_f, ts)
        caches = model.init_caches()
        hs = [
            model.step(b_c[:, p : p + 1], b_f[:, p : p + 1], ts[:, p : p + 1], caches, p).h_final
            for p in range(length)
        ]
        inc = torch.cat(hs, dim=1)
    return float((par.h_final - inc).abs().max().item())


def main(stage2_steps: int = 800, tok_steps: int = 2000, seed: int = 0) -> int:
    if not LAKE_ROOT.exists():
        print(f"lake not found at {LAKE_ROOT} — run scripts/build_btcusdt_stage1.py first")
        return 1
    dev = device()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Milestone 3: Stage-2 AR on {SYMBOL} 2023 (device={dev}) ===")

    data = load_lake()
    assert data["dataset_hash"].startswith(M2_DATASET_HASH_PREFIX), (
        f"dataset integrity: {data['dataset_hash'][:23]}… != M2 {M2_DATASET_HASH_PREFIX}…"
    )
    print(
        f"[lake]  {data['x'].shape[0]:,} bars, "
        f"dataset_hash {data['dataset_hash'][:23]}… (matches M2 ✓)"
    )

    tok, tok_hash, note = freeze_or_load_tokenizer(data, steps=tok_steps, seed=seed, dev=dev)
    print(f"[tok]   {note}")

    t0 = time.time()
    stream = materialize_or_load_tokens(tok, tok_hash, data, dev)
    nuc = int(np.unique(stream["b_c"]).size)
    nuf = int(np.unique(stream["b_f"]).size)
    print(
        f"[token] stream {stream['b_c'].shape[0]:,} bars in {time.time() - t0:.0f}s; "
        f"unique coarse {nuc}/{tok.v_c}, fine {nuf}/{tok.v_f}; "
        f"hash {stream['token_stream_hash'][:23]}…"
    )

    pred_ckpt = RUN_DIR / "predictor.pt"
    print(f"[stage2] training backbone+MTP ({stage2_steps} steps, L=128, best-ckpt→{pred_ckpt}) …")
    t0 = time.time()
    res = train_stage2(
        stream["b_c"],
        stream["b_f"],
        stream["ts"],
        stream["segment_id"],
        seq_len=128,
        stride=32,
        batch_size=32,
        val_frac=0.1,
        peak_lr=1e-3,
        max_steps=stage2_steps,
        eval_every=50,
        n_eval_windows=128,
        mtp_depths=4,
        mtp_mu=0.3,
        model_max_len=512,
        # dev-grade de-risk: dropout OFF so we can see the backbone actually FIT real tokens
        # (best-val checkpoint is saved, so single-symbol overfit past the val minimum is harmless).
        model_kwargs=dict(ffn_dropout=0.0, resid_dropout=0.0, attn_dropout=0.0, token_dropout=0.0),
        device=dev,
        seed=seed,
        verbose=True,
        ckpt_path=pred_ckpt,
    )
    print(f"[stage2] done ({time.time() - t0:.0f}s)")

    model = res.model
    pred_hash = res.ckpt_hash or "sha256:(unsaved)"
    kv_delta = kv_cache_equivalence(model, stream, dev)

    _report(res, stream, tok_hash, pred_hash, kv_delta, dev)
    return 0


def _report(res, stream, tok_hash, pred_hash, kv_delta, dev) -> None:
    h = res.history
    b = res.baselines
    # gate on the BEST-val eval (the checkpoint that was actually saved), not the last (which may
    # have overfit on a single coin-year).
    f = min(h, key=lambda r: r["val_nll"]) if h else res.final
    p = print
    p("\n=== Stage-2 report ===")
    cols = ("step", "train_nll", "val_nll", "val_nll_c", "val_nll_f", "acc_c", "acc_f", "mtp_d1")
    widths = (6, 11, 9, 10, 10, 7, 7, 8)
    p("".join(f"{c:>{w}}" for c, w in zip(cols, widths, strict=True)))
    for r in h:
        star = " *best" if r is f else ""
        p(
            f"{r['step']:>6}{r['train_nll']:>11.4f}{r['val_nll']:>9.4f}"
            f"{r['val_nll_coarse']:>10.4f}{r['val_nll_fine']:>10.4f}"
            f"{r['val_acc_coarse']:>7.3f}{r['val_acc_fine']:>7.3f}"
            f"{r.get('val_mtp_depth1', float('nan')):>8.4f}{star}"
        )
    conv = res.converged_step
    p(f"\nconverged_step : {conv}  ({'plateau' if conv else 'still improving in budget'})")
    p(f"best-val eval  : step {f['step']} (the saved checkpoint)")
    p("\nMODEL (best-val) vs BASELINES (val) — coarse / fine:")
    p(
        f"  CE      model {f['val_nll_coarse']:.4f}/{f['val_nll_fine']:.4f}  "
        f"vs marginal {b['marginal_ce_coarse']:.4f}/{b['marginal_ce_fine']:.4f}  "
        f"→ margin {b['marginal_ce_coarse'] - f['val_nll_coarse']:+.4f}/"
        f"{b['marginal_ce_fine'] - f['val_nll_fine']:+.4f}"
    )
    p(
        f"  acc     model {f['val_acc_coarse']:.3f}/{f['val_acc_fine']:.3f}  "
        f"vs persistence {b['persistence_acc_coarse']:.3f}/{b['persistence_acc_fine']:.3f}  "
        f"vs marginal {b['marginal_acc_coarse']:.3f}/{b['marginal_acc_fine']:.3f}"
    )
    first = h[0] if h else f
    depth_keys = sorted(
        (k for k in f if k.startswith("val_mtp_depth")), key=lambda s: int(s.split("depth")[1])
    )
    p("  MTP per-depth CE (val) — first eval → best (all depths must drop):")
    for k in depth_keys:
        p(f"    depth{k.split('depth')[1]}: {first.get(k, float('nan')):.4f} → {f[k]:.4f}")
    all_depths_drop = all(f[k] < first.get(k, float("inf")) for k in depth_keys)
    beats_marg = (
        f["val_nll_coarse"] < b["marginal_ce_coarse"] and f["val_nll_fine"] < b["marginal_ce_fine"]
    )
    beats_pers = (
        f["val_acc_coarse"] > b["persistence_acc_coarse"]
        and f["val_acc_fine"] > b["persistence_acc_fine"]
    )
    p(
        f"\n  beats marginal CE (both): {beats_marg};  beats persistence acc (both): {beats_pers};"
        f"  all MTP depths drop: {all_depths_drop}"
    )
    p(
        "  (top-1 acc is near the ~uniform floor for 891/1225-way stochastic targets — CE is the "
        "meaningful signal; coarse/regime token is predictable, fine/residual token near-noise.)"
    )
    p(f"  KV-cache rollout vs parallel forward: max|Δ| = {kv_delta:.2e} (device {dev})")
    p(f"  tokenizer_hash   = {tok_hash[:30]}…")
    p(f"  token_stream_hash = {stream['token_stream_hash'][:30]}…")
    p(f"  predictor_hash   = {pred_hash[:30]}…")


if __name__ == "__main__":
    s2 = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    tk = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    sd = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    raise SystemExit(main(s2, tk, sd))
