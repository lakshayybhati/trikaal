"""M6 pre-flight Item 2 ★ — ONE continuous 5-cell × N-seed toy run on the REAL CUDA box.

    PYTHONPATH=src python3 scripts/m6_toy_rehearsal.py --device cuda [--steps 300] [--wandb online]
    # --seeds 0,1,2,3,4 is REQUIRED for the verdict path (it wants 5 cells x verdict.DSR_SEEDS)
    PYTHONPATH=src python3 scripts/m6_toy_rehearsal.py --local-smoke     # cpu script validation

The entire real-run arc, once, small, on the actual rented GPU type (docs/m6_preflight.md
Item 2): real lake symbols (incl. a thin coin + a delisted tail), the REAL canonical dims
(21,301,248-param backbone, G-parity enforced), bf16 autocast, W&B, the multi-symbol fold-legal
draw at the REAL 2021→2025/0.7 boundary, tripwires armed, resumable state — then every trained
(cell, seed) is scored through the MONEY-MODE CODE PATH (money=True: continuous blocks-1–5
grid of the toy eval window, no cap, per-symbol deciles) and its verdict artifact is written
through ``verdict.write_cell_eval_artifact`` (with the codebook diagnostic) so that
``scripts/m6_verdict.py --artifacts <out>/eval --allow-toy-grid`` runs end-to-end on the box —
zero first-time events extends to the DECISION path.

Numbers are garbage by design (few hundred steps); the deliverables are: it ran unattended,
nvidia-smi + determinism mode in the log, one reloadable checkpoint hash per (cell, seed),
the thin coin drawn
with its ^0.5 down-weight, Cell 5 verifiably trained on permuted micro, one W&B group with all
cells' curves, measured into-GPU throughput + utilization (pre-flight Item 4's inputs), and a
content-hashed bundle. Writes ``<out>/rehearsal_manifest.json`` (durable, never stdout-only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from trikaal.constants import N_FEATURES
from trikaal.data.universe_loader import build_symbol_windows, calendar_boundary_ms, connect_lake
from trikaal.eval.conformance import DECILES, cell_tokenizer_failures
from trikaal.eval.costs import load_spread_deciles
from trikaal.eval.verdict import (
    DSR_HORIZONS,
    PRIMARY_H,
    write_cell_eval_artifact,
    write_eval_index,
)
from trikaal.eval.xsection import SymbolEval, XSectionConfig, primary_region_grid_ms, score_cell
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import ARM_MICRO, ARM_MICRO_SHUFFLED, ARMS
from trikaal.train.cells import CELLS, assert_cells_parity
from trikaal.train.checkpoint import load_checkpoint
from trikaal.train.orchestrator import OrchestratorConfig, run_cell
from trikaal.train.tripwire import TripwireConfig, TripwireMonitor
from trikaal.utils.throughput import (
    BASIS_END_TO_END,
    BASIS_END_TO_END_EVAL,
    throughput_record,
)

LAKE = Path("processed/universe_bars")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "KLAYUSDT", "FRONTUSDT")  # deep×3 + tail + THIN
THIN = "FRONTUSDT"
TRAIN_WINDOW = ("2021-01-01", "2025-01-01")  # the REAL fold geometry (boundary 2023-10-20)
TRAIN_FRAC = 0.7
EVAL_WINDOW = ("2024-01-01", "2025-01-01")  # the TOY money-code-path region (grid_pinned=false)
TRAIN_TAIL_BARS = 200_000
SEEDS = (0, 1, 2)


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # diagnostics only, never fatal
        return f"<{type(e).__name__}: {e}>"


class GpuSampler(threading.Thread):
    """nvidia-smi utilization/memory sampler (Item 4's GPU-not-starved evidence)."""

    def __init__(self, every_s: float = 5.0):
        super().__init__(daemon=True)
        self.every_s = every_s
        self.samples: list[dict] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            out = _sh(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            )
            parts = [p.strip() for p in out.split(",")]
            if len(parts) == 3 and parts[0].isdigit():
                self.samples.append(
                    {"util": int(parts[0]), "mem_used_mb": int(parts[1]), "t": time.time()}
                )
            self._stop.wait(self.every_s)

    def stop(self) -> dict:
        self._stop.set()
        utils = [s["util"] for s in self.samples]
        mems = [s["mem_used_mb"] for s in self.samples]
        return {
            "n_samples": len(self.samples),
            "util_mean": float(np.mean(utils)) if utils else None,
            "util_max": int(np.max(utils)) if utils else None,
            "mem_used_mb_max": int(np.max(mems)) if mems else None,
        }


def load_symbol(con, symbol: str, boundary_ms: int) -> dict:
    """Full arrays sliced to [boundary − TRAIN_TAIL_BARS, end) — train tail + full eval region."""
    xcols = ", ".join(f"x_{i}" for i in range(N_FEATURES))
    mcols = ", ".join(f"m_{i}" for i in range(N_FEATURES))
    t = con.execute(
        f"SELECT bar_open_ms, segment_id, sigma, raw_ret_close, {xcols}, {mcols} "
        "FROM bars WHERE symbol = ? ORDER BY bar_open_ms",
        [symbol],
    ).to_arrow_table()
    n = t.num_rows
    ts = t.column("bar_open_ms").to_numpy().astype(np.int64)
    lo = max(0, int(np.searchsorted(ts, boundary_ms)) - TRAIN_TAIL_BARS)
    x = np.empty((n, N_FEATURES), dtype=np.float32)
    m = np.empty((n, N_FEATURES), dtype=np.uint8)
    for i in range(N_FEATURES):
        x[:, i] = t.column(f"x_{i}").to_numpy()
        m[:, i] = t.column(f"m_{i}").to_numpy()
    return {
        "x": x[lo:],
        "mask": m[lo:],
        "ts": ts[lo:],
        "segment_id": t.column("segment_id").to_numpy().astype(np.int64)[lo:],
        "sigma": t.column("sigma").to_numpy().astype(np.float64)[lo:],
        "raw_ret_close": t.column("raw_ret_close").to_numpy().astype(np.float64)[lo:],
    }


def assert_cell5_trains_on_permuted_micro(per_symbol_by_arm: dict) -> dict:
    """Item 2: 'the shuffled-micro arm verifiably trained on *permuted* micro (not real)'.

    For every symbol: the shuffled arm's ACTIVE micro entries must differ from the real arm's
    in order, while the sorted multiset is preserved (a permutation, not a rewrite). The arm
    layouts differ (micro drops nothing; both are 16-wide before select_arm)."""
    out = {}
    for real, shuf in zip(
        per_symbol_by_arm[ARM_MICRO], per_symbol_by_arm[ARM_MICRO_SHUFFLED], strict=True
    ):
        assert real.symbol == shuf.symbol
        moved, preserved = False, True
        for dim in range(7, 13):
            a = np.asarray(real.x[:, dim])
            b = np.asarray(shuf.x[:, dim])
            if not np.array_equal(a, b):
                moved = True
            if not np.array_equal(np.sort(a), np.sort(b)):
                preserved = False
        if not (moved and preserved):
            raise AssertionError(
                f"Cell-5 input check FAILED for {real.symbol}: moved={moved}, "
                f"marginals_preserved={preserved} — the placebo arm is not a permutation"
            )
        out[real.symbol] = {"micro_permuted": True, "marginals_preserved": True}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    # default None so --local-smoke can supply its own default WITHOUT clobbering an explicit
    # --steps. It used to overwrite unconditionally, so `--local-smoke --steps 200` silently ran
    # 30 steps — and 30 steps is not enough for the loss to drop, so the tripwire (correctly)
    # aborted on whichever seed drew unluckily. A flag that is accepted and discarded is the same
    # defect as the probe's dropped --warmup.
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--wandb", default="online", choices=["disabled", "offline", "online"])
    ap.add_argument("--flash", action="store_true", help="prefer flash2 (post-bench decision)")
    ap.add_argument("--out", type=Path, default=Path("runs/m6_toy"))
    ap.add_argument(
        "--local-smoke",
        action="store_true",
        help="cpu validation of THIS SCRIPT: tiny shells, short steps, wandb disabled",
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="reuse the already-trained checkpoints in --out; skip training (the eval-cost "
        "fix: full-window sdpa rollouts measured ~40 min/(cell,seed) on the 4090 — the money "
        "CODE PATH needs exercising, not the toy window's length)",
    )
    ap.add_argument(
        "--no-eval-resume",
        action="store_true",
        help="recompute every (cell, seed) even if a valid artifact exists (default: resume)",
    )
    ap.add_argument(
        "--symbols",
        default=None,
        help="comma-separated lake symbols; default SYMBOLS. Every symbol must have pre-boundary "
        "history — post-boundary listings have zero fold-legal train windows and the sampler "
        "raises.",
    )
    ap.add_argument(
        "--seeds",
        default=None,
        help="comma-separated seeds; default keeps the historical (0,1,2). NOTE: the VERDICT path "
        "requires exactly verdict.DSR_SEEDS — m6_verdict.py builds want_names over 5 cells x "
        "DSR_SEEDS = 25 artifacts — so a run that must reach a verdict word has to pass 0,1,2,3,4.",
    )
    ap.add_argument(
        "--eval-window",
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="override the toy eval window (default EVAL_WINDOW)",
    )
    args = ap.parse_args()
    t_start = time.time()
    global SEEDS, SYMBOLS
    if args.seeds:
        SEEDS = tuple(int(x) for x in args.seeds.split(","))
    if args.symbols:
        SYMBOLS = tuple(x.strip() for x in args.symbols.split(",") if x.strip())
    eval_window = tuple(args.eval_window) if args.eval_window else EVAL_WINDOW

    if args.local_smoke:
        args.device, args.batch = "cpu", 16
        args.seq_len, args.wandb = 32, "disabled"
        if args.steps is None:
            args.steps = 30
    elif args.steps is None:
        args.steps = 300
    if not LAKE.exists():
        print("missing lake — copy processed/universe_bars in first (runbook §3)")
        return 1

    canonical = not args.local_smoke
    tok_kw = (
        {"max_len": max(512, args.seq_len)}
        if canonical
        else dict(d_model=16, n_layers=1, n_heads=2, d_ff=32, max_len=64)
    )
    # RoPE caches must cover the eval rollout: seq_len + h_max (predict_mu guards this; the
    # 512 default would crash at position ~571 — the rehearsal-sizing catch, fixed pre-CUDA)
    bb_extra = {"max_len": args.seq_len + 64}
    bb_kw = (
        bb_extra
        if canonical
        else dict(
            n_layers=1,
            d_model=16,
            d_ff=32,
            n_heads=2,
            mtp_depths=0,
            ffn_dropout=0.0,
            resid_dropout=0.0,
            attn_dropout=0.0,
            token_dropout=0.0,
            **bb_extra,
        )
    )

    print(f"=== M6 TOY REHEARSAL (device={args.device}) — NUMBERS MEANINGLESS BY DESIGN ===")
    smi = _sh(["nvidia-smi"]) if args.device.startswith("cuda") else "<cpu run — no GPU>"
    print(smi.splitlines()[0] if smi and not smi.startswith("<") else smi)

    # ---- lake → full arrays; fold-legal training windows at the REAL boundary --------------
    boundary = calendar_boundary_ms(*TRAIN_WINDOW, TRAIN_FRAC)
    con = connect_lake(LAKE)
    raw = {}
    for s in SYMBOLS:
        raw[s] = load_symbol(con, s, boundary)
        print(f"[lake] {s}: {raw[s]['x'].shape[0]:,} bars loaded")

    def windows_for_seed(seed: int) -> dict:
        return {
            arm: [
                build_symbol_windows(
                    s,
                    d["x"],
                    d["mask"],
                    d["segment_id"],
                    d["ts"],
                    arm=arm,
                    seq_len=args.seq_len,
                    boundary_ms=boundary,
                    seed=seed,
                )
                for s, d in raw.items()
            ]
            for arm in ARMS
        }

    per_seed_windows = {
        seed: windows_for_seed(seed) for seed in (SEEDS if not args.eval_only else SEEDS[:1])
    }
    cell5_check = assert_cell5_trains_on_permuted_micro(per_seed_windows[SEEDS[0]])
    print(f"[cell5] permuted-micro training input verified for {len(cell5_check)} symbols")
    for sw in per_seed_windows[SEEDS[0]][ARM_MICRO]:
        print(f"[draw]  {sw.symbol}: {sw.n_windows:,} fold-legal train windows")

    # ---- 5 cells × 3 seeds through the REAL orchestrator path ------------------------------
    cfg = OrchestratorConfig(
        seeds=SEEDS,
        seq_len=args.seq_len,
        batch_size=args.batch,
        # toy-rehearsal shells — the §7 v1.4 standing gate binds the REAL M6 run (default
        # 0.9); toy dims on lake slices are not the gated configuration
        micro_legibility_min=None,
        steps_stage1=args.steps,
        steps_stage2=args.steps,
        tok_kwargs=tok_kw,
        backbone_kwargs=bb_kw,
        expect_backbone_params=21_301_248 if canonical else None,
        enforce_parity=canonical,
        device=args.device,
        prefer_flash=args.flash,
        autocast_bf16=args.device.startswith("cuda"),
        out_dir=args.out,
        state_every=max(50, args.steps // 4),
        wandb_mode=args.wandb,
        wandb_group="m6_toy_rehearsal",
    )
    parity = assert_cells_parity(**cfg.tok_kwargs) if cfg.enforce_parity else None
    monitor = TripwireMonitor(TripwireConfig(k_first_steps=min(100, args.steps)))
    sampler = GpuSampler()
    if args.device.startswith("cuda"):
        sampler.start()
    run_walls: dict[str, float] = {}
    manifests = {}
    if args.eval_only:
        for spec in CELLS:
            for seed in SEEDS:
                mp = Path(cfg.out_dir) / f"{spec.name}_seed{seed}" / "run_manifest.json"
                manifests[f"{spec.name}_seed{seed}"] = json.loads(mp.read_text())
        print(f"[eval-only] reusing {len(manifests)} trained runs from {cfg.out_dir}")
    for spec in CELLS if not args.eval_only else []:
        for seed in SEEDS:
            t0 = time.time()
            m = run_cell(spec, seed, per_seed_windows[seed], cfg, monitor)
            run_walls[m["run"]] = round(time.time() - t0, 1)
            manifests[m["run"]] = m
            print(
                f"[train] {m['run']}: s1={m['final_loss']['stage1']:.3f} "
                f"s2={m['final_loss']['stage2']:.3f} wall={run_walls[m['run']]}s "
                f"mode={m['determinism']['attention_mode']}"
            )
    gpu_stats = sampler.stop() if args.device.startswith("cuda") else None

    # ---- reload every checkpoint (the reloadable-proof) ------------------------------------
    ckpt_hashes = {}
    for run, m in manifests.items():
        rd = Path(cfg.out_dir) / run
        load_checkpoint(rd / "tokenizer.pt", TokenizerAE, map_location=args.device)
        load_checkpoint(rd / "predictor.pt", TrikaalAR, map_location=args.device)
        ckpt_hashes[run] = m["artifacts"]
    print(f"[ckpt] all {len(ckpt_hashes)} (cell, seed) checkpoints reloaded OK")

    # ---- money-CODE-PATH eval per (cell, seed) → the verdict artifacts ---------------------
    deciles = load_spread_deciles(DECILES)
    base_cfg = XSectionConfig(
        window_start=eval_window[0],
        window_end=eval_window[1],
        train_frac=TRAIN_FRAC,
        h=PRIMARY_H,
        seq_len=args.seq_len,
        cap_per_symbol=None,
        spread_frac_by_symbol=deciles,
        money=True,  # the money MECHANICS on a toy window — grid_pinned=false by design
        device=args.device,
    )
    grid = primary_region_grid_ms(base_cfg)
    art_dir = Path(cfg.out_dir) / "eval"
    art_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, str] = {}
    eval_secs: dict[str, float] = {}  # per (cell, seed) — the last unmeasured run-cost input
    eval_decisions: dict[str, dict] = {}  # per (cell, seed) — decisions actually SCORED
    resumed_units: list[str] = []  # units skipped because a valid artifact already existed
    sym_evals = [
        SymbolEval(
            symbol=s,
            x=d["x"],
            mask=d["mask"],
            segment_id=d["segment_id"],
            ts=d["ts"],
            sigma=d["sigma"],
            raw_ret_close=d["raw_ret_close"],
        )
        for s, d in raw.items()
    ]
    for spec in CELLS:
        for seed in SEEDS:
            t_eval0 = time.time()
            # ---- EVAL RESUME, at (cell, seed) granularity -------------------------------
            # The eval loop previously had NO existence check: a restart recomputed all 25
            # units from scratch. Against a multi-hour leg on preemptible infrastructure that
            # loses the entire elapsed run to one preemption. The per-(cell, seed) artifact is
            # already atomic and content-hashed, so it IS the natural checkpoint — this just
            # uses it. An artifact is only honoured if it parses AND carries the schema key;
            # a truncated write is recomputed rather than trusted.
            done_path = art_dir / f"cell{spec.cell_id}_seed{seed}_eval.json"
            if not args.no_eval_resume and done_path.is_file():
                try:
                    doc = json.loads(done_path.read_text())
                    if doc.get("schema") == "m6_cell_eval_v1" and doc.get("cell_id") is not None:
                        entries[done_path.name] = hashlib.sha256(done_path.read_bytes()).hexdigest()
                        resumed_units.append(done_path.name)
                        print(f"[eval]  RESUMED {done_path.name} (already complete — skipped)")
                        continue
                except (json.JSONDecodeError, OSError) as exc:
                    print(f"[eval]  {done_path.name} unreadable ({exc}) — recomputing", flush=True)
            rd = Path(cfg.out_dir) / f"{spec.name}_seed{seed}"
            tok = load_checkpoint(rd / "tokenizer.pt", TokenizerAE, map_location=args.device)
            pred = load_checkpoint(rd / "predictor.pt", TrikaalAR, map_location=args.device)
            model, tokm = pred.model.to(args.device).eval(), tok.model.to(args.device).eval()
            # §7 v1.3 conformance: the eval-loaded tokenizer must BE causal — hard stop
            fails = cell_tokenizer_failures(tokm.get_config(), run=f"{spec.name}_seed{seed}")
            if fails:
                print("EVAL REFUSED — " + "; ".join(fails), file=sys.stderr)
                return 1
            val_by_h, kappa_by_h, head_score = {}, {}, None
            dec_by_h: dict[int, int] = {}
            for h in DSR_HORIZONS:
                sc = score_cell(
                    f"{spec.name}_seed{seed}_h{h}",
                    model,
                    tokm,
                    spec.arm,
                    sym_evals,
                    replace(base_cfg, h=h, seed=seed),
                    val_only=(h != PRIMARY_H),
                )
                val_by_h[h] = sc.val_ir_by_kappa
                kappa_by_h[h] = sc.kappa_chosen
                dec_by_h[h] = int(sc.n_decisions)
                if h == PRIMARY_H:
                    head_score = sc
            name, sha = write_cell_eval_artifact(
                art_dir,
                cell_id=spec.cell_id,
                seed=seed,
                grid={"h": PRIMARY_H, "start_ms": int(grid[0]), "n_periods": int(grid.size)},
                headline_series=head_score.headline_series,
                kappa_star_by_h=kappa_by_h,
                val_ir_by_kappa_by_h=val_by_h,
                codebook=head_score.codebook,
                meta={"checkpoints": ckpt_hashes[f"{spec.name}_seed{seed}"]},
            )
            entries[name] = sha
            eval_secs[f"{spec.name}_seed{seed}"] = round(time.time() - t_eval0, 3)
            eval_decisions[f"{spec.name}_seed{seed}"] = {
                "by_h": dict(dec_by_h),
                # the time bought ALL scored horizons, so the rate's denominator is their sum
                "total_scored": int(sum(dec_by_h.values())),
                "primary_h": int(dec_by_h.get(PRIMARY_H, 0)),
            }
            print(
                f"[eval]  {spec.name}_seed{seed}: IR@0.30%={head_score.ir_headline:+.2f} "
                f"κ*={head_score.kappa_chosen} decisions={head_score.n_decisions} "
                f"effbits={head_score.codebook['effective_bits_per_token']:.2f} → {name}"
            )
    write_eval_index(art_dir, entries)
    # COUNT, not a literal: --seeds makes this 5 x len(SEEDS), and a hardcoded "15" printed while
    # 10 files were written is prose contradicting its own datum — the defect this project polices.
    print(
        f"[eval] {len(entries)} verdict artifacts (5 cells x {len(SEEDS)} seeds) + index → "
        f"{art_dir} (feed to scripts/m6_verdict.py)"
    )

    # ---- throughput (Item 4 inputs) + the content-hashed bundle ----------------------------
    # Finding 0: this block is where the M6 cost basis was LOST. Under --eval-only there is no
    # training to time, so the fields were written NaN/0 — over the top of the training
    # invocation's manifest at the same path. The measurement did not fail; it was overwritten.
    # Two structural fixes: (1) the numbers are a utils.throughput record, so an unmeasured rate
    # is NaN with measured=False and can never be quoted as a datum; (2) an eval-only re-run
    # CARRIES FORWARD a prior finite training record instead of clobbering it.
    total_steps = len(manifests) * 2 * args.steps
    train_wall = sum(run_walls.values())
    steps_per_s = total_steps / train_wall if train_wall > 0 else float("nan")
    bars_per_s = steps_per_s * args.batch * args.seq_len
    tput = throughput_record(
        steps=total_steps if train_wall > 0 else None,
        batch=args.batch,
        seq_len=args.seq_len,
        wall_s=train_wall if train_wall > 0 else None,
        basis=BASIS_END_TO_END,
        device=args.device,
        note="both stages pooled across all (cell, seed) runs; wall-clock",
    )
    carried_from = None
    prior_path = Path(cfg.out_dir) / "rehearsal_manifest.json"
    if not tput["measured"] and prior_path.exists():
        try:
            prior = json.loads(prior_path.read_text()).get("throughput", {})
            prior_rec = prior.get("record") if isinstance(prior, dict) else None
            if isinstance(prior_rec, dict) and prior_rec.get("measured"):
                tput, carried_from = dict(prior_rec), "prior invocation at this out_dir"
        except (json.JSONDecodeError, OSError) as exc:  # a broken prior must not mask the NaN
            print(f"[throughput] prior manifest unreadable ({exc}) — no carry-forward", flush=True)
    total_decisions_scored = sum(d["total_scored"] for d in eval_decisions.values())
    thin_drawn = {
        run: m["draw"]["drawn_by_symbol_stage1"].get(THIN, 0) for run, m in manifests.items()
    }
    h = hashlib.sha256()
    for name in sorted(entries):
        h.update((art_dir / name).read_bytes())
    h.update((art_dir / "index.json").read_bytes())
    for run in sorted(manifests):
        h.update(json.dumps(manifests[run], sort_keys=True, default=str).encode())
    bundle_sha = h.hexdigest()

    manifest = {
        "rehearsal": "m6_preflight_item2",
        "device": args.device,
        "nvidia_smi": smi,
        "gpu_sampler": gpu_stats,
        "canonical_dims": canonical,
        "parity": parity,
        "config": {
            "symbols": list(SYMBOLS),
            "thin_coin": THIN,
            "train_window": list(TRAIN_WINDOW),
            "eval_window": list(eval_window),
            "eval_only_rerun": bool(args.eval_only),
            "seq_len": args.seq_len,
            "batch": args.batch,
            "steps_per_stage": args.steps,
            "seeds": list(SEEDS),
            "wandb": {"mode": args.wandb, "project": cfg.wandb_project, "group": cfg.wandb_group},
            "backbone_max_len": bb_kw["max_len"],
        },
        "determinism_by_run": {r: m["determinism"] for r, m in manifests.items()},
        "cell5_permuted_micro": cell5_check,
        "draw_thin_coin_stage1": thin_drawn,
        "checkpoints": ckpt_hashes,
        "eval_artifacts": entries,
        "eval_grid": {"h": PRIMARY_H, "start_ms": int(grid[0]), "n_periods": int(grid.size)},
        "throughput": {
            "record": tput,  # the numeric, basis-tagged datum (utils.throughput) — quote THIS
            "carried_forward_from": carried_from,
            "train_wall_s": round(train_wall, 1),
            "total_train_steps": total_steps,
            "steps_per_s": round(steps_per_s, 3),
            "bars_per_s_into_gpu": round(bars_per_s, 1),
            "per_run_wall_s": run_walls,
            "note": "bars/s = steps/s x batch x seq_len (both stages pooled). Quote "
            "throughput.record — the flat fields below it are legacy and carry NO basis tag. "
            "Under --eval-only there is no training to time: the record is carried forward from "
            "the prior invocation if one is finite, else it is measured=False with NaN rates. "
            "The attention bench is NOT a substitute: it is basis=compute_only_step (pre-resident "
            "batches, no loader/tokenize/eval/checkpoint) and is an UPPER BOUND, not a cost basis.",
        },
        # v1.6: the eval leg as NUMBERS. Its own receipt said the absolute rate was "PENDING the
        # 4090"; this is that measurement. Per (cell, seed) seconds plus a basis-tagged record
        # under the SAME measured/is_costable discipline as training — eval is real run cost, so
        # end_to_end_eval is costable, while a compute-only micro-benchmark still is not.
        "eval_leg": {
            # PRIMARY FIELD = SECONDS PER DECISION, not the per-(cell,seed) total. A total
            # measured over a few thousand TOY decisions cost-estimates nothing for the real
            # headline grid (~35,064 periods x 40 symbols ~= 1.4M decisions per cell-seed) —
            # quoting it as "the eval cost" would be the compute-only bench in a new costume.
            # money mode FORBIDS cap_per_symbol (xsection.py:71, §3a uncapped primary), so there
            # is NO knob to bound the real eval leg: the per-decision RATE is the only thing that
            # says what it costs, and the total is a DERIVED value.
            "seconds_per_decision": (
                round(sum(eval_secs.values()) / max(1, total_decisions_scored), 9)
                if total_decisions_scored
                else float("nan")
            ),
            "total_decisions_scored": total_decisions_scored,
            "DENOMINATOR_SEMANTICS": (
                "decisions are produced at the PRIMARY h only; the other DSR horizons are scored "
                "val_only (kappa selection) and report 0 decisions. So the rate is SECONDS PER "
                "PRIMARY-H DECISION *INCLUDING* the cost of the val-only passes at the other "
                "horizons. That is the correct unit to extrapolate with, because the real run "
                "also pays for all scored horizons per (cell, seed) — but do NOT re-divide it by "
                "a horizon count, and do not compare it to a rate measured over a different "
                "horizon set."
            ),
            "decisions_by_run": eval_decisions,
            "grid_dims": {
                "n_periods": int(grid.size),
                "n_symbols": len(sym_evals),
                "horizons_scored": list(DSR_HORIZONS),
                "primary_h": PRIMARY_H,
                "eval_window": list(eval_window),
                "periods_x_symbols": int(grid.size) * len(sym_evals),
            },
            "resumed_units": resumed_units,
            "n_resumed": len(resumed_units),
            "seconds_by_run": eval_secs,
            "n_runs": len(eval_secs),
            "mean_seconds_per_cell_seed": (
                round(sum(eval_secs.values()) / len(eval_secs), 3) if eval_secs else float("nan")
            ),
            "total_seconds": round(sum(eval_secs.values()), 3) if eval_secs else float("nan"),
            "chunk": "predict_mu default (recipe pins chunk=512)",
            "horizons_scored": list(DSR_HORIZONS),
            "primary_h": PRIMARY_H,
            "eval_window": list(eval_window),
            "n_symbols": len(sym_evals),
            "grid_n_periods": int(grid.size),
            "record": throughput_record(
                steps=len(eval_secs),
                batch=1,
                seq_len=1,
                wall_s=sum(eval_secs.values()) if eval_secs else None,
                basis=BASIS_END_TO_END_EVAL,
                device=args.device,
                note=(
                    "steps = (cell, seed) scoring passes; bars_per_s here reads as PASSES/s. "
                    "Scale by grid_n_periods x n_symbols to project the real eval leg."
                ),
            ),
        },
        "wall_s_total": round(time.time() - t_start, 1),
        "bundle_sha256": bundle_sha,
        "note": "NUMBERS MEANINGLESS BY DESIGN — the gate is the machinery, end-to-end, "
        "unattended, on the real GPU type (m6_preflight Item 2).",
    }
    out = Path(cfg.out_dir) / "rehearsal_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    print(
        f"[throughput] {steps_per_s:.2f} steps/s → {bars_per_s:,.0f} bars/s into-GPU; "
        f"util mean={gpu_stats['util_mean'] if gpu_stats else 'n/a'}"
    )
    print(f"[bundle] sha256:{bundle_sha[:16]}… — manifest → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
