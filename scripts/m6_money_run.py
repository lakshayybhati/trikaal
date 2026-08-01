"""§7 v1.6 C-5 — THE MONEY DRIVER. The only path that may produce the real M6 cell artifacts.

    PYTHONPATH=src .venv/bin/python scripts/m6_money_run.py --dry-run     # $0, full path, 0 steps
    PYTHONPATH=src .venv/bin/python scripts/m6_money_run.py --shard 0/5   # one box of five

WHY IT EXISTS. Pre-flight Item 2 promises ZERO first-time code paths on rented hardware, and
until now the money configuration had no driver at all: ``m6_toy_rehearsal.py`` differs from it on
symbols, window, seeds, step budget, grid, conformance and the legibility gate. Validating the
rehearsal validates the wrong code. So this driver exists, and the outstanding CUDA validation is
re-scoped to run THROUGH it at reduced scale — same code, smaller inputs.

THE FOUR RULES IT IS BUILT TO (C-5 A1-A3, A8):

A1  IT DOES NOT DUPLICATE THE REHEARSAL. Every step of the matrix — train, reload-proof, resume,
    score, emit — lives once in ``trikaal.run.matrix`` and both drivers call it. Duplication is
    precisely how C-6 happened: two files each holding their own copy of the truth.
A2  NO FLAG MAY ALTER A PINNED VALUE. ``argparse`` carries operational concerns ONLY — output
    directory, shard, resume, dry-run, device. There is no ``--seeds``, no ``--seq-len``, no
    ``--steps``, no ``--symbols``, no ``--cost``. A flag that can move a pin makes the pin
    decoration, which is C-7's lesson and must not be re-learned here. Enforced by a KAT that
    reads this file's own argparse.
A3  ``assert_conformance`` IS THE FIRST ACTION, BEFORE ANY COMPUTE, over the FULL money surface —
    symbols, blocks, grid, κ, cost model, deciles, bootstrap recipe — not merely the
    orchestrator's own fields. The training side has until now sat entirely outside that gate. A
    divergent config must fail in SECONDS, not after hours on a rented box.
A8  INVENT NO VALUE. Anything not pinned is READ from code and RECORDED, never chosen. See
    ``unpinned_parameters()`` below, which also records WHY each is legitimately unpinned so that
    a later reader cannot mistake an unpinned parameter for an unconsidered one.

WHAT IS DELIBERATELY NOT HERE: the verdict. This driver emits the 25 artifacts + index;
``scripts/m6_verdict.py`` remains the only code path allowed to produce SURVIVES/NULL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from trikaal.data.universe_loader import calendar_boundary_ms, connect_lake
from trikaal.eval.conformance import (
    MDE_INPUTS,
    PINNED_SEEDS,
    ConformanceError,
    assert_conformance,
    money_config,
    symbols_sha256,
)
from trikaal.eval.verdict import PRIMARY_H
from trikaal.eval.xsection import SymbolEval, primary_region_grid_ms
from trikaal.run.matrix import (
    MatrixSpec,
    all_units,
    eval_matrix,
    load_symbol_arrays,
    reload_checkpoints,
    shard_partition_failures,
    shard_units,
    train_matrix,
    windows_by_arm,
    write_index_if_complete,
)
from trikaal.train.cells import CELLS
from trikaal.train.orchestrator import OrchestratorConfig
from trikaal.train.tripwire import TripwireConfig, TripwireMonitor
from trikaal.utils.provenance import run_provenance

LAKE = Path("processed/universe_bars")


def unpinned_parameters(orch: OrchestratorConfig) -> dict:
    """Values READ from code, with the reason each is legitimately unpinned (C-5 A8).

    Recorded rather than chosen. The reasoning travels WITH the numbers so a later reader cannot
    mistake an unpinned parameter for an unconsidered one."""
    return {
        "values": {
            "steps_stage1": orch.steps_stage1,
            "steps_stage2": orch.steps_stage2,
            "peak_lr_stage1": orch.peak_lr_stage1,
            "peak_lr_stage2": orch.peak_lr_stage2,
            "warmup_frac": orch.warmup_frac,
            "batch_size": orch.batch_size,
            "alpha": orch.alpha,
        },
        "source": "trikaal.train.orchestrator.OrchestratorConfig defaults — read, never chosen",
        "why_ablation_validity_does_not_depend_on_them": (
            "ALL FIVE CELLS SHARE THEM. The 2x2+placebo contrast is matched on everything except "
            "quantizer x arm, so a step budget or learning rate that is too small weakens every "
            "arm identically and cannot manufacture or destroy the paired difference DIR(4-5). "
            "Only REPRODUCIBILITY depends on these, which is why they are recorded here rather "
            "than pinned."
        ),
        "stage2_budget_is_derivable_not_arbitrary": (
            "m6_design.md:59 — ~1 effective pass over 270-300M effective bars at canonical "
            "geometry is 16,479-18,311 steps/stage. The default 2000 is BELOW that band; the "
            "operative budget is an operator decision recorded per run, not a spec constant."
        ),
        "stage1_budget_is_genuinely_free": (
            "no design document derives it; the tokenizer is trained to a reconstruction "
            "criterion and the §7 v1.4 legibility gate is the binding quality check on it."
        ),
    }


def _pinned_inputs() -> dict:
    doc = json.loads(MDE_INPUTS.read_text())
    return {
        "symbols": tuple(doc["symbols_sampled"]),
        "window": tuple(doc["window"]),
        "train_frac": float(doc["train_frac"]),
    }


def build_parser() -> argparse.ArgumentParser:
    """OPERATIONAL FLAGS ONLY (A2). Nothing here can move a pinned value."""
    ap = argparse.ArgumentParser(description="M6 money run — the pinned §3a configuration")
    ap.add_argument("--out", type=Path, default=Path("runs/m6_money"), help="output directory")
    ap.add_argument("--shard", default="0/1", help="i/N — partition the 25 units across boxes")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--no-resume", action="store_true", help="recompute units that already exist")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="execute the FULL path with ZERO training steps (A9) — $0 local proof",
    )
    ap.add_argument("--lake", type=Path, default=LAKE, help="lake root (operational)")
    ap.add_argument("--wandb", default="disabled", choices=("disabled", "offline", "online"))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t_start = time.time()
    try:
        shard_i, n_shards = (int(x) for x in str(args.shard).split("/"))
    except ValueError:
        print(f"--shard must be i/N, got {args.shard!r}", file=sys.stderr)
        return 2

    pinned = _pinned_inputs()
    symbols = list(pinned["symbols"])
    cfg = money_config(h=PRIMARY_H, device=args.device)

    # ---- A3: CONFORMANCE IS THE FIRST ACTION, BEFORE ANY COMPUTE ---------------------------
    # The full money surface — symbols, blocks, grid, kappas, cost model, deciles, bootstrap
    # recipe — not merely the orchestrator's own fields. A divergent config fails in SECONDS.
    try:
        assert_conformance(cfg, symbols=symbols, seeds=tuple(PINNED_SEEDS))
    except ConformanceError as exc:
        print(f"CONFORMANCE FAILED — nothing was computed:\n{exc}", file=sys.stderr)
        return 1
    print(
        f"[gate]  conformance PASS over {len(symbols)} symbols, seeds {tuple(PINNED_SEEDS)}",
        flush=True,
    )
    print(f"[gate]  symbols sha256 {symbols_sha256(symbols)[:16]}…")

    # ---- the orchestrator config: pins only, plus operational fields -----------------------
    orch = OrchestratorConfig(
        device=args.device,
        out_dir=args.out,
        wandb_mode=args.wandb,
        wandb_group="m6_money",
        autocast_bf16=args.device.startswith("cuda"),
        # A9: a dry run executes every step of the path with ZERO training work.
        steps_stage1=0 if args.dry_run else OrchestratorConfig.steps_stage1,
        steps_stage2=0 if args.dry_run else OrchestratorConfig.steps_stage2,
        # A dry run cannot clear a gate that needs a trained tokenizer, and must not pretend to.
        micro_legibility_min=None if args.dry_run else OrchestratorConfig.micro_legibility_min,
        money_run=not args.dry_run,
    )

    units = all_units(CELLS, PINNED_SEEDS)
    part_fails = shard_partition_failures(units, n_shards)
    if part_fails:
        print("SHARD PARTITION INVALID: " + "; ".join(part_fails), file=sys.stderr)
        return 1
    mine = shard_units(units, shard_i, n_shards)
    print(
        f"[shard] {shard_i}/{n_shards} → {len(mine)} of {len(units)} units: "
        + ", ".join(f"c{c.cell_id}s{s}" for c, s in mine)
    )

    manifest_path = args.out / "money_run_manifest.json"
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- data --------------------------------------------------------------------------------
    con = connect_lake(args.lake) if args.lake.exists() else None
    if con is None:
        print(f"LAKE MISSING at {args.lake} — refusing to invent data", file=sys.stderr)
        return 1
    boundary = calendar_boundary_ms(pinned["window"][0], pinned["window"][1], pinned["train_frac"])
    # tail_bars=None: the money configuration trains on the FULL pinned region, unlike the
    # rehearsal which deliberately slices a tail. Same function, different argument (C-5 A1).
    raw = {s: load_symbol_arrays(con, s, boundary_ms=boundary, tail_bars=None) for s in symbols}
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

    def windows_for_seed(seed: int) -> dict:
        # LAZY (A9): eager construction across all seeds needs ~95 GiB at the money
        # configuration's 84.15M bars. Same inputs, same numbers, one seed resident.
        return windows_by_arm(raw, seq_len=orch.seq_len, boundary_ms=boundary, seed=seed)

    grid = primary_region_grid_ms(cfg)

    spec = MatrixSpec(
        orch=orch,
        base_eval_cfg=cfg,
        sym_evals=sym_evals,
        windows_for_seed=windows_for_seed,
        art_dir=args.out / "eval",
        symbols=tuple(symbols),
        eval_window=tuple(pinned["window"]),
        grid_start_ms=int(grid[0]),
        grid_n_periods=int(grid.size),
        units=mine,
        label="m6_money_run",
    )

    monitor = TripwireMonitor(TripwireConfig(k_first_steps=min(100, max(1, orch.steps_stage1))))
    manifests = train_matrix(spec, monitor)
    ckpt_hashes = reload_checkpoints(spec, manifests)
    print(f"[ckpt]  {len(ckpt_hashes)} checkpoints reloaded OK", flush=True)
    outcome = eval_matrix(spec, ckpt_hashes, resume=not args.no_resume)
    complete = write_index_if_complete(spec, outcome.entries, len(units))
    print(
        f"[eval]  {len(outcome.entries)} artifacts ({len(outcome.resumed)} resumed); "
        f"index {'WRITTEN — matrix complete' if complete else 'withheld — shard is partial'}"
    )

    doc = {
        "driver": "m6_money_run",
        "schema": "m6_money_run_v1",
        "shard": {"index": shard_i, "n_shards": n_shards, "units": len(mine)},
        "conformance": "PASS (assert_conformance, first action, full money surface)",
        "symbols_sha256": symbols_sha256(symbols),
        "n_symbols": len(symbols),
        "eval_window": list(pinned["window"]),
        "grid": {"h": PRIMARY_H, "start_ms": int(grid[0]), "n_periods": int(grid.size)},
        "seeds": list(PINNED_SEEDS),
        "seq_len": orch.seq_len,
        "dry_run": bool(args.dry_run),
        "unpinned_parameters": unpinned_parameters(orch),
        "provenance": run_provenance(args.device),
        "eval_artifacts": outcome.entries,
        "resumed_units": outcome.resumed,
        "resume_refusals": outcome.refusals,
        "eval_seconds": outcome.seconds,
        "eval_decisions": outcome.decisions,
        "index_written": complete,
        "wall_s": round(time.time() - t_start, 1),
    }
    manifest_path.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
    print(f"[manifest] → {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
