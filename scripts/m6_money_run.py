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

A9 AND A4 ARE DIFFERENT JOBS. CONFLATING THEM IS WHAT BROKE THE FIRST DRY RUN, so it is written
down here rather than left to be rediscovered:

  **A9 — the dry run.** LOCAL, $0, COMPLETE PATH COVERAGE. The question is *does every stage
  execute and hand off to the next* — conformance, config assembly, symbol load, gate arming,
  training loop, checkpoint round-trip, scoring, artifact schema, index, verdict wiring. It is a
  WIRING proof. It must take minutes.
  **A4 — the GPU validation.** REDUCED SCALE ON RENTED HARDWARE. The question is *does it produce
  the right numbers*. That is a NUMERICAL proof and it costs money.

The first version of ``--dry-run`` zeroed the training steps but ran the REAL money eval —
1,402,560 decisions per unit at h=15, 8.4 h on a 4090 — i.e. it tried to do A4's job on A9's
budget, and cost hours per unit instead of minutes while proving nothing A4 would not prove
better. A dry run therefore DECLARES ``money_run=False`` and scores a deliberately SHORTENED grid:
the real scoring path over a tiny number of decisions covers strictly more than a synthetic score
would, at a cost A9 can actually pay.

WHY THAT IS NOT AN A2 VIOLATION. A2 protects the MONEY run's config from being altered by a flag.
``money_run`` is a DECLARATION, never an inference (C-6), and a dry run declares itself not a
money run — so the money surface, including ``xsection.py:71``'s cap prohibition, does not apply
to it. The mechanism already exists; this uses it rather than inventing a second one.

AND A DRY-RUN ARTIFACT IS STRUCTURALLY UN-QUOTABLE, not merely labelled: it carries
``money_run: false`` and ``grid_pinned: false``, and ``verdict._validate_artifact`` REFUSES to
assemble any artifact so marked. Labelling alone would leave "someone reads the manifest
carefully" as the only thing between a path proof and a reported result.

WHAT IS DELIBERATELY NOT HERE: the verdict. This driver emits the 25 artifacts + index;
``scripts/m6_verdict.py`` remains the only code path allowed to produce SURVIVES/NULL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
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
from trikaal.utils.preflight import (
    STRATEGIES,
    STRATEGY_FAST,
    STRATEGY_LOW,
    memory_requirement_bytes,
    physical_ram_bytes,
    resource_failures,
    resource_record,
)
from trikaal.utils.provenance import run_provenance

LAKE = Path("processed/universe_bars")
# A9 bounds. Small enough that the WIRING proof takes minutes; large enough that scoring, kappa
# selection and the artifact schema are all genuinely exercised on real data.
# THE REAL BOUND IS THE CAP, and it is the only one. A first draft also sliced the recorded grid,
# which looked like a second bound and was not: ``score_cell`` builds its own grids from ``cfg``,
# so a truncated grid changed only the METADATA and would have made the artifact's n_periods
# disagree with its own headline_series. Removed rather than kept as decoration.
DRY_RUN_CAP = 8  # decisions per symbol — legal ONLY because money_run is declared False
# The DATA is bounded too, and for the same reason. A9 requires COMPLETE PATH COVERAGE in MINUTES;
# the pinned 40 symbols are 84,153,600 bars, ~170 min just to load and 27.8 GiB to hold. Conformance
# still asserts the FULL pinned surface first — that is the A3 proof and it is not weakened — and
# only the subsequent LOAD is bounded, on the same declaration that lifts the cap prohibition.
DRY_RUN_SYMBOLS = 4


def unpinned_parameters(orch: OrchestratorConfig) -> dict:
    """Values READ from code, with the reason each is legitimately unpinned (C-5 A8).

    Recorded rather than chosen. The reasoning travels WITH the numbers so a later reader cannot
    mistake an unpinned parameter for an unconsidered one.

    **§7 v1.6.25 (RE-AUDIT R7) — THIS RECORD SHIPPED FALSE PROSE, UNDER TEST PROTECTION.** It said
    *"The default 2000 is BELOW that band; the operative budget is an operator decision recorded
    per run, not a spec constant"* while the `values` dict beside it read **26,003** and
    `conformance.PINNED_STEPS_STAGE1/2` gated exactly that number. Both halves were written into
    the same manifest and the assertion in `tests/run/test_money_driver.py` PINNED THE STALE
    SENTENCE, so the suite defended the wrong statement. The step budgets are no longer unpinned
    at all — v1.6.22 made them a matched, conformance-gated spec constant — so they have been
    moved out of the unpinned set rather than re-described inside it.
    """
    return {
        # STILL genuinely unpinned: read from code, shared by all 25 units, never chosen per-cell.
        "values": {
            "peak_lr_stage1": orch.peak_lr_stage1,
            "peak_lr_stage2": orch.peak_lr_stage2,
            "warmup_frac": orch.warmup_frac,
            "batch_size": orch.batch_size,
            "alpha": orch.alpha,
        },
        # NOT unpinned any more, recorded here so the manifest still carries them in one place.
        "pinned_since_v1_6_22": {
            "steps_stage1": orch.steps_stage1,
            "steps_stage2": orch.steps_stage2,
            "gate": "conformance.PINNED_STEPS_STAGE1/2 — a divergence fails before any compute",
            "why": (
                "m6_design.md:59 derives ~1 effective pass over 270-300M effective bars at "
                "canonical geometry as 16,479-18,311 steps/stage, and the blueprint spec states "
                "1-3 passes. The predecessor default of 2000 was BELOW that band — it was the "
                "SMOKE-TEST budget, and the money run would have inherited it. 26,003 steps is "
                "20.00 tokens/param (Chinchilla ~20) = 1.399 passes. MATCHED AND FIXED across all "
                "25 units: saturation is MEASURED AND REPORTED, never a stopping rule, because "
                "data-dependent stopping would give 25 different budgets and confound DIR(4-5) "
                "with training length (the C-12 class in the primary)."
            ),
        },
        "source": "trikaal.train.orchestrator.OrchestratorConfig defaults — read, never chosen",
        "why_ablation_validity_does_not_depend_on_them": (
            "ALL FIVE CELLS SHARE THEM. The 2x2+placebo contrast is matched on everything except "
            "quantizer x arm, so a step budget or learning rate that is too small weakens every "
            "arm identically and cannot manufacture or destroy the paired difference DIR(4-5). "
            "Only REPRODUCIBILITY depends on these, which is why they are recorded here rather "
            "than pinned."
        ),
        "stage1_budget_is_not_independently_derivable": (
            "no design document derives a stage-1 step count; the tokenizer is trained to a "
            "reconstruction criterion and the §7 v1.4 legibility gate is the binding quality "
            "check on it. It is pinned to the SAME 26,003 as stage 2 so the two stages cannot "
            "drift apart across shards, not because 26,003 is derived for stage 1."
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
    # MEMORY STRATEGY IS A PROPERTY OF THE HOST, NOT OF THE SPECIFICATION. Both strategies compute
    # byte-identical results and differ only in residency and rebuild count, so this is operational
    # and cannot move a pinned value. "auto" picks from the machine's physical RAM.
    ap.add_argument("--mem-strategy", default="auto", choices=("auto", *STRATEGIES))
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
    print(f"[gate]  symbols sha256 {symbols_sha256(symbols)[:16]}…", flush=True)

    # A9 option (a), and ONLY AFTER the gate above has asserted the full money surface. The dry
    # run DECLARES itself not a money run, which is what lifts xsection.py:71's cap prohibition;
    # REAL scoring over a shortened grid, because a synthetic score would skip the path A9 exists
    # to prove. A SEPARATE object: `cfg` remains the money surface that was gated, `eval_cfg` is
    # what the dry run actually scores, so the two can never be confused.
    # (The first draft bounded `cfg` BEFORE the gate and conformance rejected it — the gate
    # catching the driver's own ordering bug is exactly why it runs first.)
    eval_cfg = replace(cfg, money=False, cap_per_symbol=DRY_RUN_CAP) if args.dry_run else cfg

    # ---- RESOURCE PRECONDITIONS, before any compute (§7 v1.6 C-5, ruling 2) -----------------
    # Same principle as conformance-first. A COUNT(*) is cheap; discovering the requirement
    # thirty minutes into a swap-thrash on a rented box is not, and does not stop the meter.
    con = connect_lake(args.lake) if Path(args.lake).exists() else None
    if con is None:
        print(f"LAKE MISSING at {args.lake} — refusing to invent data", file=sys.stderr)
        return 1
    # Conformance above ran over the FULL pinned 40; only the LOAD is bounded, and only in a run
    # that has declared itself not a money run.
    load_symbols = symbols[:DRY_RUN_SYMBOLS] if args.dry_run else symbols
    n_bars = int(
        con.execute("SELECT COUNT(*) FROM bars WHERE symbol IN ?", [load_symbols]).fetchone()[0]
    )
    ram = physical_ram_bytes() or 0
    strategy = args.mem_strategy
    if strategy == "auto":
        fast_fits = memory_requirement_bytes(n_bars, STRATEGY_FAST)["peak_bytes"] <= ram
        strategy = STRATEGY_FAST if fast_fits else STRATEGY_LOW
    req = memory_requirement_bytes(n_bars, strategy)
    print(
        f"[gate]  {n_bars:,} bars; memory strategy {strategy!r} needs {req['peak_gib']:.2f} GiB, "
        f"host has {ram / 2**30:.2f} GiB",
        flush=True,
    )
    rfails = resource_failures(n_bars=n_bars, strategy=strategy, out_dir=args.out)
    if rfails:
        print(
            "RESOURCE PRECONDITION FAILED — nothing was computed:\n  - " + "\n  - ".join(rfails),
            file=sys.stderr,
        )
        return 1

    # ---- the orchestrator config: pins only, plus operational fields -----------------------
    orch = OrchestratorConfig(
        device=args.device,
        out_dir=args.out,
        wandb_mode=args.wandb,
        wandb_group="m6_money",
        # backbone_kwargs is NOT restated here: OrchestratorConfig derives it from the pins
        # (max_len >= seq_len + max(h)), and a driver-side literal would be a second copy of a
        # pinned-derived value — the shape that produced C-6.
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
    boundary = calendar_boundary_ms(pinned["window"][0], pinned["window"][1], pinned["train_frac"])
    # tail_bars=None: the money configuration trains on the FULL pinned region, unlike the
    # rehearsal which deliberately slices a tail. Same function, different argument (C-5 A1).
    raw = {
        s: load_symbol_arrays(con, s, boundary_ms=boundary, tail_bars=None) for s in load_symbols
    }
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

    _cache: dict = {}

    def windows_for_seed(seed: int, arms: tuple[str, ...] | None = None) -> dict:
        """LAZY, and under 'low' only the arm the unit needs is materialised.

        Both strategies produce byte-identical windows (``shuffle_micro`` is seeded on
        ``(symbol, seed)``); they differ in residency and rebuild count only. 'fast' caches the
        seed's three arms; 'low' keeps exactly one and pays the rebuild."""
        want = tuple(arms) if (arms and strategy == STRATEGY_LOW) else None
        key = (seed, want)
        if key not in _cache:
            _cache.clear()  # one seed resident at a time — the A9 finding
            _cache[key] = windows_by_arm(
                raw, seq_len=orch.seq_len, boundary_ms=boundary, seed=seed, arms=want
            )
        return _cache[key]

    grid = primary_region_grid_ms(cfg)  # money geometry; the dry run bounds via cap_per_symbol

    spec = MatrixSpec(
        orch=orch,
        base_eval_cfg=eval_cfg,
        sym_evals=sym_evals,
        windows_for_seed=windows_for_seed,
        art_dir=args.out / "eval",
        symbols=tuple(load_symbols),
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
        "n_symbols": len(load_symbols),
        "n_symbols_conformance_asserted": len(symbols),
        "eval_window": list(pinned["window"]),
        "grid": {"h": PRIMARY_H, "start_ms": int(grid[0]), "n_periods": int(grid.size)},
        "seeds": list(PINNED_SEEDS),
        "seq_len": orch.seq_len,
        "dry_run": bool(args.dry_run),
        # LOUD, and mirrored into every artifact where the verdict loader REFUSES them
        "money_run": bool(orch.money_run),
        "grid_pinned": not bool(args.dry_run),
        "dry_run_bounds": ({"cap_per_symbol": DRY_RUN_CAP} if args.dry_run else None),
        "memory_strategy": strategy,
        "resources": resource_record(n_bars=n_bars, strategy=strategy, out_dir=args.out),
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
