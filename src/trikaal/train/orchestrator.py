"""The 5-cell × ≥3-seed training orchestrator (m6_design §6 item 5).

Runs every cell of the matrix (``cells.CELLS``) × every seed through the two-stage pipeline —
Stage 1 (tokenizer, multi-symbol fold-aligned draw) → tokenize → Stage 2 (AR backbone) — under
the run disciplines the design binds:

* **identical draw + seeds across cells** — the (symbol, window-start) sequence is a pure function
  of (seed, per-symbol starts), and the starts are arm-independent, so every cell sees the same
  draw; the ONLY varied factor is quantizer × arm (m6_design §4);
* **construction gates first**: G-parity across the quantizer arms + the 21,301,248 backbone-param
  pin (canonical dims) BEFORE any training step;
* **tripwires every step/eval** (item 8): a trip aborts the WHOLE run (a broken cell invalidates
  the matched comparison);
* **determinism recorded** (item 7): the attention mode + seed + device + env go into the per-run
  manifest AND both checkpoints' metadata;
* **per-cell content-hashed artifacts** + a durable JSON run-manifest per (cell, seed) — the same
  never-stdout-only discipline as ``gate_a_run_manifest``;
* **W&B logging** with an offline/disabled mode so local smokes need no credentials;
* **atomic resumable train state** (item 9) per (cell, seed, stage).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from trikaal.data.universe_loader import MultiSymbolWindowSampler, SymbolWindows
from trikaal.eval.conformance import (
    PINNED_BACKBONE_PARAMS,
    PINNED_DSR,
    PINNED_MICRO_LEGIBILITY_MIN,
    PINNED_MONEY_SEQ_LEN,
    PINNED_SEEDS,
    PINNED_STEPS_STAGE1,
    PINNED_STEPS_STAGE2,
    ConformanceError,
)
from trikaal.model.attention_mode import (
    determinism_record,
    resolve_attention_mode,
    set_attention_backend,
)
from trikaal.train.cells import (
    CELLS,
    CellSpec,
    assert_cells_parity,
    build_cell_backbone,
    build_cell_tokenizer,
)
from trikaal.train.checkpoint import save_checkpoint
from trikaal.train.gates import cosine_warmup_lr, micro_legibility_gate
from trikaal.train.token_stream import tokenize_features
from trikaal.train.train_state import autocast_ctx, save_train_state
from trikaal.train.tripwire import TripwireMonitor
from trikaal.utils.seeding import set_determinism
from trikaal.utils.throughput import BASIS_END_TO_END, throughput_record


def _money_backbone_kwargs() -> dict:
    """The backbone kwargs the money surface REQUIRES, derived from the pins (§7 v1.6 C-5 A9).

    The KV-cached rollout addresses position ``seq_len - 1 + k`` for ``k`` up to ``h``, so the
    model must be built with ``max_len >= seq_len + max(h)``. Found by the A9 dry run: at the
    default ``max_len = seq_len`` the money run would have TRAINED every unit and then raised at
    the first eval. ``max_len`` sizes RoPE buffers only — the realized parameter count is
    21,301,248 either way (verified) — so this enables the rollout and moves no number.

    A DEFAULT rather than something the driver restates: a driver-side literal would be a second
    copy of a pinned-derived value, which is exactly the shape that produced C-6."""
    return {"max_len": PINNED_MONEY_SEQ_LEN + max(PINNED_DSR["horizons"])}


@dataclass
class OrchestratorConfig:
    # §7 v1.6 C-6 — THESE REFERENCE THE PINS, THEY DO NOT RESTATE THEM. The defaults were
    # ``(0, 1, 2)`` and ``128`` against a pinned ``(0, 1, 2, 3, 4)`` and money ``512``: the S=5 the
    # operator approved and funded never reached the training side, and 128-trained models would
    # have been scored on 512 of context. Restoring correct literals here would have been a fix;
    # removing the second copy is the guarantee, and ``__post_init__`` asserts on top of it so a
    # caller cannot re-introduce the divergence by argument either.
    seeds: tuple[int, ...] = PINNED_SEEDS
    cells: tuple[CellSpec, ...] = CELLS
    seq_len: int = PINNED_MONEY_SEQ_LEN
    batch_size: int = 32
    # §7 v1.6.15 C-18: REFERENCE the pins, never restate them (the C-6 guarantee shape).
    steps_stage1: int = PINNED_STEPS_STAGE1
    steps_stage2: int = PINNED_STEPS_STAGE2
    peak_lr_stage1: float = 1e-3
    peak_lr_stage2: float = 3e-4
    warmup_frac: float = 0.1
    alpha: float = 0.5  # symbol sampling ∝ n_windows^alpha (m6_design §4)
    tok_kwargs: dict = field(default_factory=dict)  # shell dims (empty = canonical)
    # §7 v1.4 STANDING gate (gate-2 ruling): min per-bar logistic sign-acc for EACH of the
    # six micro dims from bar t's own id, measured post-Stage-1 / pre-Stage-2 on the run's
    # real training stream; hard stop below it. None disables (toy shells ONLY — the real
    # M6 run uses this default).
    micro_legibility_min: float | None = PINNED_MICRO_LEGIBILITY_MIN
    backbone_kwargs: dict = field(default_factory=_money_backbone_kwargs)
    expect_backbone_params: int | None = PINNED_BACKBONE_PARAMS  # None ONLY for tiny smoke shells
    enforce_parity: bool = True  # G-parity holds at canonical dims; tiny smoke shells skip
    device: str = "cpu"
    prefer_flash: bool = False
    autocast_bf16: bool = False
    out_dir: Path = Path("runs/m6")
    state_every: int = 50
    wandb_mode: str = "disabled"  # "disabled" | "offline" | "online"
    wandb_project: str = "trikaal-m6"
    wandb_group: str | None = None  # one group = one W&B page holding all 15 cell-runs' curves
    # §7 v1.6 C-6: TRUE by default so the DANGEROUS direction is the one that must be declared.
    # A toy shell announcing itself is safe; a money run silently inheriting toy values is what
    # this defect was. Toy drivers set money_run=False explicitly.
    money_run: bool = True

    def __post_init__(self) -> None:
        """HARD-FAIL if a money run diverges from the pinned surface (§7 v1.6 C-6).

        An assertion, not a corrected default: the defect existed because two files each held
        their own copy of the truth, so a right-looking default fixes today and guarantees
        nothing about tomorrow. Toy/smoke/probe shells legitimately run tiny geometry and opt out
        via ``money_run=False`` — an explicit declaration, never an inference from the values
        themselves (inferring it would re-create the hole: a 128-seq money run would simply be
        classified as a toy and skip the check)."""
        if not self.money_run:
            return
        bad = []
        if tuple(self.seeds) != tuple(PINNED_SEEDS):
            bad.append(f"seeds {tuple(self.seeds)} != pinned {tuple(PINNED_SEEDS)} (§7 v1.5 S=5)")
        if int(self.seq_len) != int(PINNED_MONEY_SEQ_LEN):
            bad.append(
                f"seq_len {self.seq_len} != the money surface {PINNED_MONEY_SEQ_LEN} — training "
                "on a shorter context than the model is SCORED on is a train/eval regime "
                "mismatch, not a stale default"
            )
        if self.micro_legibility_min != PINNED_MICRO_LEGIBILITY_MIN:
            bad.append(
                f"micro_legibility_min {self.micro_legibility_min!r} != pinned "
                f"{PINNED_MICRO_LEGIBILITY_MIN} (§7 v1.4 standing gate — a HARD STOP; None "
                "disables it entirely and is for toy shells only)"
            )
        if self.expect_backbone_params != PINNED_BACKBONE_PARAMS:
            bad.append(
                f"expect_backbone_params {self.expect_backbone_params!r} != pinned "
                f"{PINNED_BACKBONE_PARAMS} (the realized count the ablation reports)"
            )
        if not self.enforce_parity:
            bad.append("enforce_parity is False — G-parity binds at canonical dims")
        # §7 v1.6.15 C-18: the step budget was a bare dataclass default. The money driver RECORDED
        # it in its manifest, but recording is not pinning — a different budget would have run and
        # been faithfully reported as the recipe.
        for name, want in (
            ("steps_stage1", PINNED_STEPS_STAGE1),
            ("steps_stage2", PINNED_STEPS_STAGE2),
        ):
            if int(getattr(self, name)) != int(want):
                bad.append(f"{name} {getattr(self, name)} != pinned {want} (§7 v1.6.15 C-18)")
        # §7 v1.6 C-5 A9, FOUND BY THE DRY RUN: the rollout addresses positions up to
        # seq_len + h, so a backbone built at the default max_len = seq_len cannot score at all.
        # The money run would have TRAINED every unit and then died at the first eval. Derived,
        # not chosen: max(PINNED_DSR["horizons"]) is the largest h any VAL leg rolls out to, and
        # max_len is param-neutral (21,301,248 either way — verified), so this enables the
        # rollout without moving a single number.
        need = int(self.seq_len) + max(PINNED_DSR["horizons"])
        got = int(self.backbone_kwargs.get("max_len", 0)) or None
        if got is None or got < need:
            bad.append(
                f"backbone_kwargs max_len={got!r} < seq_len + max(h) = {need} — the rollout "
                f"addresses positions up to seq_len+h and would raise at the FIRST eval, after "
                f"every unit had already been trained"
            )
        if bad:
            raise ConformanceError(
                "OrchestratorConfig diverges from the pinned money surface (§7 v1.6 C-6); pass "
                "money_run=False for a toy/smoke/probe shell:\n  - " + "\n  - ".join(bad)
            )


def _wandb_run(cfg: OrchestratorConfig, name: str, run_config: dict):
    if cfg.wandb_mode == "disabled":
        return None
    import wandb  # lazy — a pinned dep, but only touched when logging is on

    return wandb.init(
        project=cfg.wandb_project,
        name=name,
        group=cfg.wandb_group,
        mode=cfg.wandb_mode,
        config=run_config,
    )


def _log(run, metrics: dict, step: int) -> None:
    if run is not None:
        run.log(metrics, step=step)


def _train_loop(
    *,
    label: str,
    model: torch.nn.Module,
    step_fn,
    steps: int,
    peak_lr: float,
    warmup_frac: float,
    cfg: OrchestratorConfig,
    monitor: TripwireMonitor,
    state_path: Path,
    np_rng: np.random.Generator,
    sampler: MultiSymbolWindowSampler,
    wandb_run,
    timing: dict | None = None,
) -> float:
    """The shared per-stage loop: cosine LR, clip, tripwire every step, atomic state saves.

    ``timing``, when supplied, is filled in place with the stage's completed-step count and
    wall-clock seconds (Finding 0: the cost basis must be a NUMBER on a manifest, and only an
    end-to-end wall time can price a run). Purely observational — no training state depends on it,
    and the fields are written only on normal completion, so an aborted stage reports no rate
    rather than a fast one.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01, betas=(0.9, 0.95))
    warmup = int(warmup_frac * steps)
    last_loss = float("nan")
    t_stage = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_warmup_lr(step, peak_lr, warmup, steps)
        model.train()
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(cfg.device, bf16=cfg.autocast_bf16):
            loss = step_fn()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach())
        monitor.check_step(label, step, last_loss, float(gn))  # a trip ABORTS the run
        if step % cfg.state_every == 0:
            save_train_state(
                state_path,
                model=model,
                optimizer=opt,
                step=step,
                np_rng=np_rng,
                sampler_state=sampler.state(),
            )
        if step % 25 == 0:
            _log(wandb_run, {f"{label}/loss": last_loss, f"{label}/grad_norm": float(gn)}, step)
    if timing is not None:
        timing["steps"] = int(steps)
        timing["wall_s"] = float(time.time() - t_stage)
    return last_loss


def run_cell(
    spec: CellSpec,
    seed: int,
    per_symbol_by_arm: dict[str, list[SymbolWindows]],
    cfg: OrchestratorConfig,
    monitor: TripwireMonitor | None = None,
) -> dict:
    """Train ONE (cell, seed): Stage-1 tokenizer → tokenize → Stage-2 backbone → artifacts.

    ``per_symbol_by_arm[arm]`` holds each symbol's arm-transformed windows (built once per arm by
    the caller so every cell sharing an arm sees byte-identical inputs). Returns the run manifest
    (also written to ``out_dir/<cell>_seed<seed>/run_manifest.json``)."""
    # §7 v1.6.22 (B1 APPROVED 2026-08-03): forced determinism is PAID. The measured penalty is
    # 13.175% (1.152x slower, docs/m6_cuda_probe_report.md) -- not the illustrative 1.3x, which had
    # been double-counted against a figure that already contained it. `deterministic_algorithms`
    # was hardcoded False here while 49/49 records asserted bit-exactness beside it; that is the
    # defect invariant 7's own text encoded, and it is now closed at the site.
    set_determinism(seed, deterministic_algorithms=True)
    monitor = monitor or TripwireMonitor()
    dev = cfg.device
    run_name = f"{spec.name}_seed{seed}"
    run_dir = Path(cfg.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    mode = resolve_attention_mode(dev, prefer_flash=cfg.prefer_flash)
    det = determinism_record(seed=seed, device=dev, attention_mode=mode)

    per_symbol = per_symbol_by_arm[spec.arm]
    t_s1: dict = {}
    t_s2: dict = {}
    sampler = MultiSymbolWindowSampler(per_symbol, alpha=cfg.alpha, seed=seed)
    wandb_run = _wandb_run(cfg, run_name, {"cell": spec.name, "seed": seed, **det})

    try:
        # ---- Stage 1: tokenizer on the fold-aligned multi-symbol draw --------------------
        tok = build_cell_tokenizer(spec, **cfg.tok_kwargs).to(dev)
        set_attention_backend(tok, mode)

        def tok_step():
            xb, mb, _ = sampler.sample_batch(cfg.batch_size, cfg.seq_len)
            xt = torch.from_numpy(xb).to(dev)
            mt = torch.from_numpy(mb.astype(np.float32)).to(dev)
            return tok(xt, mt)["loss"]

        s1_loss = _train_loop(
            label=f"{run_name}/stage1",
            model=tok,
            step_fn=tok_step,
            steps=cfg.steps_stage1,
            peak_lr=cfg.peak_lr_stage1,
            warmup_frac=cfg.warmup_frac,
            cfg=cfg,
            monitor=monitor,
            state_path=run_dir / "stage1_state.pt",
            np_rng=sampler._rng,
            sampler=sampler,
            wandb_run=wandb_run,
            timing=t_s1,
        )
        tok_hash = save_checkpoint(run_dir / "tokenizer.pt", tok, tok.get_config(), meta=det)

        # ---- tokenize each symbol with the FROZEN cell tokenizer -------------------------
        tokens: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for sw in per_symbol:
            b_c, b_f = tokenize_features(
                tok, sw.x, sw.mask, sw.segment_id, window=cfg.seq_len, device=dev
            )
            tokens[sw.symbol] = (b_c, b_f)

        # ---- §7 v1.4 STANDING micro-legibility gate: post-Stage-1, PRE-Stage-2 hard stop --
        legibility_receipt = None
        if cfg.micro_legibility_min is not None and spec.arm in ("micro", "micro_shuffled"):
            legibility_receipt = micro_legibility_gate(
                tok, per_symbol, tokens, min_acc=cfg.micro_legibility_min, run_name=run_name
            )
            print(f"[{run_name}] micro legibility gate: {legibility_receipt}", flush=True)

        # ---- Stage 2: AR backbone on the token windows (same draw geometry) --------------
        backbone = build_cell_backbone(
            tok.v_c,
            tok.v_f,
            expect_base_params=cfg.expect_backbone_params,
            **cfg.backbone_kwargs,
        ).to(dev)
        set_attention_backend(backbone, mode)
        s2_sampler = MultiSymbolWindowSampler(per_symbol, alpha=cfg.alpha, seed=seed)
        sym_arrays = {
            sw.symbol: (tokens[sw.symbol][0], tokens[sw.symbol][1], sw.ts, sw.starts)
            for sw in per_symbol
        }

        def ar_step():
            # same two-stage draw over the SAME fold-legal starts (token streams are bar-aligned)
            sym_ids = s2_sampler._rng.choice(
                len(s2_sampler.per_symbol), size=cfg.batch_size, p=s2_sampler.weights
            )
            bc_l, bf_l, ts_l = [], [], []
            for si in sym_ids:
                sw = s2_sampler.per_symbol[int(si)]
                b_c, b_f, ts, starts = sym_arrays[sw.symbol]
                s = int(starts[s2_sampler._rng.integers(0, starts.size)])
                bc_l.append(b_c[s : s + cfg.seq_len])
                bf_l.append(b_f[s : s + cfg.seq_len])
                ts_l.append(ts[s : s + cfg.seq_len])
                s2_sampler.drawn_by_symbol[sw.symbol] += 1
            s2_sampler.n_drawn += cfg.batch_size
            bc = torch.from_numpy(np.stack(bc_l).astype(np.int64)).to(dev)
            bf = torch.from_numpy(np.stack(bf_l).astype(np.int64)).to(dev)
            tst = torch.from_numpy(np.stack(ts_l).astype(np.int64)).to(dev)
            return backbone.compute_loss(bc, bf, tst)["loss"]

        s2_loss = _train_loop(
            label=f"{run_name}/stage2",
            model=backbone,
            step_fn=ar_step,
            steps=cfg.steps_stage2,
            peak_lr=cfg.peak_lr_stage2,
            warmup_frac=cfg.warmup_frac,
            cfg=cfg,
            monitor=monitor,
            state_path=run_dir / "stage2_state.pt",
            np_rng=s2_sampler._rng,
            sampler=s2_sampler,
            wandb_run=wandb_run,
            timing=t_s2,
        )
        pred_hash = save_checkpoint(
            run_dir / "predictor.pt", backbone, backbone.get_config(), meta=det
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    manifest = {
        "run": run_name,
        "cell": {"id": spec.cell_id, "quantizer": spec.quantizer, "arm": spec.arm},
        "seed": seed,
        "determinism": det,
        "artifacts": {"tokenizer_hash": tok_hash, "predictor_hash": pred_hash},
        "final_loss": {"stage1": s1_loss, "stage2": s2_loss},
        # Finding 0 — the cost basis must be a NUMBER on a manifest, never a prose label. Both
        # stages are end-to-end wall-clock (sampling, transfer, checkpoint saves included), so
        # these are the only records in the repo that may price a run (utils.throughput).
        "throughput": {
            "stage1": throughput_record(
                steps=t_s1.get("steps"),
                batch=cfg.batch_size,
                seq_len=cfg.seq_len,
                wall_s=t_s1.get("wall_s"),
                basis=BASIS_END_TO_END,
                device=dev,
                note="Stage-1 tokenizer loop, wall-clock",
            ),
            "stage2": throughput_record(
                steps=t_s2.get("steps"),
                batch=cfg.batch_size,
                seq_len=cfg.seq_len,
                wall_s=t_s2.get("wall_s"),
                basis=BASIS_END_TO_END,
                device=dev,
                note="Stage-2 AR loop, wall-clock (excludes the tokenize pass between stages)",
            ),
        },
        "micro_legibility_gate": legibility_receipt,
        "draw": {
            "alpha": cfg.alpha,
            "n_drawn_stage1": sampler.n_drawn,
            "drawn_by_symbol_stage1": sampler.drawn_by_symbol,
            "n_drawn_stage2": s2_sampler.n_drawn,
            "drawn_by_symbol_stage2": s2_sampler.drawn_by_symbol,
        },
        "config": {
            "seq_len": cfg.seq_len,
            "batch_size": cfg.batch_size,
            "steps_stage1": cfg.steps_stage1,
            "steps_stage2": cfg.steps_stage2,
            "tok_kwargs": cfg.tok_kwargs,
            "backbone_kwargs": cfg.backbone_kwargs,
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def run_all_cells(
    per_symbol_by_arm: dict[str, list[SymbolWindows]],
    cfg: OrchestratorConfig,
    monitor: TripwireMonitor | None = None,
) -> dict:
    """The full matrix: construction gates → every (cell × seed) → a top-level manifest.

    A TripwireAbort from ANY cell propagates and aborts the whole run — by design."""
    monitor = monitor or TripwireMonitor()
    parity = None
    if cfg.enforce_parity:
        parity = assert_cells_parity(**cfg.tok_kwargs)  # G-parity BEFORE any training step
    runs = []
    for spec in cfg.cells:
        for seed in cfg.seeds:
            runs.append(run_cell(spec, seed, per_symbol_by_arm, cfg, monitor))
    top = {
        "n_runs": len(runs),
        "parity": parity,
        "cells": [s.name for s in cfg.cells],
        "seeds": list(cfg.seeds),
        "runs": [r["run"] for r in runs],
        "artifacts": {r["run"]: r["artifacts"] for r in runs},
    }
    out = Path(cfg.out_dir) / "orchestrator_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(top, indent=2, sort_keys=True))
    return top


__all__ = ["OrchestratorConfig", "run_all_cells", "run_cell"]
