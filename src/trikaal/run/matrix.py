"""§7 v1.6 C-5 — the ONE (cell, seed) matrix path both drivers call.

WHY THIS MODULE EXISTS AT ALL (C-5 A1). The money driver must not be a copy of
``m6_toy_rehearsal.py``. **Duplication is exactly how C-6 happened** — two files each holding
their own copy of the truth, drifting to ``seeds (0,1,2)`` / ``seq_len 128`` on one side and the
pinned ``(0,1,2,3,4)`` / ``512`` on the other. So the shared orchestration lives here once and the
two drivers differ ONLY in where their config comes from: the rehearsal from flags, the money run
from ``money_config()`` and the pins.

WHAT IS SHARED: the train loop over units, the checkpoint-reload proof, the resume decision, the
per-unit eval → artifact emission, and the provenance stamp. What is NOT shared, by construction,
is any value — every number arrives inside the caller's ``MatrixSpec``.

THREE THINGS THIS MODULE FIXES THAT THE REHEARSAL GOT WRONG, and the rehearsal inherits the fixes
by calling here rather than being patched separately:

  * **RESUME BINDING (C-14).** The rehearsal honoured any artifact carrying the right schema key.
    Schema equality alone lets a rerun *after retraining* silently mix generations: the artifact
    from checkpoint-generation A is reused beside freshly-scored units from generation B, and
    ``load_cell_evals`` content-verifies the mixture perfectly happily, because each file's hash
    is internally consistent. An artifact is now reusable only if it also binds to the checkpoints
    that produced it, and to the symbols, window and h it was scored under.
  * **PROVENANCE PER UNIT (C-5 A7, C-18).** GPU model, driver, CUDA build, torch/numpy/**Python**,
    attention mode and every determinism flag are stamped into each artifact. The interpreter
    especially: ``uv.lock`` pins PACKAGES, not the interpreter — ``requires-python ">=3.11"``
    admits 3.14 and a rented box silently ran 3.14.6.
  * **FAN-OUT (C-5 A5).** The 25 units are independent, so a shard is a pure partition. The
    partition is checked to be exact and disjoint rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from trikaal.constants import N_FEATURES
from trikaal.data.universe_loader import build_symbol_windows
from trikaal.eval.conformance import cell_tokenizer_failures
from trikaal.eval.verdict import (
    ARTIFACT_SCHEMA,
    DSR_HORIZONS,
    PRIMARY_H,
    write_cell_eval_artifact,
    write_eval_index,
)
from trikaal.eval.xsection import SymbolEval, XSectionConfig, score_cell
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import ARMS
from trikaal.train.cells import CellSpec
from trikaal.train.checkpoint import load_checkpoint
from trikaal.train.orchestrator import OrchestratorConfig, run_cell
from trikaal.utils.provenance import run_provenance


# ---------------------------------------------------------------- shared lake I/O (C-5 A1)
def load_symbol_arrays(con, symbol: str, *, boundary_ms: int, tail_bars: int | None = None) -> dict:
    """One symbol's full feature arrays, optionally sliced to a training TAIL before the boundary.

    Factored out of ``m6_toy_rehearsal.py`` rather than copied into the money driver (C-5 A1).
    ``tail_bars=None`` loads the whole series — the money configuration; the rehearsal passes a
    finite tail because it deliberately runs a reduced training region.
    """
    xcols = ", ".join(f"x_{i}" for i in range(N_FEATURES))
    mcols = ", ".join(f"m_{i}" for i in range(N_FEATURES))
    t = con.execute(
        f"SELECT bar_open_ms, segment_id, sigma, raw_ret_close, {xcols}, {mcols} "
        "FROM bars WHERE symbol = ? ORDER BY bar_open_ms",
        [symbol],
    ).to_arrow_table()
    n = t.num_rows
    ts = t.column("bar_open_ms").to_numpy().astype(np.int64)
    lo = 0 if tail_bars is None else max(0, int(np.searchsorted(ts, boundary_ms)) - tail_bars)
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


def windows_by_arm(
    raw: dict,
    *,
    seq_len: int,
    boundary_ms: int,
    seed: int,
    arms: tuple[str, ...] | None = None,
) -> dict:
    """Per-arm, per-symbol training windows for ONE seed — built once per arm so every cell
    sharing an arm sees byte-identical inputs (m6_design §4). Shared by both drivers.

    ``arms=None`` builds all three (the 'fast' host strategy). Passing a subset builds only what
    the unit needs (the 'low' strategy): byte-identical output, ~2.5x less resident memory, paid
    for with rebuilds. Which is right is a property of the HOST, not of the specification."""
    return {
        arm: [
            build_symbol_windows(
                s,
                d["x"],
                d["mask"],
                d["segment_id"],
                d["ts"],
                arm=arm,
                seq_len=seq_len,
                boundary_ms=boundary_ms,
                seed=seed,
            )
            for s, d in raw.items()
        ]
        for arm in (arms or ARMS)
    }


@dataclass
class MatrixSpec:
    """Everything that differs between the two drivers. No value is defaulted here on purpose."""

    orch: OrchestratorConfig
    base_eval_cfg: XSectionConfig
    sym_evals: list[SymbolEval]
    # §7 v1.6 C-5 A9: a CALLABLE, not a prebuilt dict. Materialising all 5 seeds x 3 arms of
    # arm-transformed windows at the money configuration's 84.15M bars needs ~95 GiB; built one
    # seed at a time and released, the same numbers fit in ~19 GiB. Measured by the dry run, which
    # is what A9 exists for. This changes WHEN the windows are built, never what they contain.
    windows_for_seed: Callable[..., dict[str, list]]
    art_dir: Path
    symbols: tuple[str, ...]
    eval_window: tuple[str, str]
    grid_start_ms: int
    grid_n_periods: int
    units: tuple[tuple[CellSpec, int], ...]
    label: str
    horizons: tuple[int, ...] = DSR_HORIZONS
    extra_meta: dict = field(default_factory=dict)

    def unit_name(self, spec: CellSpec, seed: int) -> str:
        return f"{spec.name}_seed{seed}"


def all_units(cells, seeds) -> tuple[tuple[CellSpec, int], ...]:
    return tuple((spec, seed) for spec in cells for seed in seeds)


def shard_units(units, shard: int, n_shards: int) -> tuple[tuple[CellSpec, int], ...]:
    """Partition the units for ``--shard i/N`` (§7 v1.6 C-5 A5).

    The 25 units are independent, so wall-clock divides across shards at ~constant dollars and a
    preemption costs 1/25th. Strided rather than blocked so an odd N still balances the expensive
    cells across shards. The caller verifies exactness via :func:`shard_partition_failures`."""
    if not (1 <= n_shards):
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if not (0 <= shard < n_shards):
        raise ValueError(f"shard must be in [0, {n_shards}), got {shard}")
    return tuple(u for i, u in enumerate(units) if i % n_shards == shard)


def shard_partition_failures(units, n_shards: int) -> list[str]:
    """Prove the shards are an exact disjoint cover — a dropped unit is a missing artifact."""
    seen: list[tuple[str, int]] = []
    for i in range(n_shards):
        seen += [(s.name, sd) for s, sd in shard_units(units, i, n_shards)]
    want = [(s.name, sd) for s, sd in units]
    fails = []
    if len(seen) != len(set(seen)):
        fails.append("shards overlap — a unit would be computed twice")
    if sorted(seen) != sorted(want):
        missing = sorted(set(want) - set(seen))
        fails.append(f"shards do not cover the matrix; missing {missing}")
    return fails


def artifact_reuse_failures(doc: dict, *, expected: dict) -> list[str]:
    """Why a pre-existing artifact may NOT be resumed (empty = safe to honour). C-14.

    Schema equality is not sufficient and was the whole bug: after a retrain, a stale artifact
    still parses, still carries the right schema, and still hashes consistently with itself — so
    ``load_cell_evals`` accepts a mixture of checkpoint generations without complaint. The binding
    below is what makes the artifact a checkpoint of THIS run rather than of some run."""
    fails: list[str] = []
    if doc.get("schema") != ARTIFACT_SCHEMA:
        return [f"schema {doc.get('schema')!r} != {ARTIFACT_SCHEMA!r}"]
    if int(doc.get("cell_id", -1)) != int(expected["cell_id"]):
        fails.append(f"cell_id {doc.get('cell_id')} != {expected['cell_id']}")
    if int(doc.get("seed", -1)) != int(expected["seed"]):
        fails.append(f"seed {doc.get('seed')} != {expected['seed']}")
    meta = doc.get("meta") or {}
    got_ckpt = meta.get("checkpoints")
    if got_ckpt != expected["checkpoints"]:
        fails.append(
            "meta.checkpoints does not match the checkpoints now on disk — the artifact was "
            "scored from a DIFFERENT training generation; reusing it would silently mix "
            "generations inside one verdict"
        )
    # NOTE (§7 v1.6 C-5): ``n_periods`` is deliberately NOT bound here. It is DERIVED from the
    # scored series, so it cannot be known before scoring and cannot form a pre-scoring
    # expectation. Nothing is lost: it is a deterministic function of (symbols, eval_window, h,
    # start_ms), every one of which IS bound below, plus the checkpoints bound above.
    for key in ("symbols", "eval_window", "h", "start_ms"):
        if meta.get(key) != expected[key]:
            fails.append(f"meta.{key} {meta.get(key)!r} != {expected[key]!r}")
    return fails


def train_matrix(spec: MatrixSpec, monitor=None, *, verbose: bool = True) -> dict[str, dict]:
    """Train every unit in this shard. Returns ``{unit_name: run_manifest}``."""
    manifests: dict[str, dict] = {}
    walls: dict[str, float] = {}
    for cell, seed in spec.units:
        t0 = time.time()
        # built HERE and dropped at the end of the iteration — the peak is one seed, not five
        windows = spec.windows_for_seed(seed, (cell.arm,))
        m = run_cell(cell, seed, windows, spec.orch, monitor)
        del windows
        walls[m["run"]] = round(time.time() - t0, 1)
        m["wall_s"] = walls[m["run"]]
        manifests[m["run"]] = m
        if verbose:
            print(
                f"[train] {m['run']}: s1={m['final_loss']['stage1']:.3f} "
                f"s2={m['final_loss']['stage2']:.3f} wall={walls[m['run']]}s "
                f"mode={m['determinism']['attention_mode']}",
                flush=True,
            )
    return manifests


def reload_checkpoints(spec: MatrixSpec, manifests: dict[str, dict]) -> dict[str, dict]:
    """The reloadable-proof: every checkpoint must load before anything is scored."""
    hashes = {}
    for run, m in manifests.items():
        rd = Path(spec.orch.out_dir) / run
        load_checkpoint(rd / "tokenizer.pt", TokenizerAE, map_location=spec.orch.device)
        load_checkpoint(rd / "predictor.pt", TrikaalAR, map_location=spec.orch.device)
        hashes[run] = m["artifacts"]
    return hashes


@dataclass
class EvalOutcome:
    entries: dict[str, str]
    seconds: dict[str, float]
    decisions: dict[str, dict]
    resumed: list[str]
    refusals: list[str]


def eval_matrix(
    spec: MatrixSpec,
    ckpt_hashes: dict[str, dict],
    *,
    resume: bool = True,
    verbose: bool = True,
) -> EvalOutcome:
    """Score every unit in this shard → one content-hashed artifact each."""
    spec.art_dir.mkdir(parents=True, exist_ok=True)
    out = EvalOutcome({}, {}, {}, [], [])
    for cell, seed in spec.units:
        unit = spec.unit_name(cell, seed)
        t0 = time.time()
        expected = {
            "cell_id": cell.cell_id,
            "seed": seed,
            "checkpoints": ckpt_hashes.get(unit),
            "symbols": list(spec.symbols),
            "eval_window": list(spec.eval_window),
            "h": PRIMARY_H,
            "start_ms": int(spec.grid_start_ms),
        }
        done = spec.art_dir / f"cell{cell.cell_id}_seed{seed}_eval.json"
        if resume and done.is_file():
            try:
                doc = json.loads(done.read_text())
                why = artifact_reuse_failures(doc, expected=expected)
                if not why:
                    out.entries[done.name] = hashlib.sha256(done.read_bytes()).hexdigest()
                    out.resumed.append(done.name)
                    if verbose:
                        print(f"[eval]  RESUMED {done.name}", flush=True)
                    continue
                out.refusals.append(f"{done.name}: {'; '.join(why)}")
                if verbose:
                    print(f"[eval]  RECOMPUTING {done.name} — {why[0]}", flush=True)
            except (json.JSONDecodeError, OSError) as exc:
                out.refusals.append(f"{done.name}: unreadable ({exc})")

        rd = Path(spec.orch.out_dir) / unit
        tok = load_checkpoint(rd / "tokenizer.pt", TokenizerAE, map_location=spec.orch.device)
        pred = load_checkpoint(rd / "predictor.pt", TrikaalAR, map_location=spec.orch.device)
        model = pred.model.to(spec.orch.device).eval()
        tokm = tok.model.to(spec.orch.device).eval()
        # §7 v1.3: the eval-loaded tokenizer must itself be causal — a hard stop, not a warning
        bad = cell_tokenizer_failures(tokm.get_config(), run=unit)
        if bad:
            raise RuntimeError("EVAL REFUSED — " + "; ".join(bad))

        val_by_h, kappa_by_h, dec_by_h, head = {}, {}, {}, None
        for h in spec.horizons:
            sc = score_cell(
                f"{unit}_h{h}",
                model,
                tokm,
                cell.arm,
                spec.sym_evals,
                replace(spec.base_eval_cfg, h=h, seed=seed),
                val_only=(h != PRIMARY_H),
            )
            val_by_h[h], kappa_by_h[h] = sc.val_ir_by_kappa, sc.kappa_chosen
            dec_by_h[h] = int(sc.n_decisions)
            if h == PRIMARY_H:
                head = sc

        # §7 v1.6 C-5: n_periods is the ACTUAL series length, never a separately-carried number.
        # `_validate_artifact` requires len(headline_series) == grid.n_periods, so deriving it
        # here makes that invariant true by construction instead of by two values agreeing. The
        # A9 dry run exposed the gap: a grid truncated for the record does NOT bound what
        # `score_cell` scores (it builds its own grids from cfg — the real bound is
        # cap_per_symbol), so the recorded n_periods and the series could silently disagree.
        n_periods = int(np.asarray(head.headline_series).size)
        det = None
        try:
            det = json.loads((rd / "run_manifest.json").read_text())["determinism"]
        except (OSError, json.JSONDecodeError, KeyError):
            det = {}
        name, sha = write_cell_eval_artifact(
            spec.art_dir,
            cell_id=cell.cell_id,
            seed=seed,
            grid={"h": PRIMARY_H, "start_ms": int(spec.grid_start_ms), "n_periods": n_periods},
            headline_series=head.headline_series,
            kappa_star_by_h=kappa_by_h,
            val_ir_by_kappa_by_h=val_by_h,
            codebook=head.codebook,
            mu_diag=head.mu_diag,
            ohlcv_recon=head.ohlcv_recon,
            decode_agreement=head.decode_agreement,
            meta={
                # the RESUME BINDING (C-14): an artifact is a checkpoint of THIS run, and says so
                "checkpoints": ckpt_hashes.get(unit),
                "symbols": list(spec.symbols),
                "eval_window": list(spec.eval_window),
                "h": PRIMARY_H,
                "n_periods": n_periods,
                "start_ms": int(spec.grid_start_ms),
                # the PROVENANCE STAMP (A7/C-18) — recorded, never chosen
                "provenance": run_provenance(
                    spec.orch.device, attention_mode=det.get("attention_mode", "unknown")
                ),
                "determinism": det,
                "driver": spec.label,
                # §7 v1.6 C-5 A9: mirrored into every artifact so a dry-run unit is refused by
                # verdict._validate_artifact — structurally un-quotable, not merely labelled.
                "money_run": bool(spec.orch.money_run),
                "grid_pinned": bool(spec.orch.money_run),
                "dry_run": not bool(spec.orch.money_run),
                **spec.extra_meta,
            },
        )
        out.entries[name] = sha
        out.seconds[unit] = round(time.time() - t0, 3)
        out.decisions[unit] = {
            "by_h": dict(dec_by_h),
            "total_scored": int(sum(dec_by_h.values())),
            "primary_h": int(dec_by_h.get(PRIMARY_H, 0)),
        }
        if verbose:
            print(
                f"[eval]  {unit}: IR@0.30%={head.ir_headline:+.2f} κ*={head.kappa_chosen} "
                f"decisions={head.n_decisions} → {name}",
                flush=True,
            )
    return out


def write_index_if_complete(spec: MatrixSpec, entries: dict[str, str], expected: int) -> bool:
    """Write the eval index ONLY when every unit is present — a partial index is a trap.

    Under ``--shard`` each box holds a slice, so the index is written by whoever ends up with the
    complete set. ``load_cell_evals`` would otherwise be handed an index that looks authoritative
    while describing a fraction of the matrix."""
    if len(entries) < expected:
        return False
    write_eval_index(spec.art_dir, entries)
    return True
