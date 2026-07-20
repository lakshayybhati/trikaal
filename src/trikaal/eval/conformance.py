"""Money-run conformance gate — the pinned §3a surface, machine-checked (pre-flight Item 8).

A lock is worthless if the computed quantity can drift from the locked one: every §3a pin that
the eval config could silently violate is asserted here, and the REAL eval driver must call
:func:`assert_conformance` as its FIRST step (before loading any model). The standalone runner
is ``scripts/m6_conformance.py``; its passing output is committed as the Item-8 artifact.

Checks (hard-fail on ANY diff; every failure listed, not just the first):
window + train_frac vs ``runs_manifest/m6_mde_inputs.json`` · money mode ON with the headline =
ONE continuous grid over forward blocks 1–5 matching the JSON's ``primary_region_ms`` (VAL
block 0 excluded) · the evaluated symbol set == ``symbols_sampled`` with its pinned sha256 ·
κ grid {1.0, 1.5, 2.0, 3.0} · seeds exactly {0, 1, 2} · no ``cap_per_symbol`` · per-symbol
spread deciles in use (from the committed artifact; not flat) · ``shuffle_micro`` seed-threading
bit-identical between the TRAIN path (``build_symbol_windows``) and the EVAL path (the
``score_cell`` transform) for every pinned seed · headline cost 0.30 % · the §3a bootstrap
constants (B = 10,000, seed 20260704, α = 0.05).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from trikaal.data.universe_loader import build_symbol_windows, calendar_boundary_ms
from trikaal.eval import paired_bootstrap as pb
from trikaal.eval.xsection import XSectionConfig, primary_region_grid_ms
from trikaal.train.arms import ARM_MICRO_SHUFFLED, select_arm, shuffle_micro

# ---- the §3a pins (literals, mirroring docs/m6_prereg.md §3a) --------------------------------
PINNED_SYMBOLS_SHA256 = "60e24f598de9601260099e3f11e537814385df23c837073035b9f7ce4dc32631"
PINNED_KAPPAS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
PINNED_SEEDS: tuple[int, ...] = (0, 1, 2)
PINNED_HEADLINE_COST = 0.0030
PINNED_BOOT = {"B": 10_000, "seed": 20260704, "alpha": 0.05}
MDE_INPUTS = Path("runs_manifest/m6_mde_inputs.json")
DECILES = Path("runs_manifest/m6_spread_deciles.json")

# ---- the §7-v1.3 causal-encoder pin (the token-causality adjudication) -----------------------
# every M6 cell tokenizer — train side and the eval-loaded checkpoint alike — must report
# encoder_causal=True; bidirectional encoding puts future-bar content into the pre-tokenized
# eval context (measured 41.8%/28.3% real-lake past-token flips) and invariant 2 binds inputs.
PINNED_ENCODER_CAUSAL = True

# §7 v1.4 (supervisor adjudication of the token-control programme): the fine subtoken must be
# a PER-BAR pointwise encoding — the causal contextual encoder smears per-bar feature state
# forward across later tokens' ids (measured per-bar id visibility ~= chance, logistic 0.5135),
# so feature-space conditionals arrive per-bar-illegible to the AR.
PINNED_FINE_POINTWISE = True


def cell_tokenizer_failures(tok_config: dict, *, run: str) -> list[str]:
    """Divergences between one cell tokenizer's config and the §7 v1.3/v1.4 pins (empty = OK).

    ``tok_config`` is ``TokenizerAE.get_config()`` of the constructed (train) or
    checkpoint-loaded (eval) cell tokenizer. A bidirectional cell config — or a
    contextual-fine one — is a NAMED failure; the eval driver hard-stops on it before
    scoring anything."""
    failures = []
    got = tok_config.get("encoder_causal")
    if got is not PINNED_ENCODER_CAUSAL:
        failures.append(
            f"{run}: tokenizer encoder_causal={got!r} != pinned {PINNED_ENCODER_CAUSAL} "
            "(prereg §7 v1.3 — bidirectional tokens carry future-bar content into eval inputs)"
        )
    got_fp = tok_config.get("fine_pointwise")
    if got_fp is not PINNED_FINE_POINTWISE:
        failures.append(
            f"{run}: tokenizer fine_pointwise={got_fp!r} != pinned {PINNED_FINE_POINTWISE} "
            "(prereg §7 v1.4 — a contextual fine subtoken smears per-bar feature state "
            "across later tokens' ids, leaving it per-bar-illegible to the AR)"
        )
    return failures


# ---- the §3-clause-5 DSR pins (the verdict path's recipe — an INDEPENDENT statement of the
# prereg constants; trikaal/eval/verdict.py carries its own literals and the two must agree,
# so an edit to either side is caught before any DSR is computed) ------------------------------
PINNED_DSR = {
    "n_trials": 180,  # 5 cells × 3 seeds × 3 horizons × 4 κ — ENUMERATED, never assumed
    "cells": (1, 2, 3, 4, 5),
    "seeds": PINNED_SEEDS,
    "horizons": (5, 15, 60),  # h=1 is not evaluated anywhere in M6 (the old 240 was an error)
    "kappas": PINNED_KAPPAS,
    "threshold": 0.95,
    "var_sr": "population variance (ddof=0) of the 180 de-annualized per-trial VAL IRs",
    "statistic": "Cell 4's seed-mean pooled headline series at the 0.30% flat netting",
}


class ConformanceError(AssertionError):
    """The money-run config diverged from the pre-registered §3a surface."""


def money_config(h: int = 15, *, seq_len: int = 512, device: str = "cpu", seed: int = 0):
    """The money-run :class:`XSectionConfig` — the ONE way any driver builds it.

    Both real drivers (`scripts/m6_conformance.py`, `scripts/m6_verdict.py`) construct the
    surface through this function so the gated config and the decided config cannot diverge."""
    from trikaal.eval.costs import load_spread_deciles

    return XSectionConfig(
        h=h,
        seq_len=seq_len,
        cap_per_symbol=None,
        device=device,
        seed=seed,
        spread_frac_by_symbol=load_spread_deciles(DECILES),
        money=True,
    )


def symbols_sha256(symbols: list[str]) -> str:
    """The pinned convention: sha256 of ``json.dumps(sorted(symbols))``."""
    return hashlib.sha256(json.dumps(sorted(symbols)).encode()).hexdigest()


def _shuffle_threading_probe(seed: int, symbol: str = "PROBEUSDT") -> bool:
    """TRAIN path (build_symbol_windows) vs EVAL path (score_cell's transform) — bit-identical?

    Both must reach ``shuffle_micro`` with the SAME (seed, symbol) derivation; a functional
    probe on synthetic data catches any drift in either path's threading."""
    rng = np.random.default_rng(1234 + seed)
    n = 600
    x = rng.standard_normal((n, 16)).astype(np.float32)
    m = np.zeros((n, 16), dtype=np.uint8)
    m[:, 13:16] = 1
    m[rng.random((n, 16)) < 0.1] = 1  # holes so the co-shuffled mask carries signal
    m[:, 13:16] = 1
    seg = np.zeros(n, dtype=np.int64)
    seg[n // 2 :] = 1
    ts = (1_600_000_000_000 + np.arange(n) * 60_000).astype(np.int64)
    # train path: the loader's arm application
    sw = build_symbol_windows(
        symbol,
        x,
        m,
        seg,
        ts,
        arm=ARM_MICRO_SHUFFLED,
        seq_len=16,
        boundary_ms=int(ts[-1]) + 1,
        seed=seed,
    )
    # eval path: score_cell's prepare transform, verbatim
    xs, ms = shuffle_micro(x, m, seg, symbol=symbol, seed=seed)
    x_arm, m_arm = select_arm(np.asarray(xs, np.float32), ms, ARM_MICRO_SHUFFLED)
    return bool(np.array_equal(sw.x, x_arm) and np.array_equal(sw.mask, m_arm))


def conformance_failures(
    cfg: XSectionConfig,
    *,
    symbols: list[str],
    seeds: tuple[int, ...],
    mde_inputs_path: Path = MDE_INPUTS,
    deciles_path: Path = DECILES,
) -> list[str]:
    """Every divergence between (cfg, symbols, seeds) and the pinned §3a surface (empty = PASS)."""
    fails: list[str] = []
    try:
        mde = json.loads(Path(mde_inputs_path).read_text())
    except OSError as e:
        return [f"cannot read {mde_inputs_path}: {e} — the pinned surface artifact is REQUIRED"]

    # 1) window + train_frac == the MDE-inputs pins
    if [cfg.window_start, cfg.window_end] != list(mde["window"]):
        fails.append(f"window {cfg.window_start}..{cfg.window_end} != pinned {mde['window']}")
    if float(cfg.train_frac) != float(mde["train_frac"]):
        fails.append(f"train_frac {cfg.train_frac} != pinned {mde['train_frac']}")

    # 2) money mode + the primary region (blocks 1-5, VAL excluded, ONE continuous grid)
    if not cfg.money:
        fails.append("cfg.money is False — the real eval must run in money mode (§3a)")
    if cfg.val_block != 0 or cfg.n_blocks != 6:
        fails.append(f"fold plan (val_block={cfg.val_block}, n_blocks={cfg.n_blocks}) != (0, 6)")
    grid = primary_region_grid_ms(cfg)
    region = mde.get("primary_region_ms")
    if region is None:
        fails.append(f"{mde_inputs_path} lacks primary_region_ms — regenerate (audit item 5)")
    elif grid.size == 0 or int(grid[0]) != int(region[0]) or int(grid[-1]) >= int(region[1]):
        # the money grid must anchor AT the pinned region start (the MDE grid's anchor) and
        # stay strictly inside the region — then arange(anchor, end, h·60000) IS the MDE grid
        fails.append(f"headline grid [{grid[0]}, {grid[-1]}] not the pinned region {region}")
    boundary = calendar_boundary_ms(cfg.window_start, cfg.window_end, cfg.train_frac)
    if grid.size and int(grid[0]) <= boundary:
        fails.append("headline grid reaches VAL block 0 — §3a excludes it")

    # 3) the evaluated symbol set == symbols_sampled with the pinned hash
    pinned = sorted(mde["symbols_sampled"])
    if symbols_sha256(pinned) != PINNED_SYMBOLS_SHA256:
        fails.append(
            f"{mde_inputs_path} symbols_sampled hash {symbols_sha256(pinned)[:16]}… != "
            f"pinned {PINNED_SYMBOLS_SHA256[:16]}…"
        )
    if sorted(symbols) != pinned:
        fails.append(f"evaluated symbol set ({len(symbols)}) != the pinned 40-symbol primary set")

    # 4) κ grid, 5) seeds, 6) cap, 9) headline cost
    if tuple(float(k) for k in cfg.kappas) != PINNED_KAPPAS:
        fails.append(f"kappa grid {cfg.kappas} != pinned {PINNED_KAPPAS}")
    if tuple(int(s) for s in seeds) != PINNED_SEEDS:
        fails.append(f"seeds {tuple(seeds)} != pinned {PINNED_SEEDS}")
    if cfg.cap_per_symbol is not None:
        fails.append(f"cap_per_symbol={cfg.cap_per_symbol} — the dev bound must be ABSENT")
    if float(cfg.headline_cost) != PINNED_HEADLINE_COST:
        fails.append(f"headline_cost {cfg.headline_cost} != pinned {PINNED_HEADLINE_COST}")

    # 7) per-symbol spread deciles in use — from the committed artifact, and not flat
    if cfg.spread_frac_by_symbol is None:
        fails.append("spread_frac_by_symbol is None — flat 'major' is not money-legal")
    else:
        try:
            art = json.loads(Path(deciles_path).read_text())
            art_map = {s: float(r["spread_frac"]) for s, r in art["deciles"].items()}
        except OSError as e:
            art_map = None
            fails.append(f"cannot read {deciles_path}: {e}")
        missing = [s for s in symbols if s not in cfg.spread_frac_by_symbol]
        if missing:
            fails.append(f"{len(missing)} evaluated symbols missing from the decile map")
        elif len({cfg.spread_frac_by_symbol[s] for s in symbols}) < 2:
            fails.append("decile map is FLAT over the evaluated symbols — per-symbol required")
        if art_map is not None and any(
            cfg.spread_frac_by_symbol.get(s) != art_map.get(s) for s in symbols
        ):
            fails.append("decile map values diverge from the committed artifact")

    # 8) shuffle_micro seed-threading: train path == eval path, per pinned seed
    for s in PINNED_SEEDS:
        if not _shuffle_threading_probe(s):
            fails.append(f"shuffle_micro train/eval threading DIVERGES at seed {s}")

    # 10) the §3a bootstrap recipe constants
    if (pb.BOOT_B, pb.BOOT_SEED, pb.BOOT_ALPHA) != (
        PINNED_BOOT["B"],
        PINNED_BOOT["seed"],
        PINNED_BOOT["alpha"],
    ):
        fails.append(
            f"bootstrap constants (B={pb.BOOT_B}, seed={pb.BOOT_SEED}, alpha={pb.BOOT_ALPHA}) "
            f"!= pinned {PINNED_BOOT}"
        )
    return fails


def verdict_dsr_failures(
    *,
    n_trials: int,
    threshold: float,
    trials: dict[tuple[int, int, int, float], float],
    var_sr: float,
) -> list[str]:
    """Divergences between the verdict path's DSR inputs and the §3-clause-5 pins (empty = OK).

    ``trials`` is the ACTUAL enumerated trial set — ``{(cell_id, seed, h, κ): de-annualized VAL
    IR}`` — not a self-declared count: a 240-trial budget (a 4th horizon), a 4-κ-only var_sr
    basis (the M5 ``run_harness`` convention, which is NOT the M6 decision path), or a missing
    cell/seed all fail the cross-product check. ``var_sr`` must equal the ddof=0 variance of the
    key-sorted trial values BIT-EXACTLY — the same construction the verdict uses — so a wrong
    ddof or a subset-variance is caught even when the key set is right."""
    fails: list[str] = []
    if int(n_trials) != PINNED_DSR["n_trials"]:
        fails.append(f"DSR n_trials {n_trials} != pinned {PINNED_DSR['n_trials']} (§3 clause 5)")
    if float(threshold) != PINNED_DSR["threshold"]:
        fails.append(f"DSR threshold {threshold} != pinned {PINNED_DSR['threshold']}")
    want = {
        (c, s, h, k)
        for c in PINNED_DSR["cells"]
        for s in PINNED_DSR["seeds"]
        for h in PINNED_DSR["horizons"]
        for k in PINNED_DSR["kappas"]
    }
    got = set(trials)
    if got != want:
        extra, missing = got - want, want - got
        fails.append(
            f"DSR trial set is not the pinned {len(want)}-trial cross product: {len(got)} "
            f"trials ({len(missing)} missing, {len(extra)} outside the pin"
            + (f"; e.g. extra {sorted(extra)[:2]}" if extra else "")
            + (f"; e.g. missing {sorted(missing)[:2]}" if missing else "")
            + ")"
        )
    expected_var = float(np.var(np.array([trials[k] for k in sorted(trials)], dtype=np.float64)))
    if float(var_sr) != expected_var:
        fails.append(
            f"var_sr {var_sr!r} != the ddof=0 variance of the enumerated trial values "
            f"({expected_var!r}) — wrong source (subset?) or wrong ddof"
        )
    return fails


def assert_conformance(
    cfg: XSectionConfig,
    *,
    symbols: list[str],
    seeds: tuple[int, ...],
    mde_inputs_path: Path = MDE_INPUTS,
    deciles_path: Path = DECILES,
) -> None:
    """The money-run gate: raise :class:`ConformanceError` listing EVERY divergence.

    The real eval driver calls this FIRST — before loading any checkpoint or computing any
    number. A non-empty failure list is a hard stop, never a warning."""
    fails = conformance_failures(
        cfg,
        symbols=symbols,
        seeds=seeds,
        mde_inputs_path=mde_inputs_path,
        deciles_path=deciles_path,
    )
    if fails:
        raise ConformanceError(
            "money-run config diverges from the pre-registered §3a surface:\n  - "
            + "\n  - ".join(fails)
        )


__all__ = [
    "DECILES",
    "MDE_INPUTS",
    "PINNED_DSR",
    "PINNED_ENCODER_CAUSAL",
    "PINNED_FINE_POINTWISE",
    "PINNED_HEADLINE_COST",
    "PINNED_KAPPAS",
    "PINNED_SEEDS",
    "PINNED_SYMBOLS_SHA256",
    "ConformanceError",
    "assert_conformance",
    "conformance_failures",
    "money_config",
    "symbols_sha256",
    "verdict_dsr_failures",
]
