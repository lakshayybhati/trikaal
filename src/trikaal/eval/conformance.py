"""Money-run conformance gate — the pinned §3a surface, machine-checked (pre-flight Item 8).

A lock is worthless if the computed quantity can drift from the locked one: every §3a pin that
the eval config could silently violate is asserted here, and the REAL eval driver must call
:func:`assert_conformance` as its FIRST step (before loading any model). The standalone runner
is ``scripts/m6_conformance.py``; its passing output is committed as the Item-8 artifact.

Checks (hard-fail on ANY diff; every failure listed, not just the first):
window + train_frac vs ``runs_manifest/m6_mde_inputs.json`` · money mode ON with the headline =
ONE continuous grid over forward blocks 1–5 matching the JSON's ``primary_region_ms`` (VAL
block 0 excluded) · the evaluated symbol set == ``symbols_sampled`` with its pinned sha256 ·
κ grid {1.0, 1.5, 2.0, 3.0} · seeds exactly {0, 1, 2, 3, 4} (§7 v1.5 S=5; this line read
"{0, 1, 2}" until §7 v1.6 C-6 — stale prose in the gate's own description) · no
``cap_per_symbol`` · per-symbol
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
PINNED_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)  # §7 v1.5 item D/C.1 Option 2: S=5 UP FRONT
PINNED_HEADLINE_COST = 0.0030
# §7 v1.6 C-6 — THE MONEY-SURFACE TRAINING PINS. These exist because the eval side and the
# training side each held their OWN copy of the truth and diverged: OrchestratorConfig carried
# seeds (0,1,2) and seq_len 128 against the pinned (0,1,2,3,4) and 512. The orchestrator now
# REFERENCES these rather than restating them, so there is no second copy left to drift, and
# asserts against them on a money run so a caller cannot re-introduce the divergence by argument.
PINNED_MONEY_SEQ_LEN = 512  # the money eval surface; ALSO the training context (see C-6 finding 2)
PINNED_MICRO_LEGIBILITY_MIN = (
    0.9  # §7 v1.4 standing six-micro-dims gate — a HARD STOP, was unpinned
)
PINNED_BACKBONE_PARAMS = 21_301_248  # the realized count (CLAUDE.md invariant), was unpinned
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

# §7 v1.4.1 (gate-2 final ruling): the per-bar bottleneck leg's micro-class weight, CALIBRATED
# not chosen — the SMALLEST lambda whose three SEEDED calibration seeds clear the restated gate
# (lambda=3: 0.9060/0.9142/0.9000, mean 0.9067, min 0.9000; lambda=2 fails at mean 0.8517;
# init-seeding defect disclosed in the prereg entry; runs_manifest/m6_lambda_search_receipt.json).
PINNED_MICRO_POINT_WEIGHT = 3.0


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
    got_l = tok_config.get("micro_point_weight")
    if got_l != PINNED_MICRO_POINT_WEIGHT:
        failures.append(
            f"{run}: tokenizer micro_point_weight={got_l!r} != pinned "
            f"{PINNED_MICRO_POINT_WEIGHT} (prereg §7 v1.4.1 — lambda is calibrated, "
            "not chosen; a different weighting invalidates the legibility receipts)"
        )
    return failures


# ---- the §3-clause-5 DSR pins (the verdict path's recipe — an INDEPENDENT statement of the
# prereg constants; trikaal/eval/verdict.py carries its own literals and the two must agree,
# so an edit to either side is caught before any DSR is computed) ------------------------------
PINNED_DSR = {
    # §7 v1.5 item A.5: N counts DISTINCT CONFIGURATIONS evaluated and displayed; it does NOT count
    # REPLICATES of a configuration. Seeds are the only replicate axis, so seeds are the only axis
    # removed — cells, horizons and κ are distinct configurations and stay. A multiple-testing
    # adjustment cannot legitimately get STRICTER because you replicated the same configuration more
    # times; a spec that penalises collecting more data on one hypothesis is misspecified.
    "n_trials": 60,  # 5 cells × 3 horizons × 4 κ — SEEDS EXCLUDED
    # §7 v1.6 C-7: WAS the prose "cells x horizons x kappas (seeds are replicates, not trials)" and
    # was never read. It is now the AXIS TUPLE the multiplicity count is computed FROM, so adding
    # "seeds" here changes n_trials and fails the gate instead of merely describing it.
    "n_trials_factorization": ("cells", "horizons", "kappas"),
    "cells": (1, 2, 3, 4, 5),
    # seeds are still ENUMERATED (audit trail + the var_sr basis) but are NOT counted in n_trials
    "seeds": PINNED_SEEDS,
    "horizons": (5, 15, 60),  # h=1 is not evaluated anywhere in M6 (the old 240 was an error)
    "kappas": PINNED_KAPPAS,
    "threshold": 0.95,
    # §7 v1.5 item A.4: a NULL DISPERSION ESTIMATE MUST NOT CONTAIN THE TREATMENT CONTRAST. The old
    # basis spanned structurally different arms, so if the microstructure claim were TRUE cell 4
    # would separate, the cross-trial spread would grow, var_sr would grow and clause 5 would get
    # HARDER — the clause was anti-correlated with the hypothesis it tests. The basis is now the
    # PLACEBO arm (cell 5): signal-free BY CONSTRUCTION (the shuffle destroys temporal alignment),
    # and matched to cell 4 on quantizer AND input dimensionality. NOT cell 2 — §5's NULL-fallback
    # can CLAIM IR(2)−IR(1), which makes cell 2 a treatment arm, and 7 vs 16 dims is a dispersion
    # mismatch. Seeds REMAIN inside var_sr (dispersion) while being absent from n_trials
    # (multiplicity): deliberately different sets.
    "var_sr_basis_cell": 5,
    # §7 v1.6 C-7: WAS the prose "population variance (ddof=0) of the de-annualized VAL IRs of the
    # CELL-5 (placebo) trials" and was never read. The ddof is now the MACHINE-READABLE value the
    # expected-variance re-derivation uses, and is separately asserted to be 0 — so a drift to a
    # sample variance is caught rather than described.
    "var_sr_ddof": 0,
    # §7 v1.6 C-7: READ — the verdict manifest's clause-5 rule string is BUILT from this key, so
    # the shipped artifact cannot carry a description that disagrees with the pinned recipe.
    # §7 v1.6.13 (ruling (a) sweep): the netting rate was the LAST literal recipe number left in a
    # manifest prose string. It now derives from PINNED_HEADLINE_COST, so it cannot drift from the
    # cost the money path actually charges. Renders byte-identically to the predecessor ("0.30%").
    "statistic": (
        f"Cell 4's seed-mean pooled headline series at the {PINNED_HEADLINE_COST:.2%} flat netting"
    ),
    # §7 v1.5 A.4.3 tripwire: pb(5−2) is a LEVEL statement and var_sr is a DISPERSION statistic, so
    # placebo-victim behaviour does not move a variance at first order. The second-order risk
    # (degraded training → more variable models) errs CONSERVATIVE and is measured, not assumed.
    # §7 v1.6 C-7: READ, and cross-checked against verdict.PLACEBO_DISPERSION_TRIPWIRE — the two
    # 1.5s were independent copies of the same truth, the exact shape that produced C-6.
    "placebo_dispersion_tripwire": 1.5,  # × the median across-seed IR dispersion of cells 1,2,3
}

# §7 v1.6 C-7 — THRESHOLDS THAT DECIDE THE VERDICT AND WERE UNDER NO GATE AT ALL. Each lives as a
# module constant in the file that uses it, and until now nothing asserted those constants still
# held their pre-registered values: an edit to any of them would have moved a §3 clause silently.
# This is the same defect class as C-2 (a gate that examines nothing) one level up — a pinned
# surface that pins nothing. The registry is INDEPENDENT of the definitions, so it is a real
# second statement of the value and not an alias.
PINNED_THRESHOLDS: dict[str, float] = {
    "ECON_FLOOR_IR": 0.5,  # §3 clause 4 materiality floor, fixed pre-data (verdict.py)
    "BOOT_POWER": 0.80,  # §3 clause 2 power point inside MDE_paired (paired_bootstrap.py)
    "PLACEBO_DISPERSION_TRIPWIRE": 1.5,  # §7 v1.5 A.4.3 (verdict.py)
}


def pinned_threshold_failures() -> list[str]:
    """Divergences between the LIVE verdict/bootstrap constants and their pins (empty = OK).

    §7 v1.6 C-7. Imported locally because ``verdict`` imports THIS module — the dependency runs
    one way and a module-level import here would be circular. Called at runtime from both gates,
    long after import, so the local import is free.
    """
    from trikaal.eval.paired_bootstrap import BOOT_POWER
    from trikaal.eval.verdict import (
        DSR_VAR_SR_BASIS_CELL,
        ECON_FLOOR_IR,
        PLACEBO_DISPERSION_TRIPWIRE,
    )

    live = {
        "ECON_FLOOR_IR": float(ECON_FLOOR_IR),
        "BOOT_POWER": float(BOOT_POWER),
        "PLACEBO_DISPERSION_TRIPWIRE": float(PLACEBO_DISPERSION_TRIPWIRE),
    }
    fails = [
        f"{name} = {live[name]!r} != pinned {want!r} (§7 v1.6 C-7)"
        for name, want in PINNED_THRESHOLDS.items()
        if live[name] != float(want)
    ]
    # the duplicate-source-of-truth leg: two independent 1.5s must agree
    if float(PINNED_DSR["placebo_dispersion_tripwire"]) != live["PLACEBO_DISPERSION_TRIPWIRE"]:
        fails.append(
            f"PINNED_DSR['placebo_dispersion_tripwire'] "
            f"{PINNED_DSR['placebo_dispersion_tripwire']!r} != "
            f"verdict.PLACEBO_DISPERSION_TRIPWIRE {live['PLACEBO_DISPERSION_TRIPWIRE']!r} — two "
            "copies of one truth, the shape that produced the C-6 seed/seq_len split-brain"
        )
    if int(DSR_VAR_SR_BASIS_CELL) != int(PINNED_DSR["var_sr_basis_cell"]):
        fails.append(
            f"verdict.DSR_VAR_SR_BASIS_CELL {DSR_VAR_SR_BASIS_CELL} != pinned "
            f"{PINNED_DSR['var_sr_basis_cell']} — the §7 v1.5 A.4 placebo basis"
        )
    return fails


class ConformanceError(AssertionError):
    """The money-run config diverged from the pre-registered §3a surface."""


def money_config(
    h: int = 15,
    *,
    seq_len: int = PINNED_MONEY_SEQ_LEN,  # §7 v1.6 C-6: ONE constant feeds eval AND training
    device: str = "cpu",
    seed: int = 0,
):
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
    IR}`` — not a self-declared count: a 240-trial budget (a 4th horizon), the M5 ``run_harness``
    4-κ convention (which is NOT the M6 decision path), or a missing cell/seed all fail the
    cross-product check.

    §7 v1.5 SPLITS THREE THINGS THAT USED TO BE ONE. (1) The ENUMERATION stays the FULL
    ``cells × seeds × horizons × κ`` cross-product — the audit trail, so nothing can go missing.
    (2) ``n_trials`` is the MULTIPLICITY count and excludes seeds (replicates are not
    configurations), so it is ``cells × horizons × κ`` = 60 regardless of S. (3) ``var_sr`` is the
    DISPERSION and is computed over the CELL-5 (placebo) subset only — a null dispersion estimate
    must not contain the treatment contrast — with seeds retained inside it. All three are asserted
    independently here, and ``var_sr`` must match the ddof=0 variance of the key-sorted placebo
    subset BIT-EXACTLY, so a wrong ddof, a wrong basis, or a subset-variance is caught even when
    the key set is right."""
    fails: list[str] = []
    if int(n_trials) != PINNED_DSR["n_trials"]:
        fails.append(f"DSR n_trials {n_trials} != pinned {PINNED_DSR['n_trials']} (§3 clause 5)")
    if float(threshold) != PINNED_DSR["threshold"]:
        fails.append(f"DSR threshold {threshold} != pinned {PINNED_DSR['threshold']}")
    # §7 v1.6 C-7: the pinned surface must hold its own registered values before it is used to
    # judge anything. A pin that is never read is decoration, and decoration gives false assurance.
    fails += pinned_threshold_failures()
    # (2) the MULTIPLICITY count must equal the seed-free factorization. The AXES come from the
    # pin (§7 v1.6 C-7) rather than being hardcoded here, so "seeds" cannot be re-introduced
    # into the multiplicity count without the count changing and this check firing.
    axes = tuple(PINNED_DSR["n_trials_factorization"])
    if "seeds" in axes:
        fails.append(
            f"n_trials_factorization {axes} contains 'seeds' — §7 v1.5 A.5: seeds are REPLICATES, "
            "and a multiplicity adjustment must not get stricter because a configuration was "
            "replicated more times"
        )
    want_n = 1
    for a in axes:
        want_n *= len(PINNED_DSR[a])
    if int(PINNED_DSR["n_trials"]) != want_n:
        fails.append(
            f"PINNED_DSR n_trials {PINNED_DSR['n_trials']} != {' x '.join(axes)} = "
            f"{want_n} (§7 v1.5 A.5: seeds are replicates and must NOT be counted)"
        )
    # (1) the ENUMERATION is still the FULL cross-product INCLUDING seeds (the audit trail)
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
    # (3) var_sr is the PLACEBO-arm dispersion — a null estimate must not contain the contrast
    basis_cell = int(PINNED_DSR["var_sr_basis_cell"])
    # §7 v1.6 C-7: the ddof is READ from the pin, and separately asserted to be the population
    # value — so a drift to a sample variance cannot pass by quietly moving the expectation too.
    ddof = int(PINNED_DSR["var_sr_ddof"])
    if ddof != 0:
        fails.append(
            f"PINNED_DSR var_sr_ddof {ddof} != 0 — §7 v1.5 A.4 pins the POPULATION variance"
        )
    basis_keys = sorted(k for k in trials if k[0] == basis_cell)
    if not basis_keys:
        fails.append(f"var_sr basis cell {basis_cell} absent from the enumerated trial set")
    else:
        expected_var = float(
            np.var(np.array([trials[k] for k in basis_keys], dtype=np.float64), ddof=ddof)
        )
        if float(var_sr) != expected_var:
            fails.append(
                f"var_sr {var_sr!r} != the ddof=0 variance of the CELL-{basis_cell} (placebo) "
                f"trial values ({expected_var!r}) — wrong basis, wrong ddof, or the pre-v1.5 "
                "all-arms basis (which made clause 5 anti-correlated with its own hypothesis)"
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
    # §7 v1.6 C-7: the verdict-deciding thresholds are checked at the MONEY-RUN gate too, not only
    # on the verdict path — a run must not start under an edited floor and discover it at the end.
    fails += pinned_threshold_failures()
    if fails:
        raise ConformanceError(
            "money-run config diverges from the pre-registered §3a surface:\n  - "
            + "\n  - ".join(fails)
        )


__all__ = [
    "DECILES",
    "MDE_INPUTS",
    "PINNED_BACKBONE_PARAMS",
    "PINNED_DSR",
    "PINNED_ENCODER_CAUSAL",
    "PINNED_FINE_POINTWISE",
    "PINNED_HEADLINE_COST",
    "PINNED_KAPPAS",
    "PINNED_MICRO_LEGIBILITY_MIN",
    "PINNED_MICRO_POINT_WEIGHT",
    "PINNED_MONEY_SEQ_LEN",
    "PINNED_SEEDS",
    "PINNED_SYMBOLS_SHA256",
    "PINNED_THRESHOLDS",
    "ConformanceError",
    "assert_conformance",
    "conformance_failures",
    "money_config",
    "pinned_threshold_failures",
    "symbols_sha256",
    "verdict_dsr_failures",
]
