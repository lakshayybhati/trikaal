"""The M6 verdict assembler — prereg §3's five conjunctive clauses, evaluated EXACTLY once.

This module + ``scripts/m6_verdict.py`` are **the ONLY code path allowed to produce the
SURVIVES/NULL decision**. Everything else that looks decision-shaped is diagnostics:
``xsection.ablation_verdict`` emits point comparisons (no CI, no clauses, no final word) and
``harness.run_harness`` is the M5 machine-validation instrument — neither may ever be quoted as
the M6 outcome (docs/m6_prereg.md §3; the third pre-training audit's finding #1).

The pipeline (docs/m6_prereg.md §3, §3a, §5 — every constant pinned there):

1. the 15 per-(cell, seed) eval artifacts (5 cells × seeds {0,1,2}) are loaded **by content
   hash** — each file's sha256 must match the driver-written index; any mismatch is a hard stop;
2. per cell, the **seed-mean pooled calendar series** on the §3a money grid (full calendar,
   flat periods = 0.0 — xsection's convention keeps every cell time-aligned);
3. paired moving-block bootstraps (§3a pinned recipe, ``paired_bootstrap``) on ΔIR(4−5)
   [primary], ΔIR(4−2) [clause 3], ΔIR(5−2) [placebo-health diagnostic];
4. the five clauses, conjunctive: (1) CI lower bound of ΔIR(4−5) > 0; (2) ΔIR ≥ MDE_paired
   (no-ceiling: the operative threshold is ALWAYS the realized MDE_paired, in either direction
   vs the §2 tabled value — both are recorded); (3) CI lower bound of ΔIR(4−2) > 0, with the
   (5−2) ci_upper < 0 "shuffle harmed" disclosure flag (never gating); (4) ΔIR ≥ 0.5 annualized
   (the economic floor); (5) DSR ≥ 0.95 with SR₀ from **N = 180 enumerated trials** (5 cells ×
   3 seeds × horizons {5,15,60} × 4 κ — every VAL entry persisted in the artifacts, the trial
   set is enumerated and cross-product-asserted, never assumed), var_sr = variance of the 180
   de-annualized per-trial VAL IRs, statistic + higher moments = Cell 4's seed-mean series;
5. on NULL, §5's fallback: IR(2)−IR(1) is CLAIMED only if it passes the IDENTICAL paired rule
   (CI > 0, ≥ its own MDE_paired, ≥ 0.5, DSR ≥ 0.95 on Cell 2's series under the same N=180
   budget), else DESCRIPTIVE_ONLY — double-NULL is a stated possible outcome.

The manifest is durable and content-hashed; the final word exists nowhere else.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from trikaal.eval.conformance import ConformanceError, verdict_dsr_failures
from trikaal.eval.dsr import deflated_sharpe_ratio, expected_max_sharpe
from trikaal.eval.metrics import information_ratio, periods_per_year
from trikaal.eval.paired_bootstrap import PairedBootstrap, paired_delta_ir_bootstrap
from trikaal.eval.xsection import ablation_verdict
from trikaal.train.cells import CELLS
from trikaal.utils.hashing import content_hash

# ---- prereg pins (literals mirror docs/m6_prereg.md §3/§3a/§5) -------------------------------
PRIMARY_H = 15  # §3: h=15 is the primary; no horizon-shopping
DSR_HORIZONS: tuple[int, ...] = (5, 15, 60)  # §3 clause 5: h=1 is not evaluated anywhere in M6
DSR_KAPPAS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)  # §3a κ grid
DSR_SEEDS: tuple[int, ...] = (0, 1, 2)  # §3a seeds
DSR_N_TRIALS = 180  # §3 clause 5: 5 cells × 3 seeds × 3 horizons × 4 κ, ENUMERATED
DSR_THRESHOLD = 0.95  # §3 clause 5
ECON_FLOOR_IR = 0.5  # §3 clause 4: annualized-IR materiality floor, fixed pre-data
SURVIVES = "SURVIVES"
NULL = "NULL"
HALT_ADJUDICATE = "HALT_ADJUDICATE"  # §7 v1.4.4 degeneracy guard (HALT-only, never S<->N)
FALLBACK_CLAIMED = "CLAIMED"  # §5: passed the identical paired rule
FALLBACK_DESCRIPTIVE = "DESCRIPTIVE_ONLY"  # §5: reported with CIs, claimed as nothing
# §7 v1.4.4 DEGENERACY GUARD: a cell is a constant-direction book (not a strategy) if its
# seed-mean μ̂ sign is locked (frac_negative outside this band) OR its θ=κ·c execution filter
# never binds (seed-mean decision-activity at κ* is exactly 0 or 1 — every/no decision trades,
# so the IR is a pure function of sign(μ̂)). Any clause involving such a cell is uninterpretable,
# so the verdict HALTS for adjudication. The guard is HALT-ONLY: it can only supersede
# SURVIVES/NULL with HALT_ADJUDICATE, and can NEVER flip SURVIVES↔NULL — the clause results are
# computed identically whether or not it fires. This asymmetry is what makes it legitimate to add
# pre-run (it can never manufacture a positive or negative claim, only refuse to emit one).
FRAC_NEG_BAND: tuple[float, float] = (0.05, 0.95)

ARTIFACT_SCHEMA = "m6_cell_eval_v1"
INDEX_SCHEMA = "m6_eval_index_v1"
INDEX_NAME = "index.json"
VERDICT_SCHEMA = "m6_verdict_v1"

_CELL_BY_ID = {c.cell_id: c for c in CELLS}


class VerdictInputError(AssertionError):
    """The eval-artifact set is malformed, incomplete, or fails its content-hash check."""


# ------------------------------------------------------------------ artifact emission (shared)
def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_cell_eval_artifact(
    out_dir: Path,
    *,
    cell_id: int,
    seed: int,
    grid: dict,
    headline_series: np.ndarray,
    kappa_star_by_h: dict[int, float],
    val_ir_by_kappa_by_h: dict[int, dict[float, float]],
    codebook: dict | None = None,
    mu_diag: dict | None = None,
    meta: dict | None = None,
) -> tuple[str, str]:
    """Write ONE per-(cell, seed) eval artifact; returns ``(name, sha256)``.

    This is the single emission path — the real eval driver AND the KAT fixture builders both
    write through here, so the loader can never drift from the writer. ``headline_series`` is
    the pooled FULL-calendar money-grid series at κ* netted at the flat 0.30 % (h = PRIMARY_H);
    ``val_ir_by_kappa_by_h`` carries the ANNUALIZED VAL IR for every (h, κ) — the DSR trial
    entries §3 clause 5 requires persisted."""
    spec = _CELL_BY_ID[cell_id]
    doc = {
        "schema": ARTIFACT_SCHEMA,
        "cell_id": int(cell_id),
        "seed": int(seed),
        "quantizer": spec.quantizer,
        "arm": spec.arm,
        "grid": {
            "h": int(grid["h"]),
            "start_ms": int(grid["start_ms"]),
            "n_periods": int(grid["n_periods"]),
        },
        "kappa_star_by_h": {str(int(h)): float(k) for h, k in kappa_star_by_h.items()},
        "val_ir_by_kappa_by_h": {
            str(int(h)): {f"{float(k):g}": float(v) for k, v in by_k.items()}
            for h, by_k in val_ir_by_kappa_by_h.items()
        },
        "headline_series": [float(v) for v in np.asarray(headline_series, dtype=np.float64)],
        # non-gating codebook-usage diagnostic (CO2 item 5): score_cell's CellScore.codebook —
        # per-cell utilization + effective bits, reported in the paper, never thresholded
        "codebook": codebook or {},
        # §7 v1.4.2 standing μ̂ receipt: per-cell mean/std/frac_negative of the decision signal
        # (the mean estimator; a constant-sign μ̂ was the acceptance mode-bias pathology)
        "mu_diag": mu_diag or {},
        "meta": meta or {},
    }
    name = f"cell{cell_id}_seed{seed}_eval.json"
    path = Path(out_dir) / name
    path.write_text(json.dumps(doc, sort_keys=True))
    return name, _file_sha256(path)


def write_eval_index(out_dir: Path, entries: dict[str, str]) -> Path:
    """Write the content-hash index over the 15 artifacts: ``{name: sha256}``."""
    path = Path(out_dir) / INDEX_NAME
    path.write_text(
        json.dumps({"schema": INDEX_SCHEMA, "artifacts": dict(entries)}, sort_keys=True)
    )
    return path


# ------------------------------------------------------------------ content-addressed loading
def _validate_artifact(doc: dict, name: str) -> list[str]:
    """Schema violations in one artifact (empty = valid)."""
    bad: list[str] = []
    if doc.get("schema") != ARTIFACT_SCHEMA:
        bad.append(f"{name}: schema {doc.get('schema')!r} != {ARTIFACT_SCHEMA!r}")
        return bad
    cid, seed = int(doc["cell_id"]), int(doc["seed"])
    spec = _CELL_BY_ID.get(cid)
    if spec is None or seed not in DSR_SEEDS:
        bad.append(f"{name}: (cell_id={cid}, seed={seed}) outside the pinned 5x{{0,1,2}} matrix")
    elif (doc.get("quantizer"), doc.get("arm")) != (spec.quantizer, spec.arm):
        bad.append(f"{name}: quantizer/arm {doc.get('quantizer')}/{doc.get('arm')} != cell {cid}")
    if int(doc["grid"]["h"]) != PRIMARY_H:
        bad.append(f"{name}: grid h={doc['grid']['h']} != the primary h={PRIMARY_H}")
    if len(doc["headline_series"]) != int(doc["grid"]["n_periods"]):
        bad.append(f"{name}: series length {len(doc['headline_series'])} != grid n_periods")
    want_h = {str(h) for h in DSR_HORIZONS}
    got_h = set(doc["val_ir_by_kappa_by_h"])
    if got_h != want_h:
        bad.append(f"{name}: VAL horizons {sorted(got_h)} != pinned {sorted(want_h)}")
    else:
        want_k = {f"{k:g}" for k in DSR_KAPPAS}
        for h in sorted(want_h):
            got_k = set(doc["val_ir_by_kappa_by_h"][h])
            if got_k != want_k:
                bad.append(f"{name}: h={h} κ entries {sorted(got_k)} != pinned {sorted(want_k)}")
    return bad


def load_cell_evals(art_dir: Path) -> tuple[dict[tuple[int, int], dict], dict[str, str]]:
    """Load the 15 artifacts **by content hash** → ``({(cell_id, seed): doc}, {name: sha256})``.

    Hard-fails (:class:`VerdictInputError`, every problem listed) on: a missing index, a missing
    or extra artifact, ANY file whose sha256 diverges from the index, a schema violation, or
    grid metadata that is not identical across all 15 (the §3 pairing contract needs one shared
    calendar grid)."""
    art_dir = Path(art_dir)
    idx_path = art_dir / INDEX_NAME
    if not idx_path.exists():
        raise VerdictInputError(f"no {INDEX_NAME} in {art_dir} — artifacts must be indexed")
    idx = json.loads(idx_path.read_text())
    if idx.get("schema") != INDEX_SCHEMA:
        raise VerdictInputError(f"index schema {idx.get('schema')!r} != {INDEX_SCHEMA!r}")
    entries: dict[str, str] = dict(idx["artifacts"])

    problems: list[str] = []
    want_names = {f"cell{c.cell_id}_seed{s}_eval.json" for c in CELLS for s in DSR_SEEDS}
    missing, extra = want_names - set(entries), set(entries) - want_names
    if missing:
        problems.append(f"index missing {len(missing)} artifacts: {sorted(missing)[:3]}…")
    if extra:
        problems.append(f"index has {len(extra)} unexpected entries: {sorted(extra)[:3]}…")

    evals: dict[tuple[int, int], dict] = {}
    grids: set[tuple[int, int, int]] = set()
    for name in sorted(want_names & set(entries)):
        path = art_dir / name
        if not path.exists():
            problems.append(f"{name}: indexed but absent on disk")
            continue
        got = _file_sha256(path)
        if got != entries[name]:
            problems.append(f"{name}: sha256 {got[:16]}… != indexed {entries[name][:16]}…")
            continue
        doc = json.loads(path.read_text())
        bad = _validate_artifact(doc, name)
        if bad:
            problems.extend(bad)
            continue
        evals[(int(doc["cell_id"]), int(doc["seed"]))] = doc
        g = doc["grid"]
        grids.add((int(g["h"]), int(g["start_ms"]), int(g["n_periods"])))
    if len(grids) > 1:
        problems.append(f"artifacts disagree on the grid: {sorted(grids)} — pairing needs ONE")
    if problems:
        raise VerdictInputError("eval-artifact set rejected:\n  - " + "\n  - ".join(problems))
    return evals, entries


# ------------------------------------------------------------------ the §3 clause evaluation
def _seed_mean_series(evals: dict[tuple[int, int], dict], cell_id: int) -> np.ndarray:
    rows = [np.asarray(evals[(cell_id, s)]["headline_series"], np.float64) for s in DSR_SEEDS]
    return np.mean(np.stack(rows), axis=0)


def enumerate_dsr_trials(
    evals: dict[tuple[int, int], dict],
) -> dict[tuple[int, int, int, float], float]:
    """The §3-clause-5 trial set: every (cell, seed, horizon, κ) VAL IR, **de-annualized**.

    Returns exactly the 5 × 3 × 3 × 4 = 180 cross product keyed by (cell_id, seed, h, κ);
    the caller asserts the enumeration against the pinned recipe before any DSR is computed."""
    trials: dict[tuple[int, int, int, float], float] = {}
    for (cid, seed), doc in evals.items():
        for h in DSR_HORIZONS:
            by_k = doc["val_ir_by_kappa_by_h"][str(h)]
            for k in DSR_KAPPAS:
                ir_ann = float(by_k[f"{k:g}"])
                trials[(cid, seed, h, k)] = ir_ann / float(np.sqrt(periods_per_year(h)))
    return trials


def _pb_dict(pb: PairedBootstrap) -> dict:
    from dataclasses import asdict

    return {k: (float(v) if isinstance(v, float) else v) for k, v in asdict(pb).items()}


def degeneracy_guard(evals: dict[tuple[int, int], dict]) -> dict:
    """§7 v1.4.6 HALT-only guard: flag any cell whose IR is a constant-direction book.

    A cell is degenerate if its sign is locked (``frac_negative`` outside FRAC_NEG_BAND) or the
    θ=κ·c filter never binds (decision-activity at κ* is exactly 0 or 1), tested BOTH on the
    seed-mean AND **on every individual seed** (a union — either trips the HALT).

    WHY THE PER-SEED LEG EXISTS (§7 v1.4.6, supervisor-identified). v1.4.4 tested ONLY the
    seed-mean and threw the per-seed lists away. The v1.4.5 h-sweep measured a 1-in-2 lock rate
    across two byte-identical-recipe runs, so a realistic triple is (0.999, 0.55, 0.55): its mean
    is 0.700 — comfortably inside [0.05, 0.95], no HALT — while the locked seed's
    constant-direction book still enters the seed-mean headline series the §3 clauses are computed
    on. Averaging is exactly the wrong reduction for a degeneracy test: one poisoned seed is not
    diluted by two healthy ones, it contaminates the aggregate the verdict reads. The per-seed
    values are now PERSISTED too, so an adjudication has the distribution rather than a mean.

    PURE READ — never touches the clause computation, so it cannot alter a SURVIVES/NULL outcome;
    it only reports whether the verdict must HALT. A missing ``activity_decisions`` (pre-v1.4.4
    artifact) skips the filter legs (NaN comparisons are False); the sign legs still apply.
    """
    lo, hi = FRAC_NEG_BAND

    def _locked(v: float) -> bool:
        return v == v and not (lo <= v <= hi)  # v==v excludes NaN

    def _never_binds(v: float) -> bool:
        return v == v and (v <= 0.0 or v >= 1.0)

    per_cell: dict[str, dict] = {}
    degenerate: list[int] = []
    for c in (1, 2, 3, 4, 5):
        fns = [
            float(evals[(c, s)].get("mu_diag", {}).get("frac_negative", float("nan")))
            for s in DSR_SEEDS
        ]
        acts = [
            float(evals[(c, s)].get("mu_diag", {}).get("activity_decisions", float("nan")))
            for s in DSR_SEEDS
        ]
        fn, act = float(np.mean(fns)), float(np.mean(acts))
        bad_fn = [s for s, v in zip(DSR_SEEDS, fns, strict=True) if _locked(v)]
        bad_act = [s for s, v in zip(DSR_SEEDS, acts, strict=True) if _never_binds(v)]
        reasons: list[str] = []
        if _locked(fn):
            reasons.append(f"seed-mean frac_negative {fn:.4f} outside [{lo}, {hi}] — sign-locked")
        if _never_binds(act):
            reasons.append(
                f"seed-mean decision-activity {act:.4f} at kappa* is 0 or 1 — filter never binds"
            )
        if bad_fn:
            reasons.append(
                f"per-seed frac_negative outside [{lo}, {hi}] on seed(s) {bad_fn} "
                f"(values {[fns[DSR_SEEDS.index(s)] for s in bad_fn]}) — a locked seed's "
                "constant-direction book enters the seed-mean series the clauses read"
            )
        if bad_act:
            reasons.append(
                f"per-seed decision-activity at kappa* is 0 or 1 on seed(s) {bad_act} "
                f"(values {[acts[DSR_SEEDS.index(s)] for s in bad_act]}) — filter never binds there"
            )
        deg = bool(reasons)
        # NaN (a pre-v1.4.4 artifact missing the field) → None so the manifest stays deterministic
        # (nan != nan) and JSON-clean; a None value never triggers degeneracy (checks are nan-safe).
        per_cell[str(c)] = {
            "frac_negative_seed_mean": (fn if fn == fn else None),
            "activity_decisions_seed_mean": (act if act == act else None),
            # §7 v1.4.6: the per-seed DISTRIBUTION, persisted — an adjudication needs the seeds,
            # not an aggregate that can hide a single locked one.
            "frac_negative_by_seed": {
                str(s): (v if v == v else None) for s, v in zip(DSR_SEEDS, fns, strict=True)
            },
            "activity_decisions_by_seed": {
                str(s): (v if v == v else None) for s, v in zip(DSR_SEEDS, acts, strict=True)
            },
            "degenerate_seeds_frac_negative": list(bad_fn),
            "degenerate_seeds_activity": list(bad_act),
            "degenerate": deg,
            "reasons": reasons,
        }
        if deg:
            degenerate.append(c)
    return {
        "armed": True,
        "frac_neg_band": list(FRAC_NEG_BAND),
        "rule": "HALT-ONLY (§7 v1.4.6): a cell is a constant-direction book if frac_negative is "
        "outside the band OR decision-activity at kappa* is 0 or 1, evaluated on the seed-mean OR "
        "ON ANY INDIVIDUAL SEED (union). The per-seed leg closes the v1.4.4 hole where a triple "
        "like (0.999, 0.55, 0.55) averages to 0.700 and passes while one seed is fully locked. Any "
        "clause involving such a cell is uninterpretable, so the verdict is HALT_ADJUDICATE. The "
        "guard NEVER flips SURVIVES<->NULL — the clauses/primary below are computed identically.",
        "per_cell": per_cell,
        "degenerate_cells": degenerate,
        "halted": bool(degenerate),
    }


def power_guard(evals: dict[tuple[int, int], dict], claimed: dict[str, dict]) -> dict:
    """§7 v1.4.7 HALT-only POWER guard: refuse a claim its own seed spread cannot support.

    THE RISK THIS MEASURES (supervisor-identified, v1.4.7). The pre-registered MDE was derived
    entirely from SCORING noise — a paired moving-block bootstrap over TIME — and contains no
    training-variance term at all. But v1.4.5 measured two runs of the SAME recipe at the SAME
    SEED landing on frac_negative 0.5662 vs 0.99883: basin-hopping, not float jitter. Same-seed
    divergence is a LOWER BOUND on across-seed variance, so the tabled MDE understates the true
    detection threshold and the 3-seed design may be underpowered against its own effect size.
    Forcing deterministic algorithms would remove the same-seed component and NOT the across-seed
    component, so this cannot be fixed by determinism alone.

    THE RULE. For each claimed between-cell ΔIR, take the WITHIN-cell across-seed IR range of the
    cells that delta is built from. If that range meets or exceeds |ΔIR|, the claim is smaller than
    the seed-to-seed wobble of its own inputs and the verdict is HALT_ADJUDICATE.

    Like ``degeneracy_guard`` this is a PURE READ computed post-hoc: it never touches a clause and
    can never flip SURVIVES<->NULL. It only refuses to REPORT an outcome its own inputs cannot
    support — and it halts symmetrically on a NULL too, because an underpowered NULL is equally
    uninterpretable (the same symmetry ``degeneracy_guard`` already has).
    """
    per_cell: dict[str, dict] = {}
    for c in (1, 2, 3, 4, 5):
        irs = [
            float(
                information_ratio(
                    np.asarray(evals[(c, s)]["headline_series"], np.float64), PRIMARY_H
                )
            )
            for s in DSR_SEEDS
        ]
        finite = [v for v in irs if np.isfinite(v)]
        per_cell[str(c)] = {
            "ir_by_seed": {str(s): v for s, v in zip(DSR_SEEDS, irs, strict=True)},
            "ir_range_across_seeds": (max(finite) - min(finite)) if finite else None,
            "ir_std_across_seeds": float(np.std(finite)) if finite else None,
        }

    checks: dict[str, dict] = {}
    tripped: list[str] = []
    for name, meta in claimed.items():
        delta = float(meta["delta_ir"])
        cells = [str(c) for c in meta["cells"]]
        ranges = [per_cell[c]["ir_range_across_seeds"] for c in cells]
        worst = max((r for r in ranges if r is not None), default=None)
        trips = bool(worst is not None and np.isfinite(delta) and worst >= abs(delta))
        checks[name] = {
            "delta_ir_claimed": delta,
            "cells": cells,
            "worst_within_cell_ir_range": worst,
            "seed_spread_exceeds_claim": trips,
        }
        if trips:
            tripped.append(name)
    return {
        "armed": True,
        "rule": "HALT-ONLY (§7 v1.4.7): if the WITHIN-cell across-seed IR range of the cells a "
        "claimed between-cell dIR is built from meets or exceeds |dIR|, the claim is smaller than "
        "the seed-to-seed wobble of its own inputs and the verdict is HALT_ADJUDICATE. The tabled "
        "MDE contains SCORING noise only (paired moving-block bootstrap over time) and no "
        "training-variance term, so it cannot speak to this. PURE READ — never alters a clause, "
        "never flips SURVIVES<->NULL; it only refuses to report an unsupported outcome.",
        "per_cell": per_cell,
        "checks": checks,
        "underpowered_claims": tripped,
        "halted": bool(tripped),
    }


def assemble_verdict(
    evals: dict[tuple[int, int], dict],
    artifact_sha256: dict[str, str],
    *,
    tabled_mde_h15: float,
) -> dict:
    """Evaluate §3's five clauses on the 15 loaded artifacts → the verdict manifest body.

    Pure computation, no I/O: the driver (``scripts/m6_verdict.py``) supplies content-verified
    inputs and persists the returned manifest. The §3a bootstrap recipe is pinned inside
    ``paired_bootstrap`` (B=10,000, seed 20260704, ⌈√T⌉, percentile) — not parameterized here."""
    s = {c: _seed_mean_series(evals, c) for c in (1, 2, 3, 4, 5)}
    ir = {c: float(information_ratio(s[c], PRIMARY_H)) for c in s}

    pb45 = paired_delta_ir_bootstrap(s[4], s[5], h=PRIMARY_H)
    pb42 = paired_delta_ir_bootstrap(s[4], s[2], h=PRIMARY_H)
    pb52 = paired_delta_ir_bootstrap(s[5], s[2], h=PRIMARY_H)

    # clause 5: the pinned DSR recipe over the ENUMERATED N=180 trial set. The recipe is
    # asserted against conformance.PINNED_DSR — an INDEPENDENT statement of the §3-clause-5
    # constants — before any DSR is computed; var_sr uses the key-sorted ddof=0 construction
    # the pin re-derives bit-exactly (a subset-variance or a ddof drift cannot pass).
    trials = enumerate_dsr_trials(evals)
    var_sr = float(np.var(np.array([trials[k] for k in sorted(trials)], dtype=np.float64)))
    recipe_fails = verdict_dsr_failures(
        n_trials=DSR_N_TRIALS, threshold=DSR_THRESHOLD, trials=trials, var_sr=var_sr
    )
    if recipe_fails:
        raise ConformanceError(
            "verdict DSR recipe diverges from the §3-clause-5 pins:\n  - "
            + "\n  - ".join(recipe_fails)
        )
    sr0 = float(expected_max_sharpe(var_sr, DSR_N_TRIALS))
    dsr = float(deflated_sharpe_ratio(s[4], n_trials=DSR_N_TRIALS, var_sr=var_sr))

    shuffle_harmed = bool(pb52.ci_upper < 0.0)  # §3 clause 3: disclosure, never gating
    clauses = {
        "1_paired_ci": {
            "rule": "one-sided paired CI lower bound of ΔIR_info(4-5) > 0",
            "ci_lower": pb45.ci_lower,
            "pass": pb45.passes_ci,
        },
        "2_mde_paired": {
            "rule": "ΔIR_info ≥ MDE_paired = (z0.95+z0.80)·SE_boot; no ceiling appeal — the "
            "operative threshold is the realized MDE_paired in either direction vs §2's table",
            "delta_ir": pb45.delta_ir,
            "mde_paired": pb45.mde_paired,
            "tabled_mde_h15": float(tabled_mde_h15),
            "mde_paired_exceeds_tabled": bool(pb45.mde_paired > float(tabled_mde_h15)),
            "pass": pb45.passes_mde,
        },
        "3_placebo_validity": {
            "rule": "one-sided paired CI lower bound of IR(4)-IR(2) > 0 (placebo-independent)",
            "ci_lower": pb42.ci_lower,
            "pass": pb42.passes_ci,
            "health_diagnostic_5_minus_2": {
                "delta_ir": pb52.delta_ir,
                "ci_lower": pb52.ci_lower,
                "ci_upper": pb52.ci_upper,
                "shuffle_harmed": shuffle_harmed,
                "note": "ci_upper < 0 ⇒ the shuffle demonstrably harmed training below the "
                "OHLCV-only counterfactual — disclosed prominently, never gating (§3 clause 3)",
            },
        },
        "4_economic_floor": {
            "rule": f"ΔIR_info ≥ {ECON_FLOOR_IR} annualized IR (fixed pre-data)",
            "delta_ir": pb45.delta_ir,
            "floor": ECON_FLOOR_IR,
            "pass": bool(pb45.delta_ir >= ECON_FLOOR_IR),
        },
        "5_dsr": {
            "rule": f"DSR ≥ {DSR_THRESHOLD}; statistic = Cell 4's seed-mean pooled headline "
            f"series; SR0 from N={DSR_N_TRIALS} enumerated trials; var_sr over the 180 "
            "de-annualized VAL IRs (ddof=0)",
            "dsr": dsr,
            "threshold": DSR_THRESHOLD,
            "n_trials": DSR_N_TRIALS,
            "var_sr": var_sr,
            "sr0": sr0,
            "pass": bool(dsr >= DSR_THRESHOLD),
        },
    }
    failing = [name for name, c in clauses.items() if not c["pass"]]
    primary = SURVIVES if not failing else NULL

    # §5 NULL-fallback: evaluated ONLY when the primary is NULL (its precondition)
    fallback = None
    if primary == NULL:
        pb21 = paired_delta_ir_bootstrap(s[2], s[1], h=PRIMARY_H)
        dsr2 = float(deflated_sharpe_ratio(s[2], n_trials=DSR_N_TRIALS, var_sr=var_sr))
        fb_clauses = {
            "ci": {"ci_lower": pb21.ci_lower, "pass": pb21.passes_ci},
            "mde": {
                "delta_ir": pb21.delta_ir,
                "mde_paired": pb21.mde_paired,
                "pass": pb21.passes_mde,
            },
            "floor": {
                "delta_ir": pb21.delta_ir,
                "floor": ECON_FLOOR_IR,
                "pass": bool(pb21.delta_ir >= ECON_FLOOR_IR),
            },
            "dsr": {
                "dsr": dsr2,
                "threshold": DSR_THRESHOLD,
                "statistic": "cell2_seed_mean_headline_series",
                "pass": bool(dsr2 >= DSR_THRESHOLD),
            },
        }
        fb_word = (
            FALLBACK_CLAIMED
            if all(c["pass"] for c in fb_clauses.values())
            else FALLBACK_DESCRIPTIVE
        )
        fallback = {
            "rule": "§5: IR(2)-IR(1) claimable ONLY under the identical paired rule "
            "(CI > 0, ≥ its own MDE_paired, ≥ the 0.5 floor, DSR ≥ 0.95, same N=180 budget)",
            "bootstrap_2_minus_1": _pb_dict(pb21),
            "clauses": fb_clauses,
            "word": fb_word,
        }

    # §7 v1.4.4/v1.4.7 GUARDS — POST-HOC, HALT-only, computed AFTER the clauses/primary so they
    # can never mutate them. Either firing supersedes the EMITTED verdict with HALT_ADJUDICATE;
    # neither can flip SURVIVES<->NULL. The degeneracy guard asks "is this cell a constant book?";
    # the power guard asks "is this claim bigger than its own inputs' seed-to-seed wobble?".
    guard = degeneracy_guard(evals)
    power = power_guard(
        evals,
        {
            "clause1_delta_4_minus_5": {"delta_ir": pb45.delta_ir, "cells": (4, 5)},
            "clause3_delta_4_minus_2": {"delta_ir": pb42.delta_ir, "cells": (4, 2)},
        },
    )
    emitted = HALT_ADJUDICATE if (guard["halted"] or power["halted"]) else primary

    grid = next(iter(evals.values()))["grid"]
    manifest = {
        "schema": VERDICT_SCHEMA,
        "prereg": "docs/m6_prereg.md §3 (five conjunctive clauses), §3a (pinned surface), §5",
        "primary_h": PRIMARY_H,
        "grid": dict(grid),
        "artifact_sha256": dict(artifact_sha256),
        "cells": {
            str(c): {
                "ir_seed_mean_h15": ir[c],
                "per_seed_ir_h15": {
                    str(sd): float(
                        information_ratio(
                            np.asarray(evals[(c, sd)]["headline_series"], np.float64), PRIMARY_H
                        )
                    )
                    for sd in DSR_SEEDS
                },
                "kappa_star_by_h": evals[(c, DSR_SEEDS[0])]["kappa_star_by_h"],
            }
            for c in (1, 2, 3, 4, 5)
        },
        "bootstrap": {
            "delta_4_minus_5": _pb_dict(pb45),
            "delta_4_minus_2": _pb_dict(pb42),
            "delta_5_minus_2": _pb_dict(pb52),
        },
        "clauses": clauses,
        "degeneracy_guard": guard,
        "power_guard": power,
        "verdict": {
            # `emitted` is the FINAL verdict: HALT_ADJUDICATE if the guard fired, else the
            # clause-derived `primary`. `primary` is retained un-altered (the guard is HALT-only —
            # it never changes SURVIVES<->NULL, only supersedes with HALT).
            "emitted": emitted,
            "primary": primary,
            "halted_for_degeneracy": guard["halted"],
            "halted_for_power": power["halted"],
            "failing_clauses": failing,
            "fallback": fallback,
            "double_null": bool(
                primary == NULL and (fallback or {}).get("word") != FALLBACK_CLAIMED
            ),
        },
        "point_diagnostics": {k: v for k, v in ablation_verdict(ir).items() if k != "cell_ir"},
    }
    manifest["content_hash"] = content_hash(manifest)
    return manifest


__all__ = [
    "ARTIFACT_SCHEMA",
    "DSR_HORIZONS",
    "DSR_KAPPAS",
    "DSR_N_TRIALS",
    "DSR_SEEDS",
    "DSR_THRESHOLD",
    "ECON_FLOOR_IR",
    "FALLBACK_CLAIMED",
    "FALLBACK_DESCRIPTIVE",
    "FRAC_NEG_BAND",
    "HALT_ADJUDICATE",
    "INDEX_NAME",
    "INDEX_SCHEMA",
    "NULL",
    "PRIMARY_H",
    "SURVIVES",
    "VERDICT_SCHEMA",
    "VerdictInputError",
    "assemble_verdict",
    "degeneracy_guard",
    "enumerate_dsr_trials",
    "load_cell_evals",
    "power_guard",
    "write_cell_eval_artifact",
    "write_eval_index",
]
