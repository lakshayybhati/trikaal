"""The M6 verdict assembler — prereg §3's five conjunctive clauses, evaluated EXACTLY once.

This module + ``scripts/m6_verdict.py`` are **the ONLY code path allowed to produce the
SURVIVES/NULL decision**. Everything else that looks decision-shaped is diagnostics:
``xsection.ablation_verdict`` emits point comparisons (no CI, no clauses, no final word) and
``harness.run_harness`` is the M5 machine-validation instrument — neither may ever be quoted as
the M6 outcome (docs/m6_prereg.md §3; the third pre-training audit's finding #1).

The pipeline (docs/m6_prereg.md §3, §3a, §5 — every constant pinned there):

1. the 25 per-(cell, seed) eval artifacts (5 cells × seeds {0,1,2,3,4}) are loaded **by content
   hash** — each file's sha256 must match the driver-written index; any mismatch is a hard stop;
2. per cell, the **seed-mean pooled calendar series** on the §3a money grid (full calendar,
   flat periods = 0.0 — xsection's convention keeps every cell time-aligned);
3. paired moving-block bootstraps (§3a pinned recipe, ``paired_bootstrap``) on ΔIR(4−5)
   [primary], ΔIR(4−2) [clause 3], ΔIR(5−2) [placebo-health diagnostic];
4. the five clauses, conjunctive: (1) CI lower bound of ΔIR(4−5) > 0; (2) ΔIR ≥ MDE_paired
   (no-ceiling: the operative threshold is ALWAYS the realized MDE_paired, in either direction
   vs the §2 tabled value — both are recorded); (3) CI lower bound of ΔIR(4−2) > 0, with the
   (5−2) ci_upper < 0 "shuffle harmed" disclosure flag (never gating); (4) ΔIR ≥ 0.5 annualized
   (the economic floor); (5) DSR ≥ 0.95 with SR₀ from **N = 60 enumerated trials**
   (5 cells × horizons {5,15,60} × 4 κ — SEEDS EXCLUDED, they are replicates not configurations,
   §7 v1.5 A.5; every VAL entry persisted in the artifacts, the trial set is enumerated over the
   FULL cross-product INCLUDING seeds as the audit trail and cross-product-asserted, never
   assumed), var_sr = the ddof=0 variance of the CELL-5 (placebo) trial values — NOT all arms
   (§7 v1.5 A.4) — statistic + higher moments = Cell 4's seed-mean series;
5. on NULL, §5's fallback: IR(2)−IR(1) is CLAIMED only if it passes the IDENTICAL paired rule
   (CI > 0, ≥ its own MDE_paired, ≥ 0.5, DSR ≥ 0.95 on Cell 2's series under the same N=60
   budget), else DESCRIPTIVE_ONLY — double-NULL is a stated possible outcome.

The manifest is durable and content-hashed; the final word exists nowhere else.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from trikaal.eval.conformance import (
    PINNED_CODEBOOK_MIN_UTILIZATION,
    PINNED_DSR,
    ConformanceError,
    verdict_dsr_failures,
)
from trikaal.eval.dsr import deflated_sharpe_ratio, expected_max_sharpe
from trikaal.eval.metrics import information_ratio, periods_per_year
from trikaal.eval.paired_bootstrap import PairedBootstrap, paired_delta_ir_bootstrap
from trikaal.eval.tdist import student_t_ppf, welch_satterthwaite_df
from trikaal.eval.xsection import ablation_verdict
from trikaal.train.arms import ARM_MICRO, ARM_OHLCV, arm_n_features
from trikaal.train.cells import CELLS
from trikaal.utils.hashing import content_hash
from trikaal.utils.provenance import PROVENANCE_IDENTITY_KEYS, run_provenance

# ---- prereg pins (literals mirror docs/m6_prereg.md §3/§3a/§5) -------------------------------
PRIMARY_H = 15  # §3: h=15 is the primary; no horizon-shopping
DSR_HORIZONS: tuple[int, ...] = (5, 15, 60)  # §3 clause 5: h=1 is not evaluated anywhere in M6
DSR_KAPPAS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)  # §3a κ grid
DSR_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)  # §3a seeds (§7 v1.5: S=5 up front)
DSR_N_TRIALS = 60  # §3 clause 5 (§7 v1.5 A.5): 5 cells × 3 horizons × 4 κ — SEEDS EXCLUDED
DSR_VAR_SR_BASIS_CELL = 5  # §7 v1.5 A.4: the PLACEBO arm — a null dispersion estimate
# must not contain the treatment contrast (the old all-arms basis made clause 5 HARDER
# exactly when the ablation succeeded — anti-correlated with its own hypothesis).
PLACEBO_DISPERSION_TRIPWIRE = 1.5  # §7 v1.5 A.4.3: x median dispersion of cells 1,2,3
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

ARTIFACT_SCHEMA = "m6_cell_eval_v2"
# §7 v1.6 C-2: the fields ``degeneracy_guard`` actually READS. The v1 contract made ``mu_diag``
# optional, so an artifact could omit it, pass validation, and leave the guard reporting
# ``armed: true`` having examined nothing — the schema bump is what makes a v1 artifact (written
# before the requirement existed) fail loudly instead of resuming into a blind guard.
MU_DIAG_REQUIRED_KEYS = ("frac_negative", "activity_decisions")
INDEX_SCHEMA = "m6_eval_index_v1"
INDEX_NAME = "index.json"
VERDICT_SCHEMA = "m6_verdict_v1"

_CELL_BY_ID = {c.cell_id: c for c in CELLS}


class VerdictInputError(AssertionError):
    """The eval-artifact set is malformed, incomplete, or fails its content-hash check."""


# ------------------------------------------------------------------ artifact emission (shared)
def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mu_diag_problems(mu_diag: object, where: str) -> list[str]:
    """Everything wrong with a ``mu_diag`` payload (empty list = usable by the guard).

    §7 v1.6 C-2. ``degeneracy_guard`` is HALT-only and reads exactly two scalars; if either is
    absent it compares against NaN, every comparison is False, and the guard passes silently. So
    the contract is enforced HERE, at the single emission path, rather than discovered as a
    non-event downstream."""
    if not isinstance(mu_diag, dict) or not mu_diag:
        return [f"{where}: mu_diag is missing or empty — degeneracy_guard would examine nothing"]
    bad: list[str] = []
    for k in MU_DIAG_REQUIRED_KEYS:
        if k not in mu_diag:
            bad.append(f"{where}: mu_diag lacks {k!r} — the guard's {k} leg would be a no-op")
            continue
        try:
            v = float(mu_diag[k])
        except (TypeError, ValueError):
            bad.append(f"{where}: mu_diag[{k!r}] = {mu_diag[k]!r} is not a float")
            continue
        if not math.isfinite(v):
            bad.append(f"{where}: mu_diag[{k!r}] = {v} is not finite — NaN disarms the guard")
    return bad


def write_cell_eval_artifact(
    out_dir: Path,
    *,
    cell_id: int,
    seed: int,
    grid: dict,
    headline_series: np.ndarray,
    kappa_star_by_h: dict[int, float],
    val_ir_by_kappa_by_h: dict[int, dict[float, float]],
    mu_diag: dict,
    ohlcv_recon: dict,
    decode_agreement: dict,
    codebook: dict,
    meta: dict | None = None,
) -> tuple[str, str]:
    """Write ONE per-(cell, seed) eval artifact; returns ``(name, sha256)``.

    This is the single emission path — the real eval driver AND the KAT fixture builders both
    write through here, so the loader can never drift from the writer. ``headline_series`` is
    the pooled FULL-calendar money-grid series at κ* netted at the flat 0.30 % (h = PRIMARY_H);
    ``val_ir_by_kappa_by_h`` carries the ANNUALIZED VAL IR for every (h, κ) — the DSR trial
    entries §3 clause 5 requires persisted.

    ``mu_diag`` is REQUIRED (§7 v1.6 C-2) and must carry finite ``frac_negative`` and
    ``activity_decisions``: they are the only inputs ``degeneracy_guard`` reads, so an artifact
    without them turns a binding HALT gate into a no-op that still reports itself armed."""
    bad = _mu_diag_problems(mu_diag, f"cell{cell_id}_seed{seed}")
    if not isinstance(decode_agreement, dict) or "sign_agreement_dim0" not in decode_agreement:
        bad.append(
            f"cell{cell_id}_seed{seed}: decode_agreement missing 'sign_agreement_dim0' — the C-1 "
            "handicap disclosure is REQUIRED beside the headline"
        )
    if not isinstance(ohlcv_recon, dict) or "ohlcv_recon_mae" not in ohlcv_recon:
        bad.append(
            f"cell{cell_id}_seed{seed}: ohlcv_recon missing 'ohlcv_recon_mae' — the C-12 capacity "
            "disclosure is REQUIRED beside the headline, not an appendix"
        )
    if bad:
        raise VerdictInputError(
            "refusing to write an eval artifact the degeneracy guard could not read:\n  - "
            + "\n  - ".join(bad)
        )
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
        "codebook": codebook,
        # §7 v1.4.2 standing μ̂ receipt: per-cell mean/std/frac_negative of the decision signal
        # (the mean estimator; a constant-sign μ̂ was the acceptance mode-bias pathology).
        # REQUIRED since v1.6 C-2 — validated above, never defaulted to {}.
        "mu_diag": dict(mu_diag),
        # §7 v1.6 C-12 M1 (supervisor-adopted): the placebo CAPACITY disclosure. REQUIRED, for the
        # C-2 reason — a disclosure that may be absent is a disclosure that will be absent.
        "ohlcv_recon": dict(ohlcv_recon),
        # §7 v1.6.11 C-1: the SECOND handicap channel. REQUIRED for the same C-2 reason.
        "decode_agreement": dict(decode_agreement),
        # §7 v1.6 C-5 A7: every artifact is attributable. Auto-filled when the caller did not
        # supply one, so an unattributable unit cannot exist rather than being merely
        # discouraged — the C-2 lesson applied to provenance.
        "meta": {"provenance": run_provenance(), **(meta or {})},
    }
    name = f"cell{cell_id}_seed{seed}_eval.json"
    path = Path(out_dir) / name
    path.write_text(json.dumps(doc, sort_keys=True))
    return name, _file_sha256(path)


def write_eval_index(out_dir: Path, entries: dict[str, str]) -> Path:
    """Write the content-hash index over the 25 artifacts: ``{name: sha256}``."""
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
    # §7 v1.6 C-2: the guard's inputs are part of the contract, not an optional extra.
    bad.extend(_mu_diag_problems(doc.get("mu_diag"), name))
    rec = doc.get("ohlcv_recon")
    if not isinstance(rec, dict) or not math.isfinite(float(rec.get("ohlcv_recon_mae", "nan"))):
        bad.append(f"{name}: ohlcv_recon.ohlcv_recon_mae missing or non-finite (§7 v1.6 C-12 M1)")
    da = doc.get("decode_agreement")
    if not isinstance(da, dict) or not math.isfinite(float(da.get("sign_agreement_dim0", "nan"))):
        bad.append(f"{name}: decode_agreement.sign_agreement_dim0 missing/non-finite (§7 v1.6.11)")
    # §7 v1.6 C-5 A9: a DRY-RUN artifact is structurally un-quotable, not merely labelled. The
    # dry run declares money_run=False and therefore scores a deliberately SHORTENED grid, so its
    # numbers are wiring evidence and nothing else. Labelling alone would leave "someone reads the
    # manifest carefully" as the only thing between a path proof and a reported result.
    meta = doc.get("meta") or {}
    if meta.get("dry_run") or meta.get("grid_pinned") is False or meta.get("money_run") is False:
        bad.append(
            f"{name}: DRY-RUN artifact (money_run={meta.get('money_run')!r}, "
            f"grid_pinned={meta.get('grid_pinned')!r}) — produced on a shortened grid to prove "
            "the path, and can never be assembled into a verdict"
        )
    # §7 v1.6.22: the codebook diagnostic is REQUIRED per-(cell, seed) and its utilization is
    # gated at the spec's own >= 95% (design :1859). It was optional and unvalidated -- and we are
    # about to lean on it for the BSQ disclosure, so a disclosure with no data behind it is the
    # defect this project has closed four times.
    # §7 v1.6.25 (RE-AUDIT R1) — THIS BLOCK WAS BORN DEAD. It shipped in 591ec44 BELOW the
    # function's `return bad`, so from the moment the "fourth instance of specified-not-enforced"
    # was declared closed, its fix had never executed once. The compensating control for dropping
    # the C-4 external gate was unreachable code. Nothing above this line changed; the `return`
    # moved below it, and the negative tests that could not have existed before now exist.
    cb = doc.get("codebook")
    if not isinstance(cb, dict) or not cb:
        bad.append(f"{name}: codebook diagnostic is missing or empty (§7 v1.6.22; spec :1859)")
    else:
        for sub in ("coarse", "fine"):
            leg = cb.get(sub)
            if not isinstance(leg, dict):
                bad.append(f"{name}: codebook.{sub} missing (§7 v1.6.22)")
                continue
            u = leg.get("utilization")
            if not isinstance(u, (int, float)) or not np.isfinite(float(u)):
                bad.append(f"{name}: codebook.{sub}.utilization missing/non-finite")
            elif float(u) < PINNED_CODEBOOK_MIN_UTILIZATION:
                bad.append(
                    f"{name}: codebook.{sub}.utilization {float(u):.4f} < the pinned "
                    f"{PINNED_CODEBOOK_MIN_UTILIZATION} (dead-code collapse; spec :1859). This "
                    "catches COLLAPSE only — it does not certify the quantizer is competitive."
                )
    return bad


def placebo_capacity_disclosure(evals: dict[tuple[int, int], dict]) -> dict:
    """§7 v1.6 C-12 M1 (supervisor-adopted) — the placebo's CAPACITY handicap, measured.

    THE CONFOUND THIS BOUNDS. The Cell-5 permutation destroys the contemporaneous micro↔OHLCV
    dependence as well as the intended temporal alignment (**measured 0.2037 → 0.0005**), so
    Cell 5's micro channels are six dims of INDEPENDENT NOISE rather than "Cell 4 minus
    information". Noise with a preserved marginal is incompressible and shares no structure with
    OHLCV, so bits spent on it are bits taken from OHLCV — and ``micro_point_weight = 3.0`` aims
    triple gradient pressure at fitting it. ΔIR(4−5) therefore contains (information) + (capacity
    handicap), inseparable from the contrast alone. It INFLATES rather than manufactures: Cell 4
    must still clear the MDE and the 0.5 economic floor on its own.

    WHY THE COMPARISON IS LIKE-FOR-LIKE. ``block_time_permute`` never touches the OHLCV columns —
    they come back byte-identical — so both arms reconstruct the SAME OHLCV targets. Cell 5 doing
    it worse is capacity diverted.

    REQUIRED IN THE RESULTS, NOT AN APPENDIX, and NON-GATING: it is emitted beside the headline,
    is never a clause, has no threshold, and cannot flip SURVIVES↔NULL.
    """

    def mae(cell: int) -> float:
        vals = [
            float(evals[(cell, s)]["ohlcv_recon"]["ohlcv_recon_mae"])
            for s in DSR_SEEDS
            if (cell, s) in evals
        ]
        return float(np.mean(vals)) if vals else float("nan")

    m4, m5, m2 = mae(4), mae(5), mae(2)
    ratio = (m5 / m4) if (m4 and math.isfinite(m4) and m4 > 0) else float("nan")
    return {
        "ohlcv_recon_mae_cell4": m4,
        "ohlcv_recon_mae_cell5_placebo": m5,
        "ohlcv_recon_mae_cell2_ohlcv_only": m2,
        "cell5_over_cell4_ratio": ratio,
        "excess_ohlcv_recon_error_of_the_placebo": (m5 - m4),
        "reading": (
            "ratio > 1 means the PLACEBO reconstructs the SAME OHLCV targets worse than the "
            "treatment, i.e. capacity was diverted to encoding six dims of independent noise "
            "under a 3x micro weight — the C-12 handicap, measured rather than bounded by "
            "assumption. ratio ~ 1 means the handicap is small and dIR(4-5) is correspondingly "
            "cleaner."
        ),
        "non_gating": True,
        "status": "REQUIRED DISCLOSURE (§7 v1.6 C-12 M1) — reported beside the headline",
    }


def decode_agreement_disclosure(evals: dict[tuple[int, int], dict]) -> dict:
    """§7 v1.6.11 C-1 — the DECODE-NOISE handicap channel, measured on the real cells.

    THE SECOND HANDICAP CHANNEL, after ``placebo_capacity_disclosure``. Both would inflate
    ΔIR(4−5) in the SAME direction if real, and they compound: C-12's capacity handicap makes the
    placebo spend bits on incompressible noise, and a decode-agreement gap makes the placebo's μ̂
    noisier — each degrading cell 5 for a reason unrelated to microstructure information.

    Measured here rather than on a toy because the toy came back INDETERMINATE at n=3 (t = 1.821,
    Welch df 2.385, crit 2.6226) and resolving it there needs n ≈ 13 per arm on briefly-trained
    tokenizers that may not transfer. The real cells exist and 5 seeds are already paid for.

    DEGENERACY TRAVELS WITH THE SCORE (§7 v1.6.10): a collapsed single-bar decode scores a perfect
    agreement for free, so any degenerate (cell, seed) is named and the comparison refuses.

    NON-GATING: no clause, no threshold, cannot flip SURVIVES↔NULL.
    """

    # §7 v1.6.15 (fail-open sweep): a unit whose decode_agreement is absent/empty/None/non-dict was
    # SILENTLY DROPPED by the `if "sign_agreement_dim0" in da` filter, so a disclosure computed
    # from nothing was indistinguishable from one computed from five seeds. `_validate_artifact`
    # refuses such artifacts at load, so this was behind a closed door — which is exactly how C-2
    # read before the rehearsal path bypassed it. Named, counted, and surfaced instead.
    unreadable: list[str] = []

    def per_cell(cell: int) -> dict:
        vals, degen = [], []
        for s in DSR_SEEDS:
            da = (evals.get((cell, s)) or {}).get("decode_agreement")
            if not isinstance(da, dict) or "sign_agreement_dim0" not in da:
                unreadable.append(f"cell{cell}_seed{s}:decode_agreement")
                continue
            val = float(da["sign_agreement_dim0"])
            if not np.isfinite(val):
                unreadable.append(f"cell{cell}_seed{s}:sign_agreement_dim0={val!r}")
                continue
            vals.append(val)
            if da.get("single_bar_degenerate"):
                degen.append(s)
        return {
            "sign_agreement_mean": float(np.mean(vals)) if vals else float("nan"),
            "sign_agreement_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            "sign_agreement_by_seed": vals,
            "degenerate_seeds": degen,
        }

    cells = {c: per_cell(c) for c in (1, 2, 3, 4, 5)}
    a, b = cells[4], cells[5]
    n = min(len(a["sign_agreement_by_seed"]), len(b["sign_agreement_by_seed"]))
    degen = bool(a["degenerate_seeds"] or b["degenerate_seeds"])
    # DEGENERACY IS REPORTED FIRST AND ALWAYS (§7 v1.6.10). An earlier draft only set it inside
    # the se>0 branch, so a collapsed arm with zero across-seed spread fell through to
    # "INSUFFICIENT REPLICATES" and the flag was LOST — the degeneracy check has to survive the
    # very case it exists to catch.
    test: dict = {
        "reading": "INSUFFICIENT REPLICATES",
        "any_degenerate": degen,
        "significant": False,
    }
    if degen:
        test["reading"] = (
            "INDETERMINATE — a degenerate single-bar decode makes the agreement an artifact "
            "(§7 v1.6.10); no claim is made"
        )
    if n >= 2 and a["sign_agreement_sd"] == a["sign_agreement_sd"] and not degen:
        diff = abs(a["sign_agreement_mean"] - b["sign_agreement_mean"])
        se = float(np.sqrt(a["sign_agreement_sd"] ** 2 / n + b["sign_agreement_sd"] ** 2 / n))
        if se > 0:
            t = diff / se
            df = welch_satterthwaite_df(
                a["sign_agreement_sd"] / np.sqrt(n),
                n - 1,
                b["sign_agreement_sd"] / np.sqrt(n),
                n - 1,
            )
            crit = student_t_ppf(0.95, df)
            test = {
                "abs_diff_cell4_minus_cell5": diff,
                "se_of_diff": se,
                "t": float(t),
                "welch_df": float(df),
                "t_crit_0p95": float(crit),
                "significant": bool(t > crit),
                "any_degenerate": degen,
                "reading": (
                    "DISTINGUISHABLE — cells 4 and 5 differ in decode agreement by more than "
                    "the SE of their difference; the handicap channel is real at this n"
                    if t > crit
                    else "INDETERMINATE — not distinguishable from zero at this n; neither "
                    "'symmetric' nor 'asymmetric' is supported"
                ),
            }
    if unreadable:
        test = {
            "reading": (
                "UNMEASURABLE — one or more units carried no readable decode_agreement, so this "
                "disclosure examined less than the full design; no claim is made either way"
            ),
            "any_degenerate": degen,
            "significant": False,
        }
    return {
        "per_cell": {str(k): v for k, v in cells.items()},
        "primary_pair_test_cell4_vs_cell5": test,
        # §7 v1.6.15: fails CLOSED. A disclosure that measured nothing must not be readable as a
        # null result — assemble_verdict raises it to HALT_ADJUDICATE.
        "unmeasurable": bool(unreadable),
        "unreadable_inputs": sorted(unreadable),
        "why_a_two_sample_test": (
            "comparing two arms' means requires the SE of their DIFFERENCE, not one arm's spread. "
            "The spread form is the power guard's REFUSAL rule and does not answer an INFERENCE "
            "question; applied here on the toy it read 1.10 -> 'asymmetric' where the correct "
            "statistic read t = 1.82 vs crit 2.62 -> indeterminate."
        ),
        "non_gating": True,
        # §7 v1.6.17 (audit S-3): C-1 and the expectation-estimator approximation are NOT
        # INDEPENDENT. `predict_mu`'s expectation decode is a plug-in — decode(E[z]) != E[decode(z)]
        # for a nonlinear decoder, and p_f is conditioned on the argmax coarse rather than
        # marginalized (disclosed at predict.py:8-22, adjudicated §7 v1.4.7 item 4). That
        # delta-method error is evaluated in the SAME out-of-distribution single-bar regime this
        # disclosure measures, so the two channels compound rather than sit side by side.
        "not_independent_of_the_expectation_estimator": (
            "the expectation decode is a plug-in (decode(E[z]) != E[decode(z)]; p_f conditioned on "
            "the argmax coarse, not marginalized) and its delta-method error is evaluated in the "
            "SAME out-of-distribution single-bar regime measured here — the two are not "
            "independent sources of decode noise (§7 v1.4.7 item 4, audit S-3)"
        ),
        "status": "REQUIRED DISCLOSURE (§7 v1.6.11) — the second handicap channel",
    }


def provenance_failures(evals: dict[tuple[int, int], dict]) -> list[str]:
    """Refuse to assemble a verdict across mixed hardware (§7 v1.6 C-5 A5).

    Fan-out makes this reachable: with ``--shard i/N`` the 25 units are computed on N boxes, and
    the hard constraint is that every shard shares provenance — same GPU model, same environment,
    same INTERPRETER. The 25 units are ONE paired comparison; a cell trained and scored on a
    different instrument is a different experiment, and ΔIR(4−5) across two instruments measures
    the instruments as much as the tokenizer.

    Artifacts written before the stamp existed carry no ``meta.provenance``; those are reported as
    a divergence rather than skipped, because "some units are unattributable" is exactly the state
    this refuses to certify. Empty list = one instrument."""
    # §7 v1.6.16 (vacuity sweep): ZERO units is not "one instrument", it is NO EVIDENCE — and the
    # comprehension below would certify it clean, because a divergence check over an empty set
    # finds no divergence. Same shape as "an empty matrix is a perfect cover".
    if not evals:
        return [
            "provenance gate received ZERO units — a divergence check over an empty set reports "
            "no divergence, which is not the same as one instrument"
        ]
    by_key: dict[str, dict[str, list[str]]] = {}
    missing: list[str] = []
    for (cell, seed), doc in sorted(evals.items()):
        prov = (doc.get("meta") or {}).get("provenance")
        unit = f"cell{cell}_seed{seed}"
        if not isinstance(prov, dict) or not prov:
            missing.append(unit)
            continue
        for key in PROVENANCE_IDENTITY_KEYS:
            by_key.setdefault(key, {}).setdefault(repr(prov.get(key)), []).append(unit)
    fails = []
    if missing:
        fails.append(
            f"{len(missing)} artifact(s) carry no meta.provenance ({missing[:3]}...) — a unit "
            "that cannot be attributed to an instrument cannot be certified as sharing one"
        )
    for key, groups in sorted(by_key.items()):
        if len(groups) > 1:
            detail = "; ".join(f"{v} on {sorted(u)[:2]}" for v, u in sorted(groups.items()))
            fails.append(
                f"artifacts disagree on provenance.{key}: {detail} — the verdict REFUSES to "
                "assemble across mixed hardware/environments (§7 v1.6 C-5 A5)"
            )
    return fails


def load_cell_evals(art_dir: Path) -> tuple[dict[tuple[int, int], dict], dict[str, str]]:
    """Load the 25 artifacts **by content hash** → ``({(cell_id, seed): doc}, {name: sha256})``.

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
    problems += provenance_failures(evals)
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

    Returns the full cells × seeds × horizons × κ cross product keyed by (cell_id, seed, h, κ) —
    at the v1.5 pins that is 5 × 5 × 3 × 4 = 300 enumerated values, of which the cell-5 subset
    (5 × 3 × 4 = 60) is the ``var_sr`` basis and ``DSR_N_TRIALS`` = 60 counts CONFIGURATIONS
    (seeds excluded as replicates). The caller asserts the enumeration against the pinned recipe
    before any DSR is computed.

    **§7 v1.6.13, DISCLOSED:** this docstring read "exactly the 5 × 3 × 3 × 4 = 180 cross product"
    until the ruling-(a) sweep. That is the pre-v1.5 3-seed count, and it survived the C-17 pass
    that corrected this very file's module docstring — the fix was scoped to the strings the audit
    named instead of to the file.

    **AUDIT C-3 — AMENDED PRE-DATA (§7 v1.6.22), AND THIS DOCSTRING WAS STILL DESCRIBING THE
    PREDECESSOR (§7 v1.6.25 R8).** Every value is divided by ``sqrt(periods_per_year(PRIMARY_H))``,
    so every trial carries the unit of the statistic SR₀ is compared against — the Sharpe of the
    headline series at ``PRIMARY_H``. The superseded convention divided by
    ``sqrt(periods_per_year(h))``, mixing per-5/15/60-minute Sharpes across a √12 = 3.4641× span;
    it is REPORTED, not deleted, as the ``v1_5_mixed_unit_basis_superseded`` leg of
    ``dual_specification`` (see ``own_h_trial_values``). The paragraph above says a stale count
    survived the pass that was supposed to fix the file; **this paragraph said "REPORTED, NOT
    FIXED" and "this function is unchanged" for three versions after the function changed**, one
    line above the line that changed. See ``scripts/m6_c3_dsr_units.py`` and its receipt."""
    trials: dict[tuple[int, int, int, float], float] = {}
    for (cid, seed), doc in evals.items():
        for h in DSR_HORIZONS:
            by_k = doc["val_ir_by_kappa_by_h"][str(h)]
            for k in DSR_KAPPAS:
                ir_ann = float(by_k[f"{k:g}"])
                # §7 v1.6.22 (C-3 AMENDED, pre-data, Lakshay 2026-08-03): de-annualize at
                # PRIMARY_H, not at each trial's own horizon. SR0 is compared against
                # _sharpe(headline_series) whose grid is pinned to PRIMARY_H, so the trial set
                # must carry that same unit. The predecessor divided by sqrt(ppy(h)), mixing
                # per-5/15/60-minute Sharpes -- a sqrt(12) = 3.4641x span, outcome-material
                # (witness SR_hat 0.45705: DSR 0.9868 PASS mixed vs 0.8577 FAIL consistent).
                trials[(cid, seed, h, k)] = ir_ann / float(np.sqrt(periods_per_year(PRIMARY_H)))
    return trials


def own_h_trial_values(
    evals: dict[tuple[int, int], dict],
    keys: list[tuple[int, int, int, float]],
) -> np.ndarray:
    """The SAME trials under the **pre-v1.6.22 unit convention** — de-annualized at each
    trial's OWN horizon.

    This is the convention the C-3 amendment replaced, kept as an executable object rather than a
    description so both superseded specifications can be reported from the same artifacts.
    ``enumerate_dsr_trials`` divides by ``sqrt(ppy(PRIMARY_H))``; this divides by
    ``sqrt(ppy(h))``, which for h ∈ {5, 15, 60} is a √12 = 3.4641× span across the set.

    **§7 v1.6.25 (RE-AUDIT R3).** It existed inline for the v1.5-superseded leg only, so the
    ``v1_2_original`` leg — which is supposed to reconstruct the ORIGINAL spec — silently read the
    AMENDED units. It is a function now, and both superseded legs call it.
    """
    return np.array(
        [
            float(evals[(cid, seed)]["val_ir_by_kappa_by_h"][str(h)][f"{k:g}"])
            / float(np.sqrt(periods_per_year(h)))
            for cid, seed, h, k in keys
        ],
        dtype=np.float64,
    )


def dsr_unit_convention_failures(
    evals: dict[tuple[int, int], dict],
    trials: dict[tuple[int, int, int, float], float],
) -> list[str]:
    """RE-DERIVE every clause-5 trial value from the artifacts under the PINNED unit convention.

    **§7 v1.6.25 (RE-AUDIT R2) — the check the C-3 amendment's own signed package specified and
    which did not ship.** The amendment is one identifier wide: ``enumerate_dsr_trials`` divides by
    ``sqrt(periods_per_year(PRIMARY_H))``, and its predecessor divided by
    ``sqrt(periods_per_year(h))``. Reverting it passed all 601 tests, restored the EASIER
    mixed-unit basis, and would have moved clause 5 by a √12 = 3.4641× factor on the dispersion —
    outcome-material by the amendment's own witness (DSR 0.9868 PASS → 0.8577 FAIL).

    A pin that only a unit test reads is a pin the money run does not have, so this runs on the
    MONEY PATH: it recomputes ``ir_annualized / sqrt(ppy(pinned_h))`` for all 300 enumerated trials
    straight from ``val_ir_by_kappa_by_h`` and demands bit-equality. A revert therefore fails the
    run. Cost is 300 divisions — free, on a 270-GPU-hour experiment.
    """
    fails: list[str] = []
    pinned_h = int(PINNED_DSR["de_annualization_horizon"])
    if pinned_h != PRIMARY_H:
        fails.append(f"PINNED_DSR['de_annualization_horizon'] {pinned_h} != PRIMARY_H {PRIMARY_H}")
        return fails
    denom = float(np.sqrt(periods_per_year(pinned_h)))
    wrong_at_own_h = 0
    for (cid, seed, h, k), got in trials.items():
        ir_ann = float(evals[(cid, seed)]["val_ir_by_kappa_by_h"][str(h)][f"{k:g}"])
        want = ir_ann / denom
        if got != want:
            if got == ir_ann / float(np.sqrt(periods_per_year(h))):
                wrong_at_own_h += 1
            elif len(fails) < 3:
                fails.append(
                    f"trial (cell{cid}, seed{seed}, h={h}, κ={k:g}) = {got!r}, but the pinned "
                    f"convention re-derives {want!r} from the artifact"
                )
    if wrong_at_own_h:
        fails.append(
            f"{wrong_at_own_h} of {len(trials)} trials are de-annualized at their OWN horizon — "
            f"the PRE-v1.6.22 convention the C-3 amendment replaced. The trial set must carry the "
            f"unit of the statistic SR0 is compared against (h={pinned_h}); mixing h ∈ "
            f"{DSR_HORIZONS} spans sqrt(12) = 3.4641x and inflates var_sr"
        )
    return fails


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
    it only reports whether the verdict must HALT.

    §7 v1.6 C-2 — ``armed`` NOW MEANS IT EXAMINED DATA, AND THE GUARD FAILS CLOSED. The previous
    contract let ``mu_diag`` be absent: every read became NaN, ``nan == nan`` is False so every
    comparison was False, and the guard returned a hardcoded ``"armed": True`` with
    ``halted: False`` having inspected nothing. ``m6_toy_rehearsal.py`` never passed ``mu_diag``,
    so the end-to-end run that was to VALIDATE the guards would have produced a verdict word with
    this gate silently dead — the project's own "a check that cannot fail" pattern, sitting in the
    decision path. ``armed`` is now computed from whether every (cell, seed) yielded finite values,
    and a guard that cannot read its inputs HALTS rather than passing. Still HALT-only: it can
    never flip SURVIVES↔NULL.
    """
    lo, hi = FRAC_NEG_BAND
    unreadable: list[str] = []

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
        # §7 v1.6 C-2: record WHICH inputs were unreadable, so "armed" is a measurement.
        unreadable += [
            f"cell{c}_seed{s}:{k}"
            for s in DSR_SEEDS
            for k in MU_DIAG_REQUIRED_KEYS
            if not np.isfinite(float(evals[(c, s)].get("mu_diag", {}).get(k, float("nan"))))
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
        # §7 v1.6 C-2: a MEASUREMENT, not a literal. False iff any pinned input was unreadable.
        "armed": not unreadable,
        "unreadable_inputs": unreadable,
        "frac_neg_band": list(FRAC_NEG_BAND),
        "rule": "HALT-ONLY (§7 v1.4.6): a cell is a constant-direction book if frac_negative is "
        "outside the band OR decision-activity at kappa* is 0 or 1, evaluated on the seed-mean OR "
        "ON ANY INDIVIDUAL SEED (union). The per-seed leg closes the v1.4.4 hole where a triple "
        "like (0.999, 0.55, 0.55) averages to 0.700 and passes while one seed is fully locked. Any "
        "clause involving such a cell is uninterpretable, so the verdict is HALT_ADJUDICATE. The "
        "guard NEVER flips SURVIVES<->NULL — the clauses/primary below are computed identically.",
        "per_cell": per_cell,
        "degenerate_cells": degenerate,
        # FAIL-CLOSED (§7 v1.6 C-2): a guard that could not read its own inputs HALTS. Previously
        # an unreadable input produced halted=False, which is the failure mode this closes.
        "halted": bool(degenerate) or bool(unreadable),
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
    unreadable: list[str] = []
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
        # §7 v1.6.15 S-2: record WHICH inputs were unreadable, so "armed" is a MEASUREMENT.
        unreadable += [
            f"cell{c}_seed{s}:headline_ir"
            for s, v in zip(DSR_SEEDS, irs, strict=True)
            if not np.isfinite(v)
        ]
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
        # §7 v1.6.15 S-2: a non-finite CLAIM is as unreadable as non-finite data. Previously
        # `np.isfinite(delta)` False made `trips` False — a quiet pass on an unevaluable claim.
        if not np.isfinite(delta) or worst is None:
            unreadable.append(f"claim[{name}]:delta_ir={delta!r},worst_range={worst!r}")
        checks[name] = {
            "delta_ir_claimed": delta,
            "cells": cells,
            "worst_within_cell_ir_range": worst,
            "seed_spread_exceeds_claim": trips,
        }
        if trips:
            tripped.append(name)
    return {
        # §7 v1.6.15 S-2 — THE C-2 DEFECT'S UNSWEPT SIBLING. This was a hardcoded `True`: with
        # non-finite per-seed IRs `finite` is empty, every range is None, `worst` is None, `trips`
        # is False, and the guard returned armed/not-halted having evaluated nothing. C-2 was
        # fixed at `degeneracy_guard` and nothing swept the siblings — the ninth standing norm
        # (fix the class, not the instance) violated by the pass that wrote it.
        "armed": not unreadable,
        "unreadable_inputs": sorted(unreadable),
        "rule": "HALT-ONLY (§7 v1.4.7): if the WITHIN-cell across-seed IR range of the cells a "
        "claimed between-cell dIR is built from meets or exceeds |dIR|, the claim is smaller than "
        "the seed-to-seed wobble of its own inputs and the verdict is HALT_ADJUDICATE. The tabled "
        "MDE contains SCORING noise only (paired moving-block bootstrap over time) and no "
        "training-variance term, so it cannot speak to this. PURE READ — never alters a clause, "
        "never flips SURVIVES<->NULL; it only refuses to report an unsupported outcome.",
        "per_cell": per_cell,
        "checks": checks,
        "underpowered_claims": tripped,
        # §7 v1.6.15 S-2: fails CLOSED. A guard that could not read its inputs HALTs rather than
        # reporting a quiet pass. Still HALT-only — it can never flip SURVIVES<->NULL.
        "halted": bool(tripped) or bool(unreadable),
    }


def assemble_verdict(
    evals: dict[tuple[int, int], dict],
    artifact_sha256: dict[str, str],
    *,
    tabled_mde_h15: float,
    external_validation: dict | None = None,
    money_verdict: bool = True,
) -> dict:
    """Evaluate §3's five clauses on the 15 loaded artifacts → the verdict manifest body.

    Pure computation, no I/O: the driver (``scripts/m6_verdict.py``) supplies content-verified
    inputs and persists the returned manifest. The §3a bootstrap recipe is pinned inside
    ``paired_bootstrap`` (B=10,000, seed 20260704, ⌈√T⌉, percentile) — not parameterized here."""
    s = {c: _seed_mean_series(evals, c) for c in (1, 2, 3, 4, 5)}
    ir = {c: float(information_ratio(s[c], PRIMARY_H)) for c in s}

    # §7 v1.5 item B: the per-seed contrast supplies the across-seed TRAINING-VARIANCE term. It is
    # CLEAN — all seeds share the same eval data and grid, so the cross-seed spread is purely
    # model-induced with no scoring-noise component to subtract.
    def _per_seed_delta(ca: int, cb: int) -> np.ndarray:
        return np.array(
            [
                information_ratio(
                    np.asarray(evals[(ca, sd)]["headline_series"], np.float64), PRIMARY_H
                )
                - information_ratio(
                    np.asarray(evals[(cb, sd)]["headline_series"], np.float64), PRIMARY_H
                )
                for sd in DSR_SEEDS
            ],
            dtype=np.float64,
        )

    pb45 = paired_delta_ir_bootstrap(s[4], s[5], h=PRIMARY_H, per_seed_deltas=_per_seed_delta(4, 5))
    pb42 = paired_delta_ir_bootstrap(s[4], s[2], h=PRIMARY_H, per_seed_deltas=_per_seed_delta(4, 2))
    pb52 = paired_delta_ir_bootstrap(s[5], s[2], h=PRIMARY_H)  # diagnostic only, no MDE clause

    # clause 5: the pinned DSR recipe over the ENUMERATED trial set (N=60 multiplicity;
    # the ENUMERATION is the full cells x seeds x horizons x kappas cross-product). The recipe is
    # asserted against conformance.PINNED_DSR — an INDEPENDENT statement of the §3-clause-5
    # ---- G-§8.C.3 FIRES HERE: after Cell 1, BEFORE any between-cell Δ exists (audit C-4).
    # `money_verdict` defaults TRUE so the DANGEROUS direction is the one that must be declared —
    # the C-6 rule. A fixture announcing itself is safe; a real verdict silently skipping a binding
    # entry gate is the defect.
    #
    # §7 v1.6.25 (RE-AUDIT R4) — TWO DEFECTS, ONE SITE.
    # (a) The comment here (and at `scripts/m6_verdict.py`) said the gate "is BLOCKED today, so a
    #     money verdict HALTS here". That was TRUE when written and FALSE from v1.6.22 onward:
    #     Lakshay dropped the gate as binding, so `evaluate` returns DROPPED and
    #     `assert_clear_to_compute_deltas` returns cleanly. A comment describing a HALT that
    #     cannot happen is worse than none — it tells the next reader a control is live.
    # (b) The gate's verdict — including `required_disclosure`, the BSQ sentence that must reach
    #     the paper verbatim — was computed and thrown away. It reached no manifest, so the one
    #     compensating record for dropping a binding entry gate existed only as a transient. It is
    #     now evaluated UNCONDITIONALLY (pure, no I/O) and PERSISTED as a REQUIRED field; only the
    #     ENFORCEMENT is conditional on `money_verdict`.
    from trikaal.eval.external_validation import (
        BSQ_DISCLOSURE,
        GATE_IS_BINDING,
        ExternalReference,
        assert_clear_to_compute_deltas,
    )
    from trikaal.eval.external_validation import (
        evaluate as _eval_g8c3,
    )

    g8c3 = (
        external_validation
        if external_validation is not None
        else _eval_g8c3(None, ExternalReference())
    )
    if money_verdict:
        assert_clear_to_compute_deltas(g8c3)
    external_validation_record = {
        "gate": dict(g8c3),
        "gate_is_binding": bool(GATE_IS_BINDING),
        "enforced_this_run": bool(money_verdict),
        # The residual we owe the reader for dropping the gate. Persisted here so the manifest
        # cannot be quoted without it — `verdict_manifest_failures` rejects its absence.
        "required_disclosure": BSQ_DISCLOSURE,
    }

    # constants — before any DSR is computed; var_sr uses the key-sorted ddof=0 construction
    # the pin re-derives bit-exactly (a subset-variance or a ddof drift cannot pass).
    trials = enumerate_dsr_trials(evals)
    # §7 v1.6.25 R2: the C-3 unit convention, re-derived from the artifacts on the MONEY PATH.
    unit_fails = dsr_unit_convention_failures(evals, trials)
    if unit_fails:
        raise ConformanceError(
            "the clause-5 trial set is not on the pinned de-annualization basis:\n  - "
            + "\n  - ".join(unit_fails)
        )
    # §7 v1.5 A.4: dispersion basis = the CELL-5 placebo trials only (seeds retained inside it);
    # the ENUMERATION stays the full cross-product as the audit trail, and n_trials counts
    # configurations without seeds. Three deliberately different sets.
    basis_keys = sorted(k for k in trials if k[0] == DSR_VAR_SR_BASIS_CELL)
    var_sr = float(np.var(np.array([trials[k] for k in basis_keys], dtype=np.float64)))
    # §7 v1.6.22 C-3: the retained mixed-unit leg, from the SAME artifacts.
    var_sr_mixed = float(np.var(own_h_trial_values(evals, basis_keys)))
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
            # §7 v1.6 C-7: BUILT FROM THE PINS, never restated. The hardcoded predecessor said
            # "var_sr over the 180 de-annualized VAL IRs" — a description v1.5 superseded on BOTH
            # counts (N is 60, and the basis is the cell-5 placebo subset, not all arms), which
            # would have shipped inside the verdict manifest as a false account of how its own
            # var_sr was computed. A rule string that cannot disagree with the recipe is the fix.
            "rule": (
                f"DSR ≥ {DSR_THRESHOLD}; statistic = {PINNED_DSR['statistic']}; "
                f"SR0 from N={DSR_N_TRIALS} enumerated trials "
                f"({' x '.join(PINNED_DSR['n_trials_factorization'])}, seeds excluded as "
                f"replicates); var_sr = the ddof={PINNED_DSR['var_sr_ddof']} variance of the "
                f"CELL-{DSR_VAR_SR_BASIS_CELL} (placebo) VAL-IR trials, seeds retained"
            ),
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
            f"(CI > 0, >= its own MDE_paired, >= the {ECON_FLOOR_IR} floor, "
            f"DSR >= {DSR_THRESHOLD}, same N={DSR_N_TRIALS} budget)",
            "bootstrap_2_minus_1": _pb_dict(pb21),
            "clauses": fb_clauses,
            "word": fb_word,
        }

    # §7 v1.5 A.4.3 PLACEBO-DISPERSION TRIPWIRE. pb(5−2) is a LEVEL statement and var_sr is a
    # DISPERSION statistic, so placebo-victim behaviour does not move a variance at first order.
    # The second-order risk (a degraded training signal producing more variable models) errs
    # CONSERVATIVE — inflated var_sr raises SR0, making clause 5 HARDER — and is measured, not
    # assumed.
    def _arm_dispersion(c: int) -> float:
        irs = [
            float(
                information_ratio(
                    np.asarray(evals[(c, sd)]["headline_series"], np.float64), PRIMARY_H
                )
            )
            for sd in DSR_SEEDS
        ]
        fin = [v for v in irs if np.isfinite(v)]
        return (max(fin) - min(fin)) if fin else float("nan")

    disp = {str(c): _arm_dispersion(c) for c in (1, 2, 3, 4, 5)}
    ref = [disp[str(c)] for c in (1, 2, 3) if disp[str(c)] == disp[str(c)]]
    med = float(np.median(ref)) if ref else float("nan")
    placebo_disp = disp["5"]
    fired = bool(
        med == med
        and med > 0
        and placebo_disp == placebo_disp
        and placebo_disp > PLACEBO_DISPERSION_TRIPWIRE * med
    )
    placebo_tripwire = {
        "rule": (
            "§7 v1.5 A.4.3: DISCLOSE if the placebo arm's across-seed IR dispersion exceeds "
            f"{PLACEBO_DISPERSION_TRIPWIRE}x the median dispersion of cells 1,2,3 — a possible "
            "shuffle-degradation artifact in the var_sr basis. Threshold fixed PRE-DATA. "
            "Non-gating: it triggers DISCLOSURE and a dual-basis report, never a verdict change."
        ),
        "threshold": PLACEBO_DISPERSION_TRIPWIRE,
        "across_seed_ir_dispersion_by_cell": disp,
        "median_dispersion_cells_1_2_3": med,
        "placebo_dispersion": placebo_disp,
        "fired": fired,
        "action_if_fired": (
            "report clause 5 under BOTH the cell-5 and the within-cell-4 bases and disclose; "
            "NEVER fall back to cell 2 — §5's NULL-fallback can CLAIM IR(2)-IR(1), which makes "
            # §7 v1.6.13 (ruling (a) sweep): the two arm widths derive from the arm registry, so
            # the manifest cannot describe a width the cells do not have. Renders byte-identically.
            f"cell 2 a treatment arm, and {arm_n_features(ARM_OHLCV)} vs "
            f"{arm_n_features(ARM_MICRO)} dims is a dispersion mismatch"
        ),
    }

    # §7 v1.5 §0.3 DUAL-SPECIFICATION REPORT — the SAME artifacts under BOTH specifications.
    # The v1.2 leg reconstructs the pre-amendment rules: seeds COUNTED as trials, var_sr over ALL
    # arms, z-quantile MDE with scoring noise only, AND the pre-C-3 own-h de-annualization.
    #
    # §7 v1.6.25 (RE-AUDIT R3) — IT WAS A HYBRID CALLING ITSELF FAITHFUL. This leg read `trials`,
    # which the v1.6.22 C-3 amendment had re-based onto PRIMARY_H units, so what shipped was
    # {all-arms basis (v1.2)} x {PRIMARY_H units (v1.5-amended)} — a combination NO specification
    # ever had, presented as "the pre-amendment rules faithfully". The bias has a direction:
    # mixing per-5/15/60-minute Sharpes INFLATES the dispersion, so the faithful leg has a larger
    # var_sr, a larger SR0 and a LOWER DSR than the hybrid. The hybrid therefore agreed with the
    # v1.5 primary more readily than the real v1.2 spec would have — i.e. it under-reported
    # disagreement exactly when the primary says SURVIVES, which is the one direction a
    # dual-specification report exists to guard.
    n_trials_v12 = len(DSR_SEEDS) * len(_CELL_BY_ID) * len(DSR_HORIZONS) * len(DSR_KAPPAS)
    var_sr_v12 = float(np.var(own_h_trial_values(evals, sorted(trials))))
    dsr_v12 = float(deflated_sharpe_ratio(s[4], n_trials=n_trials_v12, var_sr=var_sr_v12))
    clauses_v12 = {
        "1_paired_ci": bool(pb45.passes_ci),
        "2_mde_paired": bool(pb45.delta_ir >= pb45.mde_paired_scoring_only),
        "3_placebo_validity": bool(pb42.passes_ci),
        "4_economic_floor": bool(pb45.delta_ir >= ECON_FLOOR_IR),
        "5_dsr": bool(dsr_v12 >= DSR_THRESHOLD),
    }
    primary_v12 = SURVIVES if all(clauses_v12.values()) else NULL
    dual_specification = {
        "rule": (
            "§7 v1.5 §0.3: the headline is computed under BOTH the v1.2 ORIGINAL and the v1.5 "
            "AMENDED specification from the SAME artifacts. v1.5 is the PRE-REGISTERED PRIMARY. "
            "If they DISAGREE that disagreement is a FIRST-CLASS FINDING, reported in the abstract "
            "and results and NEVER resolved in our favour or relegated to an appendix. A manifest "
            "lacking this field cannot be quoted as the M6 outcome (same footing as grid_pinned)."
        ),
        "v1_2_original": {
            "n_trials": n_trials_v12,
            "n_trials_basis": "cells x SEEDS x horizons x kappas (seeds COUNTED as trials)",
            "var_sr_basis": "all_arms, de-annualized at EACH TRIAL'S OWN horizon — the units v1.2 "
            "actually used (§7 v1.6.25 R3: this leg previously read the PRIMARY_H-based trial set, "
            "making it a hybrid of two specifications and biased toward agreeing with the primary)",
            "var_sr": var_sr_v12,
            "sr0": float(expected_max_sharpe(var_sr_v12, n_trials_v12)),
            "dsr": dsr_v12,
            "mde_rule": "z-quantiles, scoring-noise SE only",
            "mde_paired": pb45.mde_paired_scoring_only,
            "clauses_pass": clauses_v12,
            "primary": primary_v12,
        },
        # §7 v1.6.22 C-3: the superseded MIXED-UNIT basis, retained and REPORTED so nobody has
        # to take our unit judgement on trust. Same artifacts, same clause, different convention.
        "v1_5_mixed_unit_basis_superseded": {
            "n_trials": DSR_N_TRIALS,
            "var_sr_basis": f"cell{DSR_VAR_SR_BASIS_CELL}_placebo, de-annualized at EACH TRIAL'S "
            "OWN horizon (the pre-v1.6.22 convention)",
            "var_sr": var_sr_mixed,
            "sr0": float(expected_max_sharpe(var_sr_mixed, DSR_N_TRIALS)),
            "dsr": float(deflated_sharpe_ratio(s[4], n_trials=DSR_N_TRIALS, var_sr=var_sr_mixed)),
            "why_superseded": "SR0 is compared against a per-PRIMARY_H Sharpe; a per-5/15/60 "
            "blend is a unit error (sqrt(12) = 3.4641x span). Disagreement between this leg and "
            "the primary is a FIRST-CLASS FINDING, on the same footing as v1.2-vs-v1.5.",
        },
        "v1_5_amended": {
            "n_trials": DSR_N_TRIALS,
            "n_trials_basis": "cells x horizons x kappas (seeds are REPLICATES, not trials)",
            "var_sr_basis": f"cell{DSR_VAR_SR_BASIS_CELL}_placebo",
            "var_sr": var_sr,
            "sr0": sr0,
            "dsr": dsr,
            "mde_rule": "t-quantiles (Welch-Satterthwaite), SE_boot (+) SE_train",
            "mde_paired": pb45.mde_paired,
            "clauses_pass": {k: bool(c["pass"]) for k, c in clauses.items()},
            "primary": primary,
        },
        "agree": bool(primary_v12 == primary),
        "disagreement_is_a_first_class_finding": True,
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
    # §7 v1.6.15 (fail-open sweep): a REQUIRED disclosure that could not read its inputs joins the
    # HALT set. It stays NON-GATING in the only sense that matters — it can never flip
    # SURVIVES<->NULL, the clauses below are computed identically — but "we measured nothing" must
    # not be emitted as a verdict, which is the same rule the two guards already follow.
    decode_disc = decode_agreement_disclosure(evals)
    emitted = (
        HALT_ADJUDICATE
        if (guard["halted"] or power["halted"] or decode_disc.get("unmeasurable"))
        else primary
    )

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
        "dual_specification": dual_specification,
        # §7 v1.6.25 R4: REQUIRED. The gate's own verdict + the BSQ disclosure it obliges, carried
        # in the manifest rather than evaluated and discarded.
        "external_validation": external_validation_record,
        # §7 v1.6 C-12 M1: REQUIRED beside the headline. Non-gating, no threshold, cannot flip
        # SURVIVES<->NULL — it bounds the placebo's capacity handicap with a measured magnitude.
        "placebo_capacity_disclosure": placebo_capacity_disclosure(evals),
        # §7 v1.6.11 C-1: the SECOND handicap channel. Both inflate dIR(4-5) if real; they compound.
        "decode_agreement_disclosure": decode_disc,
        "placebo_dispersion_tripwire": placebo_tripwire,
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
            "halted_for_unmeasurable_disclosure": bool(decode_disc.get("unmeasurable")),
            "failing_clauses": failing,
            "fallback": fallback,
            "double_null": bool(
                primary == NULL and (fallback or {}).get("word") != FALLBACK_CLAIMED
            ),
        },
        "point_diagnostics": {k: v for k, v in ablation_verdict(ir).items() if k != "cell_ir"},
    }
    manifest["content_hash"] = content_hash(manifest)
    # §7 v1.6.25 R4: the emitter validates its OWN output. `grid_pinned` is stamped by the driver
    # one level up, so it is checked by the loader rather than here.
    self_check = verdict_manifest_failures(manifest, require_grid_pinned=False)
    if self_check:
        raise VerdictInputError(
            "assemble_verdict produced a manifest missing REQUIRED fields:\n  - "
            + "\n  - ".join(self_check)
        )
    return manifest


# ------------------------------------------------------------------ manifest required fields
# §7 v1.6.25 (RE-AUDIT R4). A manifest lacking any of these cannot be quoted as the M6 outcome —
# the rule `dual_specification` already carried in prose, now enforced for the whole set.
REQUIRED_MANIFEST_FIELDS = (
    "schema",
    "clauses",
    "verdict",
    "dual_specification",
    "external_validation",
    "placebo_capacity_disclosure",
    "decode_agreement_disclosure",
    "degeneracy_guard",
    "power_guard",
    "content_hash",
)


def verdict_manifest_failures(manifest: dict, *, require_grid_pinned: bool = True) -> list[str]:
    """Everything missing from a verdict manifest (empty = quotable).

    WHY A LOADER AT ALL. The M6 outcome is read back out of a JSON file — by us, by the paper, and
    by anyone reproducing it. Until v1.6.25 nothing on the READ side asserted anything, so a field
    that stopped being written (R4: the G-§8.C.3 verdict and its required BSQ disclosure) simply
    stopped existing, silently. Emission-side validation alone cannot catch a manifest produced by
    an older revision; this can, because it runs against the file.
    """
    fails: list[str] = []
    if not isinstance(manifest, dict):
        return [f"manifest is {type(manifest).__name__}, not a dict"]
    for key in REQUIRED_MANIFEST_FIELDS:
        if key not in manifest or manifest[key] in (None, {}, ""):
            fails.append(f"REQUIRED field {key!r} is missing or empty")
    if manifest.get("schema") not in (None, VERDICT_SCHEMA):
        fails.append(f"schema {manifest.get('schema')!r} != {VERDICT_SCHEMA!r}")
    ev = manifest.get("external_validation")
    if isinstance(ev, dict):
        gate = ev.get("gate")
        if not isinstance(gate, dict) or not gate.get("status"):
            fails.append("external_validation.gate is missing or carries no status (§7 v1.6.25 R4)")
        disc = ev.get("required_disclosure")
        if not isinstance(disc, str) or not disc.strip():
            fails.append(
                "external_validation.required_disclosure is absent — dropping G-§8.C.3 as binding "
                "was conditioned on shipping this sentence, and a MISSING disclosure is the one "
                "defect disclosure cannot neutralize"
            )
    elif "external_validation" not in fails and ev is not None:
        fails.append("external_validation is not a dict")
    if require_grid_pinned and manifest.get("grid_pinned") is not True:
        fails.append(
            f"grid_pinned is {manifest.get('grid_pinned')!r}, not True — a toy/fixture manifest "
            "can never be quoted as the M6 outcome"
        )
    return fails


def load_verdict_manifest(path: Path, *, require_grid_pinned: bool = True) -> dict:
    """Read a verdict manifest and REFUSE it if any required field is absent."""
    doc = json.loads(Path(path).read_text())
    fails = verdict_manifest_failures(doc, require_grid_pinned=require_grid_pinned)
    if fails:
        raise VerdictInputError(
            f"verdict manifest {path} cannot be quoted as the M6 outcome:\n  - "
            + "\n  - ".join(fails)
        )
    return doc


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
    "REQUIRED_MANIFEST_FIELDS",
    "SURVIVES",
    "VERDICT_SCHEMA",
    "VerdictInputError",
    "assemble_verdict",
    "degeneracy_guard",
    "dsr_unit_convention_failures",
    "enumerate_dsr_trials",
    "load_cell_evals",
    "load_verdict_manifest",
    "own_h_trial_values",
    "power_guard",
    "verdict_manifest_failures",
    "write_cell_eval_artifact",
    "write_eval_index",
]
