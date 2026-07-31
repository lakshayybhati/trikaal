"""§7 v1.6 ITEM 3 — is each INCONCLUSIVE rule's firing condition tight enough to need NO judgment?

    PYTHONPATH=src .venv/bin/python scripts/m6_taxonomy_audit.py

The bounds (≤1 re-run round, ≤5 seeds, cells named by the tripped guard) are already written as
numbers and were verified separately. THIS audits the other half: given a verdict manifest, can
R1 / R2 / R2b / R3 be selected MECHANICALLY, or does some step require a human to decide?

Method: implement the rules exactly as the prereg words them, run them against fixtures built to
sit on each boundary, and REPORT AMBIGUITY WHERE THE WORDS ADMIT TWO READINGS — deliberately
without choosing between them. A rule that needs a choice is a hole to be flagged, not filled.

Writes ``runs_manifest/m6_taxonomy_audit.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from trikaal.eval.verdict import DSR_SEEDS

OUT = Path("runs_manifest/m6_taxonomy_audit.json")
S = len(DSR_SEEDS)
MAJORITY = S // 2 + 1  # 3 of 5


def select_rule(guards: dict) -> dict:
    """The prereg rule set, applied in order — first match governs.

    Returns the selected rule, or ``AMBIGUOUS`` with both readings when the text admits two.
    """
    deg_halted = bool(guards["degeneracy"]["halted"])
    pow_halted = bool(guards["power"]["halted"])
    if not (deg_halted or pow_halted):
        return {"rule": None, "why": "no guard halted — SURVIVES/NULL is emitted, taxonomy N/A"}

    # R1 — power halt ONLY.
    if pow_halted and not deg_halted:
        return {
            "rule": "R1",
            "why": "power_guard halted and degeneracy_guard did not",
            "determinate": True,
        }

    # R2 / R2b — a degeneracy halt, split on majority vs strict minority of seeds.
    per_cell = guards["degeneracy"]["per_cell"]
    findings, ambiguities = [], []
    for cell, rec in sorted(per_cell.items()):
        if not rec.get("degenerate"):
            continue
        fn_seeds = set(rec.get("degenerate_seeds_frac_negative") or [])
        act_seeds = set(rec.get("degenerate_seeds_activity") or [])
        union = fn_seeds | act_seeds
        readings = {
            "union_of_legs": len(union),
            "max_single_leg": max(len(fn_seeds), len(act_seeds)),
        }
        verdicts = {k: ("majority" if v >= MAJORITY else "minority") for k, v in readings.items()}
        # HOLE 1 — "degenerate at a MAJORITY of seeds" does not say majority of WHICH leg.
        if verdicts["union_of_legs"] != verdicts["max_single_leg"]:
            ambiguities.append(
                {
                    "cell": cell,
                    "hole": "MAJORITY_OVER_WHICH_LEG",
                    "counts": readings,
                    "readings": verdicts,
                    "detail": (
                        "The guard has TWO independent degeneracy legs (sign-lock and "
                        "activity-never-binds) and persists them as separate seed lists. The rule "
                        "says 'a cell degenerate at a MAJORITY of seeds' without saying whether "
                        "that is a majority on ONE leg or over the UNION. Here the two readings "
                        "disagree, so R2 and R2b both have a claim."
                    ),
                }
            )
        # HOLE 2 — R2b's headline and its parenthetical are different tests.
        mean_in_band = rec.get("seed_mean_in_band")
        strict_minority = verdicts["union_of_legs"] == "minority"
        if mean_in_band is not None and strict_minority != bool(mean_in_band):
            ambiguities.append(
                {
                    "cell": cell,
                    "hole": "R2B_HEADLINE_VS_PARENTHETICAL",
                    "strict_minority_of_seeds": strict_minority,
                    "seed_mean_in_band": bool(mean_in_band),
                    "detail": (
                        "R2b is written as 'degenerate at a strict MINORITY of seeds' with the "
                        "parenthetical '(the per-seed leg fires while the seed-mean is in band)'. "
                        "Those are two different tests and here they disagree, so which one "
                        "governs is a choice the text does not make."
                    ),
                }
            )
        findings.append({"cell": cell, "counts": readings, "readings": verdicts})

    if ambiguities:
        return {
            "rule": "AMBIGUOUS",
            "why": "the degeneracy halt matches R2 and R2b under different readings of the text",
            "determinate": False,
            "candidates": ["R2", "R2b"],
            "ambiguities": ambiguities,
            "findings": findings,
        }
    if any(f["readings"]["union_of_legs"] == "majority" for f in findings):
        return {
            "rule": "R2",
            "why": "a cell is degenerate at a majority of seeds",
            "determinate": True,
            "findings": findings,
        }
    if findings:
        return {
            "rule": "R2b",
            "why": "degeneracy confined to a strict minority of seeds",
            "determinate": True,
            "findings": findings,
        }
    return {
        "rule": "R3",
        "why": "degeneracy halt with no cell resolving to either branch",
        "determinate": True,
    }


def _guards(*, deg=False, pow_=False, per_cell=None) -> dict:
    return {
        "degeneracy": {"halted": deg, "per_cell": per_cell or {}},
        "power": {"halted": pow_},
    }


def main() -> int:
    seeds = list(DSR_SEEDS)
    fixtures = {
        "no_halt": _guards(),
        "power_only": _guards(pow_=True),
        "degeneracy_majority_both_legs": _guards(
            deg=True,
            per_cell={
                "4": {
                    "degenerate": True,
                    "degenerate_seeds_frac_negative": seeds[:4],
                    "degenerate_seeds_activity": seeds[:3],
                    "seed_mean_in_band": False,
                }
            },
        ),
        "degeneracy_minority_both_legs": _guards(
            deg=True,
            per_cell={
                "4": {
                    "degenerate": True,
                    "degenerate_seeds_frac_negative": seeds[:1],
                    "degenerate_seeds_activity": [],
                    "seed_mean_in_band": True,
                }
            },
        ),
        # HOLE 1 made concrete: 2 seeds sign-locked, 2 DIFFERENT seeds never-binding.
        # Union = 4 of 5 (majority ⇒ R2); no single leg exceeds 2 (minority ⇒ R2b).
        "split_legs_union_majority": _guards(
            deg=True,
            per_cell={
                "4": {
                    "degenerate": True,
                    "degenerate_seeds_frac_negative": seeds[:2],
                    "degenerate_seeds_activity": seeds[2:4],
                    "seed_mean_in_band": False,
                }
            },
        ),
        # HOLE 2 made concrete: a strict minority of seeds degenerate, seed-mean OUT of band.
        "minority_seeds_but_mean_out_of_band": _guards(
            deg=True,
            per_cell={
                "4": {
                    "degenerate": True,
                    "degenerate_seeds_frac_negative": seeds[:2],
                    "degenerate_seeds_activity": [],
                    "seed_mean_in_band": False,
                }
            },
        ),
        "both_guards_halt_majority": _guards(
            deg=True,
            pow_=True,
            per_cell={
                "4": {
                    "degenerate": True,
                    "degenerate_seeds_frac_negative": seeds[:4],
                    "degenerate_seeds_activity": seeds[:4],
                    "seed_mean_in_band": False,
                }
            },
        ),
    }
    results = {name: select_rule(g) for name, g in fixtures.items()}
    holes = {n: r for n, r in results.items() if r.get("determinate") is False}

    per_rule = {
        "R1": {
            "pre_declared_condition": "power_guard halted AND degeneracy_guard did not",
            "what_executes": (
                "verdict.py computes both guards and writes guards.power.halted / "
                "guards.degeneracy.halted as booleans; the rule reads those two fields"
            ),
            "needs_human_choice": False,
            "note": (
                "DETERMINATE as a TEST. Its ACTION ('add seeds to the pre-declared cap') is "
                "unreachable at S=5 — the cap is already spent, so R1 collapses into R3, which "
                "the prereg states explicitly."
            ),
        },
        "R2": {
            "pre_declared_condition": (
                "degeneracy halt with a cell degenerate at a MAJORITY of seeds"
            ),
            "what_executes": (
                "degeneracy_guard persists degenerate_seeds_frac_negative and "
                "degenerate_seeds_activity as SEPARATE per-cell seed lists; NO code computes "
                "'majority', so the count is performed by the adjudicator reading those lists"
            ),
            "needs_human_choice": True,
            "hole": "MAJORITY_OVER_WHICH_LEG — see holes; the text does not say union or per-leg",
        },
        "R2b": {
            "pre_declared_condition": (
                "degeneracy halt with a cell degenerate at a strict MINORITY of seeds "
                "(parenthetical: the per-seed leg fires while the seed-mean is in band)"
            ),
            "what_executes": "same persisted lists as R2, plus the seed-mean fields",
            "needs_human_choice": True,
            "hole": (
                "R2B_HEADLINE_VS_PARENTHETICAL — two different tests joined as if equivalent; "
                "they can disagree"
            ),
        },
        "R3": {
            "pre_declared_condition": "catch-all: report INCONCLUSIVE",
            "what_executes": (
                "nothing to decide — verdict.py already emits HALT_ADJUDICATE and persists every "
                "field R3's report requires (per-seed frac_negative/activity/IR, ranges, primary, "
                "realized MDE, dual_specification)"
            ),
            "needs_human_choice": False,
            "note": "At S=5 this is the OPERATIVE rule for every INCONCLUSIVE outcome.",
        },
    }

    out = {
        "receipt": "m6_taxonomy_audit",
        "amendment": "prereg §7 v1.6 item 3 (supervisor-ordered, local $0)",
        "S": S,
        "majority_threshold": MAJORITY,
        "fixture_results": results,
        "per_rule": per_rule,
        "holes_found": list(holes),
        "s5_collapse_confirmed": {
            "claim": "at S=5, R1 and R2b collapse into R3 and R3 is the operative rule",
            "verified": True,
            "why": (
                "Both R1 and R2b prescribe the SAME remedy — add seeds within the cap. The cap is "
                "'total distinct seeds <= 5' and PINNED_SEEDS already has 5, so the remedy set is "
                "empty and each rule's own text routes it to R3 ('if it still halts -> R3'; R2b "
                "'treat as R1'). R2 already routes to R3 by construction ('no re-run, go to R3'). "
                "Every INCONCLUSIVE path therefore terminates at R3 with zero re-runs."
            ),
            "consequence_for_the_holes": (
                "The two holes do NOT change the outcome at S=5 — R2, R2b and R1 all terminate at "
                "R3 — but they DO change what gets REPORTED: R2 additionally requires the "
                "degeneracy to be written up as a PRIMARY mechanism finding, while R2b treats the "
                "same halt as a training lottery. So the ambiguity is live for the paper's claims "
                "even though it is dead for the re-run decision."
            ),
        },
        "verdict": (
            "R1 and R3 are determinate. R2 and R2b are NOT: (1) 'majority of seeds' does not say "
            "whether the majority is over one degeneracy leg or the union of both, and the guard "
            "deliberately keeps the legs separate; (2) R2b's headline test (strict minority of "
            "seeds) and its parenthetical test (per-seed leg fires, seed-mean in band) are "
            "different conditions presented as one. Both are FLAGGED, NOT FILLED — the fix is a "
            "specification choice and the design is frozen."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print("=== TAXONOMY COMPLETENESS AUDIT ===")
    for name, r in results.items():
        print(f"  {name:38s} -> {r.get('rule')}  determinate={r.get('determinate')}")
    print(f"\n  holes: {list(holes)}")
    print(f"[receipt] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
