"""G-§8.C.3 — the Kronos external-validation entry gate (audit C-4).

**WHY THIS FILE EXISTS.** The gate is named binding in three places (`m6_design.md:51`,
`docs/m6_prereg.md` §6, `ROADMAP.md` M6 entry gates) and had **no implementation at all** — the
auditor's C-4. A gate whose failure protocol lives only in prose is the shape this project has now
closed four times, so the protocol is ENCODED here rather than described.

What the auditor said breaks, in their words: *"a weak Cell 1 manufactures a positive FSQ result
and nothing detects it."* Cell 1 is the §5 NULL-fallback's comparator (`IR(2) − IR(1)`), so an
under-trained Cell 1 does not merely weaken the baseline — it inflates the fallback claim.

**IT FAILS CLOSED, AND THAT IS THE POINT.** The published reference RankIC has NOT been obtained
(`ExternalReference.published_rankic is None`), and the comparison is BLOCKED on a ruling that is
not the builder's — see §"THE INVARIANT-8 PROBLEM" below. An unconfigured gate therefore returns
``BLOCKED``, which is a HALT. It must never return "pass" because nobody filled the number in;
that is precisely how C-2, C-10 and C-15 happened.

**THE INVARIANT-8 PROBLEM (NEEDS-RULING, LAKSHAY'S).** The prereg requires Kronos-small to be
*"run through its own published preprocessing on the same bars"*. The published weights are a bare
`state_dict` — the model card's own loading instructions are ``from model import Kronos,
KronosTokenizer`` — so **running them requires Kronos's model code**. CLAUDE.md invariant 8 says
*"No Kronos code or weights are ever part of Trikaal … its public weights appear in exactly one
place: the eval harness."* Weights are permitted there; **code is not permitted anywhere.** The
gate as specified cannot be executed without resolving that, and it is not a scheduling problem.
Licence is not the obstacle: Kronos is MIT (see `docs/m6_c4_kronos_gate_requirements.md`).
"""

from __future__ import annotations

from dataclasses import dataclass

# The three pre-committed responses (m6_design.md:51 / prereg §6). Encoded, not described.
HALT_BEFORE_ANY_DELTA = "HALT_BEFORE_ANY_DELTA"
CELL1_ONLY_FIX = "CELL1_ONLY_FIX"
FULL_5_CELL_SAME_SEED_RETRAIN = "FULL_5_CELL_SAME_SEED_RETRAIN"

GATE_RATIO = 0.85  # prereg v1.2: fails iff RankIC < 0.85 × published Kronos-small

BLOCKED = "BLOCKED"
PASS = "PASS"
FAIL = "FAIL"


class ExternalValidationBlocked(RuntimeError):
    """Raised when a Δ would be computed while G-§8.C.3 is unresolved."""


@dataclass(frozen=True)
class ExternalReference:
    """The published Kronos-small figure this gate compares against.

    ``published_rankic = None`` is the CURRENT, HONEST state: the number has not been transcribed,
    the slice it applies to is not pinned, and the requirements doc already records that *which*
    reported figure to use is itself unresolved. A ``None`` here HALTS.
    """

    published_rankic: float | None = None
    source: str = "NOT OBTAINED — see docs/m6_c4_kronos_gate_requirements.md"
    slice_id: str | None = None
    weights_sha256: str | None = None  # content-hash the weights like every other input


def gate_threshold(ref: ExternalReference) -> float | None:
    if ref.published_rankic is None:
        return None
    return GATE_RATIO * float(ref.published_rankic)


def evaluate(cell1_rankic: float | None, ref: ExternalReference) -> dict:
    """The gate. Returns a verdict dict; never raises on a normal FAIL — the CALLER halts.

    Order matters and is deliberate: an unobtainable reference is reported BEFORE an unmeasured
    Cell 1, because "we never got the number" and "the baseline is weak" are different states and
    collapsing them is how a blocked gate reads as a passing one.
    """
    thr = gate_threshold(ref)
    if thr is None:
        return {
            "gate": "G-§8.C.3",
            "status": BLOCKED,
            "reason": (
                "no published Kronos-small RankIC has been obtained, so the threshold does not "
                "exist. BLOCKED is a HALT: an unconfigured gate must never read as a pass"
            ),
            "reference": ref.source,
            "protocol": HALT_BEFORE_ANY_DELTA,
            "blocking_question": (
                "running the published weights requires Kronos's model code (the card's own "
                "loader is `from model import Kronos, KronosTokenizer`), and invariant 8 permits "
                "Kronos WEIGHTS in the eval harness but no Kronos CODE anywhere — Lakshay's ruling"
            ),
        }
    if cell1_rankic is None:
        return {
            "gate": "G-§8.C.3",
            "status": BLOCKED,
            "reason": "Cell 1 has not been trained/scored — the gate fires AFTER Cell 1",
            "threshold": thr,
            "protocol": HALT_BEFORE_ANY_DELTA,
        }
    passed = float(cell1_rankic) >= thr
    return {
        "gate": "G-§8.C.3",
        "status": PASS if passed else FAIL,
        "cell1_rankic": float(cell1_rankic),
        "published_rankic": float(ref.published_rankic),  # type: ignore[arg-type]
        "ratio": float(cell1_rankic) / float(ref.published_rankic),  # type: ignore[arg-type]
        "gate_ratio": GATE_RATIO,
        "threshold": thr,
        "reference": ref.source,
        "slice_id": ref.slice_id,
        "weights_sha256": ref.weights_sha256,
        "protocol": None
        if passed
        else (
            f"{HALT_BEFORE_ANY_DELTA} -> then ONE of: {CELL1_ONLY_FIX} | "
            f"{FULL_5_CELL_SAME_SEED_RETRAIN} (pre-committed, m6_design.md:51; the choice is the "
            "operator's and the earlier 'materially off / fix' wording is superseded)"
        ),
    }


def assert_clear_to_compute_deltas(verdict: dict) -> None:
    """Call BEFORE any between-cell Δ is computed. Halts unless the gate PASSED.

    This is the whole point of the gate: it is an ENTRY gate, so the failure must arrive before a
    Δ exists, not after one has been looked at.
    """
    if verdict.get("status") != PASS:
        raise ExternalValidationBlocked(
            f"G-§8.C.3 is {verdict.get('status')} — {verdict.get('reason', 'gate not passed')}. "
            f"Protocol: {verdict.get('protocol')}. No between-cell Δ may be computed while this "
            "gate is unresolved (prereg §6; m6_design.md:51)."
        )


__all__ = [
    "BLOCKED",
    "CELL1_ONLY_FIX",
    "FAIL",
    "FULL_5_CELL_SAME_SEED_RETRAIN",
    "GATE_RATIO",
    "HALT_BEFORE_ANY_DELTA",
    "PASS",
    "ExternalReference",
    "ExternalValidationBlocked",
    "assert_clear_to_compute_deltas",
    "evaluate",
    "gate_threshold",
]
