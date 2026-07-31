"""Training-throughput records that are NUMBERS, not prose (prereg §7 v1.6 Finding 0).

WHY THIS EXISTS. The M6 cost basis — "47.2k bars/s → full M6 ≈ 2–3 GPU-days ≈ $20–30" — lived
repo-wide in exactly ONE place: a *prose string* in a receipt's ``labels_required_by_item_5``
field. The manifest that string cited (``m6_rehearsal_manifest.json``) records
``throughput.bars_per_s_into_gpu = NaN`` and ``train_wall_s = 0``. A number labelled MEASURED
whose cited receipt is empty is the same defect as the 49 determinism records: a claim asserted in
prose with no datum behind it. See ``docs/m6_throughput_provenance.md`` for the forensics.

THE FIX IS STRUCTURAL, NOT EDITORIAL. Every throughput claim now goes through
``throughput_record``, which:

* emits **numeric** fields (``steps_per_s``, ``bars_per_s``) — never a formatted string;
* forces a **closed-vocabulary ``basis``**, so a compute-only micro-benchmark can never be
  silently quoted as end-to-end training throughput (the actual confusion behind 47.2k);
* sets ``measured=False`` and NaN-fills the rates when the inputs cannot support a number, so an
  absent measurement reads as absent instead of as zero.

``basis`` values, and exactly what each excludes:

``compute_only_step``
    Timed forward/backward/optimizer on batches ALREADY RESIDENT in memory. Excludes data
    loading, tokenization, evaluation, and checkpointing. An **upper bound** on the training rate
    of the timed stage — it is not an end-to-end training throughput and must never be quoted as
    the cost basis on its own.
``end_to_end_training``
    Wall-clock of a real training invocation, everything the run actually pays for included.
    This is the only basis that prices a run.
``unmeasured``
    A structural placeholder: the fields exist, the rates are NaN, ``measured`` is False.
"""

from __future__ import annotations

import math

BASIS_COMPUTE_ONLY = "compute_only_step"
BASIS_END_TO_END = "end_to_end_training"
BASIS_END_TO_END_EVAL = "end_to_end_eval"
BASIS_UNMEASURED = "unmeasured"

VALID_BASES = (BASIS_COMPUTE_ONLY, BASIS_END_TO_END, BASIS_END_TO_END_EVAL, BASIS_UNMEASURED)

#: The bases that may price a run. Training and eval are BOTH real run cost — the eval leg was the
#: last unmeasured input (its own receipt said "PENDING the 4090"). ``compute_only_step`` stays out
#: deliberately: quoting a micro-benchmark as a cost basis is the Finding-0 defect.
COSTABLE_BASES = (BASIS_END_TO_END, BASIS_END_TO_END_EVAL)

#: What each basis does NOT include — carried in the record so a reader cannot lose it.
BASIS_EXCLUDES = {
    BASIS_COMPUTE_ONLY: (
        "data loading, tokenization, eval, checkpointing — UPPER BOUND, not a cost basis"
    ),
    BASIS_END_TO_END: "nothing; wall-clock of the real invocation",
    BASIS_END_TO_END_EVAL: (
        "nothing; wall-clock of the real scoring pass (rollout + costs + netting) at the pinned "
        "chunk size — the eval leg of the run cost"
    ),
    BASIS_UNMEASURED: "everything — no measurement was taken",
}


def _finite_positive(x: float | int | None) -> bool:
    return x is not None and math.isfinite(float(x)) and float(x) > 0.0


def throughput_record(
    *,
    steps: int | None,
    batch: int | None,
    seq_len: int | None,
    wall_s: float | None,
    basis: str,
    device: str = "",
    note: str = "",
) -> dict:
    """A numeric throughput record. ``bars_per_s = steps/wall_s × batch × seq_len``.

    Every rate is a float (possibly NaN) — never a string. ``measured`` is True only when all of
    ``steps``, ``batch``, ``seq_len``, ``wall_s`` are finite and positive AND the basis is a real
    measurement basis; otherwise the rates are NaN and ``measured`` is False, so an unmeasured
    quantity cannot masquerade as a zero.
    """
    if basis not in VALID_BASES:
        raise ValueError(f"basis must be one of {VALID_BASES}, got {basis!r}")
    ok = (
        basis != BASIS_UNMEASURED
        and _finite_positive(steps)
        and _finite_positive(batch)
        and _finite_positive(seq_len)
        and _finite_positive(wall_s)
    )
    steps_per_s = float(steps) / float(wall_s) if ok else float("nan")
    bars_per_s = steps_per_s * float(batch) * float(seq_len) if ok else float("nan")
    return {
        "basis": basis,
        "excludes": BASIS_EXCLUDES[basis],
        "measured": bool(ok),
        "device": device,
        "steps": int(steps) if steps is not None else None,
        "batch": int(batch) if batch is not None else None,
        "seq_len": int(seq_len) if seq_len is not None else None,
        "wall_s": float(wall_s) if wall_s is not None else float("nan"),
        "steps_per_s": steps_per_s,
        "bars_per_s": bars_per_s,
        "note": note,
    }


def unmeasured_record(*, reason: str, device: str = "") -> dict:
    """The explicit no-datum record — structurally present, numerically NaN, ``measured`` False."""
    return throughput_record(
        steps=None,
        batch=None,
        seq_len=None,
        wall_s=None,
        basis=BASIS_UNMEASURED,
        device=device,
        note=reason,
    )


def is_costable(record: dict | None) -> bool:
    """True only for a finite END-TO-END measurement — the sole basis that may price a run.

    A ``compute_only_step`` record is deliberately NOT costable: that conflation is the exact
    defect this module exists to prevent.
    """
    if not record or not record.get("measured"):
        return False
    return record.get("basis") in COSTABLE_BASES and math.isfinite(
        float(record.get("bars_per_s", float("nan")))
    )


__all__ = [
    "BASIS_COMPUTE_ONLY",
    "BASIS_END_TO_END",
    "BASIS_END_TO_END_EVAL",
    "BASIS_EXCLUDES",
    "BASIS_UNMEASURED",
    "COSTABLE_BASES",
    "VALID_BASES",
    "is_costable",
    "throughput_record",
    "unmeasured_record",
]
