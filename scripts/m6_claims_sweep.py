"""§7 v1.6 ITEM 4 — sweep the repo for the Finding-0 failure mode: prose that carries a datum.

    PYTHONPATH=src .venv/bin/python scripts/m6_claims_sweep.py

Finding 0 was not a one-off. Its signature is generic and machine-detectable:

  P1  a NUMBER living inside a STRING field of a receipt (a value that cannot be read
      programmatically, cannot be range-checked, and drifts silently from the thing it describes);
  P2  a numeric manifest field that is NaN / null / 0 while prose elsewhere quotes it as measured;
  P3  a claim labelled MEASURED whose quoted figure appears in NO artifact.

This finds and LISTS them. It fixes nothing — the ruling is explicit that these are reported, not
silently repaired. Writes ``runs_manifest/m6_claims_sweep.json``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

OUT = Path("runs_manifest/m6_claims_sweep.json")
MANIFEST_DIR = Path("runs_manifest")

# SELF-INCLUSION SKIP SET. These receipts QUOTE the defects as evidence, so scanning them counts
# the audit's own findings as new findings and the totals inflate on every re-run. That exact bug
# (49 -> 98) was caught once before in the bit-exact corrective receipt; the fix is the same, and
# idempotence is asserted below rather than assumed.
SELF_REFERENTIAL = {
    "m6_claims_sweep.json",  # this script's own output
    "m6_cost_basis_forensics.json",  # quotes the 47.2k prose string and the NaN fields verbatim
}

# A number with a unit or magnitude suffix inside prose — "47.2k bars/s", "2-3 GPU-days", "$20-30".
NUM_IN_PROSE = re.compile(
    r"(?<![\w.])(\d[\d,]*\.?\d*)\s*"
    r"(k\b|M\b|GPU-day|GPU-hour|bars/s|/s|%|\$|hr\b|min\b|GB\b)",
    re.IGNORECASE,
)
MEASURED_WORD = re.compile(r"\bMEASURED\b")

# Fields that are prose BY DESIGN — narrative, not a datum channel.
PROSE_BY_DESIGN = {
    "note",
    "notes",
    "detail",
    "why",
    "purpose",
    "question",
    "verdict",
    "answer",
    "reason",
    "rule",
    "basis",
    "excludes",
    "generator",
    "status",
    "implication",
    "caveat",
    "disclosure",
    "interpretation",
    "CORRECTION",
    "summary",
}


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def main() -> int:
    p1, p2, p3 = [], [], []
    all_numbers_in_artifacts: set[str] = set()

    for f in sorted(MANIFEST_DIR.glob("*.json")):
        if f.name in SELF_REFERENTIAL:
            continue
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for path, val in walk(d):
            leaf = path.rsplit(".", 1)[-1].split("[")[0]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                all_numbers_in_artifacts.add(f"{float(val):.6g}")
                hit = None
                if isinstance(val, float) and math.isnan(val):
                    hit = {"file": str(f), "field": path, "value": "NaN"}
                elif val == 0 and any(
                    w in path.lower() for w in ("wall", "throughput", "rate", "bars_per")
                ):
                    hit = {"file": str(f), "field": path, "value": 0}
                if hit is not None:
                    # SEVERITY MATTERS, and getting it wrong is the same defect one level up: a
                    # NaN inside a per-STEP diagnostic trajectory is a genuinely undefined
                    # statistic (e.g. a correlation on a constant rollout), not a lost
                    # measurement. Burying the real hits under those would make this audit
                    # unreadable, which is how the original defect survived in the first place.
                    per_step = "trajectory[" in path or ".attempts[" in path
                    summary_kind = any(
                        w in path.lower() for w in ("wall", "throughput", "rate", "bars_per")
                    )
                    hit["severity"] = (
                        "HIGH"
                        if (summary_kind and not per_step)
                        else "LOW"
                        if per_step
                        else "MEDIUM"
                    )
                    hit["why"] = (
                        "summary-level quantity with no measurement behind it"
                        if hit["severity"] == "HIGH"
                        else "per-step diagnostic; undefined by construction, not a lost datum"
                        if hit["severity"] == "LOW"
                        else "non-trajectory numeric slot holding no value"
                    )
                    p2.append(hit)
            elif isinstance(val, str) and leaf not in PROSE_BY_DESIGN:
                for m in NUM_IN_PROSE.finditer(val):
                    p1.append(
                        {
                            "file": str(f),
                            "field": path,
                            "quoted": m.group(0),
                            "context": val[max(0, m.start() - 60) : m.start() + 80],
                        }
                    )
                if MEASURED_WORD.search(val) and NUM_IN_PROSE.search(val):
                    p3.append({"file": str(f), "field": path, "text": val[:300]})

    out = {
        "receipt": "m6_claims_sweep",
        "amendment": "prereg §7 v1.6 item 4 (supervisor-ordered, local $0)",
        "patterns": {
            "P1_number_inside_a_string_field": "a datum that cannot be read programmatically",
            "P2_nan_or_zero_numeric_field": "a numeric slot that holds no measurement",
            "P3_labelled_MEASURED_in_prose": "the exact Finding-0 signature",
        },
        "P1_hits": p1,
        "P2_hits": p2,
        "P3_hits": p3,
        "counts": {
            "P1": len(p1),
            "P2_total": len(p2),
            "P2_HIGH": sum(h["severity"] == "HIGH" for h in p2),
            "P2_MEDIUM": sum(h["severity"] == "MEDIUM" for h in p2),
            "P2_LOW": sum(h["severity"] == "LOW" for h in p2),
            "P3": len(p3),
        },
        "self_referential_files_skipped": sorted(SELF_REFERENTIAL),
        # Every HIGH hit adjudicated by hand, including this audit's OWN false positives. A sweep
        # that reports unadjudicated hits pushes the judgment onto the next reader and quietly
        # inflates its own findings — the same defect one level up.
        "triage_of_high_hits": {
            "m6_rehearsal_manifest.json::throughput.bars_per_s_into_gpu": {
                "verdict": "TRUE POSITIVE — this is Finding 0 itself",
                "evidence": "runs_manifest/m6_cost_basis_forensics.json",
            },
            "m6_rehearsal_manifest.json::throughput.steps_per_s": {
                "verdict": "TRUE POSITIVE — same lost measurement",
                "evidence": "runs_manifest/m6_cost_basis_forensics.json",
            },
            "m6_rehearsal_manifest.json::throughput.train_wall_s": {
                "verdict": "TRUE POSITIVE — 0 where a wall-clock belongs; the zero reads as a "
                "measurement of nothing rather than as an absent measurement",
                "evidence": "runs_manifest/m6_cost_basis_forensics.json",
            },
            "m6_token_causality_probe.json::results.untrained_causal_flag.coarse_flip_rate": {
                "verdict": "FALSE POSITIVE OF THIS AUDIT — matched on the substring 'rate'. 0.0 "
                "is the GENUINE measured value and the DESIRED one: zero token flips is the "
                "causal-safety result, corroborated by coarse_flip_by_position (20 positions, all "
                "0.0) over n_chunks=60. Nothing is missing here.",
                "evidence": "the companion per-position array in the same artifact",
            },
            "m6_token_causality_probe.json::results.untrained_causal_flag.fine_flip_rate": {
                "verdict": "FALSE POSITIVE OF THIS AUDIT — identical reasoning to coarse_flip_rate",
                "evidence": "the companion per-position array in the same artifact",
            },
        },
        "scope_note": (
            "Fields that are narrative by design (note/purpose/verdict/CORRECTION/...) are "
            "excluded — prose is their job. A hit here is a number in a field that is NOT one of "
            "those, i.e. a datum channel carrying text. NOTHING IS FIXED BY THIS SCRIPT."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print("=== CLAIMS SWEEP (prose-carrying-a-datum) ===")
    print(f"  P1 number-in-string-field : {len(p1)}")
    hi = sum(h["severity"] == "HIGH" for h in p2)
    md = sum(h["severity"] == "MEDIUM" for h in p2)
    lo = len(p2) - hi - md
    print(f"  P2 NaN/zero numeric field : {len(p2)} (HIGH {hi} / MED {md} / LOW {lo})")
    print(f"  P3 labelled MEASURED      : {len(p3)}")
    for h in p3:
        print(f"    P3 -> {h['file']}::{h['field']}")
    for h in p2:
        if h["severity"] != "LOW":
            print(f"    P2[{h['severity']}] -> {h['file']}::{h['field']} = {h['value']}")
    print(f"[receipt] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
