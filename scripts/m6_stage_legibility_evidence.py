"""Stage the cell-4 / cell-5 Stage-1 evidence for an operator upload. STAGES ONLY — never uploads.

WHY IT MATTERS. The paper's headline number — 97.3% of cell 4's shortfall on the two signed
channels — is measured on CELL 4, and cell 4 exists nowhere public: 0 files in the repository
across every commit, 0 of the 13 files on HuggingFace. A reader cannot check the central claim of
the paper against anything. These six files are what that claim is computed from.

WHAT IS HERE AND WHAT IS NOT. Six files, ~146 MB: for each of cell4_seed0 and cell5_seed0, the
Stage-1 tokenizer, its optimizer state, and the unit log. THERE IS NO PREDICTOR, because Stage 2
was never entered — the gate refused both arms before it. So this is evidence, not a model: nothing
here can forecast, and the log is the gate's own refusal with all six per-dim sign accuracies in it.

★ THE OPERATOR UPLOADS. This script writes a staging directory and a manifest and stops. No token
is read, written, echoed or handled here.

    .venv/bin/python scripts/m6_stage_legibility_evidence.py [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trikaal.utils.paths import display_path  # noqa: E402
from trikaal.utils.receipts import ReceiptRefused, write_receipt  # noqa: E402

RECEIPT = REPO / "runs_manifest/m6_micro_legibility_stop.json"
SOURCE = REPO / "runs_cloud/legibility_stop"
MANIFEST = REPO / "runs_manifest/m6_legibility_evidence_manifest.json"
BUNDLED_MANIFEST = "m6_legibility_evidence_manifest.json"

ARMS = {"cell4_seed0": "micro", "cell5_seed0": "micro_shuffled"}
ROLES = {
    "tokenizer.pt": "Stage-1 FSQ tokenizer — the object the legibility gate measured",
    "stage1_state.pt": "Stage-1 optimizer/scheduler state — resume only",
    "unit.log": "the run's own log, ending in the gate's RuntimeError with all six per-dim "
    "sign accuracies",
}


def _git_head() -> str:
    import subprocess

    r = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    )
    return r.stdout.strip() or "unavailable"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "runs_cloud/legibility_evidence_bundle")
    ap.add_argument("--force", action="store_true", help="overwrite the tracked manifest")
    args = ap.parse_args()

    stop = json.loads(RECEIPT.read_text())
    expected = stop["files"]

    files, missing, mismatched = {}, [], []
    for name, meta in sorted(expected.items()):
        src = SOURCE / name
        if not src.is_file():
            missing.append(name)
            continue
        got = sha256(src)
        rec = {
            "bytes": src.stat().st_size,
            "sha256": got,
            "sha256_in_receipt": meta["sha256"],
            "verifies": got == meta["sha256"],
            "bytes_in_receipt": meta["bytes"],
            "source": display_path(src, REPO),
        }
        if not rec["verifies"]:
            mismatched.append(name)
        files[name] = rec

    if missing or mismatched:
        print(
            f"REFUSING TO STAGE: {len(missing)} missing, {len(mismatched)} mismatched",
            file=sys.stderr,
        )
        for n in missing + mismatched:
            print(f"  {n}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    for name in files:
        dst = args.out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, dst)
    # prove the copies, not the originals — a staging step that verifies its input is not verifying
    for name, rec in files.items():
        rec["sha256_after_copy"] = sha256(args.out / name)
        rec["copy_verifies"] = rec["sha256_after_copy"] == rec["sha256"]

    total = sum(r["bytes"] for r in files.values())
    by_cell: dict[str, dict] = {}
    for name, rec in files.items():
        cell, fname = name.split("/", 1)
        by_cell.setdefault(
            cell,
            {"cell": int(cell[4]), "seed": 0, "arm": ARMS[cell], "quantizer": "fsq", "files": {}},
        )
        by_cell[cell]["files"][fname] = {
            "bytes": rec["bytes"],
            "sha256": rec["sha256"],
            "role": ROLES[fname],
            "needed_to_reproduce_the_gate": fname == "tokenizer.pt",
        }
    doc = {
        "schema": "m6_legibility_evidence_v1",
        "script": "scripts/m6_stage_legibility_evidence.py",
        "git_head": _git_head(),
        "license_weights": "Apache-2.0 — same terms as the code and the cell-1 weights",
        "WHAT_THIS_IS": (
            "The Stage-1 artifacts of CELL 4 (FSQ + microstructure) and CELL 5 (FSQ + SHUFFLED "
            "microstructure, the placebo), seed 0, from the M6 money run. The paper's headline — "
            "97.3% of cell 4's shortfall on the two signed channels — is measured on CELL 4, which "
            "existed nowhere public: 0 files in the code repository across every commit, 0 of the "
            "13 files in the weights release. These six files are what that number is computed "
            "from, and unit.log is the gate's own refusal with all six per-dim sign accuracies in "
            "it."
        ),
        "★_WHAT_THIS_IS_NOT": (
            "NOT A MODEL, AND NOT AN ARM THAT WORKED. These are Stage-1 tokenizers of an arm the "
            "pre-registered micro-legibility gate REFUSED on real data on 2026-08-12. Stage 2 was "
            "never entered (stage2_entered=false, artifacts_produced=0), so NO PREDICTOR EXISTS "
            "for either cell and nothing here can forecast anything. Publishing them is publishing "
            "the EVIDENCE FOR A NEGATIVE RESULT — do not read it as the microstructure arm having "
            "succeeded. The three cell-1 units in the sibling seed0/seed2/seed4 directories are "
            "the only trained predictors this project released."
        ),
        "how_to_verify": (
            "sha256sum each file and compare against .cells[cell].files[name].sha256; every one "
            "also appears under the SAME path key in runs_manifest/m6_micro_legibility_stop.json "
            "in the code repository, which was committed before this bundle existed."
        ),
        "cross_check_receipt": (
            "runs_manifest/m6_micro_legibility_stop.json — the gate's own stop record; all six "
            "hashes here were verified against it before staging and again after copying"
        ),
        "n_files": len(files),
        "bytes_total": total,
        "mib_total": round(total / 2**20, 2),
        "all_hashes_verify_against_m6_micro_legibility_stop": all(
            r["verifies"] for r in files.values()
        ),
        "all_copies_verify": all(r["copy_verifies"] for r in files.values()),
        "staged_at": display_path(args.out, REPO),
        "cells": by_cell,
        "UPLOAD_IS_THE_OPERATOR'S": (
            "This script stages and stops. It reads no credential and performs no network call."
        ),
    }
    try:
        write_receipt(MANIFEST, doc, measured=files, force=args.force)
    except ReceiptRefused as e:
        print(e, file=sys.stderr)
        return 2

    (args.out / BUNDLED_MANIFEST).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"staged {len(files)} files, {total / 2**20:.1f} MiB -> {display_path(args.out, REPO)}")
    print(f"  + {BUNDLED_MANIFEST} written INTO the bundle, so one upload carries everything")
    verified = doc["all_hashes_verify_against_m6_micro_legibility_stop"]
    print(f"  all hashes verify against the stop receipt: {verified}")
    print(f"  all copies verify after staging:            {doc['all_copies_verify']}")
    print(f"manifest: {display_path(MANIFEST, REPO)}")
    print("UPLOAD IS THE OPERATOR'S. This script did not and will not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
