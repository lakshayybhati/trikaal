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
    doc = {
        "receipt": "m6_legibility_evidence_manifest",
        "script": "scripts/m6_stage_legibility_evidence.py",
        "what": (
            "The cell-4 and cell-5 Stage-1 artifacts the micro-legibility gate produced before it "
            "refused both arms. The paper's headline (97.3% of cell 4's shortfall) is measured on "
            "cell 4, which exists nowhere public: 0 files in the repository across every commit, "
            "0 of 13 on HuggingFace."
        ),
        "★_THIS_IS_EVIDENCE_NOT_A_MODEL": (
            "There is NO predictor here and none exists on the money path: Stage 2 was never "
            "entered for these cells (stage2_entered=false, artifacts_produced=0), so nothing in "
            "this bundle can forecast. The tokenizers ARE microstructure-arm weights — "
            "quantizer=fsq, n_features=16, micro_point_weight=3.0 — and unit.log is the gate's own "
            "RuntimeError carrying all six per-dim sign accuracies the 97.3% is computed from."
        ),
        "UPLOAD_IS_THE_OPERATOR'S": (
            "This script stages and stops. It reads no credential and performs no network call."
        ),
        "n_files": len(files),
        "bytes_total": total,
        "mib_total": round(total / 2**20, 2),
        "all_hashes_verify_against_m6_micro_legibility_stop": all(
            r["verifies"] for r in files.values()
        ),
        "all_copies_verify": all(r["copy_verifies"] for r in files.values()),
        "staged_at": display_path(args.out, REPO),
        "files": files,
    }
    try:
        write_receipt(MANIFEST, doc, measured=files, force=args.force)
    except ReceiptRefused as e:
        print(e, file=sys.stderr)
        return 2

    print(f"staged {len(files)} files, {total / 2**20:.1f} MiB -> {display_path(args.out, REPO)}")
    verified = doc["all_hashes_verify_against_m6_micro_legibility_stop"]
    print(f"  all hashes verify against the stop receipt: {verified}")
    print(f"  all copies verify after staging:            {doc['all_copies_verify']}")
    print(f"manifest: {display_path(MANIFEST, REPO)}")
    print("UPLOAD IS THE OPERATOR'S. This script did not and will not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
