"""Pre-teardown gate: verify every pulled artifact against the box's own sha256 manifest.

    PYTHONPATH=src .venv/bin/python scripts/m6_pull_verify.py \
        --local-dir runs_cloud/cuda_probe --sums runs_cloud/cuda_probe/SHA256SUMS.box

WHY THIS IS FAIL-CLOSED. The standing rule exists because the money-leg run's primary artifacts did
not survive teardown. "I rsync'd it and it looked fine" is not verification: a truncated transfer, a
partial write, or a file that was never produced all look like success to a copy command. The box
computes the digests; this recomputes them locally and requires an EXACT set match — every listed
file present and identical, and (reported, not fatal) anything present locally that the box never
listed.

Exit 0 ONLY if every file verifies. Any missing file, any digest mismatch, any unreadable entry →
exit 1 and DO NOT DESTROY THE BOX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

OUT = Path("runs_manifest/m6_teardown_verification.json")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sums(path: Path) -> dict[str, str]:
    """Parse `sha256sum`-style output: '<hex>  <relative/path>'."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"unparseable SHA256SUMS line: {line[:120]!r}")
        digest, name = parts[0], parts[1].lstrip("*").lstrip("./")
        out[name] = digest.lower()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-dir", type=Path, required=True)
    ap.add_argument("--sums", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    expected = parse_sums(args.sums)
    verified, missing, mismatched, unreadable = [], [], [], []
    for name, want in sorted(expected.items()):
        p = args.local_dir / name
        if not p.is_file():
            missing.append(name)
            continue
        try:
            got = sha256_file(p)
        except OSError as e:
            unreadable.append({"file": name, "error": str(e)})
            continue
        (verified if got == want else mismatched).append(
            name if got == want else {"file": name, "expected": want, "got": got}
        )

    listed = set(expected)
    on_disk = {
        str(p.relative_to(args.local_dir))
        for p in args.local_dir.rglob("*")
        if p.is_file() and p.name != args.sums.name
    }
    extra = sorted(on_disk - listed)

    ok = not (missing or mismatched or unreadable)
    rec = {
        "receipt": "m6_teardown_verification",
        "amendment": "standing pre-teardown gate (prereg §7 v1.4.4 rider) — v1.6 CUDA probe",
        "local_dir": str(args.local_dir),
        "sums_file": str(args.sums),
        "n_expected": len(expected),
        "n_verified": len(verified),
        "verified_files": verified,
        "missing": missing,
        "mismatched": mismatched,
        "unreadable": unreadable,
        "present_locally_but_not_listed_by_the_box": extra,
        "all_verified": ok,
        "verdict": (
            "PASS — every artifact the box listed is present locally and byte-identical by "
            "sha256. Teardown is permitted."
            if ok
            else "FAIL — DO NOT DESTROY THE BOX. See missing/mismatched/unreadable."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2, sort_keys=True))
    print(f"=== PRE-TEARDOWN VERIFICATION: {'PASS' if ok else 'FAIL'} ===")
    print(f"  verified {len(verified)}/{len(expected)}")
    for k in ("missing", "mismatched", "unreadable"):
        if rec[k]:
            print(f"  {k}: {rec[k]}")
    if extra:
        print(f"  (not listed by the box, reported only: {len(extra)} file(s))")
    print(f"[receipt] -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
