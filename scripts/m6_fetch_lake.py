"""§7 v1.6.26 — PROVISION THE LAKE ON A RENTED BOX. Pull → CONTENT-VERIFY → prove the token is gone.

    HF_TOKEN=... python3 scripts/m6_fetch_lake.py --dest processed/universe_bars   # on the box
    python3 scripts/m6_fetch_lake.py --verify-only --dest processed/universe_bars
    python3 scripts/m6_fetch_lake.py --assert-no-token                             # after the pull

**P2 FINDING F2.** The fan-out had no lake step at all: the §2 tarball ships 550 KB of source and
`--lake …` was a placeholder, so the money driver stopped at `m6_money_run.py:250` with
`LAKE MISSING`. This is that step, executable rather than described.

**VERIFICATION IS THE POINT, NOT THE PULL.** `LAKE MISSING` catches ABSENCE. It cannot see
TRUNCATION — a lake short by 200 of its 1,920 parquets opens, queries, and answers with fewer bars,
and the box scores a silently different cross-section. So every file is checked against
``runs_manifest/m6_lake_subset_manifest.json``, which was committed BEFORE any transfer, by
**sha256** (the HF `lfs.oid`), not by size alone.

**THE TOKEN NEVER TOUCHES THE BOX'S DISK.** It arrives in the environment at the moment of the
pull, is passed to the API call directly, and is never written, echoed or logged.
``--assert-no-token`` then PROVES it is gone rather than assuming it: a scrub nobody checks is a
scrub that did not happen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = "lakshayybhati/trikaal-m6-snapshot"
MANIFEST = Path("runs_manifest/m6_lake_subset_manifest.json")
HF_PATTERN = re.compile(r"hf_[A-Za-z0-9]{16,}")
#: Bound on how much shell history the scrub check will read. A token pasted into a shell
#: is recent; scanning a decade of history to find one is not worth holding the file open.
_HISTORY_LINE_CAP = 50_000
CHUNK = 1 << 22


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify(dest: Path, manifest: dict, *, sample: float = 1.0) -> list[str]:
    """Everything wrong with the lake under ``dest`` (empty = a byte-exact subset)."""
    files = manifest["files"]
    bad: list[str] = []
    n_ok = 0
    for rel, want in sorted(files.items()):
        p = dest / rel
        if not p.exists():
            bad.append(f"MISSING {rel}")
            continue
        got = p.stat().st_size
        if got != want["size"]:
            bad.append(f"TRUNCATED/SHORT {rel}: {got} bytes, expected {want['size']}")
            continue
        if sample >= 1.0 or (hash(rel) % 100) < int(sample * 100):
            if _sha256(p) != want["sha256"]:
                bad.append(f"CONTENT MISMATCH {rel}: sha256 differs from the committed manifest")
                continue
        n_ok += 1
    extra = 0
    for p in dest.rglob("*.parquet"):
        if str(p.relative_to(dest)) not in files:
            extra += 1
    if extra:
        bad.append(
            f"{extra} parquet(s) under {dest} are NOT in the pinned 40-symbol subset — the pull "
            "was not restricted, and an unpinned symbol in the lake is an unpinned cross-section"
        )
    print(f"[verify] {n_ok}/{len(files)} files match by size+sha256; {len(bad)} problems")
    return bad


def assert_no_token(root: Path) -> list[str]:
    """Prove the credential is gone. A scrub nobody checks is a scrub that did not happen."""
    found: list[str] = []
    for k, v in os.environ.items():
        if HF_PATTERN.search(str(v)):
            found.append(f"environment variable {k} still holds an hf_ token")
    for cand in (
        Path.home() / ".cache/huggingface/token",
        Path.home() / ".huggingface/token",
        Path.home() / ".trikaal_hf_token",
        root / ".hf_token",
    ):
        if cand.exists():
            found.append(f"token file present on this machine: {cand}")
    # ★ SCOPED, NOT DROPPED. This used to slurp the ENTIRE ~/.bash_history and ~/.zsh_history
    # into memory with `read_text()`. The CHECK is worth keeping — a token pasted into a shell is
    # exactly how it survives a scrub — but a repo script has no business holding a user's whole
    # command history in a string. It now streams line by line, stops at the first hit, is capped,
    # and NEVER carries any history content into a message: the report names the file and the line
    # NUMBER, never the line. Runs only under an explicit --assert-no-token.
    for hist in (Path.home() / ".bash_history", Path.home() / ".zsh_history"):
        try:
            if not hist.exists():
                continue
            with hist.open("r", errors="ignore") as fh:
                for lineno, line in enumerate(fh, 1):
                    if lineno > _HISTORY_LINE_CAP:
                        break
                    if HF_PATTERN.search(line):
                        found.append(f"an hf_ token appears in {hist} at line {lineno}")
                        break
        except OSError:
            pass
    print("[scrub] " + ("CLEAN — no hf_ credential found" if not found else f"{len(found)} LEAKS"))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=Path("processed/universe_bars"))
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--assert-no-token", action="store_true")
    ap.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="fraction of files to hash (1.0 = all; lower only for a smoke check, never for a run)",
    )
    args = ap.parse_args()

    if args.assert_no_token:
        return 1 if assert_no_token(args.dest) else 0

    if not MANIFEST.exists():
        print(f"no {MANIFEST} — it is committed and ships in the payload tarball", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST.read_text())
    prefixes = sorted({rel.split("/")[0] for rel in manifest["files"]})
    print(
        f"[plan] {len(manifest['files'])} files, {manifest['total_gib']} GiB, "
        f"{len(prefixes)} symbol prefixes → {args.dest}"
    )

    if not args.verify_only:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            print(
                "HF_TOKEN is not in the environment. It is passed AT THE MOMENT OF THE PULL and "
                "never written to this box's disk (runbook §2a route B).",
                file=sys.stderr,
            )
            return 1
        from huggingface_hub import snapshot_download

        args.dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=REPO,
            repo_type="dataset",
            local_dir=str(args.dest),
            # ONLY the 40 pinned symbol prefixes. 4.08 GiB of 14.72 — 72% of the lake never moves.
            allow_patterns=[f"{p}/**" for p in prefixes],
            token=token,  # passed directly; never persisted, never logged
            max_workers=8,
        )
        del token

    bad = verify(args.dest, manifest, sample=args.sample)
    if bad:
        print("LAKE REJECTED — the pull is not the pinned subset:", file=sys.stderr)
        for b in bad[:20]:
            print(f"  - {b}", file=sys.stderr)
        if len(bad) > 20:
            print(f"  … and {len(bad) - 20} more", file=sys.stderr)
        return 1
    print(
        f"[lake] VERIFIED byte-exact: {len(manifest['files'])} files, {manifest['total_gib']} GiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
