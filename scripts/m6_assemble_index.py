"""GATE 0 — build the eval index OFF-BOX, from artifacts pulled per unit. $0, local.

    python3 scripts/m6_assemble_index.py --art-dir runs_cloud/m6/eval \
        --onbox-sha runs_cloud/m6/onbox.sha

**WHY THIS EXISTS.** ``write_index_if_complete`` writes ``index.json`` only when ONE invocation
holds every unit (``len(entries) >= 25``). Under 25 sequential ``--shard i/25`` invocations — and
under the original 5-box fan-out equally — no single invocation ever does, so the index is
withheld every time and ``load_cell_evals`` hard-fails with *"no index.json ... artifacts must be
indexed"*. Assembly was always going to happen off-box; this is that step, executable rather than
assumed.

**★ THE SHA CHECK IS TAUTOLOGICAL IF THE HASHES COME FROM THE LOCAL FILES.** ``load_cell_evals``
re-hashes each artifact and compares it to the index. If the index were built by hashing the same
local files, that comparison compares a file to itself: it passes on a truncated or corrupted pull
and detects nothing, while LOOKING like content verification. That is the fail-open class, in the
one place the whole fan-out's integrity rests. So the hashes are taken from the BOX — the
``sha256sum`` recorded at production time — and this step REFUSES when a local file disagrees.
``--trust-local`` exists only for fabricated fixtures and says so in the receipt it writes.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from trikaal.eval.verdict import DSR_SEEDS, write_eval_index
from trikaal.train.cells import CELLS


def expected_names() -> set[str]:
    return {f"cell{c.cell_id}_seed{s}_eval.json" for c in CELLS for s in DSR_SEEDS}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_onbox_sha(text: str) -> dict[str, str]:
    """``sha256sum`` output → ``{basename: sha}``. Accepts full paths, keeps the basename."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            out[Path(parts[-1]).name] = parts[0]
    return out


def assembly_failures(art_dir: Path, onbox: dict[str, str] | None) -> list[str]:
    """Everything wrong with this directory (empty = safe to index)."""
    want = expected_names()
    have = {p.name for p in art_dir.glob("cell*_seed*_eval.json")}
    bad: list[str] = []
    for n in sorted(want - have):
        bad.append(f"{n}: MISSING from {art_dir}")
    for n in sorted(have - want):
        seeds = ",".join(map(str, DSR_SEEDS))
        bad.append(f"{n}: UNEXPECTED — outside the pinned 5x{{{seeds}}} matrix")
    if onbox is not None:
        for n in sorted(want & have):
            local = _sha256(art_dir / n)
            ref = onbox.get(n)
            if ref is None:
                bad.append(f"{n}: no on-box sha recorded — cannot prove the pull was faithful")
            elif ref != local:
                bad.append(f"{n}: local sha256 {local[:16]}… != on-box {ref[:16]}… — CORRUPT PULL")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--art-dir", type=Path, required=True)
    ap.add_argument("--onbox-sha", type=Path, help="sha256sum output captured ON THE BOX")
    ap.add_argument(
        "--trust-local",
        action="store_true",
        help="hash the local files instead (FIXTURES ONLY — cannot detect a corrupt pull)",
    )
    args = ap.parse_args(argv)

    if not args.onbox_sha and not args.trust_local:
        print(
            "refusing to index without on-box hashes: an index built from the local files makes "
            "load_cell_evals compare each file to itself. Pass --onbox-sha, or --trust-local "
            "and accept that this proves nothing about the transfer.",
            file=sys.stderr,
        )
        return 2
    onbox = parse_onbox_sha(args.onbox_sha.read_text()) if args.onbox_sha else None

    bad = assembly_failures(args.art_dir, onbox)
    if bad:
        print(f"ASSEMBLY REFUSED — {len(bad)} problem(s):", file=sys.stderr)
        for b in bad:
            print(f"  - {b}", file=sys.stderr)
        return 1

    entries = {
        n: (onbox[n] if onbox else _sha256(args.art_dir / n)) for n in sorted(expected_names())
    }
    path = write_eval_index(args.art_dir, entries)
    print(f"[index] {len(entries)} artifacts → {path}")
    src = "ON-BOX sha256sum" if onbox else "LOCAL hashes — proves NOTHING about the transfer"
    print(f"[index] provenance of hashes: {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
