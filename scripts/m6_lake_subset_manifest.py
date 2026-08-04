"""§7 v1.6.26 — THE EXPECTED-CONTENT MANIFEST for the 40-symbol lake subset a rented box needs.

    PYTHONPATH=src python3 scripts/m6_lake_subset_manifest.py

WHY IT EXISTS. P2 finding F2: the fan-out had no lake-provisioning step, and the driver's only
check is `LAKE MISSING` — **which catches ABSENCE and cannot see TRUNCATION**. A pull that drops
200 of 1,920 parquets leaves a lake that opens, queries, and answers with fewer bars; the money run
would score a silently different cross-section on that box. So the pull is verified against a
manifest committed BEFORE the transfer, not against itself.

**THE VERIFICATION IS CONTENT-EXACT, NOT SIZE-ONLY.** The HF tree API returns `lfs.oid` for every
parquet, and that oid IS the file's sha256 — 7,025 of 7,029 entries carry one. So a truncated,
corrupted or substituted file fails on its hash rather than merely on its length.

The token is read from ``~/.trikaal_hf_token`` at point of use and never printed, logged, or
written into the output.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

REPO = "lakshayybhati/trikaal-m6-snapshot"
TOKEN_PATH = pathlib.Path.home() / ".trikaal_hf_token"
PINS = pathlib.Path("runs_manifest/m6_mde_inputs.json")
OUT = pathlib.Path("runs_manifest/m6_lake_subset_manifest.json")


def _tree() -> list[dict]:
    """Every blob in the dataset repo, with size + lfs oid, following pagination."""
    import tempfile

    tok = TOKEN_PATH.read_text().strip()
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=1&expand=1"
    out: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        hdr = pathlib.Path(td) / "h.txt"
        while url:
            # headers to a FILE, not to stdout: a redirect emits several header blocks and
            # splitting stdout on the first blank line then parses a header block as JSON.
            p = subprocess.run(
                ["curl", "-sSL", "-D", str(hdr), "-H", f"Authorization: Bearer {tok}", url],
                capture_output=True,
                text=True,
                check=True,
            )
            out.extend(json.loads(p.stdout))
            link = [ln for ln in hdr.read_text().splitlines() if ln.lower().startswith("link:")]
            m = re.search(r'<([^>]+)>;\s*rel="next"', link[-1]) if link else None
            url = m.group(1) if m else None
    return [b for b in out if b.get("type") == "file"]


def symbol_of(path: str) -> str | None:
    """The symbol a repo path belongs to, from its PATH COMPONENT.

    **NOT a filename substring.** The layout is
    ``symbol=X/frequency=1m/month=YYYY-MM/data_0.parquet``: the symbol never appears in the
    basename, so a filename glob attributes NOTHING and silently reports an empty subset. That
    exact mistake produced a "0.0 % of lake" reading during the sizing pass.
    """
    for part in path.split("/"):
        m = re.fullmatch(r"symbol=(.+)", part)
        if m:
            return m.group(1)
    return None


def main() -> int:
    if not TOKEN_PATH.exists():
        print(f"no token at {TOKEN_PATH} — see runbook §2a route B", file=sys.stderr)
        return 1
    pins = list(json.loads(PINS.read_text())["symbols_sampled"])
    blobs = _tree()

    files: dict[str, dict] = {}
    for b in blobs:
        if symbol_of(b["path"]) in pins:
            files[b["path"]] = {
                "size": int(b["size"]),
                "sha256": (b.get("lfs") or {}).get("oid"),
            }
    missing_syms = [s for s in pins if not any(symbol_of(p) == s for p in files)]
    no_hash = [p for p, v in files.items() if not v["sha256"]]

    rep = {
        "receipt": "m6_lake_subset_manifest",
        "what": "the EXPECTED content of the 40-symbol lake subset, committed BEFORE any transfer "
        "so a pull is verified against a fixed reference rather than against itself",
        "repo": REPO,
        "repo_kind": "dataset (private, ungated)",
        "n_symbols": len(pins),
        "n_files": len(files),
        "total_bytes": sum(v["size"] for v in files.values()),
        "total_gib": round(sum(v["size"] for v in files.values()) / 2**30, 3),
        "symbols_missing_from_repo": missing_syms,
        "files_without_a_content_hash": no_hash,
        "verification": (
            "sha256 per file (the HF `lfs.oid`), not size alone. `LAKE MISSING` catches ABSENCE; "
            "TRUNCATION is the failure mode it cannot see — a short lake opens, queries and "
            "answers with fewer bars, and the run scores a silently different cross-section."
        ),
        "files": files,
    }
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(
        f"{len(files)} files, {rep['total_gib']} GiB, "
        f"{len(files) - len(no_hash)} with content hashes → {OUT}"
    )
    if missing_syms:
        print(f"★ pinned symbols absent from the repo: {missing_syms}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
