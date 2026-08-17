"""Mirror the cited runs_cloud/ evidence into a TRACKED receipt, so a fresh clone can check it.

WHY. `runs_cloud/` is gitignored by design — 1.7 GB of checkpoints and per-cell eval JSON is
evidence, not source. But §7.5 and §7.10 of the paper cite it for the numbers a sceptical reader
most wants to check: the execution filter's realized decision activity, and the realized parameter
counts. Local verification passes and a reader of the published repository cannot reproduce a
single one of them. The citation convention breaks exactly at the money-adjacent numbers.

WHAT THIS MIRRORS, and why in full rather than in summary. The three cell-1 eval documents are
copied VERBATIM, `headline_series` included: that array is what every information ratio in the
paper is computed from, so a summary would leave the reader able to read our numbers and unable to
recompute them. Each carries the SHA-256 of its source file, so the mirror can be checked against
the artifact shipped with the weights rather than taken on trust.

The two predictor checkpoints cannot be mirrored — they are 127 MB each. What IS mirrored is the
claim made about them: the parameter-group arithmetic (total, the `mtp.*` group, and the
difference), recomputed here from the tensors rather than transcribed, plus each file's SHA-256 so
the released weights can be matched to the ones these counts came from.

    .venv/bin/python scripts/m6_mirror_cloud_eval.py
    .venv/bin/python scripts/m6_mirror_cloud_eval.py --check   # re-derive and diff, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trikaal.utils.paths import display_path  # noqa: E402

OUT = REPO / "runs_manifest" / "m6_cell1_eval_mirror.json"

# The three scored cell-1 units. Cell 1 = BSQ + OHLCV-only, the only arm with scored artifacts.
EVALS: dict[str, str] = {
    "seed0": "runs_cloud/results/r0/cell1_seed0_eval.json",
    "seed2": "runs_cloud/results/r1/cell1_seed2_eval.json",
    "seed4": "runs_cloud/results/r2/cell1_seed4_eval.json",
}
# The two checkpoints §7.10 names for the realized parameter counts (BSQ arm / FSQ arm).
PREDICTORS: dict[str, str] = {
    "bsq_cell1_seed0": "runs_cloud/rescue/r0/cell1_bsq_ohlcv_seed0/predictor.pt",
    "fsq_cell4_seed0": "runs_cloud/ckpt_cell4_seed0/predictor.pt",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def param_groups(path: Path) -> dict:
    """Recompute the parameter arithmetic FROM THE TENSORS — never transcribe it."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    total = sum(int(t.numel()) for t in sd.values())
    mtp = sum(int(t.numel()) for k, t in sd.items() if k.startswith("mtp."))
    return {
        "n_params_realized_total": total,
        "n_params_mtp": mtp,
        "n_params_backbone_excluding_mtp": total - mtp,
        "stored_artifact_hash": ckpt.get("hash") if isinstance(ckpt, dict) else None,
        "file_sha256": sha256(path),
        "file_bytes": path.stat().st_size,
    }


def build() -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()

    evals: dict[str, dict] = {}
    for name, rel in EVALS.items():
        p = REPO / rel
        if not p.exists():
            raise SystemExit(f"missing cited evidence: {rel}")
        evals[name] = {
            "source_path": rel,
            "source_sha256": sha256(p),
            "source_bytes": p.stat().st_size,
            "document": json.loads(p.read_text()),
        }

    preds: dict[str, dict] = {}
    for name, rel in PREDICTORS.items():
        p = REPO / rel
        if not p.exists():
            raise SystemExit(f"missing cited checkpoint: {rel}")
        preds[name] = {"source_path": rel, **param_groups(p)}

    return {
        "schema": "m6_cell1_eval_mirror_v1",
        "script": "scripts/m6_mirror_cloud_eval.py",
        "git_head": head,
        "what": (
            "A tracked mirror of the runs_cloud/ artifacts the paper cites in sections 7.5 and "
            "7.10. runs_cloud/ is gitignored by design (1.7 GB of evidence, not source), which "
            "left those citations unverifiable from a fresh clone. The eval documents are copied "
            "verbatim including headline_series so every information ratio can be recomputed; the "
            "checkpoints are 127 MB each and are represented by their recomputed parameter "
            "arithmetic plus a SHA-256 that matches them to the released weights."
        ),
        "what_this_is_not": (
            "NOT an independent measurement. These are the same bytes the money run wrote; "
            "mirroring makes them CHECKABLE, not more true. The scored arm is cell 1 (BSQ + "
            "OHLCV-only) and no other cell was ever scored — the legibility gate fired first."
        ),
        "cell1_evals": evals,
        "predictor_parameter_groups": preds,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="re-derive and diff; write nothing")
    args = ap.parse_args()

    doc = build()
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not OUT.exists():
            print(f"MIRROR ABSENT: {display_path(OUT, REPO)}", file=sys.stderr)
            return 2
        old = json.loads(OUT.read_text())
        # git_head moves every commit and is provenance, not content — compare everything else.
        a = {k: v for k, v in old.items() if k != "git_head"}
        b = {k: v for k, v in doc.items() if k != "git_head"}
        if a != b:
            print("MIRROR STALE — re-derived content differs from the committed mirror")
            return 2
        print("MIRROR CURRENT — re-derived content matches the committed mirror")
        return 0

    OUT.write_text(text)
    for name, p in doc["predictor_parameter_groups"].items():
        print(
            f"  {name:16s} total {p['n_params_realized_total']:,}  mtp {p['n_params_mtp']:,}  "
            f"backbone {p['n_params_backbone_excluding_mtp']:,}"
        )
    print(f"\nMirror: {display_path(OUT, REPO)}  ({len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
