"""Build the weights-release manifest: what ships, what it hashes to, and what it is NOT.

WHY. §8.9 of the paper promises "the trained weights" and the repository contains no release path
at all — no manifest, no uploader, no fetcher, and no statement of what a released bundle would
even consist of. The weights exist only under gitignored `runs_cloud/`, so a fresh clone cannot run
the demo and a reader cannot check a single parameter count. "Specified but not enforced", at the
release boundary.

WHAT THIS EMITS. One tracked JSON naming every file in the release bundle with its SHA-256, byte
count, and the role it plays, plus the recomputed parameter arithmetic for each predictor. The
manifest is the contract: a downloader hashes what they received and compares. It is deliberately
NOT an uploader — pushing to HuggingFace is an operator action with credentials attached, and the
one thing this project has learned about operator actions is that a script which performs them
unattended is how the wrong thing gets published.

★ WHAT THE BUNDLE IS NOT, AND THIS TRAVELS INSIDE THE MANIFEST. Three units, all CELL 1 — BSQ
quantizer, OHLCV-only inputs. That is the BASELINE arm. There is no microstructure arm to release,
because the legibility gate fired on real data before Stage 2: Stage 2 was never entered for
cells 2-5 and none of them produced a scored artifact.
A reader who receives these weights and reads the paper's headline claim must not be able to infer
that these are the artifact of the ablation; they are the artifact of the arm that ran.

    .venv/bin/python scripts/m6_weights_release.py
    .venv/bin/python scripts/m6_weights_release.py --verify   # re-hash every file, write nothing
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

OUT = REPO / "runs_manifest" / "m6_weights_release.json"

# The three complete cell-1 money units. Seeds 0, 2, 4 — the shard labels r0/r1/r2 carried those
# seeds, and renaming them 0/1/2 would misidentify which runs these are.
UNITS: dict[int, str] = {
    0: "runs_cloud/rescue/r0/cell1_bsq_ohlcv_seed0",
    2: "runs_cloud/rescue/r1/cell1_bsq_ohlcv_seed2",
    4: "runs_cloud/rescue/r2/cell1_bsq_ohlcv_seed4",
}
ROLES = {
    "tokenizer.pt": "Stage-1 FSQ/BSQ tokenizer autoencoder — REQUIRED by the demo",
    "predictor.pt": "Stage-2 AR backbone + MTP heads — REQUIRED by the demo",
    "stage1_state.pt": "Stage-1 optimizer/scheduler state — resume only, not needed for inference",
    "stage2_state.pt": "Stage-2 optimizer/scheduler state — resume only, not needed for inference",
    "run_manifest.json": "the unit's own provenance stamp (environment, pins, hashes)",
}
INFERENCE_ONLY = ("tokenizer.pt", "predictor.pt", "run_manifest.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def param_groups(path: Path) -> dict:
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
    }


def build() -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()

    units: dict[str, dict] = {}
    total_bytes = inference_bytes = 0
    for seed, rel in UNITS.items():
        d = REPO / rel
        if not d.is_dir():
            raise SystemExit(f"unit directory missing: {rel}")
        files = {}
        for name, role in ROLES.items():
            p = d / name
            if not p.exists():
                raise SystemExit(f"{rel}: missing {name}")
            n = p.stat().st_size
            total_bytes += n
            if name in INFERENCE_ONLY:
                inference_bytes += n
            files[name] = {
                "sha256": sha256(p),
                "bytes": n,
                "role": role,
                "needed_for_inference": name in INFERENCE_ONLY,
            }
        units[str(seed)] = {
            "source_path": rel,
            "cell": 1,
            "quantizer": "bsq",
            "arm": "ohlcv",
            "files": files,
            "predictor_parameter_groups": param_groups(d / "predictor.pt"),
        }

    return {
        "schema": "m6_weights_release_v1",
        "script": "scripts/m6_weights_release.py",
        "git_head": head,
        "license_code": "Apache-2.0 (LICENSE)",
        "license_weights": "Apache-2.0 — same terms as the code; Binance source data is NOT "
        "redistributed, only the pipeline and its content hashes (M8 gate, docs/ROADMAP.md)",
        "WHAT_THIS_IS": (
            "Three trained units from CELL 1 of the M6 design: BSQ quantizer, OHLCV-only inputs, "
            "seeds 0/2/4. Each unit is a Stage-1 tokenizer plus a Stage-2 AR backbone with MTP "
            "heads. THE DRAW IS NOT THE LAKE: they were trained on a 40-symbol draw of 832,096 "
            "windows x 512 bars = 84,153,600 bars (5.06 passes at 26,003 steps x batch 32), drawn "
            "FROM the 200-symbol / 304,625,181-bar universe lake (Merkle sha256:5dfd667d...) -- "
            "27.6% of it. See .training_draw below."
        ),
        "★_WHAT_THIS_IS_NOT": (
            "NOT the artifact of the 2x2 ablation, and NOT a microstructure-aware tokenizer. Cell "
            "1 is the BASELINE arm. The micro-legibility gate fired on real data on 2026-08-12 "
            "and §7 v1.5 item E took effect: the ablation does not run as designed, Stage 2 was "
            "never entered for cells 2-5 and none of them produced a scored artifact (cells 4 and "
            "5 did run through Stage 1, and that run produced the legibility receipt), and the "
            "paper's primary result is the MECHANISM finding: the tokenizer keeps the "
            "microstructure that DUPLICATES OHLCV and DROPS THE SIGNED CHANNELS. 97.3% of CELL "
            "4's shortfall (one seed; the 18 stored calibration replicates in "
            "runs_manifest/m6_lambda_sweep.json span 77-93%) sits on TFI and "
            "signed_count_imbalance, while the four magnitude channels -- the ones that co-vary "
            "with VOLUME, not with price -- essentially clear the gate. These weights demonstrate "
            "the measurement vehicle, not the claim."
        ),
        "training_draw": {
            "n_symbols": 40,
            "n_windows": 832096,
            "seq_len": 512,
            "n_bars": 84153600,
            "n_bars_note": "DISTINCT bars in the 40 drawn symbols (each carries the full "
            "2,103,840-bar span); ledger sum over draw.drawn_by_symbol_stage1",
            "n_bar_positions_consumed": 426033152,
            "n_bar_positions_note": "n_windows x seq_len = 26,003 steps x batch 32 x 512; "
            "this is TRAINING THROUGHPUT, not the size of the draw",
            "passes_over_the_draw": 5.0625659745988285,
            "share_of_the_200_symbol_lake": 0.2762530689418587,
            "read_it_back_from": "draw.drawn_by_symbol_stage1 in each unit's "
            "run_manifest.json -- identical across seeds 0/2/4",
            "window": ["2021-01-01", "2025-01-01"],
            "train_frac": 0.7,
            "train_through": "2023-10-20T16:48:00+00:00",
            "held_out": "2023-10-20T16:48:00+00:00 -> 2025-01-01, i.e. ALL OF 2024 is "
            "out-of-sample",
        },
        "how_to_verify": (
            "sha256sum each file and compare against .units[seed].files[name].sha256; then load "
            "predictor.pt and sum numel() over state_dict to reproduce n_params_realized_total, "
            "and over keys starting 'mtp.' for n_params_mtp."
        ),
        "minimum_inference_bundle": list(INFERENCE_ONLY),
        "bytes_total": total_bytes,
        "bytes_inference_only": inference_bytes,
        "byte_verification_receipt": "runs_manifest/m6_rescue_inventory.json — each file's "
        "on-box SHA-256 vs its local SHA-256 after transfer, all matching",
        "units": units,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="re-hash every file; write nothing")
    args = ap.parse_args()

    doc = build()
    if args.verify:
        if not OUT.exists():
            print(f"RELEASE MANIFEST ABSENT: {display_path(OUT, REPO)}", file=sys.stderr)
            return 2
        old = json.loads(OUT.read_text())
        if old.get("units") != doc["units"]:
            print("RELEASE MANIFEST STALE — re-hashed files differ from the committed manifest")
            return 2
        n = sum(len(u["files"]) for u in doc["units"].values())
        print(f"RELEASE MANIFEST VERIFIED — {n} files re-hashed, all identical")
        return 0

    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(
        f"{len(doc['units'])} units, {doc['bytes_total'] / 1e9:.2f} GB total "
        f"({doc['bytes_inference_only'] / 1e9:.2f} GB inference-only)"
    )
    print(f"Manifest: {display_path(OUT, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
