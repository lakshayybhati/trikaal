"""§7 v1.4.6 item 2 — ATTRIBUTE the run-to-run model divergence ($0, pure read).

    PYTHONPATH=src .venv/bin/python scripts/m6_divergence_attribution.py

THE QUESTION. The acceptance and money-leg runs have byte-identical recipes and the same seed, yet
their cell4 models differ enormously (frac_negative 0.5662 vs 0.99883; IR +15.008 vs -3.4258 at the
same h, cap and estimator). Is that hardware, or a determinism defect?

THE PRE-DECLARED READING (supervisor):
  * DIFFERENT GPU model or different attention mode -> hardware explains it, invariant 7 holds
    (it scopes bit-exactness to same-hardware + deterministic attention); the residual risk is a
    cross-hardware training lottery, and the v1.4.6 per-seed degeneracy guard is its mitigation.
  * SAME GPU model AND sdpa_deterministic on both -> a genuine nondeterminism defect; report and
    STOP, because determinism is a stated deliverable and that would block the 15-day run.
  * If the logs cannot settle it -> SAY SO PLAINLY rather than inferring. (The lesson of C1.)

This script gathers the evidence and applies that reading. It does not guess.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path("runs_manifest/m6_divergence_attribution.json")

RUNS = {
    "acceptance": {
        "commit": "f2a4792",
        "setup_log": "runs_cloud/acceptance/setup.log",
        "stage2_manifest": "runs_manifest/m6_acceptance_stage2_manifest.json",
    },
    "moneyleg": {
        "commit": "a441cfa",
        "setup_log": "runs_cloud/moneyleg/setup.log",  # EXPECTED ABSENT — that is the finding
        "stage2_manifest": "runs_manifest/m6_moneyleg_rerun_manifest.json",
    },
}

GPU_RE = re.compile(r"NVIDIA GeForce RTX \d{4}|NVIDIA (?:A100|H100|L40S?|A6000)[\w-]*", re.I)
DRIVER_RE = re.compile(r"Driver Version:\s*([\d.]+)")
CUDA_RE = re.compile(r"CUDA Version:\s*([\d.]+)")
TORCH_RE = re.compile(r"torch[= ]+(\d[\w.+]*)")
NUMPY_RE = re.compile(r"numpy[= ]+(\d[\w.+]*)")
ATTN_RE = re.compile(r"sdpa_deterministic|flash2", re.I)


def _scan(path: Path) -> dict:
    if not path.exists():
        return {"present": False}
    txt = path.read_text(errors="replace")

    def one(rx, group=0):
        m = rx.search(txt)
        return m.group(group) if m else None

    return {
        "present": True,
        "gpu_model": one(GPU_RE),
        "driver_version": one(DRIVER_RE, 1),
        "cuda_version_smi": one(CUDA_RE, 1),
        "torch": one(TORCH_RE, 1),
        "numpy": one(NUMPY_RE, 1),
        "attention_mode": one(ATTN_RE),
    }


def _determinism_posture() -> dict:
    """The code-established half of the answer: is forced determinism ever engaged?"""
    orch = Path("src/trikaal/train/orchestrator.py").read_text()
    canary = Path("scripts/m6_canary.py").read_text()
    calls = {
        "orchestrator (the REAL 15-day run path)": re.findall(r"set_determinism\([^)]*\)", orch),
        "scripts/m6_canary.py (the fixture path)": re.findall(r"set_determinism\([^)]*\)", canary),
    }
    forced = {
        k: all("deterministic_algorithms=False" in c for c in v) is False for k, v in calls.items()
    }
    return {
        "set_determinism_calls": calls,
        "forced_determinism_anywhere": any(forced.values()),
        "meaning": (
            "set_determinism(..., deterministic_algorithms=False) leaves "
            "torch.use_deterministic_algorithms UNSET, CUBLAS_WORKSPACE_CONFIG UNSET and "
            "cudnn.deterministic UNSET. CUDA reduction order is then unconstrained, so two runs "
            "of the same recipe on the SAME hardware are NOT expected to agree bit-for-bit."
        ),
    }


def _bit_exact_contradiction() -> dict:
    """Empirical proof that the recorded bit-exactness claim is unsound as written."""
    p = Path("runs_manifest/m6_rehearsal_manifest.json")
    if not p.exists():
        return {"checked": False}
    recs = json.loads(p.read_text()).get("determinism_by_run", {})
    bad = {
        k: {
            "bit_exact_claim": v.get("bit_exact_claim"),
            "attention_mode": v.get("attention_mode"),
            "deterministic_algorithms": v.get("deterministic_algorithms"),
            "device": v.get("device"),
        }
        for k, v in recs.items()
        if v.get("bit_exact_claim") and not v.get("deterministic_algorithms")
    }
    return {
        "checked": True,
        "source": str(p),
        "n_runs_total": len(recs),
        "n_runs_claiming_bit_exact_without_forced_determinism": len(bad),
        "examples": dict(list(bad.items())[:3]),
    }


def main() -> int:
    provenance = {
        name: {
            "commit": meta["commit"],
            "setup_log": meta["setup_log"],
            "from_setup_log": _scan(Path(meta["setup_log"])),
            "stage2_manifest_records": sorted(
                k
                for k in json.loads(Path(meta["stage2_manifest"]).read_text())["recipe"]
                if k in ("device", "seed", "attention_mode", "gpu", "torch", "numpy")
            ),
        }
        for name, meta in RUNS.items()
    }
    wandb_dirs = (
        sorted(Path("runs_cloud/wandb").glob("offline-run-*"))
        if Path("runs_cloud/wandb").exists()
        else []
    )
    wandb_dates = sorted(
        {d.name.split("-")[2][:8] for d in wandb_dirs if len(d.name.split("-")) > 2}
    )

    a = provenance["acceptance"]["from_setup_log"]
    m = provenance["moneyleg"]["from_setup_log"]
    gpu_settled = bool(a.get("gpu_model") and m.get("gpu_model"))
    attn_settled = bool(a.get("attention_mode") and m.get("attention_mode"))
    posture = _determinism_posture()
    contradiction = _bit_exact_contradiction()

    if gpu_settled and attn_settled:
        branch = "A or B — both fields recovered; compare them directly."
    elif not posture["forced_determinism_anywhere"]:
        branch = (
            "NEITHER pre-declared branch as written. The GPU comparison CANNOT be settled from the "
            "receipts (the money-leg run has no setup log at all), so branch A/B cannot be "
            "adjudicated on hardware. But the question is answered from CODE instead of logs: "
            "forced determinism is NEVER engaged in either the fixture path or the REAL 15-day "
            "orchestrator path, so run-to-run divergence is EXPECTED even on identical hardware. "
            "That lands on branch B's DISPOSITION (a determinism-deliverable defect, blocking for "
            "the 15-day run) by a different and code-established mechanism than branch B assumed: "
            "SDPA attention was never the weak link -- the missing "
            "torch.use_deterministic_algorithms is."
        )
    else:
        branch = "Unsettled: forced determinism is engaged somewhere; inspect the calls."

    out = {
        "receipt": "m6_divergence_attribution",
        "amendment": "prereg s7 v1.4.6 item 2 (supervisor-ordered)",
        "question": (
            "acceptance vs money-leg cell4: frac_negative 0.5662 vs 0.99883 and IR +15.008 vs "
            "-3.4258 under byte-identical recipes and the same seed. Hardware, or a defect?"
        ),
        "run_provenance": provenance,
        "wandb_coverage": {
            "offline_run_dates": wandb_dates,
            "covers_acceptance_or_moneyleg": False if wandb_dates == ["20260719"] else None,
            "note": (
                "The only wandb offline runs are from the 2026-07-19 toy rehearsal, and even those "
                "record gpu: None. wandb does not cover either run in question."
            ),
        },
        "checkpoint_metadata": (
            "The .pt files carry only {format, config, hash, state_dict} — model config and a "
            "weight hash, no environment, no GPU, no attention mode."
        ),
        "gpu_comparison_settled": gpu_settled,
        "attention_mode_comparison_settled": attn_settled,
        "determinism_posture_from_code": posture,
        "bit_exact_claim_contradiction": contradiction,
        "pre_declared_reading_applied": branch,
        "blocking_for_the_real_run": bool(not posture["forced_determinism_anywhere"]),
        "what_the_builder_did_NOT_do": (
            "The determinism posture was NOT changed. Enabling forced deterministic algorithms "
            "slows CUDA training and the 2-3 GPU-day / $20-30 budget was measured WITHOUT it, so "
            "the throughput/cost trade is a supervisor decision. bit_exact_claim was likewise NOT "
            "redefined; a bit_exact_preconditions_met field plus a caveat were ADDED beside it so "
            "the contradiction is self-announcing (attention_mode.determinism_record)."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))

    print("=== run provenance ===")
    for name, p in provenance.items():
        s = p["from_setup_log"]
        print(
            f"  {name:<11} log_present={s['present']!s:<5} gpu={s.get('gpu_model')} "
            f"driver={s.get('driver_version')} torch={s.get('torch')} "
            f"attn={s.get('attention_mode')}"
        )
        print(f"              stage2 manifest records: {p['stage2_manifest_records']}")
    print(f"\ngpu comparison settled: {gpu_settled}   attention settled: {attn_settled}")
    print("\n=== determinism posture (code) ===")
    for k, v in posture["set_determinism_calls"].items():
        print(f"  {k}: {v}")
    print(f"  forced determinism anywhere: {posture['forced_determinism_anywhere']}")
    c = contradiction
    if c.get("checked"):
        print(
            f"\nbit_exact_claim contradiction: "
            f"{c['n_runs_claiming_bit_exact_without_forced_determinism']}/{c['n_runs_total']} "
            f"committed runs claim bit-exactness with forced determinism OFF"
        )
    print(f"\nREADING: {branch}")
    print(f"BLOCKING for the real run: {out['blocking_for_the_real_run']}")
    print(f"[receipt] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
