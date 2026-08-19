"""§7 v1.6.26 (P2 finding F3) — A PLACEHOLDER IDENTITY KEY PROTECTS NOTHING WHILE APPEARING TO.

P2 rented a real RTX 4090 whose `nvidia-smi` reported driver **580.159.03** and the provenance
stamp came back `driver_version: "unavailable"` — `torch._C._cuda_getDriverVersion` no longer
behaves as assumed at torch 2.12.1, and the lookup failed into its own placeholder.

**This is the R6 class recurring inside the fix for R6.** R6's whole argument is that an identity
key exists to make shards DISAGREE when they differ. A key that reads the same `"unavailable"` on
all 25 units can never disagree, so it contributes nothing to the refusal while being counted
among the 16 that make it sound complete.

Two halves, and the second is the one that generalizes:
  1. the lookup is fixed — `nvidia-smi` is the driver's own reporter and is tried FIRST;
  2. **the assertion that would have caught it exists now**, on a CUDA device, at two gates: the
     money driver's pre-spend check and `write_cell_eval_artifact`'s durable one.

CPU is EXEMPT by construction: the CUDA keys are genuinely unresolvable there, and recording
`"unavailable"` is the honest value the C-18 rule asks for.
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.eval.verdict import (
    DSR_HORIZONS,
    DSR_KAPPAS,
    PRIMARY_H,
    VerdictInputError,
    write_cell_eval_artifact,
)
from trikaal.utils.provenance import (
    PROVENANCE_IDENTITY_KEYS,
    identity_placeholder_failures,
    run_provenance,
)

# the real values P2 observed on the box, minus the one that failed
CUDA_PROV = {
    "image": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
    "git_commit": "532a6691426e5b644109da32461ef269d573c501",
    "steps_stage1": 26003,
    "steps_stage2": 26003,
    "lockfile_sha256": "bd1297e1ba67afafc5c3edb3559df2fcdd17f42f1b715b914c3aa69f0fe2dff4",
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "cuda_build": "13.0",
    "driver_version": "580.159.03",
    "device": "cuda",
    "torch": "2.12.1+cu130",
    "numpy": "2.4.6",
    "python": "3.11.10",
    "platform": "Linux-5.15.0-181-generic-x86_64-with-glibc2.35",
    "platform_abi": "x86_64-with-glibc2.35",
    "attention_mode": "sdpa_deterministic",
    "deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
}
MU = {"frac_negative": 0.5, "activity_decisions": 0.5}
RECON = {"ohlcv_recon_mae": 0.10}
DECODE = {"sign_agreement_dim0": 0.95}
CODEBOOK = {"coarse": {"utilization": 0.99}, "fine": {"utilization": 0.98}}


# ------------------------------------------------------------------ DISCRIMINATION FIRST
def test_a_fully_resolved_cuda_stamp_passes():
    """If this failed, every refusal below would be vacuous."""
    assert identity_placeholder_failures(CUDA_PROV) == []


def test_cpu_is_exempt_because_those_keys_are_genuinely_unresolvable():
    cpu = dict(CUDA_PROV, device="cpu", gpu_name="unavailable", driver_version="unavailable")
    assert identity_placeholder_failures(cpu) == []
    # and the real CPU stamp on this host must pass for the same reason
    assert identity_placeholder_failures(run_provenance("cpu")) == []


# ------------------------------------------------------------------ the refusal
@pytest.mark.parametrize("key", PROVENANCE_IDENTITY_KEYS)
@pytest.mark.parametrize("placeholder", ["unavailable", "unknown"])
def test_ANY_placeholder_identity_key_on_CUDA_is_refused(key, placeholder):
    fails = identity_placeholder_failures(dict(CUDA_PROV, **{key: placeholder}))
    assert fails and any(key in f for f in fails)


def test_the_EXACT_P2_observation_is_refused():
    """The literal stamp P2 brought back: 15 keys real, `driver_version` a placeholder."""
    fails = identity_placeholder_failures(dict(CUDA_PROV, driver_version="unavailable"))
    assert len(fails) == 1 and "driver_version" in fails[0]


def test_the_artifact_writer_REFUSES_a_placeholder_cuda_unit(tmp_path):
    """The durable gate — a unit that would assemble into a verdict cannot carry one."""
    with pytest.raises(VerdictInputError, match="driver_version"):
        write_cell_eval_artifact(
            tmp_path,
            cell_id=1,
            seed=0,
            grid={"h": PRIMARY_H, "start_ms": 0, "n_periods": 8},
            headline_series=np.zeros(8),
            kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
            val_ir_by_kappa_by_h={h: {k: 0.4 for k in DSR_KAPPAS} for h in DSR_HORIZONS},
            mu_diag=MU,
            ohlcv_recon=RECON,
            decode_agreement=DECODE,
            codebook=CODEBOOK,
            meta={"provenance": dict(CUDA_PROV, driver_version="unavailable")},
        )


def test_the_same_unit_with_a_REAL_driver_writes(tmp_path):
    """Discrimination on the writer too: the refusal must be the placeholder, nothing else."""
    name, _sha = write_cell_eval_artifact(
        tmp_path,
        cell_id=1,
        seed=0,
        grid={"h": PRIMARY_H, "start_ms": 0, "n_periods": 8},
        headline_series=np.zeros(8),
        kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
        val_ir_by_kappa_by_h={h: {k: 0.4 for k in DSR_KAPPAS} for h in DSR_HORIZONS},
        mu_diag=MU,
        ohlcv_recon=RECON,
        decode_agreement=DECODE,
        codebook=CODEBOOK,
        meta={"provenance": CUDA_PROV},
    )
    assert name == "cell1_seed0_eval.json"


def test_the_driver_lookup_prefers_nvidia_smi_AND_ASKS_IT_THE_RIGHT_QUESTION(monkeypatch):
    """The fix itself: `nvidia-smi` is the driver's own reporter and is tried FIRST, so a torch
    private symbol that changes behaviour cannot silently produce a placeholder again.

    **§7 v1.6.26 — THE MUTATION HARNESS REJECTED THE FIRST VERSION OF THIS TEST, AND IT WAS
    RIGHT.** That version stubbed `subprocess.run` with `lambda *a, **k: _Ok()` and asserted only
    on the return value, so it never looked at what was ASKED. Corrupting the query field to
    `--query-gpu=NOTHING` left it green: a test that mocks the call it is verifying, and checks
    only the mock's own answer, verifies the mock. The argv is the contract — a real `nvidia-smi`
    given the wrong field prints an error and the lookup falls back to the placeholder, which is
    precisely the F3 failure. So the argv is asserted."""
    import subprocess

    import trikaal.utils.provenance as P

    seen: list[list[str]] = []

    class _Ok:
        returncode = 0
        stdout = "580.159.03\n"

    def _run(argv, *a, **k):
        seen.append(list(argv))
        return _Ok()

    monkeypatch.setattr(subprocess, "run", _run)
    assert P._driver_version() == "580.159.03"
    assert seen, "the lookup never called out to nvidia-smi at all"
    assert seen[0][0] == "nvidia-smi"
    assert "--query-gpu=driver_version" in seen[0], (
        f"nvidia-smi was asked {seen[0]!r} — the wrong field makes it print an error and the "
        "lookup falls back to the placeholder, which IS the F3 defect"
    )
    assert "--format=csv,noheader" in seen[0]


def test_a_failing_nvidia_smi_does_not_crash_the_stamp(monkeypatch):
    """Fail-soft is correct HERE — the value becomes the honest placeholder, and the SEPARATE
    assertion above is what refuses it on a CUDA device. Recording a placeholder and refusing to
    run on one are two different jobs; conflating them is how the original defect hid."""
    import subprocess

    import trikaal.utils.provenance as P

    def _boom(*a, **k):
        raise OSError("no nvidia-smi here")

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.delattr(P.torch._C, "_cuda_getDriverVersion", raising=False)
    assert P._driver_version() == "unavailable"
