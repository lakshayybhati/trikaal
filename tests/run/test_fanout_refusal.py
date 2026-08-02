"""§7 v1.6.23 — THE FAN-OUT REFUSAL. 25 units across N shards must be ONE instrument.

The 25 (cell, seed) units are independent, so the run fans out across boxes: 270 GPU-h on one is
~54 on five at the same dollars, and a preemption costs 1/25th instead of everything. **The price
is that five boxes are five chances for an unrecorded difference to sit inside one paired
comparison.** ΔIR(4−5) across two instruments measures the instruments as much as the tokenizer.

So the contract is: **identical GPU model, driver, container image, interpreter and lockfile per
shard, recorded per unit, or the verdict REFUSES TO ASSEMBLE.** These tests assert the refusal on
every identity key, and — because a refusal that cannot fire is worse than none — each one is
mutation-proven: the healthy set must assemble, and the mismatched set must not.
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.eval.verdict import (
    DSR_HORIZONS,
    DSR_KAPPAS,
    DSR_SEEDS,
    PRIMARY_H,
    VerdictInputError,
    load_cell_evals,
    write_cell_eval_artifact,
    write_eval_index,
)
from trikaal.run.matrix import shard_partition_failures
from trikaal.utils.provenance import PROVENANCE_IDENTITY_KEYS

CODEBOOK_OK = {
    "coarse": {"utilization": 0.99, "vocab": 891},
    "fine": {"utilization": 0.98, "vocab": 1225},
}
MU_OK = {"frac_negative": 0.5, "activity_decisions": 0.5, "mu_mean": 0.0, "mu_std": 1.0}
RECON_OK = {"ohlcv_recon_mae": 0.5, "n_bars": 1000}
DECODE_OK = {"sign_agreement_dim0": 0.9, "single_bar_degenerate": False}

PROV = {
    "image": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
    "lockfile_sha256": "bd1297e1ba67afafc5c3edb3559df2fcdd17f42f1b715b914c3aa69f0fe2dff4",
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "cuda_build": "13.0",
    "driver_version": "580159",
    "device": "cuda",
    "torch": "2.12.1+cu130",
    "numpy": "2.4.6",
    "python": "3.11.10",
    "python_executable": "/opt/conda/bin/python3",
    "platform": "Linux-x86_64",
    "attention_mode": "sdpa_deterministic",
    "deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
}
N_SHARDS = 5


def _units():
    """The 25 (cell, seed) units, in the deterministic order the shard partition uses."""
    return [(c, s) for c in (1, 2, 3, 4, 5) for s in DSR_SEEDS]


def _shard_of(idx: int) -> int:
    return idx % N_SHARDS


def build_fanout(tmp_path, *, mutate_unit: int | None = None, mutate_key: str | None = None):
    """Fabricate 25 artifacts as if produced by N_SHARDS boxes. ``mutate_key`` makes ONE unit's
    provenance differ, simulating a shard that drifted."""
    out = tmp_path / "art"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    common = rng.standard_normal(256) * 1e-3
    entries: dict[str, str] = {}
    for i, (c, s) in enumerate(_units()):
        prov = dict(PROV)
        if mutate_unit == i and mutate_key is not None:
            v = prov[mutate_key]
            prov[mutate_key] = (not v) if isinstance(v, bool) else f"{v}-DRIFTED"
        name, sha = write_cell_eval_artifact(
            out,
            cell_id=c,
            seed=s,
            grid={"h": PRIMARY_H, "start_ms": 0, "n_periods": 256},
            headline_series=common + rng.standard_normal(256) * 1e-3,
            kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
            val_ir_by_kappa_by_h={h: {f"{k:g}": 0.4 for k in DSR_KAPPAS} for h in DSR_HORIZONS},
            mu_diag=MU_OK,
            ohlcv_recon=RECON_OK,
            decode_agreement=DECODE_OK,
            codebook=CODEBOOK_OK,
            meta={"provenance": prov, "shard": _shard_of(i), "n_shards": N_SHARDS},
        )
        entries[name] = sha
    write_eval_index(out, entries)
    return out


# ------------------------------------------------------------------ the partition itself
def test_the_shard_partition_is_an_exact_disjoint_cover():
    class U:
        def __init__(self, name):
            self.name = name

    units = [(U(f"cell{c}"), s) for c, s in _units()]
    assert shard_partition_failures(units, N_SHARDS) == []
    assert shard_partition_failures([], N_SHARDS), "an empty matrix is not a covered partition"


# ------------------------------------------------------------------ DISCRIMINATION FIRST
def test_a_uniform_fanout_assembles(tmp_path):
    """If this failed, every refusal below would be vacuous."""
    evals, shas = load_cell_evals(build_fanout(tmp_path))
    assert len(evals) == 25
    assert (
        len({tuple(sorted(d["meta"]["provenance"].items(), key=str)) for d in evals.values()}) == 1
    )


@pytest.mark.parametrize("key", PROVENANCE_IDENTITY_KEYS)
def test_ONE_drifted_unit_on_ANY_identity_key_refuses_the_whole_verdict(tmp_path, key):
    """The mutation proof, per key: 24 units agree, ONE differs, the assembly must REFUSE.

    Unit 7 is on shard 2 — a mid-run drift, not the first box, because an off-by-one in the
    partition would make unit 0 a special case and hide the general failure."""
    with pytest.raises(VerdictInputError, match="provenance"):
        load_cell_evals(build_fanout(tmp_path, mutate_unit=7, mutate_key=key))


def test_the_refusal_NAMES_the_key_and_the_units(tmp_path):
    try:
        load_cell_evals(build_fanout(tmp_path, mutate_unit=7, mutate_key="gpu_name"))
    except VerdictInputError as e:
        msg = str(e)
        assert "gpu_name" in msg and "REFUSES" in msg
        assert "cell2_seed2" in msg or "4090" in msg
    else:
        pytest.fail("a drifted gpu_name did not refuse")


def test_a_unit_with_NO_provenance_refuses(tmp_path):
    """The fan-out case that is not a mismatch but an absence — a shard whose launcher never
    stamped. 'Some units are unattributable' is exactly the state this refuses to certify."""
    out = build_fanout(tmp_path)
    import json

    p = next(out.glob("cell3_seed1_eval.json"))
    doc = json.loads(p.read_text())
    doc["meta"].pop("provenance")
    p.write_text(json.dumps(doc, indent=2, sort_keys=True))
    # the content hash now disagrees with the index too — both refusals are correct; assert the
    # provenance one is reachable by rebuilding the index over the tampered file
    entries = {
        q.name: __import__("hashlib").sha256(q.read_bytes()).hexdigest()
        for q in out.glob("cell*_eval.json")
    }
    write_eval_index(out, entries)
    with pytest.raises(VerdictInputError, match="provenance"):
        load_cell_evals(out)


def test_image_and_lockfile_are_IN_the_identity_surface():
    """§7 v1.6.23: both were missing. Two boxes can carry identical torch/driver strings and
    still differ in container image or resolved dependency set."""
    assert "image" in PROVENANCE_IDENTITY_KEYS
    assert "lockfile_sha256" in PROVENANCE_IDENTITY_KEYS
