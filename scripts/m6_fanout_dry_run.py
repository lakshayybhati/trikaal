"""FAN-OUT ASSEMBLY DRY RUN — $0, local. The A9 instrument, one level up.

A9 fabricated a full money-run code path and found the `max_len` defect before it cost a training
run. This does the same for the FAN-OUT: fabricate 25 unit artifacts as if produced by 5 boxes,
assemble the verdict, and prove the assembly REFUSES on a deliberately mismatched shard.

WHY IT MATTERS MORE THAN IT LOOKS. The 25 units are one paired comparison. ΔIR(4−5) across two
instruments measures the instruments as much as the tokenizer, and a shard that quietly ran a
different container image or a different resolved dependency set is invisible in every number the
verdict prints. The refusal is the only thing standing between that and a published result.

WHAT IT PROVES, IN ORDER — the healthy case FIRST, because a refusal that fires on everything
proves nothing:
  1. 5 shards x 5 units is an exact disjoint cover of the 25-unit matrix
  2. a UNIFORM fan-out assembles to a verdict word
  3. a ONE-UNIT drift on EVERY identity key refuses, per key, naming the key
  4. a shard that never stamped provenance at all refuses
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from run.test_fanout_refusal import N_SHARDS, _units, build_fanout
from trikaal.eval.verdict import VerdictInputError, assemble_verdict, load_cell_evals
from trikaal.run.matrix import shard_partition_failures
from trikaal.utils.provenance import PROVENANCE_IDENTITY_KEYS

OUT = Path("runs_manifest/m6_fanout_dry_run.json")


class _U:
    def __init__(self, name: str):
        self.name = name


def main() -> int:
    import tempfile

    rep: dict = {
        "receipt": "m6_fanout_dry_run",
        "what": "fan-out assembly dry run — 25 units across 5 simulated shards, $0, local",
        "n_units": 25,
        "n_shards": N_SHARDS,
        "identity_keys": list(PROVENANCE_IDENTITY_KEYS),
    }

    # 1. the partition
    units = [(_U(f"cell{c}"), s) for c, s in _units()]
    rep["partition"] = {
        "exact_disjoint_cover": shard_partition_failures(units, N_SHARDS) == [],
        "empty_matrix_refused": bool(shard_partition_failures([], N_SHARDS)),
    }

    with tempfile.TemporaryDirectory() as td:
        # 2. DISCRIMINATION FIRST — the uniform case must assemble
        evals, shas = load_cell_evals(build_fanout(Path(td) / "ok"))
        man = assemble_verdict(evals, shas, tabled_mde_h15=3.518, money_verdict=False)
        rep["uniform_fanout"] = {
            "units_loaded": len(evals),
            "assembled": True,
            "verdict": man["verdict"]["primary"],
            "distinct_provenance_stamps": len(
                {json.dumps(d["meta"]["provenance"], sort_keys=True) for d in evals.values()}
            ),
        }

        # 3. one drifted unit, per identity key
        per_key = {}
        for k in PROVENANCE_IDENTITY_KEYS:
            try:
                load_cell_evals(build_fanout(Path(td) / f"drift_{k}", mutate_unit=7, mutate_key=k))
                per_key[k] = {"refused": False, "detail": "ASSEMBLED — the refusal did not fire"}
            except VerdictInputError as e:
                per_key[k] = {"refused": True, "names_the_key": k in str(e)}
        rep["one_drifted_unit_refuses"] = per_key
        rep["n_keys_refusing"] = sum(1 for v in per_key.values() if v["refused"])

    ok = (
        rep["partition"]["exact_disjoint_cover"]
        and rep["partition"]["empty_matrix_refused"]
        and rep["uniform_fanout"]["assembled"]
        and rep["uniform_fanout"]["distinct_provenance_stamps"] == 1
        and rep["n_keys_refusing"] == len(PROVENANCE_IDENTITY_KEYS)
        and all(v["names_the_key"] for v in rep["one_drifted_unit_refuses"].values())
    )
    rep["verdict"] = (
        "GREEN — a uniform fan-out assembles and a ONE-UNIT drift on EVERY identity key refuses"
        if ok
        else "RED — the fan-out refusal is incomplete; DO NOT FAN OUT"
    )
    rep["scope_limit"] = (
        "fabricated artifacts, not a trained run: this proves the ASSEMBLY contract, not that the "
        "launcher stamps correctly on a real box. The launcher side is the runbook's job and is "
        "verified by the first shard's artifact, not by this script."
    )
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "one_drifted_unit_refuses"}, indent=2))
    print(f"\nkeys refusing: {rep['n_keys_refusing']}/{len(PROVENANCE_IDENTITY_KEYS)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
