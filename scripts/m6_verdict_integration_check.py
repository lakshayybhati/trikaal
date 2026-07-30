"""§7 v1.4.7 — END-TO-END verdict integration check with BOTH guards armed ($0, CPU).

    PYTHONPATH=src .venv/bin/python scripts/m6_verdict_integration_check.py

WHY THIS EXISTS. Both guards were added to the ONLY code path allowed to emit SURVIVES/NULL, and the
pre-flight verdict dry run PREDATES both (``runs_manifest/m6_toy_verdict_manifest.json`` carries
neither ``degeneracy_guard`` nor ``power_guard``). Unit KATs prove the guard FUNCTIONS behave; they
do not prove the DRIVER still runs, still writes a manifest, still surfaces the HALT, and still
carries the new per-seed fields. That gap is the last place an integration break could hide.

This invokes ``scripts/m6_verdict.py`` as a SUBPROCESS — the real driver, the real §3a conformance
gate, the real content-hash load, the real manifest write — on three toy fixtures:

  * HEALTHY  -> the verdict must still EMIT (no HALT), and the manifest must carry
               frac_negative_by_seed / activity_decisions_by_seed / ir_by_seed /
               ir_range_across_seeds, with grid_pinned FALSE and loud.
  * DEGENERATE (one locked seed; the v1.4.6 per-seed leg) -> HALT_ADJUDICATE end-to-end.
  * LOW-POWER  (wide across-seed IR spread; the v1.4.7 power guard) -> HALT_ADJUDICATE end-to-end.

Exit code is non-zero if ANY assertion fails, so this is usable as a gate and not merely a report.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trikaal.eval.verdict import (  # noqa: E402
    DSR_HORIZONS,
    DSR_KAPPAS,
    DSR_SEEDS,
    HALT_ADJUDICATE,
    PRIMARY_H,
    write_cell_eval_artifact,
    write_eval_index,
)

OUT = Path("runs_manifest/m6_verdict_integration_check.json")
T = 1500
SIGMA_COMMON, SIGMA_IDIO = 0.02, 0.01
MU_PLANTED = {1: 0.0005, 2: 0.0005, 3: 0.0010, 4: 0.0020, 5: 0.0000}
BENIGN_MU_DIAG = {
    "mean": 0.0,
    "std": 1.0,
    "frac_negative": 0.5,
    "n": T,
    "estimator": "expectation",
    "activity_decisions": 0.5,
}
REQUIRED_PER_SEED_FIELDS = (
    ("degeneracy_guard", "frac_negative_by_seed"),
    ("degeneracy_guard", "activity_decisions_by_seed"),
    ("power_guard", "ir_by_seed"),
    ("power_guard", "ir_range_across_seeds"),
)


def _fixture(out_dir: Path, *, mu_by_cell_seed, mudiag_by_cell_seed, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(T) * SIGMA_COMMON
    entries: dict[str, str] = {}
    for cell in (1, 2, 3, 4, 5):
        for sd in DSR_SEEDS:
            series = common + rng.standard_normal(T) * SIGMA_IDIO + mu_by_cell_seed[cell][sd]
            val = {h: {k: float(rng.normal(0.5, 0.3)) for k in DSR_KAPPAS} for h in DSR_HORIZONS}
            name, sha = write_cell_eval_artifact(
                out_dir,
                cell_id=cell,
                seed=sd,
                grid={"h": PRIMARY_H, "start_ms": 1_704_132_000_000, "n_periods": T},
                headline_series=series,
                kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
                val_ir_by_kappa_by_h=val,
                mu_diag=mudiag_by_cell_seed[cell][sd],
                meta={"fixture": True, "integration_check": True},
            )
            entries[name] = sha
    write_eval_index(out_dir, entries)
    return out_dir


def _flat_mu():
    return {c: dict.fromkeys(DSR_SEEDS, MU_PLANTED[c]) for c in (1, 2, 3, 4, 5)}


def _flat_diag():
    return {c: {sd: dict(BENIGN_MU_DIAG) for sd in DSR_SEEDS} for c in (1, 2, 3, 4, 5)}


def _run_driver(art_dir: Path, out_path: Path) -> subprocess.CompletedProcess:
    """The REAL driver as a subprocess — conformance gate, hash load, manifest write, prints."""
    return subprocess.run(
        [
            sys.executable,
            "scripts/m6_verdict.py",
            "--artifacts",
            str(art_dir),
            "--out",
            str(out_path),
            "--allow-toy-grid",
        ],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=900,
    )


def _case(tmp: Path, name: str, mu, diag, *, expect_halt: bool, seed: int) -> dict:
    art = _fixture(tmp / name, mu_by_cell_seed=mu, mudiag_by_cell_seed=diag, seed=seed)
    out_path = tmp / f"{name}_verdict.json"
    proc = _run_driver(art, out_path)
    checks: dict[str, bool] = {"driver_exit_zero": proc.returncode == 0}
    rec: dict = {"case": name, "returncode": proc.returncode, "stderr_tail": proc.stderr[-600:]}
    if proc.returncode != 0 or not out_path.exists():
        rec["checks"] = {**checks, "manifest_written": out_path.exists()}
        rec["passed"] = False
        return rec

    m = json.loads(out_path.read_text())
    v = m["verdict"]
    checks["manifest_written"] = True
    checks["conformance_gate_ran_first"] = "§3a conformance PASS" in proc.stdout
    checks["grid_pinned_false_and_loud"] = m["grid_pinned"] is False
    checks["both_guards_in_manifest"] = "degeneracy_guard" in m and "power_guard" in m
    checks["both_guards_armed"] = bool(m["degeneracy_guard"]["armed"] and m["power_guard"]["armed"])
    for guard, field in REQUIRED_PER_SEED_FIELDS:
        present = all(field in m[guard]["per_cell"][str(c)] for c in (1, 2, 3, 4, 5))
        checks[f"{guard}.{field}_present_for_every_cell"] = present
    # per-seed maps must actually be keyed by the pinned seeds, not empty shells
    checks["per_seed_maps_keyed_by_pinned_seeds"] = all(
        set(m["degeneracy_guard"]["per_cell"][str(c)]["frac_negative_by_seed"])
        == {str(s) for s in DSR_SEEDS}
        and set(m["power_guard"]["per_cell"][str(c)]["ir_by_seed"]) == {str(s) for s in DSR_SEEDS}
        for c in (1, 2, 3, 4, 5)
    )
    emitted_halt = v["emitted"] == HALT_ADJUDICATE
    checks["emitted_matches_expectation"] = emitted_halt is expect_halt
    # HALT-only invariant, checked end-to-end and not only in a unit KAT
    checks["primary_preserved_as_a_clause_word"] = v["primary"] in ("SURVIVES", "NULL")
    if expect_halt:
        checks["halt_surfaced_on_stdout"] = "guard] HALT" in proc.stdout
        checks["a_guard_flag_is_true"] = bool(v["halted_for_degeneracy"] or v["halted_for_power"])
    else:
        checks["no_guard_halted"] = not (v["halted_for_degeneracy"] or v["halted_for_power"])
        checks["emitted_equals_primary"] = v["emitted"] == v["primary"]

    rec.update(
        {
            "emitted": v["emitted"],
            "primary": v["primary"],
            "halted_for_degeneracy": v["halted_for_degeneracy"],
            "halted_for_power": v["halted_for_power"],
            "degenerate_cells": m["degeneracy_guard"]["degenerate_cells"],
            "underpowered_claims": m["power_guard"]["underpowered_claims"],
            "grid_pinned": m["grid_pinned"],
            "cell4_ir_range_across_seeds": m["power_guard"]["per_cell"]["4"][
                "ir_range_across_seeds"
            ],
            "checks": checks,
            "passed": all(checks.values()),
        }
    )
    return rec


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cases = []

        # 1) HEALTHY — must still EMIT, with every new field present
        cases.append(_case(tmp, "healthy", _flat_mu(), _flat_diag(), expect_halt=False, seed=101))

        # 2) DEGENERATE — ONE locked seed on the signal cell (the v1.4.6 per-seed leg). Its
        #    seed-mean is (0.999 + 0.55 + 0.55)/3 = 0.6997, in-band, so ONLY the per-seed leg fires.
        diag = _flat_diag()
        diag[4][DSR_SEEDS[0]]["frac_negative"] = 0.999
        diag[4][DSR_SEEDS[1]]["frac_negative"] = 0.55
        diag[4][DSR_SEEDS[2]]["frac_negative"] = 0.55
        cases.append(
            _case(tmp, "degenerate_one_locked_seed", _flat_mu(), diag, expect_halt=True, seed=101)
        )

        # 3) LOW-POWER — wide across-seed IR spread on the signal cell (the v1.4.7 power guard)
        mu = _flat_mu()
        mu[4] = {DSR_SEEDS[0]: 0.0200, DSR_SEEDS[1]: 0.0020, DSR_SEEDS[2]: -0.0150}
        cases.append(
            _case(tmp, "low_power_wide_seed_spread", mu, _flat_diag(), expect_halt=True, seed=303)
        )

    all_passed = all(c["passed"] for c in cases)
    out = {
        "receipt": "m6_verdict_integration_check",
        "amendment": "prereg s7 v1.4.7 integration item (supervisor-ordered, $0)",
        "why": (
            "Both guards were added to the ONLY path that may emit SURVIVES/NULL, and the "
            "pre-flight verdict dry run (m6_toy_verdict_manifest.json) predates both. Unit KATs "
            "prove the guard functions behave; they do not prove the DRIVER still emits, still "
            "writes a manifest, still surfaces the HALT and still carries the per-seed fields."
        ),
        "method": (
            "scripts/m6_verdict.py invoked as a SUBPROCESS with --allow-toy-grid on three toy "
            "fixtures (healthy / one-locked-seed degenerate / wide-seed-spread low-power). Real "
            "conformance gate, real content-hash load, real manifest write."
        ),
        "all_passed": all_passed,
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))

    for c in cases:
        print(
            f"[{'PASS' if c['passed'] else 'FAIL'}] {c['case']}: emitted={c.get('emitted')} "
            f"primary={c.get('primary')} deg={c.get('degenerate_cells')} "
            f"lowpower={c.get('underpowered_claims')}"
        )
        for k, ok in c.get("checks", {}).items():
            if not ok:
                print(f"         FAILED CHECK: {k}")
    print(f"\nall_passed={all_passed}")
    print(f"[receipt] -> {OUT}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
