"""§7 v1.4.5 CONSTANT-BOOK LATTICE — an OBSERVATION receipt, not a test ($0, pure manifest read).

    PYTHONPATH=src .venv/bin/python scripts/m6_lattice_observation.py

WHAT THIS IS. Across the two full 10-cell Stage-2 runs on disk (acceptance 36560dc and money-leg
7784df2, whose recipes are byte-identical) there are 20 (run, arm, cell) headline-IR
observations. Most of them land on a SMALL SET OF REPEATED float64 values that recur ACROSS the
two runs at full precision. A repeated exact value is the signature of a CONSTANT-DIRECTION BOOK:
when every scored bar trades in the same direction, the pooled net series — and therefore the IR
— is a property of the RETURN BLOCK and the direction alone, identical for any model that
produces that direction. The money-leg manifest carries per-cell ``frac_negative`` (mu_diag
landed in v1.4.2), so the mapping direction -> lattice value is MEASURED there, not assumed;
the acceptance run predates mu_diag, so its cells are classified by exact float64 identity with
money-leg cells whose direction WAS measured. That is an inference, and it is labelled as one.

WHAT THIS IS NOT. n = 2 runs. There are no error bars, no repeated seeds, and no hypothesis
test. Nothing here is evidence FOR the microstructure claim: cell 5 is the SHUFFLED-micro
placebo, so its behaviour tracks input CAPACITY, not micro INFORMATION. This receipt exists to
name a pattern precisely enough that it cannot quietly harden into a claim.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

RUNS = {
    "acceptance": Path("runs_manifest/m6_acceptance_stage2_manifest.json"),
    "moneyleg": Path("runs_manifest/m6_moneyleg_rerun_manifest.json"),
}
OUT = Path("runs_manifest/m6_constant_book_lattice.json")

# cell id -> (quantizer, feature arm). Cells 3 and 4 receive REAL microstructure; cell 5 receives
# SHUFFLED microstructure (the placebo) — it is micro-FED but carries no micro information.
CELL_ARM = {
    1: ("bsq", "ohlcv"),
    2: ("fsq", "ohlcv"),
    3: ("bsq", "micro"),
    4: ("fsq", "micro"),
    5: ("fsq", "micro_shuffled"),
}
MICRO_FED = {3, 4, 5}


def main() -> int:
    obs = []
    for run, path in RUNS.items():
        d = json.loads(path.read_text())
        for arm in ("planted", "noise"):
            cell_ir = d["arms"][arm]["cell_ir"]
            mu_diag = d["arms"][arm].get("mu_diag")  # absent on the acceptance run
            for c in "12345":
                obs.append(
                    {
                        "run": run,
                        "arm": arm,
                        "cell": int(c),
                        "quantizer": CELL_ARM[int(c)][0],
                        "feature_arm": CELL_ARM[int(c)][1],
                        "ir": float(cell_ir[c]),
                        "frac_negative": (float(mu_diag[c]["frac_negative"]) if mu_diag else None),
                    }
                )

    # ---- the lattice is defined by MEASURED constant books, never by how often a value recurs.
    # A singleton is still on the lattice if its book is constant (money-leg planted cell2 is the
    # only all-LONG planted cell, so its value appears once); conversely a recurring value is
    # only a lattice point because the direction behind it was measured. Per arm the points are:
    #   all-short  frac_negative == 1.0
    #   all-long   frac_negative == 0.0
    #   flat       IR == 0.0 exactly => the pooled net series is identically zero => nothing
    #              traded at kappa* (any traded period nets y - cost != 0 almost surely)
    lattice: dict[str, dict] = {}
    for o in obs:
        fn = o["frac_negative"]
        direction = None
        if o["ir"] == 0.0:
            direction = "flat_never_trades"
        elif fn == 1.0:
            direction = "all_short"
        elif fn == 0.0:
            direction = "all_long"
        if direction is None:
            continue
        key = f"{o['arm']}:{direction}"
        lattice.setdefault(
            key, {"arm": o["arm"], "direction": direction, "ir": o["ir"], "measured_on": []}
        )
        assert lattice[key]["ir"] == o["ir"], f"{key}: constant book must be model-independent"
        lattice[key]["measured_on"].append(f"{o['run']}/{o['arm']}/cell{o['cell']}")

    by_arm: dict[str, set[float]] = {}
    for meta in lattice.values():
        by_arm.setdefault(meta["arm"], set()).add(meta["ir"])
    by_arm.setdefault("planted", set()).add(0.0)  # the flat point exists in either arm
    by_arm.setdefault("noise", set()).add(0.0)

    counts = Counter(o["ir"] for o in obs)
    for o in obs:
        o["on_lattice"] = o["ir"] in by_arm.get(o["arm"], set())
        o["ir_multiplicity"] = counts[o["ir"]]
    off = [o for o in obs if not o["on_lattice"]]
    lattice = {
        k: {**v, "cross_run_multiplicity": counts[v["ir"]]} for k, v in sorted(lattice.items())
    }

    # the money-leg arm is where direction -> IR is MEASURED (mu_diag present)
    measured = {}
    for o in obs:
        if o["run"] != "moneyleg" or o["frac_negative"] is None:
            continue
        fn = o["frac_negative"]
        label = "all_short" if fn == 1.0 else "all_long" if fn == 0.0 else f"mixed({fn:.5f})"
        measured.setdefault(o["arm"], {}).setdefault(label, []).append(
            {"cell": o["cell"], "ir": o["ir"]}
        )

    out = {
        "receipt": "m6_constant_book_lattice",
        "status": "OBSERVATION — n=2 runs, no error bars, NOT a hypothesis test",
        "provenance": {
            "acceptance": "36560dc runs_manifest/m6_acceptance_stage2_manifest.json",
            "moneyleg": "7784df2 runs_manifest/m6_moneyleg_rerun_manifest.json",
            "recipes_byte_identical": json.dumps(
                json.loads(RUNS["acceptance"].read_text())["recipe"], sort_keys=True
            )
            == json.dumps(json.loads(RUNS["moneyleg"].read_text())["recipe"], sort_keys=True),
        },
        "n_observations": len(obs),
        "mechanism": (
            "A constant-direction book makes the pooled net series a function of the RETURN "
            "BLOCK and the direction alone, so its IR is model-independent and recurs exactly. "
            "Direction->IR is MEASURED on the money-leg arm (mu_diag frac_negative); acceptance "
            "cells are classified by exact float64 identity with those measured cells, which is "
            "an INFERENCE (the acceptance run predates the mu_diag receipt)."
        ),
        "measured_direction_to_ir_moneyleg": measured,
        "lattice_definition": (
            "A cell is ON the lattice iff its book is CONSTANT: all-short (frac_negative==1.0), "
            "all-long (frac_negative==0.0), or flat (IR==0.0 exactly => the pooled net series is "
            "identically zero => nothing traded at kappa*). Recurrence is EVIDENCE of a constant "
            "book, not the definition of one — money-leg planted cell2 is the only all-long "
            "planted cell, so its lattice value appears exactly once and is still a lattice point."
        ),
        "lattice_points": lattice,
        "off_lattice": [
            {k: o[k] for k in ("run", "arm", "cell", "feature_arm", "ir", "frac_negative")}
            for o in off
        ],
        "off_lattice_caveat": (
            "OFF-LATTICE MEANS 'NOT PERFECTLY CONSTANT', NOT 'A STRATEGY'. The only off-lattice "
            "cell with a measured direction is money-leg planted cell4 at frac_negative "
            "0.99883 — 14 long decisions out of 12,000. That is a 99.88% one-sided book that "
            "misses the lattice by a hair, not a discriminating signal."
        ),
        "pattern": {
            "off_lattice_count": len(off),
            "all_off_lattice_are_micro_fed": all(o["cell"] in MICRO_FED for o in off),
            "all_off_lattice_are_planted_arm": all(o["arm"] == "planted" for o in off),
            "ohlcv_only_observations": sum(1 for o in obs if o["cell"] not in MICRO_FED),
            "ohlcv_only_off_lattice": sum(1 for o in off if o["cell"] not in MICRO_FED),
            "noise_arm_off_lattice": sum(1 for o in off if o["arm"] == "noise"),
        },
        "model_independence_is_MEASURED_not_inferred": (
            "The three money-leg noise cells that are all-long (cell2 FSQ+ohlcv, cell3 BSQ+micro, "
            "cell4 FSQ+micro) span BOTH quantizers and BOTH feature arms and return the IDENTICAL "
            "float64 IR -7.563377128272953. Three different trained models, three different "
            "tokenizers, one bit-identical number. That is direct evidence that the value belongs "
            "to the block and the direction, not to any model. The script asserts this."
        ),
        "correction_to_the_posted_enumeration": (
            "The ruling enumerated the off-lattice set as 'cell4 in both runs, acceptance cell3 "
            "once'. Verified independently: acceptance planted cell5 is ALSO off-lattice — its "
            "-1.0308263783219347 is NOT the planted all-long constant -1.050127274973947 that "
            "money-leg cell2 measures. The off-lattice set is 4 observations: acceptance planted "
            "cells 3, 4, 5 and money-leg planted cell4. Cell4 remains the ONLY cell off-lattice "
            "in BOTH runs. Money-leg planted cells 1 and 2 are NOT off-lattice (cell1 is the flat "
            "never-trades point, cell2 is the all-long point) — a singleton value is not an "
            "off-lattice value."
        ),
        "what_this_does_NOT_support": (
            "Not evidence for the microstructure claim. Cell 5 is the SHUFFLED-micro placebo and "
            "is off-lattice in the acceptance run, so the pattern is equally consistent with "
            "extra input CAPACITY widening mu-hat dispersion as with micro INFORMATION. With "
            "n=2 runs and no seed replication there is no error bar on any of it."
        ),
        "observations": obs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))

    print(f"observations: {len(obs)}   measured constant-book points: {len(lattice)}")
    for k, meta in lattice.items():
        print(
            f"  {k:<22} IR={meta['ir']!r:<22} recurs x{meta['cross_run_multiplicity']} "
            f"measured_on={meta['measured_on']}"
        )
    print(f"off-lattice: {len(off)}")
    for o in off:
        print(
            f"  {o['run']}/{o['arm']}/cell{o['cell']} ({o['feature_arm']}) "
            f"IR={o['ir']!r} frac_neg={o['frac_negative']}"
        )
    p = out["pattern"]
    print(
        f"pattern: all off-lattice micro-fed={p['all_off_lattice_are_micro_fed']} "
        f"all in planted arm={p['all_off_lattice_are_planted_arm']} "
        f"ohlcv-only off-lattice={p['ohlcv_only_off_lattice']}/{p['ohlcv_only_observations']}"
    )
    print(f"[receipt] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
