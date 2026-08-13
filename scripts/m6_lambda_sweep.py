"""§7 v1.5 ITEM E — THE ONCE-ONLY LAMBDA RE-DERIVATION. There is no second one.

CRITERION, RULED PRE-DATA AND UNMOVED: the SMALLEST lambda whose THREE SEEDED RUNS clear
**0.9 on ALL SIX dims on BOTH arms**. Not cell 4 alone. Not a mean. Not min >= 0.85.

CONFIGS: lambda in {5, 8, 12}. Chosen to SPAN, not to optimise — the fixture landscape is
non-monotone (2 -> 0.8517, 3 -> 0.9067, 4 -> 0.8752, 6 -> 0.8999), so the question is whether ANY
lambda clears, not which is best. NO OTHER AXIS MOVES: no level reallocation, no bit re-split, no
change to total bits. Those are v2.

ARMS: BOTH. Cell 5 is the binding constraint by roughly 2x (its dim 8 must move +0.31 from
0.5896), and a config clearing only cell 4 unblocks nothing.

SLICE (item E rule 3): legibility is measured on the HELD-OUT slice at the END of the TRAIN
region — `gates.lambda_calibration_boundary_ms` — and training excludes it. Tuning lambda against
a quantity and then reporting that same quantity on the same data is fitting; item E created this
slice precisely to prevent that.

★ OHLCV RECON IS REPORTED AT EVERY CONFIG, beside legibility. Lambda buys legibility by SPENDING
OHLCV fidelity — measured at lambda=3 as a 1.51x cell5/cell4 penalty (m6_ohlcv_recon_ratio.json).
A lambda that clears the gate while degrading OHLCV badly produces a WORSE INSTRUMENT, not a
rescued one, and the ablation it would enable compares two degraded tokenizers. The trade-off is
reported so it is visible rather than inferred.

★ WHAT A PASS CANNOT BUY — written here so no future reader mistakes it: EVEN A CLEARING CONFIG
RETURNS THE ABLATION ONLY AS SECONDARY/EXPLORATORY at N 60->120, needing ~15 units and ~$120-165
against ~$125 remaining. THE PRIMARY IS THE MECHANISM FINDING EITHER WAY.

STOPPING RULE, BINDING: if no listed config clears within the cap, item E's disposition is FINAL,
no further attempt occurs, and the attempts ship as measured evidence of how hard the budget binds.
No fourth lambda. No widening after a near-miss.

The money pin is NOT touched: this runs through `build_calibration_tokenizer` with
`money_run=False`, and `run_cell` RAISES if a calibration lambda is set on a money run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

from trikaal.data.universe_loader import calendar_boundary_ms, connect_lake
from trikaal.eval.conformance import MDE_INPUTS, PINNED_MONEY_SEQ_LEN, PINNED_STEPS_STAGE1
from trikaal.eval.diagnostics import ohlcv_recon_diagnostic
from trikaal.run.matrix import load_symbol_arrays, windows_by_arm
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.cells import CELLS
from trikaal.train.checkpoint import load_checkpoint
from trikaal.train.gates import lambda_calibration_boundary_ms, micro_legibility_gate
from trikaal.train.orchestrator import OrchestratorConfig, run_cell
from trikaal.train.token_stream import tokenize_features

CRITERION_MIN_ACC = 0.9  # the STANDING gate. Not moved, not re-scoped, not re-based.
LAMBDAS = (5.0, 8.0, 12.0)
SEEDS = (0, 1, 2)
CELL4 = next(c for c in CELLS if c.cell_id == 4)
CELL5 = next(c for c in CELLS if c.cell_id == 5)


def _slice_only(sw, tok_pair, slice_start_ms: int):
    """Bars at or after the calibration-slice start, with their tokens filtered identically."""
    keep = sw.ts >= slice_start_ms
    b_c, b_f = tok_pair
    return (
        dataclasses.replace(
            sw,
            x=sw.x[keep],
            mask=sw.mask[keep],
            segment_id=sw.segment_id[keep],
            ts=sw.ts[keep],
            starts=np.zeros(0, dtype=np.int64),
        ),
        (b_c[keep], b_f[keep]),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lake", type=Path, default=Path("processed/universe_bars"))
    ap.add_argument("--n-symbols", type=int, default=40)
    ap.add_argument("--out-dir", type=Path, default=Path("/root/m6_lambda"))
    ap.add_argument("--out", type=Path, default=Path("runs_manifest/m6_lambda_sweep.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=PINNED_STEPS_STAGE1)
    a = ap.parse_args()
    if not a.lake.exists():
        print(f"LAKE MISSING at {a.lake}", file=sys.stderr)
        return 1

    pinned = json.loads(MDE_INPUTS.read_text())
    symbols = list(pinned["symbols_sampled"])[: a.n_symbols]
    win, train_frac = tuple(pinned["window"]), float(pinned["train_frac"])
    boundary = calendar_boundary_ms(win[0], win[1], train_frac)
    slice_start = lambda_calibration_boundary_ms(
        int(np.datetime64(win[0], "ms").astype("int64")), int(boundary)
    )
    print(f"[slice] train ends {boundary}, calibration slice starts {slice_start}", flush=True)

    con = connect_lake(a.lake)
    raw = {s: load_symbol_arrays(con, s, boundary_ms=boundary, tail_bars=None) for s in symbols}

    out: dict = {
        "receipt": "m6_lambda_sweep",
        "what": "§7 v1.5 item E ONCE-ONLY lambda re-derivation. Legibility measured on "
        "the HELD-OUT calibration slice; OHLCV recon reported at every config.",
        "CRITERION": "smallest lambda whose THREE seeded runs clear 0.9 on ALL SIX dims on BOTH "
        "arms. Ruled pre-data; not moved.",
        "WHAT_A_PASS_CANNOT_BUY": "Even a clearing config returns the ablation only as "
        "SECONDARY/EXPLORATORY at N 60->120 (~15 units, ~$120-165 against ~$125 remaining). "
        "THE PRIMARY IS THE MECHANISM FINDING EITHER WAY.",
        "STOPPING_RULE": "If no listed config clears, item E's disposition is FINAL. No fourth "
        "lambda, no widening after a near-miss.",
        "NOT_COMPARABLE_TO_THE_GATE_FIRING_RUN": "The lambda=3.0 values 0.8223/0.7528 were "
        "trained on the full train region INCLUDING this slice and measured off-slice-held-out. "
        "No delta against them is a measured quantity. The criterion here is ABSOLUTE (0.9) and "
        "requires no baseline.",
        "money_pin_untouched": "runs via build_calibration_tokenizer with money_run=False; "
        "run_cell RAISES if a calibration lambda is set on a money run; "
        "PINNED_MICRO_POINT_WEIGHT unchanged at 3.0.",
        "lambdas": list(LAMBDAS),
        "seeds": list(SEEDS),
        "slice": {
            "train_boundary_ms": int(boundary),
            "calibration_slice_start_ms": int(slice_start),
        },
        "n_symbols": len(symbols),
        "steps_stage1": a.steps,
        "configs": {},
    }

    for lam in LAMBDAS:
        for seed in SEEDS:
            for spec in (CELL4, CELL5):
                key = f"lambda{lam:g}_seed{seed}_cell{spec.cell_id}"
                t0 = time.time()
                # TRAIN on the train region MINUS the slice: boundary_ms = slice_start.
                per_arm = windows_by_arm(
                    raw,
                    seq_len=PINNED_MONEY_SEQ_LEN,
                    boundary_ms=int(slice_start),
                    seed=seed,
                    arms=(spec.arm,),
                )
                cfg = OrchestratorConfig(
                    seeds=(seed,),
                    seq_len=PINNED_MONEY_SEQ_LEN,
                    steps_stage1=a.steps,
                    steps_stage2=0,
                    device=a.device,
                    out_dir=a.out_dir / key,
                    money_run=False,  # REQUIRED: run_cell raises otherwise
                    micro_legibility_min=None,  # measured on the SLICE below, not here
                    calibration_micro_point_weight=lam,
                )
                run_cell(spec, seed, per_arm, cfg)
                ck = a.out_dir / key / f"{spec.name}_seed{seed}" / "tokenizer.pt"
                tok = load_checkpoint(ck, TokenizerAE, map_location=a.device).model.eval()

                # MEASURE on the held-out slice only.
                sl_sw, sl_tok = [], {}
                for sw in per_arm[spec.arm]:
                    pair = tokenize_features(
                        tok,
                        sw.x,
                        sw.mask,
                        sw.segment_id,
                        window=PINNED_MONEY_SEQ_LEN,
                        device=a.device,
                    )
                    s2, p2 = _slice_only(sw, pair, int(slice_start))
                    if s2.x.shape[0] == 0:
                        continue
                    sl_sw.append(s2)
                    sl_tok[s2.symbol] = p2
                # min_acc=0.0 so the instrument REPORTS instead of raising; the 0.9 criterion is
                # applied below, unchanged. Using the production gate, not a re-implementation.
                receipt = micro_legibility_gate(tok, sl_sw, sl_tok, min_acc=0.0, run_name=key)
                accs = {d: receipt["per_dim"][d].get("sign_acc") for d in receipt["per_dim"]}
                clears = all(v is not None and v >= CRITERION_MIN_ACC for v in accs.values())
                xs = np.stack([s.x[:PINNED_MONEY_SEQ_LEN] for s in sl_sw[:64]]).astype(np.float32)
                ms = np.stack([s.mask[:PINNED_MONEY_SEQ_LEN] for s in sl_sw[:64]]).astype(
                    np.float32
                )
                recon = ohlcv_recon_diagnostic(tok, xs, ms, device=a.device)
                out["configs"][key] = {
                    "lambda": lam,
                    "seed": seed,
                    "cell": spec.cell_id,
                    "arm": spec.arm,
                    "per_dim": receipt["per_dim"],
                    "min_acc_over_dims": min(v for v in accs.values() if v is not None),
                    "clears_0_9_all_six": bool(clears),
                    "ohlcv_recon_mae": recon["ohlcv_recon_mae"],
                    "ohlcv_recon_mae_by_dim": recon.get("ohlcv_recon_mae_by_dim"),
                    "wall_s": round(time.time() - t0, 1),
                }
                print(
                    f"[{key}] min_acc={out['configs'][key]['min_acc_over_dims']:.4f} "
                    f"clears={clears} recon={recon['ohlcv_recon_mae']:.6f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
                a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    # VERDICT: smallest lambda clearing on BOTH arms across ALL THREE seeds.
    winner = None
    for lam in LAMBDAS:
        ks = [k for k, v in out["configs"].items() if v["lambda"] == lam]
        if len(ks) == len(SEEDS) * 2 and all(out["configs"][k]["clears_0_9_all_six"] for k in ks):
            winner = lam
            break
    out["VERDICT"] = {
        "clearing_lambda": winner,
        "criterion": "0.9 on all six dims, both arms, all three seeds",
        "disposition": "ITEM E DISPOSITION IS FINAL — no further attempt"
        if winner is None
        else "a clearing lambda exists; see WHAT_A_PASS_CANNOT_BUY before reading it as a rescue",
    }
    for req in ("NOT_COMPARABLE_TO_THE_GATE_FIRING_RUN", "WHAT_A_PASS_CANNOT_BUY", "STOPPING_RULE"):
        if not out.get(req):
            raise RuntimeError(f"receipt REFUSED — required disclosure {req!r} is missing")
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["VERDICT"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
