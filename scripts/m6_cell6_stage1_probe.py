"""§7 v1.6.27 — THE ~$0.50 MEASUREMENT THAT DECIDES THE $16–22. Cell 6, Stage 1 only, one seed.

    PYTHONPATH=src python3 scripts/m6_cell6_stage1_probe.py --device cuda \
        --lake processed/universe_bars

**WHY THIS EXISTS.** The C-12 bracket needs a Cell 6 (FSQ, 16 dims, micro slots 7–12 CONSTANT) so
that ΔIR(4−6) and ΔIR(4−5) can be reported as two ends of an interval. But cell 6 may not be
"cell 4 minus information" at all — it may be **a different tokenizer by our own pinned
definitions**, on two counts that are both **Stage-1 outputs** and therefore cheap to settle:

  * **codebook utilization** — six constant input dims shrink realized code entropy, and
    ``PINNED_CODEBOOK_MIN_UTILIZATION`` (0.95) REFUSES every artifact below it. Cell 6 could be
    un-assemblable before a single Stage-2 step is paid for.
  * **realized bits-per-token** — constant dims lower H(coarse)+H(fine). Outside
    ``FSQ_BPT_BAND`` (19.5–20.5) the bracket's lower end is measured on a **different-capacity
    instrument**, which is exactly the matched-bits control invariant 6 exists to protect.

One Stage-1 run at the pinned budget is ~1.5 GPU-h ≈ $0.45–0.60 against the 54.1 GPU-h ≈ $15.68–
21.63 that five cell-6 seeds cost. **It runs on box 1, which is rented anyway, and it runs BEFORE
the shard structure is fixed — so the matrix is never re-cut 25 ↔ 30 mid-flight.**

**THE DECISION RULE IS PRE-COMMITTED AND LIVES IN CODE, not in a judgement made on the box.**
Stage 2 is never entered here (``steps_stage2 = 0``) and no verdict clause is touched: cell 6 is a
CONTROL, and ``verdict._CELL_BY_ID`` must still hold exactly the five cells under test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from trikaal.constants import FSQ_BPT_BAND
from trikaal.data.universe_loader import calendar_boundary_ms, connect_lake
from trikaal.eval.conformance import (
    MDE_INPUTS,
    PINNED_CODEBOOK_MIN_UTILIZATION,
    PINNED_MONEY_SEQ_LEN,
    PINNED_STEPS_STAGE1,
)
from trikaal.eval.diagnostics import cell_codebook_diagnostic
from trikaal.run.matrix import load_symbol_arrays, windows_by_arm
from trikaal.train.arms import ARM_MICRO_CONSTANT
from trikaal.train.cells import CellSpec
from trikaal.train.orchestrator import OrchestratorConfig, run_cell
from trikaal.train.token_stream import tokenize_features

OUT = Path("runs_manifest/m6_cell6_stage1_probe.json")
CELL6 = CellSpec(6, "fsq", ARM_MICRO_CONSTANT)

WIRE = "WIRE_CELL_6"
DECLINE = "DO_NOT_BUY_CELL_6"


def decide(utilization_coarse: float, utilization_fine: float, bpt: float) -> dict:
    """The PRE-COMMITTED rule, written before the numbers exist. No judgement on the box.

    BOTH pass → wire cell 6, buy 5 seeds, report the BRACKET with both ends **and the explicit
    pre-commitment that they may invert** — ΔIR(4−6) ≤ ΔIR(4−5) is an empirical claim about two
    trained models, NOT an identity, and if it inverts the interval is empty and the larger number
    flatters us. Disclose that the two ends are confounded in OPPOSITE directions and that the
    lower end is capacity-matched only because bpt held.

    EITHER fails → **DO NOT BUY CELL 6.** Report the capacity handicap as UNQUANTIFIED IN IR UNITS
    — which is exactly what the manifest now says — and record the MEASURED reason the bracket was
    not constructed. That is an honest, publishable outcome and it costs nothing.
    """
    lo, hi = FSQ_BPT_BAND
    util_ok = (
        min(float(utilization_coarse), float(utilization_fine)) >= PINNED_CODEBOOK_MIN_UTILIZATION
    )
    bpt_ok = lo <= float(bpt) <= hi
    reasons = []
    if not util_ok:
        reasons.append(
            f"codebook utilization min({utilization_coarse:.4f}, {utilization_fine:.4f}) < "
            f"{PINNED_CODEBOOK_MIN_UTILIZATION} — every cell-6 artifact would be REFUSED by "
            "_validate_artifact, so the arm cannot enter a verdict at all"
        )
    if not bpt_ok:
        reasons.append(
            f"realized bits-per-token {bpt:.4f} outside FSQ_BPT_BAND [{lo}, {hi}] — the bracket's "
            "lower end would be measured on a DIFFERENT-CAPACITY instrument, which is the "
            "matched-bits control invariant 6 exists to protect"
        )
    return {
        "utilization_ok": util_ok,
        "bpt_ok": bpt_ok,
        "decision": WIRE if (util_ok and bpt_ok) else DECLINE,
        "reasons": reasons,
        "if_declined": (
            "report the capacity handicap as UNQUANTIFIED IN IR UNITS (what the clause strings "
            "now say) and record this measured reason. Publishable, and it costs nothing."
        ),
        "if_wired": (
            "buy 5 seeds (54.1 GPU-h, $15.68-21.63), report BOTH ends of the bracket whichever "
            "way they fall — the ordering is an empirical claim, not an identity — and disclose "
            "that the ends are confounded in OPPOSITE directions"
        ),
    }


def stage2_never_entered(manifest: dict) -> bool:
    """Did Stage 2 genuinely not run? A CHECK THAT COULD NEVER PASS, until now.

    The field was ``manifest["final_loss"]["stage2"] is None``. The key ALWAYS EXISTS and carries
    ``nan`` when ``steps_stage2 == 0``, so the receipt reported ``stage2_never_entered: False`` —
    "Stage 2 WAS entered" — on a run where Stage 2 provably never ran. It could not return True
    under any input. That is the mirror image of a check that cannot fail, and it sat on the
    receipt that documents the WIRE/DECLINE decision: a reader either believes the probe silently
    became a full training run, or learns to ignore the field. Both are worse than no field.

    A trained Stage 2 leaves a FINITE loss. ``nan`` (or a missing key) is the signature of "never
    entered", so that is what this tests. Discrimination is asserted in
    ``tests/train/test_cell6_decision_rule.py`` rather than assumed.
    """
    s2 = (manifest.get("final_loss") or {}).get("stage2")
    if s2 is None:
        return True
    try:
        return not math.isfinite(float(s2))
    except (TypeError, ValueError):
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lake", type=Path, default=Path("processed/universe_bars"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/m6_cell6_probe"))
    ap.add_argument("--n-symbols", type=int, default=8, help="symbols to tokenize for the read")
    args = ap.parse_args()

    if not args.lake.exists():
        print(f"LAKE MISSING at {args.lake} — refusing to invent data", file=sys.stderr)
        return 1
    pinned = json.loads(MDE_INPUTS.read_text())
    symbols = list(pinned["symbols_sampled"])[: args.n_symbols]
    window, train_frac = tuple(pinned["window"]), float(pinned["train_frac"])
    boundary = calendar_boundary_ms(window[0], window[1], train_frac)
    con = connect_lake(args.lake)

    # THE DRIVER'S PATH, NOT A SECOND ONE. `load_symbol_arrays` -> `windows_by_arm` is exactly
    # what m6_money_run.py:361/388 does. The previous form invented its own loading path and was
    # wrong in every argument — it passed the CONNECTION as `symbol`, the symbol string as `x`,
    # and a `window=` kwarg that `build_symbol_windows` does not accept — then applied
    # `constant_micro` to already-windowed [N, seq_len, F] data, which indexes the wrong axis.
    # None of it had ever executed; only `decide()` was tested. The Cell-6 transform now lives in
    # the loader beside `shuffle_micro`, so this path gets it for free and by the same route the
    # money run would.
    t0 = time.time()
    raw = {s: load_symbol_arrays(con, s, boundary_ms=boundary, tail_bars=None) for s in symbols}
    by_arm = windows_by_arm(
        raw,
        seq_len=PINNED_MONEY_SEQ_LEN,
        boundary_ms=boundary,
        seed=args.seed,
        arms=(ARM_MICRO_CONSTANT,),
    )
    per_symbol = by_arm[ARM_MICRO_CONSTANT]

    cfg = OrchestratorConfig(
        seeds=(args.seed,),
        seq_len=PINNED_MONEY_SEQ_LEN,
        steps_stage1=PINNED_STEPS_STAGE1,
        steps_stage2=0,  # STAGE 1 ONLY — the two numbers are Stage-1 outputs
        device=args.device,
        out_dir=args.out_dir,
        money_run=False,  # a probe declares itself; it is not a money run and never assembles
        micro_legibility_min=None,  # cell 6 is EXEMPT by arm — see MICRO_LEGIBILITY_APPLIES
    )
    manifest = run_cell(CELL6, args.seed, {ARM_MICRO_CONSTANT: per_symbol}, cfg)

    # PATH, then FACTORY. `Path / str + str` is a TypeError and `load_checkpoint` takes a model
    # factory, not None — both caught by reading the signature rather than by the box.
    from trikaal.tokenizer.model import TokenizerAE
    from trikaal.train.checkpoint import load_checkpoint

    ckpt = args.out_dir / f"{CELL6.name}_seed{args.seed}" / "tokenizer.pt"
    tok = load_checkpoint(ckpt, TokenizerAE, map_location="cpu")
    bcs, bfs = [], []
    for sw in per_symbol:
        b_c, b_f = tokenize_features(
            tok.model, sw.x, sw.mask, sw.segment_id, window=PINNED_MONEY_SEQ_LEN, device="cpu"
        )
        bcs.append(b_c)
        bfs.append(b_f)
    diag = cell_codebook_diagnostic(
        np.concatenate(bcs), np.concatenate(bfs), v_c=tok.model.v_c, v_f=tok.model.v_f
    )
    d = decide(
        diag["coarse"]["utilization"],
        diag["fine"]["utilization"],
        diag["effective_bits_per_token"],
    )
    rep = {
        "receipt": "m6_cell6_stage1_probe",
        "what": "the ~$0.50 Stage-1 measurement that decides whether the $16-22 C-12 bracket is "
        "buildable at all",
        "cell": CELL6.name,
        "seed": args.seed,
        "steps_stage1": PINNED_STEPS_STAGE1,
        "steps_stage2": 0,
        "n_symbols_tokenized": len(symbols),
        "wall_s": round(time.time() - t0, 1),
        "codebook": diag,
        "gates": {
            "codebook_min_utilization": PINNED_CODEBOOK_MIN_UTILIZATION,
            "fsq_bpt_band": list(FSQ_BPT_BAND),
        },
        "PRE_COMMITTED_DECISION": d,
        "stage2_never_entered": stage2_never_entered(manifest),
        "cell_6_is_a_control": "it enters NO verdict clause; PINNED_DSR['cells'] stays (1,2,3,4,5) "
        "and verdict._CELL_BY_ID must still hold exactly those five (asserted in the conformance "
        "gate, because n_trials_v12 is computed from the LIVE registry)",
    }
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "codebook"}, indent=2, default=str))
    print(f"\n★ DECISION: {d['decision']} → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
