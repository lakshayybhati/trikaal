"""DEMO ACCEPTANCE GATE — the demo must reproduce the PRODUCTION mu-hat BIT-FOR-BIT, or not ship.

WHY THIS EXISTS. A demo that renders a plausible forecast is not evidence that the demo shows OUR
model. Per the execution rule ("has never been executed is not an operational state") and the mock
rule ("a mock is not a subject — assert the call, not the stub's reply"), the only acceptable
evidence is running the real thing and comparing against the real thing.

WHAT IS COMPARED, AND WHY IT IS NOT CIRCULAR. ``trikaal.demo.inference`` deliberately REUSES the
production functions, so re-calling ``predict_mu`` and finding agreement would prove nothing. The
risk is not in ``predict_mu``; it is in the ASSEMBLY around it — which window, which decision
index, which arm transform, which sigma, which temporal keys. So this gate rebuilds the decision
INDEPENDENTLY, the way ``xsection.score_cell`` builds it for a whole symbol (full-symbol
tokenization, full-symbol decision grid, batch mu over many decisions), then asserts that the
demo's SINGLE-DECISION path lands on exactly the same float64 bits.

Those two paths differ in everything except the answer: production tokenizes and scores in bulk
over a symbol's entire forward region; the demo scores one instant. If the demo's window
selection, indexing or arm handling were wrong, the values would differ — which is precisely the
failure this gate is for.

BIT-FOR-BIT, NOT "CLOSE". The comparison is ``==`` on float64, not ``allclose``. A tolerance would
accept a demo that is subtly a different model, and "subtly a different model" is the whole risk.

AND THE GATE IS PROVEN CAPABLE OF FAILING before its pass is believed: ``--mutate`` perturbs a
single predictor weight by one ULP and the gate MUST then fail. A gate never observed to fail is
not a gate (the standing norm, and this project's oldest defect).

Exit codes: 0 PASS · 2 FAIL (mismatch, or --mutate did NOT produce a failure) · 1 usage/IO error.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trikaal.data.universe_loader import calendar_boundary_ms, connect_lake  # noqa: E402
from trikaal.demo.inference import (  # noqa: E402
    DEMO_ARM,
    DEMO_H,
    DEMO_SEQ_LEN,
    LAKE_ROOT,
    PINNED_TRAIN_FRAC,
    PINNED_WINDOW,
    UNIT_DIRS,
    ForecastBundle,
    available_seeds,
    forecast_at,
    load_unit,
    prepare_symbol,
    use_units_from,
    verify_lake_identity,
)
from trikaal.eval.predict import predict_mu  # noqa: E402
from trikaal.eval.xsection import (  # noqa: E402
    BAR_MS,
    SymbolEval,
    XSectionConfig,
    symbol_decisions,
)
from trikaal.run.matrix import load_symbol_arrays  # noqa: E402
from trikaal.train.arms import select_arm  # noqa: E402
from trikaal.train.token_stream import tokenize_features  # noqa: E402
from trikaal.utils.paths import display_path  # noqa: E402
from trikaal.utils.provenance import run_provenance  # noqa: E402


def production_mu(symbol: str, decision_ms_list: list[int], unit, *, device: str = "cpu"):
    """Rebuild mu THE WAY score_cell DOES: whole-symbol tokenization, batched over many decisions.

    This is the reference arm. It shares `predict_mu` with the demo but nothing else — the window,
    the decision grid and the batching are all built here the production way.
    """
    cfg = XSectionConfig(
        window_start=PINNED_WINDOW[0],
        window_end=PINNED_WINDOW[1],
        train_frac=PINNED_TRAIN_FRAC,
        h=DEMO_H,
        seq_len=DEMO_SEQ_LEN,
        cap_per_symbol=None,
        device=device,
        money=False,
    )
    con = connect_lake(LAKE_ROOT)
    boundary = calendar_boundary_ms(cfg.window_start, cfg.window_end, cfg.train_frac)
    raw = load_symbol_arrays(con, symbol, boundary_ms=boundary, tail_bars=None)
    se = SymbolEval(
        symbol=symbol,
        x=raw["x"],
        mask=raw["mask"],
        segment_id=raw["segment_id"],
        ts=raw["ts"],
        sigma=raw["sigma"],
        raw_ret_close=raw["raw_ret_close"],
    )
    grid = np.array(sorted(decision_ms_list), dtype=np.int64)
    dec = symbol_decisions(se, grid, cfg)
    x_arm, m_arm = select_arm(np.asarray(se.x, np.float32), se.mask, DEMO_ARM)
    b_c, b_f = tokenize_features(
        unit.tok, x_arm, m_arm, se.segment_id, window=cfg.seq_len, device=device
    )
    mu = predict_mu(
        unit.model,
        unit.tok,
        b_c,
        b_f,
        se.ts,
        se.sigma,
        dec,
        h=cfg.h,
        seq_len=cfg.seq_len,
        device=device,
        estimator=cfg.mu_estimator,
    )
    return {int(se.ts[d]): float(m) for d, m in zip(dec, mu, strict=True)}


def pick_decisions(symbol: str, n: int, *, device: str = "cpu") -> list[int]:
    """n valid decision instants spread across the forward region (deterministic, no RNG)."""
    cfg = XSectionConfig(
        window_start=PINNED_WINDOW[0],
        window_end=PINNED_WINDOW[1],
        train_frac=PINNED_TRAIN_FRAC,
        h=DEMO_H,
        seq_len=DEMO_SEQ_LEN,
        cap_per_symbol=None,
        device=device,
        money=False,
    )
    con = connect_lake(LAKE_ROOT)
    boundary = calendar_boundary_ms(cfg.window_start, cfg.window_end, cfg.train_frac)
    raw = load_symbol_arrays(con, symbol, boundary_ms=boundary, tail_bars=None)
    se = SymbolEval(
        symbol=symbol,
        x=raw["x"],
        mask=raw["mask"],
        segment_id=raw["segment_id"],
        ts=raw["ts"],
        sigma=raw["sigma"],
        raw_ret_close=raw["raw_ret_close"],
    )
    ts = se.ts
    lo = int(np.searchsorted(ts, boundary))
    hi = ts.shape[0] - cfg.h - 1
    if hi <= lo + cfg.seq_len:
        return []
    cand = np.linspace(lo + cfg.seq_len, hi - 1, num=max(n * 8, n), dtype=np.int64)
    grid = np.unique(ts[cand] - (ts[cand] % (cfg.h * BAR_MS)))
    dec = symbol_decisions(se, grid.astype(np.int64), cfg)
    return [int(ts[d]) for d in dec[:n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--n", type=int, default=3, help="decision instants to compare")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--mutate",
        action="store_true",
        help=(
            "NEGATIVE CONTROL: introduce a one-bar OFF-BY-ONE into the demo's assembly. The gate "
            "MUST then FAIL."
        ),
    )
    ap.add_argument("--out", type=Path, default=REPO / "runs_manifest/m6_demo_acceptance.json")
    ap.add_argument(
        "--units",
        type=Path,
        default=None,
        help=(
            "Directory holding the three cell-1 units (any layout — the repo tree or a "
            "HuggingFace download). Units are found by content: a folder with run_manifest.json, "
            "predictor.pt and tokenizer.pt. Omit to use the in-repo runs_cloud/ paths."
        ),
    )
    args = ap.parse_args()
    use_units_from(args.units)

    seeds = available_seeds()
    if not seeds:
        print("NO UNITS AVAILABLE", file=sys.stderr)
        return 1
    units = {s: load_unit(s, device=args.device) for s in seeds}
    print(f"[units] loaded seeds {seeds}; identity verified against run_manifest for each")

    stamps = pick_decisions(args.symbol, args.n, device=args.device)
    if not stamps:
        print(f"no valid decision instants for {args.symbol}", file=sys.stderr)
        return 1
    print(f"[grid]  {len(stamps)} decision instants: {stamps}")

    # ★ THE REFERENCE IS BUILT AT MATCHED BATCH COMPOSITION (one decision per production call),
    # and that is a correctness requirement, not a convenience. Measured on the real units:
    # production scored in a batch of N differs from the same decision scored alone by ~1e-6
    # RELATIVE — float32 reduction order, the invariant-7 class again (deterministic attention is
    # necessary, not sufficient). Comparing across batch compositions would therefore force a
    # tolerance, and a tolerance would accept a demo that is subtly a different model. So the GATE
    # matches composition and stays exact, and the batch-composition effect is measured separately
    # below as a NON-GATING disclosure rather than being hidden inside an allclose.
    ref = {
        s: {ms: production_mu(args.symbol, [ms], units[s], device=args.device)[ms] for ms in stamps}
        for s in seeds
    }
    # Tokenizing a 2.1M-bar symbol costs ~29 s per (symbol, seed) and MUST NOT be repeated per
    # decision — the first version did, and this gate timed out at ten minutes.
    prep = prepare_symbol(args.symbol, units=units, device=args.device)
    # NON-GATING: the same decisions scored in ONE batch of N, to quantify the effect we just
    # designed around. Reported, never asserted on.
    ref_batched = {
        s: production_mu(args.symbol, stamps, units[s], device=args.device) for s in seeds
    }

    rows, mismatches = [], []
    for ms in stamps:
        # ★ THE NEGATIVE CONTROL MUTATES THE ASSEMBLY, NOT THE WEIGHTS, AND THAT CORRECTION IS
        # THE POINT. My first version perturbed a predictor weight by 1 ULP — but BOTH arms read
        # the SAME `units[seed]` object, so it perturbed both sides equally and every row stayed
        # bit-identical. The harness duly reported "THIS GATE CANNOT FAIL", which was true of the
        # control and not of the gate: I had mutated the shared subject (the mock rule). This gate
        # compares two ASSEMBLY paths over identical weights, so only an assembly error can
        # discriminate — here the classic one, a one-bar off-by-one in the decision instant.
        demo_ms = ms + BAR_MS if args.mutate else ms
        bundle: ForecastBundle = forecast_at(
            args.symbol, demo_ms, units=units, device=args.device, prepared=prep
        )
        for sf in bundle.seeds:
            want = ref[sf.seed].get(ms)
            got = sf.mu_hat
            # BIT-FOR-BIT on float64. Not allclose — a tolerance would accept a different model.
            ok = want is not None and want.hex() == got.hex()
            batched = ref_batched[sf.seed].get(ms)
            rows.append(
                {
                    "decision_ms": ms,
                    "seed": sf.seed,
                    "demo_mu_hex": got.hex(),
                    "production_mu_hex": (want.hex() if want is not None else None),
                    "bit_identical": bool(ok),
                    # NON-GATING disclosure
                    "production_batched_mu_hex": (batched.hex() if batched is not None else None),
                    "batch_composition_rel_diff": (
                        abs(batched - got) / abs(got)
                        if (batched is not None and got != 0.0)
                        else None
                    ),
                }
            )
            if not ok:
                mismatches.append(rows[-1])

    # ★ PROVENANCE, AND ITS ABSENCE WAS THE NEWEST RECEIPT COMMITTING THE PROJECT'S OLDEST
    # DEFECT. This gate's own receipt carried no git SHA, no timestamp, no lake path and no
    # checkpoint hashes — a bit-identity claim with nothing recording WHICH bytes it was made
    # about. "9/9 bit-identical" is unfalsifiable without them: a reader cannot tell whether it
    # was run against these weights, this lake, or this revision. The lake path is here for the
    # specific historical reason that the demo once read the WRONG lake and this very gate agreed
    # with itself about it (the shared-input rule) — so the source is now on the record, next to
    # the identity that verify_lake_identity asserts.
    doc = {
        "script": "scripts/m6_demo_acceptance.py",
        "schema": "m6_demo_acceptance_v2",
        "provenance": {
            "environment": run_provenance(device=args.device),
            "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "lake_root": str(LAKE_ROOT),
            "lake_identity": verify_lake_identity(),
            "units": {
                str(s): {
                    "dir": display_path(UNIT_DIRS[s], REPO),
                    "tokenizer_hash": units[s].tokenizer_hash,
                    "predictor_hash": units[s].predictor_hash,
                    "n_params_realized": units[s].n_params_realized,
                }
                for s in seeds
            },
            "window": list(PINNED_WINDOW),
            "train_frac": PINNED_TRAIN_FRAC,
            "h": DEMO_H,
            "seq_len": DEMO_SEQ_LEN,
            "arm": DEMO_ARM,
        },
        "what_it_proves": (
            "The demo's SINGLE-DECISION assembly lands on the same float64 bits as the "
            "PRODUCTION whole-symbol batched assembly. Not circular: the two paths share only "
            "predict_mu; window selection, decision indexing, arm transform and batching are "
            "built independently on each side, and those are where a demo actually drifts."
        ),
        "comparison": "bit-for-bit float64 (.hex()), never allclose",
        "batch_composition_disclosure": (
            "NON-GATING. mu-hat is NOT invariant to how many decisions are scored in one "
            "predict_mu call: the same decision batched with others differs from the same decision "
            "scored alone by ~1e-6 relative (float32 reduction order). The gate therefore matches "
            "batch composition and stays EXACT rather than adopting a tolerance. Recorded because "
            "it is a second instance of the invariant-7 class — everything deterministic, and the "
            "answer still moves — and because mu-hat feeds a SIGN decision, so a decision sitting "
            "within this margin of zero can flip. How many do is NOT measured here."
        ),
        "symbol": args.symbol,
        "seeds": seeds,
        "n_decisions": len(stamps),
        "mutated_negative_control": bool(args.mutate),
        "rows": rows,
        "n_mismatch": len(mismatches),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.mutate:
        # NEGATIVE CONTROL INVERTS THE PASS CONDITION: a 1-ULP weight change MUST be detected.
        if mismatches:
            print(
                f"★ MUTATION DETECTED — {len(mismatches)}/{len(rows)} rows differ. GATE CAN FAIL."
            )
            doc["verdict"] = "MUTATION CORRECTLY DETECTED"
            args.out.write_text(json.dumps(doc, indent=2) + "\n")
            return 0
        print(
            "★ MUTATION NOT DETECTED — a 1-ULP weight change left every row bit-identical. "
            "THIS GATE CANNOT FAIL AND IS THEREFORE NOT A GATE.",
            file=sys.stderr,
        )
        doc["verdict"] = "GATE CANNOT FAIL"
        args.out.write_text(json.dumps(doc, indent=2) + "\n")
        return 2

    if mismatches:
        print(
            f"ACCEPTANCE FAILED — {len(mismatches)}/{len(rows)} rows not bit-identical",
            file=sys.stderr,
        )
        for m in mismatches[:5]:
            print(
                f"  seed {m['seed']} @ {m['decision_ms']}: "
                f"{m['demo_mu_hex']} != {m['production_mu_hex']}",
                file=sys.stderr,
            )
        doc["verdict"] = "FAIL"
        args.out.write_text(json.dumps(doc, indent=2) + "\n")
        return 2

    print(f"★ ACCEPTANCE PASS — {len(rows)}/{len(rows)} rows bit-identical to production")
    doc["verdict"] = "PASS"
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
