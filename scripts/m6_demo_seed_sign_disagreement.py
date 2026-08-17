"""How often do the three cell-1 seeds disagree about which way to trade? — and its baseline.

WHY THIS SCRIPT EXISTS AT ALL, which is the point worth reading. The receipt it writes was already
COMMITTED, and no code in the repository produced it: it was computed in a session, pasted into
runs_manifest/, and cited onward. That is the same class as the hand-authored
`bits_per_dim_at_rho_0.951` block — a receipt whose provenance is a person's memory is not a
receipt — and `scripts/m6_receipt_provenance_sweep.py`, which exists to catch exactly that, reported
it CLEAN. Not because its keys were washed in by prose (I assumed that, measured it, and was wrong:
excluding prose moves this receipt from 11 to 14 untraceable keys and still would not flag it) but
because the sweep's detector was block-shaped and this receipt is flat. Both are fixed there; the
missing producer is fixed here.

WHAT IS MEASURED. For each decision instant, the three seeds' mu-hat, and whether their signs
agree. Since §7 v1.4.4 established that theta = kappa*c NEVER BINDS (traded == scored, so the
money-leg IR is a pure function of sign(mu-hat)), sign disagreement is not a diagnostic ABOUT the
IR spread — it is the IR spread, at its source.

★ THE RAW RATE IS NOT THE FINDING AND THE SCRIPT REFUSES TO LET IT BE READ ALONE. 86.2% of
decisions have the seeds disagreeing; INDEPENDENT signs with these same per-seed marginals give
84.2%. The headline rate sits at its own chance baseline and carries nothing. What carries is the
per-seed MARGINAL: P(mu>0) is 0.3500 / 0.3250 / 0.8625, and seeds 0 and 4 take opposite directions
more often than they agree. The baseline is therefore computed here and emitted beside the rate,
per the self-scaling rule — never a hand-chosen band, always the statistic's own baseline.

    .venv/bin/python scripts/m6_demo_seed_sign_disagreement.py
    .venv/bin/python scripts/m6_demo_seed_sign_disagreement.py --check   # reproduce, write nothing
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from m6_demo_acceptance import pick_decisions  # noqa: E402

from trikaal.demo.inference import (  # noqa: E402
    DEMO_H,
    available_seeds,
    forecast_at,
    load_unit,
    prepare_symbol,
    verify_lake_identity,
)
from trikaal.utils.paths import display_path  # noqa: E402

OUT = REPO / "runs_manifest" / "m6_demo_seed_sign_disagreement.json"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
N_PER_SYMBOL = 40


def measure(symbols, n_per_symbol: int, *, device: str = "cpu") -> dict:
    verify_lake_identity()
    seeds = available_seeds()
    if len(seeds) < 2:
        raise SystemExit(f"need >=2 units to compare signs, have {seeds}")
    units = {s: load_unit(s, device=device) for s in seeds}

    rows = []
    for symbol in symbols:
        stamps = pick_decisions(symbol, n_per_symbol, device=device)
        prep = prepare_symbol(symbol, units=units, device=device)
        for ms in stamps:
            b = forecast_at(symbol, ms, units=units, device=device, prepared=prep)
            mus = [sf.mu_hat for sf in b.seeds]
            rows.append(
                {
                    "symbol": symbol,
                    "ms": int(ms),
                    "mus": mus,
                    "unanimous": all(m > 0 for m in mus) or all(m < 0 for m in mus),
                }
            )

    n = len(rows)
    n_dis = sum(1 for r in rows if not r["unanimous"])
    # THE BASELINE, computed from the realized marginals — the rate means nothing without it.
    p_pos = {s: sum(1 for r in rows if r["mus"][i] > 0) / n for i, s in enumerate(sorted(units))}
    ps = [p_pos[s] for s in sorted(units)]
    expected_unanimity = _prod(ps) + _prod([1.0 - p for p in ps])
    pairwise = {
        f"{a}&{b}": sum(
            1
            for r in rows
            if (r["mus"][sorted(units).index(a)] > 0) == (r["mus"][sorted(units).index(b)] > 0)
        )
        / n
        for a, b in itertools.combinations(sorted(units), 2)
    }
    return {
        "n_decisions": n,
        "n_sign_disagreement": n_dis,
        "frac": n_dis / n,
        "note": (
            f"{len(units)} seeds, cell1, h={DEMO_H}, universe lake; sign(mu_hat) is what the "
            "money path trades on"
        ),
        "READ_THIS_FIRST": (
            f"THE RAW DISAGREEMENT RATE IS NOT THE FINDING AND MUST NOT BE QUOTED ALONE. "
            f"{n_dis / n:.1%} of decisions have the three seeds disagreeing on sign, but "
            f"INDEPENDENT signs with these same per-seed marginals would give "
            f"{1.0 - expected_unanimity:.1%} — so the headline rate is at the chance baseline and "
            "carries no information by itself. Reporting it without the baseline would be "
            "inventing alarm. (Self-scaling rule: scale against the statistic's own baseline, "
            "never a hand-chosen band.)"
        ),
        "THE_ACTUAL_FINDING": (
            "The three seeds carry STRUCTURALLY DIFFERENT DIRECTIONAL BIASES. P(mu>0) is "
            + " / ".join(f"{p:.4f}" for p in ps)
            + " for seeds "
            + " / ".join(str(s) for s in sorted(units))
            + " — seed 4 is long most of the time while seeds 0 and 2 are long about a third of "
            "the time. Pairwise sign agreement between seeds 0 and 4 is "
            f"{pairwise.get('0&4', float('nan')):.4f}, i.e. they take OPPOSITE directions more "
            "often than they agree. Since v1.4.4 established that theta = kappa*c never binds — "
            "traded == scored, and the money-leg IR is a pure function of sign(mu_hat) — this is "
            "the seed instability located at its SOURCE: the -28/-67/-146 IR spread and the 24.4x "
            "activity spread are not a subtle downstream artifact, they are three models that "
            "disagree about which way to trade."
        ),
        "per_seed_marginal_p_mu_positive": {str(k): v for k, v in p_pos.items()},
        # (n - n_dis)/n, NOT 1 - n_dis/n. The two are equal in arithmetic and not in float64:
        # 1 - 69/80 = 0.13749999999999996 while 11/80 = 0.1375 exactly. The first version of this
        # producer wrote the former and the --check against the committed receipt failed on this
        # one field while all 240 mu values reproduced bit-for-bit — which is the check doing its
        # job, and a reminder that "same formula" and "same float" are different claims.
        "observed_unanimity": (n - n_dis) / n,
        "expected_unanimity_if_independent": expected_unanimity,
        "pairwise_sign_agreement": pairwise,
        "LIMITS": (
            f"{len(symbols)} symbols, {n} decisions, CELL 1 ONLY (BSQ + OHLCV-only), "
            f"{len(units)} seeds, h={DEMO_H}. Suggestive, not a measurement of the ablation — no "
            "micro-arm artifacts exist to check whether this transfers. Decisions were picked on "
            "a deterministic linspace, not sampled at random."
        ),
        "rows": rows,
    }


def _prod(xs) -> float:
    out = 1.0
    for x in xs:
        out *= x
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    ap.add_argument("--n", type=int, default=N_PER_SYMBOL, help="decisions per symbol")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--check", action="store_true", help="reproduce and diff; write nothing")
    args = ap.parse_args()

    doc = measure(tuple(args.symbols.split(",")), args.n, device=args.device)

    if args.check:
        if not OUT.exists():
            print(f"RECEIPT ABSENT: {display_path(OUT, REPO)}", file=sys.stderr)
            return 2
        old = json.loads(OUT.read_text())
        # The numeric content is the claim; the prose is regenerated from it and may be reworded.
        keys = (
            "n_decisions",
            "n_sign_disagreement",
            "frac",
            "per_seed_marginal_p_mu_positive",
            "observed_unanimity",
            "expected_unanimity_if_independent",
            "pairwise_sign_agreement",
            "rows",
        )
        bad = [k for k in keys if old.get(k) != doc.get(k)]
        if bad:
            print(f"RECEIPT NOT REPRODUCED — these differ: {bad}", file=sys.stderr)
            return 2
        print(f"RECEIPT REPRODUCED — all {len(keys)} measured fields identical, rows bit-for-bit")
        return 0

    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"{doc['n_sign_disagreement']}/{doc['n_decisions']} disagree "
        f"({doc['frac']:.1%}); chance baseline "
        f"{1.0 - doc['expected_unanimity_if_independent']:.1%}"
    )
    print(f"Receipt: {display_path(OUT, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
