"""C-20 leg 1 — THE EVIDENCE FOR THE embargo_flatness RULING.

`embargo_flatness` (folds.py) is an EMPIRICAL leakage gate: run the headline at E ∈ {60,120,240}
and require the IR to be flat past 120. It has no caller outside its own test. Wiring it in means
running the money leg at three embargo values — and because the embargo binds
`fold_valid_starts`, which gates which TRAINING windows are legal, each value needs its own
training. That is 3× the M6 run.

What the embargo actually rests on is a PREMISE: `EMBARGO_DEFAULT = H_MAX_DEFAULT (60) +
L_CORR_DEFAULT (60)`, i.e. an assumption that serial correlation in 1-minute returns dies within
60 bars. `embargo_flatness` tests that premise end-to-end and expensively. This measures it
DIRECTLY, on the real lake, for $0 — which does not replace the end-to-end gate but does convert
"assumed 60" into "measured".

CONTROL ARM: a synthetic AR(1) series with a KNOWN correlation length is passed through the same
estimator and must recover it. If the control fails, the lake numbers mean nothing and the probe
says so instead of reporting them.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

from trikaal.eval.folds import EMBARGO_DEFAULT, H_MAX_DEFAULT, L_CORR_DEFAULT

LAKE = Path("processed/universe_bars")
OUT = Path("runs_manifest/m6_c20_embargo_premise.json")
LAGS = (1, 5, 15, 30, 60, 120, 240)
MDE_INPUTS = Path("runs_manifest/m6_mde_inputs.json")


def _acf(x: np.ndarray, lags) -> dict[int, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    denom = float(x @ x)
    return {
        int(k): (float(x[:-k] @ x[k:]) / denom if denom > 0 and k < x.size else float("nan"))
        for k in lags
    }


def main() -> int:
    # ---- CONTROL: a known AR(1). rho^k must be recovered. ----
    rng = np.random.default_rng(0)
    rho = 0.97  # e-folding length ~ 1/(1-rho) ~ 33 bars
    n = 200_000
    e = rng.standard_normal(n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = rho * ar[i - 1] + e[i]
    ctl = _acf(ar, LAGS)
    ctl_ok = all(abs(ctl[k] - rho**k) < 0.05 for k in (1, 5, 15, 30))
    rep: dict = {
        "what": "C-20 leg 1 — is the L_corr = 60 premise behind EMBARGO = 120 supported?",
        "control_arm": {
            "series": f"AR(1) rho={rho}, n={n}",
            "recovered_acf": ctl,
            "expected_rho_pow_k": {int(k): rho**k for k in LAGS},
            "recovers_known_answer": ctl_ok,
        },
    }
    if not ctl_ok:
        rep["verdict"] = "PROBE INVALID — the estimator cannot recover a known correlation length"
        OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 1

    # ---- the real lake: |return| (volatility clustering) and signed return ----
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    all_syms = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT symbol FROM read_parquet('{LAKE}/**/*.parquet', hive_partitioning=1)"
            " ORDER BY symbol"
        ).fetchall()
    ]
    # §7 v1.6.17 — BUILDER DEFECT, DISCLOSED AND FIXED. The first run took `all_syms[:40]`, i.e.
    # the first forty symbols ALPHABETICALLY, and the report called them "the pinned primary
    # cross-section size". They were not the pinned set. The embargo defends the HEADLINE
    # comparison, which is computed on the pinned 40 — so that is the set that matters, and it is
    # now read from the artifact rather than approximated by a slice. The full 200 is measured
    # too, because it costs minutes and removes the question entirely.
    pinned = sorted(json.loads(MDE_INPUTS.read_text())["symbols_sampled"])
    syms = [s for s in pinned if s in set(all_syms)]
    assert len(syms) == 40, f"expected the pinned 40, got {len(syms)}"

    signed, absr = [], []
    signed_all, absr_all = [], []
    for s in all_syms:
        r = con.execute(
            f"SELECT raw_ret_close FROM read_parquet('{LAKE}/symbol={s}/**/*.parquet')"
            " ORDER BY bar_open_ms"
        ).fetchnumpy()["raw_ret_close"]
        r = np.asarray(r, dtype=np.float64)
        r = r[np.isfinite(r)]
        sa, aa = _acf(r, LAGS), _acf(np.abs(r), LAGS)
        signed_all.append(sa)
        absr_all.append(aa)
        if s in set(syms):
            signed.append(sa)
            absr.append(aa)

    def agg(rows):
        return {
            str(k): {
                "mean": float(np.nanmean([d[k] for d in rows])),
                "max_abs": float(np.nanmax([abs(d[k]) for d in rows])),
            }
            for k in LAGS
        }

    sig, ab = agg(signed), agg(absr)
    sig_all, ab_all = agg(signed_all), agg(absr_all)
    rep["scope"] = {
        "primary_set": "the PINNED 40 (m6_mde_inputs.json symbols_sampled) — the symbols the "
        "headline IR is computed on, which is the comparison the embargo defends",
        "also_measured": f"ALL {len(all_syms)} lake symbols, so the choice of 40 cannot be "
        "questioned; it cost minutes, not a decision",
        "n_pinned": len(syms),
        "n_all": len(all_syms),
    }
    rep["all_200_symbols"] = {"signed_return_acf": sig_all, "abs_return_acf": ab_all}
    rep["lake"] = {
        "n_symbols": len(syms),
        "signed_return_acf": sig,
        "abs_return_acf_volatility_clustering": ab,
    }
    rep["reading"] = {
        "EMBARGO_DEFAULT": EMBARGO_DEFAULT,
        "H_MAX_DEFAULT": H_MAX_DEFAULT,
        "L_CORR_DEFAULT": L_CORR_DEFAULT,
        "signed_acf_at_L_corr_60_mean": sig["60"]["mean"],
        "signed_acf_at_L_corr_60_worst_symbol": sig["60"]["max_abs"],
        "abs_acf_at_L_corr_60_mean": ab["60"]["mean"],
        "abs_acf_at_L_corr_60_worst_symbol": ab["60"]["max_abs"],
        "premise_supported_for_SIGNED_returns": bool(sig["60"]["max_abs"] < 0.05),
        "premise_supported_for_ABS_returns": bool(ab["60"]["max_abs"] < 0.05),
        "ALL200_signed_acf_at_60_mean": sig_all["60"]["mean"],
        "ALL200_signed_acf_at_60_worst": sig_all["60"]["max_abs"],
        "ALL200_premise_supported_for_SIGNED_returns": bool(sig_all["60"]["max_abs"] < 0.05),
    }
    rep["what_this_does_and_does_not_show"] = (
        "SIGNED-return autocorrelation is what a purge/embargo must outrun for a LABEL to be "
        "independent of a training window, and it is what L_corr names. |return| clustering is "
        "long-memory by construction in any financial series and persists far past 60 bars — that "
        "is a volatility property, not a label-leakage channel, and an embargo sized to it would "
        "have to be days long. Both are reported so the distinction is visible rather than "
        "assumed."
    )
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "lake"}, indent=2, sort_keys=True))
    print("\nsigned ACF (mean / worst symbol):")
    for k in LAGS:
        print(f"   lag {k:4d}:  {sig[str(k)]['mean']:+.5f}  /  {sig[str(k)]['max_abs']:.5f}")
    print("abs ACF (mean / worst symbol):")
    for k in LAGS:
        print(f"   lag {k:4d}:  {ab[str(k)]['mean']:+.5f}  /  {ab[str(k)]['max_abs']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
