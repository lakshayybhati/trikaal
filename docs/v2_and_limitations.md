# v2 and limitations — the terminal register

**THIS FILE IS NOT A QUEUE. Nothing here blocks the run.**

Created 2026-08-03 under the termination condition. After the four decisions land, the only
permitted work is: the re-audit, fixes that clear the run-blocking bar, and the run. **Anything
either party finds from here goes in this file.**

## The run-blocking bar (pre-committed 2026-08-03, before the re-audit existed)

> A finding delays the run **if and only if** it would cause us to publish a **FALSE VERDICT** —
> SURVIVES when the truth is NULL, or NULL when the truth is SURVIVES — **and cannot be neutralized
> by disclosure. Everything else goes in limitations and the run proceeds.**
>
> Explicitly **NON-blocking**: secondary/diagnostic marginals; anything that widens a CI or weakens
> a claim; code quality, prose, coverage, reproducibility hygiene that moves no number; anything in
> a path the money run never executes.

**The bar was fixed before its first application and may not be adjusted in response to what the
re-audit finds.** That is the whole point of pre-committing it.

---

## A. Limitations that MUST reach the paper

These are already drafted into `docs/paper_skeleton.md` §10; listed here so the register is
complete.

| # | limitation | where measured |
|---|---|---|
| A1 | **The headline is ONE CALENDAR YEAR** — 2024-01-01T18:00 → 2025-01-01, 365 days, no cross-year replication. 40-symbol cross-section. | C-20 leg 2 |
| A2 | **The embargo is justified on the SIGNED-return channel only.** Signed ACF at lag 60 is 0.0067–0.0072 mean (≈24× margin), but \|return\| ACF is 0.20 at lag 60 and 0.15 at lag 240. *"A leakage channel that isn't signed-return autocorrelation wouldn't be caught by it."* The end-to-end alternative was costed at 3× the run and not funded. | `m6_c20_embargo_premise.json` |
| A3 | **The 16-dim input has 13 live dims.** Funding and open interest occupy dims 13–15 and are constant-zero, masked on 100 % of 304,625,181 bars. Specified and wired, structurally absent. | `m6_c15b_lake_surface_check.json` |
| A4 | **Two of the eight causal-safety surfaces were never exercised on the published lake until after it was built** — the ingest swept 800 bars against a 1,440-bar warm-up, on 2 of 200 symbols, unpersisted. Closed retrospectively (1,600 anchors, 12,798 checks, both planted leaks caught). | `m6_c15b_production_sweep.json` |
| A5 | **BSQ and FSQ arms differ by 69,632 backbone params** (0.327 %), an artifact of vocabulary-dependent embedding/head rows. **Not convertible to IR units.** | C-19 |
| A6 | **The BSQ baseline is not externally anchored.** *"We therefore cannot exclude that our BSQ baseline is weaker than a reference BSQ implementation, which would inflate the FSQ-vs-BSQ comparison reported in §5."* | C-4 |
| A7 | **The two handicap channels on ΔIR(4−5)** — C-12 capacity and C-1 decode-noise — are measured per-(cell, seed) and read under the pre-registered claim-rule bands. Magnitudes in their own units; **no reconstruction-to-IR mapping exists** and none is implied. | §7 v1.6.13 |
| A8 | **C-1 and the expectation estimator are NOT independent** — the delta-method error is evaluated in the same out-of-distribution single-bar regime C-1 describes. | S-3 |
| A9 | **The eval leg's cost was inferred by subtraction until 2026-08-03.** Training is 2.6–6.1 % of spend; the rest was never measured at the pinned geometry until the eval-throughput probe. | §7 v1.6.19 |

---

## B. Withdrawn claims — recorded so they are not quietly revived

| # | what | why |
|---|---|---|
| B1 | **The three 2×2 marginals with a BSQ arm** — IR(2)−IR(1), IR(4)−IR(3), IR(3)−IR(1) — are **reported with CIs and claimed as nothing** (Band-B treatment, not deletion). | The FSQ leg is the only claim in the design with no internal control (you cannot build a "placebo BSQ" — the quantizer *is* the implementation), and its sole external control is unexecutable. **Directions differ and the disclosure says so:** a weak BSQ *inflates* the first two in our favour; IR(3)−IR(1) is BSQ-minus-BSQ, affected but undetermined and probably attenuating. The "we do not keep a favourable defect" argument applies only to the first two. |
| B2 | ΔIR(4−2) as a substitute for the primary if band C fires | capacity-mismatched (7 dims vs 16); it never pays the noise-encoding cost, so it is not a clean substitute and must not be presented as one. |
| B3 | *"free microstructure — TFI, funding, open interest"* | two of the three named channels are constant-zero on 100 % of bars. Corrected repo-wide, §7 v1.6.17. |

---

## C. v2 / out of scope — firewalled, not forgotten

| # | item | note |
|---|---|---|
| C1 | **True OFI** (orderbook depth) | the standing v1 firewall; TFI is the v1 quantity and the naming invariant exists to protect exactly this distinction |
| C2 | **Funding / OI ingestion** (futures API) | would make dims 13–15 live and turn A3 from a limitation into a feature. The `assert_perp_dims_masked` tripwire already fails loudly if they activate without the cell-5 shuffle set being widened. |
| C3 | **G-§8.C.3 re-specified**: run Kronos-small on OUR crypto slice and compare Cell 1 against *that*, rather than against a published SSE-equities figure | well-posed, needs the invariant-8 loader; the right version of the gate, for a version of the project that has time for it |
| C4 | **The Stage-1/2 early-stop machinery** (`:1912`, `:1924`) | never implemented. At the raised 26,003-step budget its trigger *could* fire (Stage 2 needs ≥25,000), so v2 could adopt it — but only with the matched-budget confound resolved (see C5). |
| C5 | **Early-stop vs matched budget** — the spec says early-stop; `m6_design.md:18` says all cells share the same training draw | reconciled for v1 as **fixed matched budget, saturation measured and reported**. Data-dependent stopping would give 25 different budgets and confound ΔIR(4−5) with "cell 4 trained longer" — the C-12 class in the primary. |
| C6 | **The eval prefill optimization** (22.3× local-measured, behind a non-default flag) | ~97 % of rollout wall-clock is context fill that could go through one batched causal forward. Gated on a bit-exactness proof at GPU scale; a pure-performance change that provably moves no number is legitimate, anything that moves a number is forbidden. |
| C7 | **`q_over_adv = 1e-3`** hardcoded at `harness.py:83` | unpinned, unreceipted — but it feeds the **modelled-cost secondary**, never the 0.30 %-netting headline, and `harness.py` is frozen. Non-blocking by construction. |
| C8 | **`student_t_ppf` root-finder residual** ~1.1e-8 at (0.99, df=30), ~6.4e-7 at (0.95, df=1000) | pre-existing, unmoved by the `_betacf` rewrite. The design operates at Welch df 2–4 where agreement is ~1e-12. Outside the operating range. |

---

## D. Operational constraints that survive the run

| # | constraint |
|---|---|
| D1 | **OC-1 — S=3 is unreachable in code.** If budget runs short mid-run the options are *fund it* or *declare INCONCLUSIVE per R3*. Dropping to three seeds would require moving `PINNED_SEEDS` post-hoc under freeze. Deliberate; not to be "fixed". |
| D2 | **No token should be issued for HuggingFace.** Kronos repos are public, ungated, MIT; an unauthenticated pull works. Issuing a credential would add risk for no capability. |
| D3 | **The audit register's verbatim region is sha256-pinned.** Editing the auditor's prose — including "correcting" its drifted line numbers — fails the suite by design. Drifted references are annotated *beside* the text. |

---

## E. Fan-out (added 2026-08-03, §7 v1.6.23)

| # | item |
|---|---|
| F1 | **`image` and `lockfile_sha256` were missing from `PROVENANCE_IDENTITY_KEYS`.** Two boxes can carry identical torch/numpy/driver strings and still differ in container image or resolved dependency set. Added; all 13 keys mutation-proven to refuse. Non-blocking under the bar (it *strengthens* a refusal), recorded because the gap existed while fan-out was already planned. |
| F2 | **Setup overhead must be costed, not discovered.** Measured 2026-08-03: ~17 min per box (10 min image pull + 6m41s pinned-torch install). It was **90 % of the probe's cost**. Across 5 boxes it is ~1.4 GPU-h ≈ $0.41–0.56 — trivial when planned. **Standing lesson: cost a rental as `setup + compute`.** |
| F3 | **The image ships torch 2.5.1; we pin 2.12.1.** Not a defect — but a shard that skips the install produces an identity-key refusal *after* paying for the compute. R3 in the runbook. |

## F. Housekeeping

| # | item |
|---|---|
| E1 | `runs_cloud/` and the paused prefill work (`m6_prefill_zero_mean.py`, its manifest) remain untracked. Decide at paper time whether they are artifacts or scratch. |
| E2 | The claims sweep is tracked-files-only and carries the sha256 of all swept files. Untracked receipts are invisible to it **by design** — a receipt that is not committed is not a claim we have made. |
