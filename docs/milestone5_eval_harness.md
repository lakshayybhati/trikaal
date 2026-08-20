<!-- DRAFT — adversarial-review section filled after the review workflow + fixes. -->
# Milestone 5 — Eval harness (the co-equal subsystem), machine-validated on the M3 checkpoint

Goal (ROADMAP M5): build the entire §8 eval harness as reusable `trikaal/eval/*` modules and prove
**the machine is correct and leak-free** on the M3 single-symbol checkpoint — *before* M6 spends
~15 GPU-days. The bar is correctness of the apparatus, **not any result**.

**Verdict: Gate A satisfied (machine validated).** The full harness runs end-to-end and leak-free
on the M3 checkpoint; every metric passes a known-answer test; the §8.A.5/§8.E.5 causal exhaustive
sweep passes on a Q4 sample. **It produces NO meaningful number** — dev-grade model, one symbol,
synthetic Q3/Q4. Real numbers come at M6.

## What was built (`trikaal/eval/*` — peers of `model/`, called by M6)

| Module | §8 | Contents |
|---|---|---|
| `metrics.py` | B.3 | net-IR on the portfolio period-return series (population std, `√(525600/h)` annualization), incremental-IR, `break_even_cost` |
| `costs.py` | B.2 | `f_taker + 0.5·spread·max(1,σ_t/σ̄) + k_impact·(Q/ADV)`, `c_total=2·c_side`, perp `funding_cost` |
| `strategy.py` | B.1/B.3 | execution filter `θ=κ·c_total`, ±1/0 positions, per-trade net P&L, equal-weight portfolio-period aggregation + breadth |
| `dsr.py` | E.4 | Deflated Sharpe (PSR + expected-max-Sharpe), PBO via CSCV, from-scratch normal CDF/inverse-CDF |
| `diagnostics.py` | D | vol MAE/R² (vs persistence), pinball, PICP/MPIW, discriminative score, TSTR |
| `folds.py` | A | train-once-eval-forward (1 train span + K=6 forward blocks), sample-level purge, leading embargo E=120, Q1–Q4, embargo-flatness gate |
| `placebo.py` | C.4 | Cell-5 shuffled-microstructure surrogate (block time-permute + phase-randomize) + Δ_micro/Δ_placebo |
| `predict.py` | B.1 | multi-horizon μ̂ via the M3 KV-cache rollout + tokenizer decode |
| `harness.py` | — | the end-to-end orchestrator wiring all of the above |
| `ic_screen.py` | B/E.8 | (M2) IC/RankIC + moving-block bootstrap + MDE/effective-N — reused |

Plus `tokenizer.decode_tokens` (the inverse of `encode_tokens`, the rollout→features path) and the
runner `scripts/m5_eval_harness.py`.

## Phase 1 — every metric passes a KNOWN-ANSWER test (the highest-stakes code)

| Primitive | Known answer (hand-computed) |
|---|---|
| `periods_per_year(h)` | 1→525600, 5→105120, 15→35040, 60→8760 |
| `information_ratio` | r_p=[2,0,2,0] (mean 1, pop-std 1) @h=15 → IR_net=√35040 |
| `break_even_cost` | IR=[3,1,−1,−3] on c=[.001,.002,.003,.004] → c_break=0.0025 (interp); all+→+inf, all−→−inf |
| cost components | taker-only c_total=8e-4; half-spread σ_t=2σ̄→1e-4 (floor at 1×); impact 0.1·0.01=1e-3; round-trip 0.21% |
| `funding_cost` | long [1e-4,1e-4]→+2e-4; short→−2e-4; no settlement→0 |
| filter `positions` | μ̂=[.01,−.01,.0005,−.0005,.002], θ=.001 → [1,−1,0,0,1] |
| `net_trade_returns` | long(+.02,c.001,f.0002)=.0188; short=−.0208 |
| portfolio aggregation | period=[0,0,1] net=[.01,.03,−.02] → r_p=[.02,−.02], breadth=[2,1] |
| `norm_cdf`/`norm_ppf` | Φ(1.96)=.975, Φ⁻¹(.975)=1.96, roundtrip; PSR(.1,0,101,0,3)≈.8407 |
| `expected_max_sharpe` | var=1,N=10 → SR₀≈1.576; grows with N and with var |
| `pbo_cscv` | strictly-dominant config → PBO=0; pure noise → 0.2<PBO<0.8 |
| folds | train_end=700,E=120 → valid train anchors t<580 (max 579); purge-only t<640; blocks partition [700,1000) |
| diagnostics | vol R²: perfect→1, at-mean→0; pinball 0.9 under-pred=0.9; PICP=0.4 on [0,2] vs [1..5] |
| placebo | block-permute preserves micro marginal, destroys alignment; phase-randomize preserves \|FFT\| |

**52 eval-suite tests green** (all KATs + Gate-A + the end-to-end harness smoke + `decode_tokens`;
`tests/eval` 50 — incl. the §6-item-10 instrument KATs added at M6 Phase 0 — + `decode_tokens` 2).

## Phase 4 — Gate-A causal re-validation

The §8.A.5/§8.E.5 pipeline-purity exhaustive truncation sweep is re-run as a merge gate on the
harness, on a Q4-representative synthetic slice (bad-tick / segment-boundary / break / stale
regions): **PASS — leak-free, 100% bar coverage** — and it still has teeth (a planted lookahead is
caught). On the runner's real pass: `gate-A PASS ✓ (420/420 bars)`.

> **★ SUPERSEDED — THE M6 DEFERRAL BELOW NEVER HAPPENED AND MUST NOT BE ACTED ON.**
> This paragraph says the Kronos integration "is folded into M6's Cell-1 work, where
> Kronos weights load anyway". **They do not load anywhere.** G-§8.C.3 was dropped as
> binding on 2026-08-03 (prereg §7 v1.6.22) and never executed
> (`external_validation.GATE_IS_BINDING = False`), and downloading Kronos weights is
> forbidden by invariant 8 in `docs/ENGINEERING.md`, which outranks this document.
> Retained as the M5 record.

**Metric-code cross-check vs Kronos — deferred to M6 (documented).** Loading official Kronos-small
weights through our harness validates our IC/RankIC *implementations* against an external reference
(§8.C.3 steps 1–2,4). The binding metric validation at M5 is the Phase-1 known-answer tests; the
Kronos integration (weights download + their input pipeline) is folded into M6's Cell-1 work, where
Kronos weights load anyway, to avoid a rabbit hole. (The "our Cell-1 within ~10–15% of Kronos"
tolerance gate needs a *trained* Cell-1 → it is an M6 gate, not M5.)

## End-to-end smoke on the M3 checkpoint (NUMBERS ARE MEANINGLESS)

Ran on the real BTCUSDT-2023 lake (dataset_hash `1d3ec4f8…` ✓) with the M3 fixture (tokenizer
`ce7c8c4a…`, predictor `6d118ede…`), `cap=200` decisions/block, h ∈ {5,15,60}. The full pipeline
executed at every horizon — folds, rollout predictions, κ-search on VAL, headline backtest @0.30%,
cost-stress curve, break-even, DSR, PBO (over the κ configs), the Cell-5 placebo (real vs
shuffled-micro tokens), the secondary diagnostics, and the Q3/Q4 synthetic 2-symbol code path.

Representative (h=15, after the μ̂ fix): `κ*=2.0, net-IR@0.30%=−46.32, cost-stress −35/−112/−226,
DSR=0.001, PBO=nan (too few common κ-config periods at this h), placebo IR(real)=−46.3 vs
IR(shuffled)=−71.3, RankIC=+0.046`. These are **garbage** (a dev-grade model that barely beats
token-frequency, one symbol, ~200 decisions) — the deeply negative IRs and degenerate `nan`/`±inf`
break-evens are exactly what a meaningless run should look like. **The machine ran every stage
without error and leak-free; that is the deliverable.** Outputs content-hashed
(`results_hash 75caa9ef…` at M5 time).

> **Anchor history (M6 Phase 0, 2026-07):** `75caa9ef` and its unpinned-env successor `91ed9054`
> are both **RETIRED**. `75caa9ef` failed to reproduce under dependency drift (numpy 1.x→2.3.3 /
> torch→2.11.0 flipped κ* 2.0→3.0 deterministically); before a lockfile landed the env drifted
> again (numpy 2.4.6 / torch 2.12.1), and the 2026-07-04 audit found instrument defects (headline
> netted at the modeled ~0.11 % cost while labeled "@0.30 %"; active-only-series annualization
> inflating IR by ~1/√activity; a time-misaligned PBO matrix) — fixed in m6_design §6 item 10,
> each with a KAT. The NEW anchor, established under the **committed** env (`uv.lock`;
> numpy==2.4.6 / torch==2.12.1 / py 3.11.7, device=mps, seed 0) with the corrected instrument,
> **run twice → bit-identical**:
> `results_hash sha256:5eead7b6cb90ffe2da0ac48752ebb819de186dbae665e954baec2e7d8b7bea46`
> — durable record in `runs_manifest/gate_a_run_manifest.json` (committed). Any later
> non-reproduction is a fail-closed STOP (m6_preflight Item 1).
>
> **Anchor history — RE-ANCHORED 2026-07-21 (prereg §7 v1.4.2, μ̂ estimator = conditional
> mean).** `predict.py` is inside the anchored instrument; the acceptance-run adjudication
> changed the default μ̂ estimator from greedy-argmax (the conditional MODE — sign-saturated
> under skew) to `"expectation"` (the conditional MEAN μ̂ is pre-registered as), so the
> results_hash re-derives under the standing procedure. New anchor, same committed env
> (numpy==2.4.6 / torch==2.12.1 / py 3.11.7, device=mps, seed 0), **run twice → bit-identical**:
> `results_hash sha256:3f86882a63dd06c780e7d73f61a5253e39f5b07564d4c84152bbaad62c886dc3`
> (supersedes `5eead7b6…`, retained as history; `gate_a_run_manifest.json` updated). The KAT
> suite (incl. the new skewed-toy known-answer estimator KAT) + both Gate-A causal tests carry
> logic-integrity across the transition; any later non-reproduction of `3f86882a…` is the
> fail-closed STOP.

## Adversarial review (required before declaring M5 done)

A 7-agent multi-agent review swept the highest-stakes code across 5 dimensions (net-IR/annualization,
cost, DSR/PBO, folds/purge/embargo, harness-integration/decode), each finding independently verified
against the §8 spec formulas. **It caught exactly the failure class M5 exists to prevent — a
lookahead in the headline-metric signal path** — fixed before declaring M5 done.

| # | Finding | Severity | Resolution |
|---|---|---|---|
| 1 | **Rollout μ̂ had a one-bar forward bias** — accumulated decoded `ret_close` over k=1..h → `log(C_{t+h}/C_t)`, including the t→t+1 move that entry-at-`C_{t+1}` cannot capture; the realized label is `log(C_{t+h}/C_{t+1})`. The extra term drove the position sign and threshold, inflating apparent μ̂↔y alignment. | **HIGH** | **Fixed** in `predict.py` — exclude the k=1 term from the sum (still feed its token to advance the cache), so μ̂ covers `[t+1, t+h]`. New KAT pins `μ̂(h=1) ≡ 0` against the harness label. |
| 2 | cost-model `σ̄` from a full-series mean (future bars) in the vol-scaled half-spread | (pre-fixed) | **Already fixed** before the verify pass — `σ̄` is a **causal expanding mean** (bars ≤ t). The review's verifier read the fixed code and dismissed it. |

Both are precisely the bug class the review targeted (a net-IR / lookahead defect that, surviving to
M6, would silently bias every cell). The μ̂ fix is structural correctness of the machine, independent
of M5's meaningless numbers.

## What M5 does and does NOT

- **Does:** make the measurement apparatus trustworthy — every metric KAT'd, folds/purge/embargo
  leak-guarded, the causal sweep re-run as a harness merge gate, the placebo + DSR/PBO machinery
  exercised end-to-end on a real checkpoint.
- **Does NOT:** produce any headline number. Dev-grade model, one symbol, **synthetic** Q3/Q4 (real
  cross-sectional numbers need the M4 universe). It must not be cited as a result.

## Honest caveats

- **M5 prediction decode is a documented contract, not the final head.** μ̂ rolls out the model and
  treats the decoded standardized `ret_close` as a vol-relative move de-normalized by causal σ_t
  (§3.4 contract); M6 finalizes the exact prediction head. Fine for a machine test (numbers
  meaningless).
- **Single symbol → Q3/Q4 are a synthetic-fixture code path**, not real cross-sectional results.
- **`σ̄` / ADV proxies:** the per-symbol trailing-vol mean and Q/ADV use full-series / constant
  proxies at M5; M4 supplies the causal per-symbol ADV and σ̄. (Flagged for the review.)
- **MPS dev-grade**, not a determinism-gated artifact; seed-pinned + content-hashed outputs.

## Out of scope (deferred per ROADMAP)

M4 universe ingest (the cloud boundary); M6 ablation (the 5-cell experiment + real net-IR / placebo
verdict / DSR-deflated headline); the Kronos external-validation tolerance gate (M6).
