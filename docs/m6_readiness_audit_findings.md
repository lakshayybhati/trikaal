# M6 readiness audit — the external auditor's report, VERBATIM, plus our disposition

> ## ⚠️ THE BOUNDARY IN THIS FILE IS LOAD-BEARING. READ THIS BEFORE QUOTING ANYTHING FROM IT.
> **PART 1 is the EXTERNAL AUDITOR'S OWN TEXT, unedited.** Not ours. Not paraphrased. Its line
> references are against baseline commit **`7a4d13b`** and have since drifted under our own fixes —
> per the standing baseline rule, `git show 7a4d13b:<path>` before disputing any of them. **The
> drifted numbers are LEFT AS WRITTEN**; where a reference has moved it is annotated *beside* the
> verbatim text, never inside it.
> **PART 2 is OUR VERDICTS.** A reader — including a re-auditor — must not read Part 2 as the
> audit's own record. It is the builder's disposition, argued in the prereg §7 log.
> The boundary is machine-enforced: `tests/test_audit_findings_complete.py` pins the sha256 of the
> region between the two HTML markers, so tidying, correcting or paraphrasing a single character of
> the auditor's prose fails the suite.

**Provenance.** Recovered by the supervisor from session transcript `780b64e7` on 2026-08-03. It
was never committed at the time; that omission is recorded in the §7 log as the supervisor's
defect, and it is why the builder could not answer the S-2..S-4 question during the Tier-4 triage
and correctly refused to reconstruct it.

---

# PART 1 — THE AUDIT, VERBATIM

> **★ TRANSLATION TABLE FOR THE VERBATIM TEXT BELOW (added 2026-08-19, BESIDE it and never
> inside it).** History was rewritten on 2026-08-19: `CLAUDE.md` became `docs/ENGINEERING.md` and
> every commit SHA changed. The auditor's prose below still uses the pre-rewrite names, **and that
> is correct** — it is what they wrote. Read it with this table:
>
> | the audit says | now |
> |---|---|
> | baseline commit `74b6094` | `7a4d13b` |
> | `CLAUDE.md` | `docs/ENGINEERING.md` |
>
> `git show 7a4d13b:<path>` is the baseline-rule command for disputing any line reference below.
> **A re-stamp pass DID edit this region once**, on 2026-08-19, and `test_the_verbatim_region_is_
> byte_identical_to_what_was_recovered` caught it and named the revert as the correct response.
> Someone else's words are not a find-and-replace target, however mechanical the change looks.

<!-- ===== BEGIN VERBATIM AUDIT — DO NOT EDIT BELOW THIS LINE ===== -->

<!-- RECOVERED VERBATIM by the supervisor from session transcript
     780b64e7-a58a-4824-bd92-800e17366acc.jsonl, 2026-08-03.
     THIS IS THE EXTERNAL AUDITOR'S OWN TEXT, UNEDITED. It was never committed at the time —
     that omission is the supervisor's defect and is why the builder could not answer the
     S-2..S-4 question in the Tier-4 triage.
     Audit baseline commit: 74b6094. Line references are against THAT commit and have since
     drifted; per the baseline rule, `git show 74b6094:<path>` before disputing any of them.
     NOTHING below is builder-authored or supervisor-authored. Do not edit the body. -->

Read-only audit complete. Full test suite run (337 passed), Gate-A anchor reproduced, ruff verified, and 14 targeted probes executed. Nothing in the repo was changed — one receipt (`runs_manifest/m6_claims_sweep.json`) was regenerated as a side effect of running its own script and I restored it to HEAD.
---
# INDEPENDENT AUDIT — Trikaal M6, branch `real-data-slice` @ `74b6094`
## Baseline facts — verified, not assumed
| Claim | Verdict |
|---|---|
| 337 tests collected | ✅ 337 collected, **337 passed** in 441s |
| ruff clean | ✅ `ruff check` + `ruff format --check` (169 files) clean |
| ~11,200 lines / 75 modules in `src/` | ✅ 11,204 / 75 |
| Gate-A anchor `sha256:3f86882a…` reproduces bit-identically | ✅ **re-ran `m5_eval_harness.py`; regenerated manifest is byte-identical to the committed one** |
| §3a conformance gate passes live | ✅ `conformance_failures()` returns empty against the real pinned artifacts |
| bpt parity FSQ vs BSQ | ✅ 20.0578 vs 20.0000, \|Δ\|=0.058 bits; tokenizer params 0.25% apart |
| DSR n_trials=60, threshold 0.95, var_sr basis = cell 5 | ✅ all three machine-gated — mutations caught |
| h=15, λ=3.0, `encoder_causal=True`, `fine_pointwise=True` | ✅ hard-errors in `cells.py` + conformance |
| 40 symbols, money mode, no cap, blocks 1–5 | ✅ gated |
| seeds (0,1,2,3,4) | ⚠️ **in force on the eval side only — see C-6** |
| design FROZEN | ⚠️ working tree is dirty (26 uncommitted lines in `m6_toy_rehearsal.py`) — see C-14 |
---
# CONFIRMED FINDINGS (reproduced)
## C-1 · The money path decodes single bars through a decoder that only ever saw full windows
`src/trikaal/eval/predict.py:178,185,206` · `src/trikaal/tokenizer/model.py:104-106` · `src/trikaal/tokenizer/decoder.py:26`
**CLAIMED:** `decode_tokens` is "the exact inverse of `encode_tokens` into the decoder (the AR rollout→features path)" (`tests/tokenizer/test_decode_tokens.py:1`); μ̂ is the conditional mean of the forward return.
**TRUE:** `TokenizerDecoder` is hardcoded **bidirectional** (`causal=False`, passed as a literal at `model.py:105`, not routed from `encoder_causal`). Training and the reconstruction loss always decode a full `[B, L, d]` window with full cross-bar attention. The rollout calls `tok.decode_tokens(pc, pf)` / `tok.decode_latent(z)` on tensors of shape `[n, 1, …]` — **a length-1 sequence**. The decoder at rollout is evaluated on an input distribution the objective never optimized. Every KAT decodes the full window; none decodes a single bar.
**HOW VERIFIED:** trained a tokenizer (d_model 64, 2 layers, `fine_pointwise=True`, λ=3.0) for 1,200 steps on autocorrelated 16-dim data, then compared in-window vs single-bar decode of `ret_close` — the exact quantity the rollout accumulates:
| | in-window (trained regime) | single-bar (rollout regime) |
|---|---|---|
| std | 1.0240 | **1.4794 (1.44×)** |
| recon MAE vs truth | 0.4835 | **0.7778 (+61%)** |
| sign agreement | — | **0.8379 → 16.2% of steps flip sign** |
| mean \|Δ\| | — | 0.6431 (63% of the in-window std) |
At random init the gap is worse (std ratio 2.45×, sign agreement 0.758), so training reduces but does not remove it.
**WHAT BREAKS:** μ̂ = Σ_{k=2..h} decode(step k), and the execution filter is `|μ̂| > κ·c_modeled`. A systematically **inflated-variance, 1-in-6-mis-signed** per-step estimate propagates directly into which trades fire and which direction they take — i.e. into the headline net-IR of every cell. It is a plausible mechanical cause of the pathologies that consumed weeks here (`frac_negative` 0.99883, activity locked at 0/1, the mode-vs-mean drift): a decoder run out of distribution, compounded over h steps. Nothing in the repo measures this, and it is not disclosed anywhere.
## C-2 · The degeneracy guard silently disarms, reports `armed: true`, and the only artifact-writing driver triggers it
`src/trikaal/eval/verdict.py:294,298` (nan defaults) · `:344` (`"armed": True`) · `:154-180` (`_validate_artifact`) · `scripts/m6_toy_rehearsal.py:445-455`
**CLAIMED:** "§7 v1.4.6 HALT-only guard … PURE READ — never touches the clause computation"; the docstring concedes only that a missing `activity_decisions` skips the filter legs — "the sign legs still apply".
**TRUE:** both legs read `evals[(c,s)].get("mu_diag", {}).get(…, nan)`. `write_cell_eval_artifact` defaults `mu_diag or {}`. `_validate_artifact` does **not** require `mu_diag`. With `mu_diag` absent, every value is NaN, `_locked`/`_never_binds` are NaN-safe-False, and the guard reports `armed: true, halted: false, degenerate_cells: []` — having examined nothing. **`scripts/m6_toy_rehearsal.py:445` — the only end-to-end driver that writes verdict artifacts — omits the `mu_diag=` kwarg** (it passes `codebook` and `meta` but not `mu_diag`, which `score_cell` returns and it already holds as `head_score.mu_diag`). No test asserts a missing `mu_diag` is rejected.
**HOW VERIFIED:** probe 2/3 — built 25 artifacts through the real `write_cell_eval_artifact` with `mu_diag=None`, loaded through the real `load_cell_evals`, ran the real `assemble_verdict`:
```
clause pass: {1_paired_ci: True, 2_mde_paired: True, 3_placebo_validity: True,
              4_economic_floor: True, 5_dsr: True}
degeneracy_guard: armed=True halted=False        <-- zero data
power_guard:      armed=True halted=False
>>> EMITTED VERDICT: SURVIVES
```
**WHAT BREAKS:** the verdict path emits **SURVIVES** with both HALT guards non-functional while the manifest asserts they are armed. The guards are the project's only protection against the exact failure it already observed twice. This is the "check that cannot fail" pattern, in the decision path, live today.
## C-3 · `var_sr` averages three different time-unit Sharpes; clause 5's threshold is largely a unit artifact
`src/trikaal/eval/verdict.py:252,470-471,519-521` · `src/trikaal/eval/conformance.py:118-120`
**CLAIMED:** "var_sr = variance of the … de-annualized per-trial VAL IRs (ddof=0)"; SR₀ is the expected-max Sharpe under N trials.
**TRUE:** `enumerate_dsr_trials` divides each trial's annualized IR by `sqrt(periods_per_year(h))` **for that trial's own h**. The result is a per-5-minute, per-15-minute, or per-60-minute Sharpe depending on the trial — scale factors 0.003084 / 0.005342 / 0.010684, spanning **3.46×**. The variance of that mixed set is dominated by the horizon mix, not by trial dispersion. The statistic being deflated (`_sharpe(s[4])`) is a per-15-minute Sharpe, so SR₀ and the statistic are not in the same units.
**HOW VERIFIED:** probe 4 — under a placebo whose *annualized* VAL IRs have SD 0.25 / 0.5 / 1.0, the mixed-horizon `var_sr` is **7.8× / 3.0× / 4.0×** the same-unit (h=15-only) variance, inflating SR₀ by 1.7–2.8×. In the degenerate case the project has already observed (all placebo VAL IRs bit-identical at −7.56), `var_sr` = 5.804e-4 and SR₀ = 0.0565 is **100% artifact** — reproduced exactly by hand: the three de-annualized values are −0.02332 / −0.04039 / −0.08078, variance 5.8056e-4.
**WHAT BREAKS:** clause 5's bar is set by *which horizons were enumerated* rather than by search dispersion. It errs conservative in the scenarios tested, but the direction is not guaranteed and has never been measured. `verdict_dsr_failures` re-derives the same wrong quantity, so the "independent statement of the constants" cannot catch it. In a paper, a reviewer who checks the units kills clause 5.
## C-4 · G-§8.C.3 — the Kronos external validation gate has no implementation
`docs/m6_prereg.md:178-190` · `CLAUDE.md` invariant 4
**CLAIMED:** a **binding gate**: "Fails iff Cell 1's RankIC < 0.85 × published Kronos-small RankIC … with Kronos-small run through its own published preprocessing on the same bars", whose failure "halts the ablation before any Cell-2–5 eval statistic is computed". CLAUDE.md: Cell 1 is "**externally validated** against published Kronos-small (validation lives only in the eval harness)".
**TRUE:** grep across `src/`, `scripts/`, `tests/` finds **no** code that loads external weights, runs Kronos preprocessing, or compares against a published RankIC. No `from_pretrained`, no `hf_hub`, no transcribed published number anywhere. `src/trikaal/eval/harness.py` computes our own RankIC only.
**WHAT BREAKS:** the paper's central legitimacy claim for the BSQ baseline — that Cell 1 is anchored to published Kronos-small rather than being a private re-implementation of unknown quality — is unimplemented. Cell 1 is also the NULL-fallback's comparator (§5 claims `IR(2)−IR(1)`). Without it, a weak Cell 1 manufactures a positive FSQ result and nothing detects it.
## C-5 · No driver runs the pre-registered money configuration
`scripts/` (all) · `src/trikaal/train/orchestrator.py`
**TRUE:** `write_cell_eval_artifact` is called from exactly three places: `m6_toy_rehearsal.py`, `m6_verdict_integration_check.py` (a fixture builder), and tests. `run_all_cells` is called only from `m6_smoke.py`. `m6_toy_rehearsal.py` is the closest thing to a real driver and does run canonical dims, but it differs from the money spec on every axis that matters:
- `SEEDS = (0,1,2)` (`:70`), not the pinned five
- `--symbols` subset, not the pinned 40; `EVAL_WINDOW = 2024-01-01→2025-01-01` toy region requiring `--allow-toy-grid`
- `micro_legibility_min=None` (`:316`) — the standing §7 v1.4 hard-stop is **disabled**
- it **never calls `assert_conformance` / `money_config`** — it imports only `DECILES, cell_tokenizer_failures` from conformance (`:41`)
- "Numbers are garbage by design (few hundred steps)"
**WHAT BREAKS:** the verdict path is complete and tested; the thing that feeds it does not exist. Whoever writes it inherits C-2 (the template omits `mu_diag`), must independently re-wire the conformance gate, the legibility gate, the 5-seed loop and the real grid — all first-time events on rented hardware, which pre-flight Item 2 exists to prevent.
## C-6 · Train/eval seed split-brain: S=5 was approved and funded but never reached the training code
`src/trikaal/train/orchestrator.py:56` vs `src/trikaal/eval/conformance.py:35` and `verdict.py:52`
**TRUE:** `OrchestratorConfig.seeds = (0, 1, 2)`; `PINNED_SEEDS = DSR_SEEDS = (0,1,2,3,4)`. Nothing in the orchestrator asserts against the pin. Verified by direct import. `docs/m6_prereg.md:130` still reads "**Seeds:** exactly **3**, literal values **{0, 1, 2}** (the committed orchestrator defaults)" — the doc and the training code agree with each other and both contradict the eval-side pins and the recorded 2026-07-31 operator approval of S=5.
Same class, same file: `seq_len: int = 128` (`:58`) against the money surface's 512; `steps_stage1/2 = 2000` (`:60-61`) pinned by nothing at all.
**WHAT BREAKS:** a run launched on defaults produces 15 checkpoints; `load_cell_evals` demands 25 and hard-fails. Fail-loud, so not a silent corruption — but it means the approved, budgeted S=5 design is not encoded on the side that spends the money.
## C-7 · Three decision constants are unpinned; three `PINNED_DSR` keys are decorative
`verdict.py:57,59` · `paired_bootstrap.py:37` · `conformance.py:37,285-351`
**HOW VERIFIED:** probe 10/11 — mutated each constant and re-ran the full `load_cell_evals → assemble_verdict` path:
| constant | mutation | gate |
|---|---|---|
| `DSR_N_TRIALS` 60→12 | | ✅ caught |
| `DSR_THRESHOLD` 0.95→0.10 | | ✅ caught |
| `DSR_VAR_SR_BASIS_CELL` 5→2 | | ✅ caught (fixture discriminates — the v1.5 fix works) |
| `PRIMARY_H` 15→60 | | ✅ caught by `_validate_artifact` |
| **`ECON_FLOOR_IR` 0.5→0.0** | clause 4's "fixed pre-data" floor | ❌ **NOT caught → SURVIVES** |
| **`PLACEBO_DISPERSION_TRIPWIRE` 1.5→99.0** | | ❌ **NOT caught** — and `PINNED_DSR["placebo_dispersion_tripwire"]=1.5` exists as a sibling literal that is never read |
| **`BOOT_POWER` 0.80→0.50** | −34% on MDE_paired | ❌ **NOT caught** — `PINNED_BOOT` covers only `B`, `seed`, `alpha` |
`PINNED_DSR` keys never read by `verdict_dsr_failures`: `n_trials_factorization`, `var_sr` (prose), **`statistic`** (which cell's series is deflated — a real decision), `placebo_dispersion_tripwire`.
## C-8 · The v1.5 amendment entry contradicts itself about what was pinned
`docs/m6_prereg.md:473-475`
**TRUE:** inside the *same* v1.5 bullet that records "A.5 — N = 60 (`PINNED_DSR["n_trials"]` 180 → 60)" and "D — `PINNED_SEEDS` (0,1,2) → (0,1,2,3,4)", the log states:
> **NOTHING PINNED HAS CHANGED:** `conformance.PINNED_DSR` (N=180), `verdict.DSR_N_TRIALS`, `PINNED_SEEDS` (3), `PINNED_MICRO_POINT_WEIGHT` (3.0) … are all as ruled.
Read in context this is the tail of the earlier *window-opened, drafts-not-yet-ruled* entry, concatenated into the signed-off v1.5 bullet (the following sentences — "Nothing in the drafts is quotable as the pre-registration until it is ruled on") belong to that earlier state. The result is that §7 v1.5 asserts both N=60/S=5 and N=180/S=3.
**WHAT BREAKS:** the pre-registration is the artifact a referee reads to check that thresholds were not moved after seeing results. A self-contradicting record of what was pinned is the single most damaging thing that section can contain.
## C-9 · §3 and §3a body text were never updated to v1.5
`docs/m6_prereg.md:92-95,130,141`
The governing clause text still reads "**N = 180 enumerated trials** (5 cells × 3 seeds × …)", "var_sr = the variance of the 180 recorded per-trial de-annualized IRs", "**Seeds:** exactly **3**", "ONE mode produces all **15** runs". All four are false against the code. The amendments live only in §7.
## C-10 · The micro-legibility hard stop passes when it measures nothing
`src/trikaal/train/gates.py:193-195,202-203`
**CLAIMED:** "a dim masked ~everywhere is SKIPPED with a named receipt entry, **never trivially passed**".
**TRUE:** on `v.size < 10_000` the dim is `continue`d and `ok` is left unchanged. If all six micro dims are thin, the gate returns `{"pass": True}` and Stage-2 proceeds. The receipt records the skip; the gate does not fail. Also: `id_legibility_sign_acc` caps at `n = min(150_000, size)` taken from the **head** of the symbol-concatenated stream and splits 80/20 by position, so on a 40-symbol run the standing gate is measured on the first one or two symbols only.
## C-11 · Determinism is not merely off — it is not a parameter
`src/trikaal/train/orchestrator.py:173`
`set_determinism(seed, deterministic_algorithms=False)` is a hardcoded literal with no `OrchestratorConfig` field. `set_determinism`'s own docstring says the flag is "required for the G2 determinism gate **and any published run**". This is B1 and it is still open, but the code state is worse than "a flag left off": under amendment A there is no knob to turn on. `attention_mode.py:121` `claim = attention_mode == MODE_SDPA` still stamps `bit_exact_claim: true` beside `deterministic_algorithms: false` (the `bit_exact_caveat` string is emitted, which is honest, but the boolean a downstream reader parses is false).
## C-12 · The placebo destroys within-bar micro↔OHLCV dependence, not only predictive information
`src/trikaal/eval/placebo.py:56-82` · `src/trikaal/train/arms.py:51-64`
**CLAIMED:** Cell 5 is "bit-for-bit Cell 4 except the microstructure sub-vector … preserves its marginal distribution and cross-channel structure while **destroying temporal alignment**"; it isolates *information* from *capacity*.
**TRUE:** the block permutation moves dims 7–12 as a unit to a different bar. Within-micro cross-channel structure survives; **micro↔OHLCV contemporaneous dependence does not.** The tokenizer must therefore represent a strictly higher-entropy per-bar source at identical bits-per-token — and cell 5 additionally carries λ=3.0 micro weighting in the per-bar bottleneck leg (`model.py:120-125`), spending 3× reconstruction weight on now-independent noise. Cell 5's OHLCV reconstruction is degraded for a reason that has nothing to do with predictive microstructure information.
**HOW VERIFIED:** by construction from the code, and corroborated by the project's own measurement — `pb(5−2) = −2.59` (the placebo scored *below* the OHLCV-only counterfactual) — the "placebo victim" signature.
**WHAT BREAKS:** ΔIR(4−5) > 0 does not distinguish "microstructure carries information" from "the shuffle handicapped the placebo's tokenizer". Clause 3 (`ΔIR(4−2) > 0`) is the stated defence, but cell 2 has 7 input dims to cell 4's 16, so it controls information at the cost of controlling capacity. **There is no cell in the design that is matched on both axes.** The prereg discusses this as a *level* concern and resolves it by disclosure; the mechanism above is a *capacity-at-fixed-bitrate* concern that the disclosure does not cover.
## C-13 · `harness.py` is inside the M6 decision path while being described as frozen M5-only
`src/trikaal/eval/xsection.py:32`
`from trikaal.eval.harness import HEADLINE_COST, KAPPAS, _per_bar_cost, forward_log_returns` — the headline cost constant, the κ grid, the **label definition**, and the **cost model's causal σ̄ construction** all come from harness.py, including a private symbol imported across modules. `verdict.py:5-7` says harness "is the M5 machine-validation instrument — neither may ever be quoted as the M6 outcome", and prereg §7 v1.5 says "**`harness.py` is NOT touched** — its … 240 is the M5 instrument frozen under Gate-A". Freezing it is correct; describing it as outside the M6 path is not. Any future M5-motivated edit to those four symbols silently moves the M6 headline.
## C-14 · Uncommitted eval-resume reuses artifacts without binding them to the checkpoints that produced them
`scripts/m6_toy_rehearsal.py:401-418` (working-tree only, not committed)
The resume honours an existing artifact on `schema == "m6_cell_eval_v1" and cell_id is not None`. It never compares `meta.checkpoints` (which the fresh path writes) against the current `ckpt_hashes`, nor the symbols/window/h. A rerun after retraining, or with different `--symbols`, silently mixes artifacts from different training generations; `load_cell_evals` then content-verifies them happily because the index hashes match. The design is declared FROZEN; the working tree is not clean.
## C-15 · The causal gate is exercised only at non-production feature parameters
`tests/conftest.py:30-33` · `src/trikaal/data/config.py:86-103`
The G0 merge gate runs on `synthetic_test_config` (`half_life_fast=64`, `n_warm=16`, `w_tau=64`, `badtick_mad_window=64`, `min_segment_len=32`) over 600 bars. Production `FeatureConfig` is `half_life_fast=1440`, `n_warm=64`, `w_tau=1440`, `badtick_mad_window=1440`, `min_segment_len=128`. The gate genuinely has teeth on its own fixture — I confirmed `sigma_includes_next`, `localized_sigma_next` and `localized_validity_next` are all caught, and that the clean transforms pass 100% coverage. But under the **production** config on the same stream, `effective_n_warm_vol() = max(30, 1440) = 1440`, so `target_valid` is False for **0 of 569 bars** and the `target` / `target_valid` legs compare NaN↔NaN and False↔False — vacuous. Several checked transforms are window-length dependent (the τ percentile pool with `tau_pool_cap=64`, the trailing MAD window, the rolling z-scores); a lookahead that only manifests at production window sizes is outside the gate's coverage.
## C-16 · The modelled impact term is a constant
`src/trikaal/eval/harness.py:64-79`
`_per_bar_cost` passes `q_over_adv=1e-3` as a hardcoded literal for every symbol and every bar, while the docstring says "M4 supplies the causal per-symbol σ̄/ADV at scale". `CostModel.impact` and `k_impact` therefore contribute a flat 2e-4 round-trip offset — the "linear temporary impact" component of the cost model is not modelled. σ̄ itself *is* causal (expanding mean, verified).
## C-17 · Stale numerics in the decision path's own prose
`verdict.py:11,21-24,145,163,244,519-521` ("15 artifacts", "5 cells × 3 seeds", "N = 180", "5 × 3 × 3 × 4 = 180", "var_sr over the 180 de-annualized VAL IRs") · `conformance.py:12` ("seeds exactly {0, 1, 2}") · `m6_verdict.py:10,17,47,73` ("15 artifacts", "the pinned N=180 DSR") · `orchestrator.py:82` ("all 15 cell-runs") · `docs/m6_cuda_probe_report.md:144` ("Suite 333 green" — now 337).
The line at `verdict.py:519-521` is not a comment: it is the `rule` string **persisted into the verdict manifest**, so the shipped decision artifact would carry a false description of how its own `var_sr` was computed.
## C-18 · Parameters that look pinned and are not
`pyproject.toml:10` / `uv.lock:3` — `requires-python = ">=3.11"` with no `.python-version`; the interpreter is still unpinned (the known 3.14-vs-3.11 hazard is unfixed). `OrchestratorConfig.steps_stage1/2 = 2000` — the Stage-1/2 step budget is a code default, referenced by no spec, gate or receipt. `predict_mu(estimator="expectation")` is a function default (`predict.py:58`) that `score_cell` relies on implicitly; no gate asserts it.
## C-19 · The realized backbone differs between quantizer arms
`src/trikaal/train/cells.py:88-118`
FSQ cells (2/4/5): **21,301,248**. BSQ cells (1/3): **21,231,616** — 69,632 fewer (0.327%), from the 2048 vs 2116 embedding/head rows. Inside the documented ±2% band and disclosed in the code comment, but `CLAUDE.md` and the paper quote "21,301,248 realized params" as *the* measurement vehicle, which is true for three of the five cells.
## C-20 · Two disclosed-but-load-bearing scope facts
- **The headline is one calendar year.** `runs_manifest/m6_mde_inputs.json:primary_region_ms = [1704132000000, 1735689600000]` = 2024-01-01 → 2025-01-01. T = 35,063 periods, but one macro regime.
- **`embargo_flatness` — described in `folds.py:88` as "the leakage gate" — is never called by anything.** `folds.py` is imported only by `harness.py` (M5). The M6 embargo *geometry* is applied on the training side via `universe_loader.fold_valid_starts` (E=120, verified), so the invariant holds structurally; but the flatness verification that would *demonstrate* it is not part of M6.
- The claims-sweep receipt is stale: committed counts P1 23 / P2 40 / P2_HIGH 5; re-running on the current tree gives **29 / 43 / 8** (`m6_cuda_probe.json` postdates the receipt). Three of the new P2_HIGH hits are false positives — the detector flags `seed = 0` as a zero-sentinel.
---
# SUSPECTED (reasoned, not reproduced at scale)
- **S-1 — C-1's magnitude at canonical dims is unknown.** I measured the decode gap on a d_model=64 / 2-layer tokenizer on synthetic data. The canonical tokenizer (d_model 256, 3 layers) on the real lake may narrow it. The *structure* is certain; the *size* is not. This is the highest-value cheap experiment available and costs $0.
- **S-2 — `power_guard` is nan-fragile in the same way as C-2.** With non-finite per-seed IRs, `ir_range_across_seeds` is `None`, `worst` is `None`, and `trips` is False; with a NaN Δ it reports `seed_spread_exceeds_claim: false`. Reproduced in probe 1, but I could not construct a realistic artifact set that reaches it, so I am not claiming it fires in practice.
- **S-3 — `expectation` is a plug-in, not the pre-registered quantity.** `decode_latent(E[z])` ≠ `E[decode_latent(z)]` for a nonlinear decoder, and `p_f` is conditioned on the argmax coarse rather than marginalized. Both are disclosed in `predict.py:8-22` and were adjudicated in v1.4.7 item 4 (retained because mc@32's noise floor exceeds the disagreement). I flag it only because C-1 compounds it: the delta-method error is evaluated in the same out-of-distribution single-bar regime.
- **S-4 — provenance of two numerical routines.** `dsr.norm_ppf` is Acklam's published rational approximation (attributed) and `tdist._betacf` is the Numerical Recipes modified-Lentz `betacf` in structure. These are standard math, not research surface, and both are cited — but NR code carries a restrictive licence, and a released artifact should not transcribe it. `attention.apply_rope`/`_rotate_half` is the standard LLaMA/HF RoPE formulation; unavoidable and not a violation, but worth one attribution line given invariant 8's strength.
---
# THE THREE QUESTIONS
**1. Is the build finished?** No. The verdict path is genuinely complete, well-tested and — for the constants it does gate — genuinely tamper-evident. What is missing is everything upstream of it and one binding gate downstream. There is no driver that runs the pre-registered money configuration: nothing calls `assert_conformance` as a first step, nothing loops the five pinned seeds (the orchestrator default is still three), nothing arms the standing legibility gate on a real run, and the only artifact-writing script is a self-described toy that omits the field the degeneracy guard needs. G-§8.C.3, a binding gate that can halt the ablation, has no implementation at all. `m6_toy_rehearsal.py` is close to the shape of the missing driver but differs on symbols, window, seeds, step budget, grid, and two gates — so the "zero first-time events on rented hardware" property that pre-flight Item 2 exists to establish does not currently hold for the money run.
**2. Single most likely way this produces a wrong published result?** A positive ΔIR(4−5) that is an artifact of the placebo's tokenizer rather than of microstructure information (C-12), reported as SURVIVES because both HALT guards were silently disarmed by an absent `mu_diag` (C-2), on a μ̂ whose per-step magnitude is inflated ~1.4× and mis-signed ~1 time in 6 by an out-of-distribution decode (C-1). Those three compose: C-1 makes the money path noisy and biased in exactly the quantity the execution filter thresholds; C-12 gives cell 5 a systematic handicap at fixed bits-per-token that clause 3 cannot fully absorb because cell 2 is not capacity-matched; C-2 removes the tripwire that would refuse to emit a word. Each alone is survivable. Together they produce a SURVIVES with a clean-looking manifest and no reader-visible signal that anything was unchecked.
**3. What did the builder and supervisor both miss?** They audited the *decision* and never audited the *measurement*. Enormous, genuinely excellent rigour went into the last 200 lines of the pipeline — the clause algebra, the DSR pins, the dual specification, the HALT guards, the mutation KATs — and the discipline there is real: I mutated four of the seven pinned constants and all four were caught. But the same scrutiny was never applied to the several thousand lines that produce the numbers those clauses read. `decode_tokens` carries a KAT asserting it inverts `encode_tokens` **on a full window**, and everyone accepted that as covering the rollout, which decodes one bar at a time through a bidirectional decoder — the KAT tests the regime that isn't used and skips the regime that is. That is the project's own "a check that cannot fail" pattern, one layer below where anyone was looking. The second shared blind spot is directional: every guard added since v1.4.4 protects against emitting a *false positive verdict*, and none protects against *feeding the verdict a corrupted input*. `_validate_artifact` checks schema, hashes, grid and κ-enumeration — and does not check that the field the degeneracy guard depends on is present. The guard was designed, argued and KAT'd on the assumption its input exists; the one driver that writes that input forgot it, and nothing in 337 tests notices. Finally, both parties repeatedly verified the pre-registration against itself rather than against the code: §3's body still says N=180 and three seeds, §7 v1.5 contradicts its own amendment two paragraphs later, and the orchestrator still defaults to the three seeds the operator paid to replace — three independent statements of the pinned surface, none of them checked against the other two.

<!-- ===== END VERBATIM AUDIT ===== -->

---

# PART 2 — OUR DISPOSITION (NOT the auditor's text)

**Everything below this line is the builder's, written after the fact.** It records what we did
about each finding. The auditor did not write, review or agree to any of it.

## Disposition — the builder's verdicts, with receipts

Findings are closed against the **prereg §7 amendment log**, which carries the full reasoning.
This table is an index, not a substitute.

| ID | verdict | where it is closed |
|---|---|---|
| C-1 | REPORTED — INDETERMINATE at n=3; now a REQUIRED real-cell disclosure | §7 v1.6.7/8/11 |
| C-2 | CONFIRMED, FIXED — blind HALT gate | §7 v1.6.1 |
| C-3 | CONFIRMED, outcome-material — **amendment DRAFTED and HELD** | §7 v1.6.13/14, `m6_c3_amendment_decision.md` |
| C-4 | REPORTED — G-§8.C.3 requirements + retrain contingency costed | §7 v1.6.9, `m6_c4_kronos_gate_requirements.md` |
| C-5 | CONFIRMED, FIXED — the money driver, A1–A9 | §7 v1.6.5/6 |
| C-6 | CONFIRMED, FIXED — train/eval seed+seq_len split-brain | §7 v1.6.4 |
| C-7 | CONFIRMED, FIXED — unguarded verdict thresholds | §7 v1.6.2 |
| C-8 | **REFUTED as stated**; the real finding is OC-1 (S=3 unreachable) | §7 v1.6.12/13 |
| C-9 | CONFIRMED, FIXED — §3/§3a body carried superseded numerics | §7 v1.6.12 |
| C-10 | CONFIRMED, FIXED — both legs; materiality measured on the real lake | §7 v1.6.15, `m6_c10_micro_density.json` |
| C-11 | CONFIRMED — **is B1**, drafted, awaiting Lakshay | `invariant7_amendment_decision.md` |
| C-12 | REPORTED — M1 adopted as a REQUIRED disclosure | §7 v1.6.9, `m6_c12_placebo_mechanism.md` |
| C-13 | CONFIRMED — freeze stands, scope description corrected | §7 v1.6.12 |
| C-14 | CLOSED by C-5/A6 — eight binding fields | §7 v1.6.5 |
| C-15 | CONFIRMED (a) + **CLOSED retrospectively (b)** on real bars | §7 v1.6.16 |
| C-16 | CONFIRMED, bounded — feeds a secondary, `harness.py` frozen | §7 v1.6.14 |
| C-17 | CONFIRMED, FIXED — incl. the shipped manifest rule string | §7 v1.6.12/13 |
| C-18 | 1 leg CLOSED, 2 CONFIRMED and now PINNED | §7 v1.6.15 |
| C-19 | CONFIRMED EXACTLY — now an ASSERTION, prose corrected | §7 v1.6.17 |
| C-20 | 3 legs: `embargo_flatness` KILLED; 1 calendar year CONFIRMED; receipt CONFIRMED stale, fixed | §7 v1.6.16/17 |
| S-1 | PROMOTED into C-1 | §7 v1.6.11 |
| S-2 | CONFIRMED, FIXED — C-2's unswept sibling | §7 v1.6.15 |
| S-3 | ADJUDICATED — non-independence now recorded in the C-1 disclosure | §7 v1.6.17 |
| S-4 | CONFIRMED, FIXED — `_betacf` rewritten, MDE bit-identical | §7 v1.6.17, `m6_s4_betacf_rewrite.json` |
