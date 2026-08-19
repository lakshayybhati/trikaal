# Claims-audit appendix — every substantive claim, tagged and sourced

**Status:** built 2026-07-31, pre-run, while the receipts are fresh (prereg §7 v1.6 item 4).
**Rule:** every claim is **MEASURED** (with artifact, the field the number lives in, and the commit)
or **ESTIMATED** (labelled as such, with what it rests on). A claim that is neither does not go in
the paper.

This appendix is **the audit of us**. It exists because *prose-without-datum is our demonstrated
failure mode* — Finding 0 below is entry one and sets the standard for the rest.

---

## 0. THE STANDARD-SETTING ENTRY — the cost basis (ESTIMATED, was labelled MEASURED)

| | |
|---|---|
| **Claim as it stood** | "MEASURED on the real 4090 (prior toy-CUDA rehearsal): 47.2k bars/s → full M6 ≈ 2–3 GPU-days" |
| **Where it lived** | `runs_manifest/m6_eval_throughput_expectation.json` → `labels_required_by_item_5.training_throughput` — a **prose string**, the only place in the repo the figure appears as a value |
| **Receipt it cites** | `runs_manifest/m6_rehearsal_manifest.json` → `throughput.bars_per_s_into_gpu` = **NaN**, `train_wall_s` = **0** |
| **Tag** | **ESTIMATED — NOT MEASURED** |
| **Forensics** | `runs_manifest/m6_cost_basis_forensics.json` |

**What the evidence settles.** The figure is *not* fabricated: 47,218.5 bars/s is exactly
reproducible as `steps_per_s × batch × seq_len` from `runs_manifest/m6_attention_bench.json`
(`sdpa_deterministic.steps_per_s` = 2.8819876814331926, batch 32, seq 512, CUDA, RTX 4090). So the
ruling's dichotomy — *measured but not persisted* vs *never measured* — resolves to a **third
option**: a real measurement **of a different quantity**, persisted numerically **in a different
artifact**, whose derived product was never persisted anywhere and whose prose label points at the
wrong receipt.

**Why it still cannot price the run.** The attention bench is a **compute-only micro-benchmark**:
60 steps (10 warmup) of forward/backward/optimizer on **pre-generated random-token batches already
resident in memory**, AR backbone only. It excludes data loading, Stage-1 tokenizer training, the
tokenize pass between stages, evaluation and checkpointing. It is an **upper bound** on the Stage-2
step rate. The cost extrapolation treats it as end-to-end training throughput.

**What the logs cannot settle, stated rather than reconstructed.** `runs_cloud/box_logs/item2.log`
— the training invocation — contains **zero non-wandb lines**: it captured only wandb's stderr, so
the script's own `[throughput]` stdout is absent from it entirely. Whether that invocation printed
a finite rate **cannot be determined from anything on disk**. What *is* settled: training happened
(per-cell stage1/stage2 summaries for all 15 runs); the manifest on disk is the `--eval-only`
re-run (`config.eval_only_rerun = true`), which writes to the **same path**, so the training
invocation's manifest was **overwritten**; and `item2b.log` shows that re-run printing
`[throughput] nan steps/s → nan bars/s`. **The failure is a lost measurement, not a failed one.**

**A second defect found in the same sweep, which the ruling did not ask for.** The attention bench
ran under **forced determinism** and never recorded that it had: `bench_mode` called
`set_determinism(0)`, whose default is `deterministic_algorithms=True`, while the production path
(`orchestrator.py:159`) passes `False`. **47.2k is therefore a *forced*-determinism rate**, and the
"~1.3× forced-determinism penalty applied to 47.2k → $43–65" arithmetic **double-counts the
penalty**. The posture is now recorded in the artifact (`config.deterministic_algorithms`).

**Every downstream cost claim is re-labelled ESTIMATED-NOT-MEASURED** until a real datum exists:
$20–30, $33–50, $43–65, "2–3 GPU-days", and the 1.3× multiplier.

**Attribution, on the record.** The supervisor propagated this ("measured throughput", twice, to
the operator, without checking the artifact) — and so did I, in every prior report that repeated
the figure. Neither of us opened the receipt.

---

## 1. MEASURED — claims with a number in an artifact

| # | Claim | Artifact → field | Commit |
|---|---|---|---|
| 1.1 | Gate-A anchor `results_hash` = `sha256:3f86882a…`, causal sweep **420/420** | `runs_manifest/gate_a_run_manifest.json` → `results_hash`, `gate_a_causal_sweep` | `c1786f4`, re-proven this pass |
| 1.2 | Backbone realized params **21,301,248** | `runs_manifest/m6_interface_respec_design_pass.json` → `g_parity_new_layout`; enforced by `train/cells.py` construction gate | `e1feb64` |
| 1.3 | Bits-per-token parity FSQ **20.058** / BSQ **20.000** | `eval/conformance.py` pins + `m6_interface_respec_design_pass.json` | `e1feb64` |
| 1.4 | Lake: **200 symbols / 304,625,181 bars / 7,024 parquet**, Merkle `5dfd667d`, push verified 100% | `runs_manifest/m6_lake_push_verification.json` → `verdict_parquets`, `manifest_ledger_verdict` | `a1729b8` |
| 1.5 | Token-control branch (a) DETECT: probe Spearman crossed 0.3 at step 2,000, final val = H0 − 0.8496 (94% of planted nats) | `runs_manifest/m6_token_control_run_manifest.json` | `2c41b67` |
| 1.6 | h-sweep: planted cell4 IR **+65.098 / +43.461 / +28.878 / +15.008** at h ∈ {2,3,5,15}; noise arm quiet at every h | `runs_manifest/m6_h_sweep.json` → `cells.*.by_h.*` | `ea8ab5f` |
| 1.7 | Guard band stability: worst frac_negative drift **0.0077** across n = 4,000 → 140,013 | `runs_manifest/m6_guard_band_stability.json` → `variants.*` | `ea8ab5f` |
| 1.8 | Run-to-run divergence: same seed, byte-identical recipe → frac_neg **0.5662** vs **0.99883** | `runs_manifest/m6_h_sweep.json` + `m6_moneyleg_rerun_manifest.json` | `ea8ab5f`, `7784df2` |
| 1.9 | **49 of 49** determinism records assert `bit_exact_claim` beside `deterministic_algorithms=False` | `runs_manifest/m6_divergence_attribution.json` → `bit_exact_claim_contradiction` | `a0a870c` |
| 1.10 | Estimator: traded-set sign agreement **0.9102**; mc@32 own noise floor **0.8164** | `runs_manifest/m6_estimator_forensics.json` | `d0bbaeb` |
| 1.11 | Eviction mechanism: per-dim point-decoder corr — return **0.98**, fillers **0.82–0.92**, independent state dim **0.001–0.014** (3 seeds) | `runs_manifest/m6_interface_respec_design_pass.json` → `gates` | `e1feb64` |
| 1.12 | Analytic MDE at h=15 = **3.518** annualized IR, reproduced to 3dp from `T_eff` = 35,002.3 | `runs_manifest/m6_mde_inputs.json`; reproduction in `m6_mde_floor_verification.json` | `4525543` |

**1.11 carries a binding framing commitment:** it is measured on a **SYNTHETIC fixture**, not on
real microstructure. It must not be elevated to headline anywhere.

---

## 2. ESTIMATED — labelled, with what each rests on

| # | Claim | Rests on | Why not measured |
|---|---|---|---|
| 2.1 | Run cost **$33–50** at S=5 (**$43–65** with forced determinism) | the compute-only bench rate × an assumed step count × an assumed spot price | no end-to-end throughput datum exists (entry 0); the 1.3× is illustrative and additionally double-counts (entry 0) |
| 2.2 | "full M6 ≈ 2–3 GPU-days" | same | same |
| 2.3 | **SE_boot ≈ 1.4149** | inverting the **analytic** MDE, not a bootstrap output | see §3 — it is an estimate of an estimate |
| 2.4 | Real-data legibility gate is **likely to fire** | the eviction mechanism (1.11) + real TFI's variance profile | a prediction, pre-registered as the modal outcome; the run tests it |
| 2.5 | Vast spot price ~$0.26–0.40/hr on a 4090 | market observation, not an artifact | spot-price dependent by nature |

---

## 3. The pre-run MDE floor — a correction, verified before use

Receipt: `runs_manifest/m6_mde_floor_verification.json`.

**What 3.518 is.** Not `(z_.95+z_.80) × SE_boot`. It is an **analytic iid-normal** MDE,
`z_sum · √(2/T_eff) · √(periods_per_year(h))` (`scripts/m6_prereg.py:74`), reproduced to 3dp
(**3.5183**) from the receipt's own `T_eff`. Inverting the multiplier therefore recovers the
**analytic** SE (1.4149), **not** the paired-bootstrap `SE_boot` that the v1.5 MDE multiplies.
Substituting one for the other is a defensible pre-run assumption but is **an estimate of an
estimate, not an identity**.

**4.3475 is not a floor.** The 3.0728 multiplier is the **ν→4 limit**, which obtains only when the
**training term dominates** — but the product is taken against an SE that **ignores** the training
term. Those two limits are mutually exclusive. Verified numerically through the repo's own
`paired_bootstrap` arithmetic: the MDE is **monotone increasing** in `se_train`, so its infimum is
at `se_train = 0`, where Welch–Satterthwaite gives ν → ν_boot = 9,999, the multiplier returns to
z_sum = 2.4865, and the MDE returns to the v1.2 number itself.

> **The pre-run computable lower bound is 3.518, not 4.3475.** The MDE reaches 4.3475 only once
> `se_train ≥ 0.685 × se_boot`, which is not knowable before the run.

**Withdrawal, attributed (2026-07-31).** The "≥ 4.35 annualized IR at S=5" floor originated with
the **supervisor** and **was stated to the operator**. It is **withdrawn in full**; the supervisor
verified the correction independently at `paired_bootstrap.py:156-160` — at `se_train = 0` the code
takes the scoring-only branch, ν = B−1 = 9,999, and the multiplier collapses to z_sum = 2.4865 —
and adopted all three layers, including the builder's further point that 3.518 is analytic rather
than a bootstrap product. Recorded visibly rather than absorbed, the same treatment C1 got.

---

## 4. Sweep for the same failure mode elsewhere — LISTED, NOT FIXED

Receipt: `runs_manifest/m6_claims_sweep.json` (idempotent; re-run is byte-identical).

| pattern | hits | disposition |
|---|---|---|
| **P3** — a field labelled MEASURED carrying a number in prose | **1** | `m6_eval_throughput_expectation.json::labels_required_by_item_5.training_throughput` — **entry 0**. The only one in the repo. |
| **P2 HIGH** — summary-level numeric slot holding NaN/0 | **5** | 3 true positives (all `m6_rehearsal_manifest.json::throughput.*` — entry 0). **2 are false positives of this audit**: `m6_token_causality_probe.json::…coarse_flip_rate` / `fine_flip_rate` = 0 matched on the substring "rate", but 0.0 is the **genuine and desired** value — zero token flips is the causal-safety pass, corroborated by the companion 20-position array over 60 chunks. |
| **P2 LOW** — NaN in a per-step diagnostic trajectory | 35 | `rollout_h2_corr` on a constant rollout: **undefined by construction, not a lost datum**. Not defects. |
| **P1** — a number inside a non-narrative string field | 23 | Listed in the receipt. None is a load-bearing claim; they are labels and descriptions. Reported for completeness, not fixed. |

**Two self-inflicted defects in the audit itself, disclosed:** (i) the sweep initially scanned its
own output and the forensics receipt, counting its findings as new findings and inflating the
totals — the same self-inclusion bug caught once before at 49→98; fixed with a skip set and
**idempotence asserted, not assumed**; (ii) the first severity pass buried the 3 real hits under 35
benign per-step NaNs, which is how the original defect survived in the first place.

**Also found, not part of the sweep patterns:** `m6_eval_throughput_expectation.json` →
`real_scale_decision_count_h15_headline.seeds = 3` is stale under v1.5 (`PINNED_SEEDS` has 5), so
`total_decisions = 21,037,800` understates by 5/3. **Listed, not fixed** — it is a committed
receipt, and patching a committed receipt is itself a disclosed prior error of mine.

---

## 5. Standing rule this appendix establishes

> **A number that appears in prose and nowhere else is not a result.** Before any figure enters the
> paper it must resolve to a field in an artifact, with the artifact named and the field path
> given. If it does not, it is tagged ESTIMATED with its basis stated — or it is cut.
