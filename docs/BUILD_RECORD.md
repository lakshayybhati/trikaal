# Trikaal — Build Record

**Canonical build history. Reconstructed from artifacts on 2026-08-05 at HEAD `0cffb4f`.**

**Method.** This document was reconstructed from the git log (121 commits, 2026-06-18 → 2026-08-05),
the milestone reports, the pre-registration amendment log (`docs/m6_prereg.md` §7, entries v1.1 →
v1.6.27), the persisted external audit (`docs/m6_readiness_audit_findings.md`), the claims-audit
appendix, the limitations register, and the receipts under `runs_manifest/`. Where a statement is
supported only by conversational history and has no artifact behind it, it is marked
**`[recollection — no artifact]`**. That convention exists because prose outrunning receipts is this
project's demonstrated failure mode; §7 of this document enumerates the instances.

**Standard.** Nothing in this record should be findable-false by an auditor opening the cited
artifact. Where an artifact and a recollection disagree, the artifact is the fact.

---

## 1. Timeline

| milestone | dates | commits | outcome |
|---|---|---|---|
| M0 blueprint | 2026-06-18 | `026cf58` | v1 design frozen as research baseline |
| M1 synthetic slice | 2026-06-18 | `59dea1c`, `302d651` | G0/G1/G2 green; adversarial review found 5 defects |
| M2 single-symbol | 2026-06-19 | `0061243` … `9f72047` | Stage-1 converges on real BTCUSDT 2023; causal IC screen |
| M3 Stage-2 AR | 2026-06-19 | `eae303c` | backbone fits real tokens; KV-cache rollout |
| M5 eval harness | 2026-06-19 | `e6c8030` | Gate A satisfied — apparatus validated, produces no meaningful number |
| M4a scaffolding | 2026-06-20 | `d80a5ca` | N-symbol generalization, local dry-run |
| M4b universe lake | 2026-06-20 → 06-22 | `33637fb` … `58bef30` | 200/200 symbols, 304,625,181 bars |
| M6 Phase 0 | 2026-07-04 | `8b0e4b7` … `49b1b22` | environment pinned, prereg timestamped, Gate-A anchor set |
| M6 Phase 1 | 2026-07-04 | `4b58da6` … `d5c6aad` | BSQ arm, cell arms, loader, orchestrator, SMOKE gate |
| audit round 1 | 2026-07-06 | `122716d` … `541ac9b` | six findings fixed pre-training |
| CO2 pass | 2026-07-18 | `ba94856` … `3581c92` | verdict assembler, DSR conformance pins |
| canary campaign | 2026-07-19 → 07-22 | `81bf02b` … `7784df2` | v3 → v6 full stop → token control → interface re-spec |
| adjudication | 2026-07-29 → 07-30 | `eb0b9e3` … `d0bbaeb` | canary gate closed 8/8, B1 attached |
| v1.5 freeze | 2026-07-30 | `6b2ad82` | last specification change; design frozen |
| pre-flight | 2026-07-31 → 08-01 | `8ef5568` … `d15584a` | claims audit, CUDA probe, prefill |
| audit round 2 | 2026-08-01 → 08-03 | `9ac2943` … `d5aa20c` | 20 findings, tiered and closed |
| re-audit | 2026-08-03 → 08-04 | `a974393`, `dedfe4c` | 12 findings; NOT READY; all blockers closed |
| run preparation | 2026-08-04 → 08-05 | `9bd102f` … `0cffb4f` | fan-out, lake provisioning, C-12 |

### M0 — blueprint (2026-06-18, `026cf58`)

`docs/specs/2026-06-18-trikaal-v1-design.md`. Design drafted and adversarially verified
by a multi-agent workflow: 9 subsystem drafts → a 4-lens audit (causal-safety, architecture/math,
financial-ML rigor, scope/consistency) → targeted revision → re-verification. The spec records that
this audit caught real defects before any code existed: wrong FSQ bits-per-token tables, a bad-tick
lookahead, a mis-scaled net-IR annualization, a missing funding-cost term, a break-even execution
filter, an inert trailing embargo. Two residuals were patched during assembly (param base unified to
21.30M; a 0.07M typo).

**Cost: $0.**

### M1 — synthetic vertical slice (2026-06-18, `59dea1c` → `302d651`)

End-to-end path on synthetic data. Gates G0 (shape/causality), G1 (overfit a single batch), G2
(determinism smoke) green; leak-free on synthetic.

A second adversarial review then confirmed **five findings**, four in scope, all fixed in `302d651`.
These are the first entries in the defect ledger (D1–D4) and the origin of the exhaustive causal
sweep that has gated every subsequent milestone.

**Cost: $0.**

### M2 — BTCUSDT single-symbol slice (2026-06-19, `0061243` → `9f72047`)

Real data pipeline (§2–§4). Stage-1 tokenizer converges on real BTCUSDT 2023 bars. Exhaustive
truncation sweep — every bar a boundary, 100% coverage — passes on real data
(`docs/milestone2_btcusdt_stage1.md`).

The per-feature causal IC screen (`9f72047`) is the origin of the project's central prior. Measured
RankIC with 95% CI, BTCUSDT 2023:

| feature | h=5 | h=15 |
|---|---|---|
| `ret_close` | −0.0335 [−0.0369, −0.0303] | −0.0220 [−0.0245, −0.0193] |
| `body` | −0.0258 [−0.0290, −0.0227] | −0.0155 [−0.0179, −0.0132] |
| **`TFI`** | **−0.0270 [−0.0304, −0.0236]** | **−0.0197 [−0.0230, −0.0166]** |

TFI carries signal at a magnitude comparable to price features, and it is thin in absolute terms —
|RankIC| ≈ 0.02–0.03. Every power discussion in this project descends from that table.

`ada5980` encoded Tier-1 review decisions into the spec: the placebo requirement, train-once /
eval-forward, the independence framing, survivorship handling, and the cost model.

**Cost: $0.**

### M3 — Stage-2 AR (2026-06-19, `eae303c`)

Backbone learns real K-line tokens on BTCUSDT-2023; KV-cache rollout implemented. Exit gate:
**beats both trivial baselines** on best-val, per coarse and fine
(`docs/milestone3_stage2_btcusdt.md`). Verdict recorded as **PASSED (qualified)** — the backbone
demonstrably fits real tokens; nothing about generalization is claimed.

**Cost: $0.**

### M5 — eval harness (2026-06-19, `e6c8030`)

Built before M4 by a deliberate re-ordering (`c37eacd`). The full §8 harness — purged walk-forward
folds, Q1–Q4 quadrants, cost-aware net IR, placebo machinery, DSR/PBO, secondary diagnostics — runs
end-to-end and leak-free on the M3 checkpoint. Every metric passes a known-answer test. The
§8.A.5/§8.E.5 causal exhaustive sweep passes on a Q4 sample.

`docs/milestone5_eval_harness.md` records the exit criterion explicitly: *"The bar is correctness of
the apparatus, not any result… It produces NO meaningful number — dev-grade model, one symbol."*
This is **Gate A**, and its `results_hash` became the project's reproducibility anchor.

**Cost: $0.**

### M4a / M4b — the universe lake (2026-06-20 → 06-22)

M4a (`d80a5ca`): ingest scaffolding, local dry-run, N-symbol generalization, exit gate in
`docs/milestone4a_universe_ingest.md`. Reviewer B verified regression byte-equivalence bit-exact.

M4b (`33637fb` → `6bc88aa`): full ingest at scale.

| metric | realized |
|---|---|
| symbols | **200 / 200** (199 first pass + AXSUSDT on resume) |
| bars | **304,625,181** |
| lake on disk | 23 GB Hive-partitioned zstd Parquet |
| wall clock | **~52.8 h** (190,111 s) + ~43 min resume |
| throughput | ~2.3 MB/s sustained — **~3.6× below** the 8.3 MB/s burst estimate |
| universe Merkle | **`sha256:5dfd667d05b97bda…`** |

The throughput shortfall is recorded in the milestone doc as a corrected projection: the home link's
*sustained* rate, not its burst, set the wall clock, so the earlier ~15 h projection was optimistic.

`11652bd` hard-gated the real-bar causal sweep (live + tail) and added an orchestrator-independence
test. `58bef30` added hash-preserving lake compaction.

**Cost: $0** (local ingest; bandwidth only).

### M6 Phase 0 — pre-registration (2026-07-04)

Six commits in one day, in a deliberate order:

- `8b0e4b7` — environment pinned: exact numpy/torch pins, committed lockfile
- `f933623` — the six governing docs committed (**this is the pre-registration timestamp**)
- `d87aa2b` — eval-instrument fixes (m6_design §6 item 10 a–f), each with a KAT
- `631bcfd` — **new Gate-A anchor**, 2× bit-identical under the committed environment
- `d42d57c` — CI gate for invariant 2; `setup_cloud` installs from the lockfile
- `49b1b22` — doc refresh

The ordering matters and is deliberate: the environment is pinned *before* the anchor is set, so the
anchor means something.

### M6 Phase 1 — the ablation machinery (2026-07-04)

`4b58da6` BSQ quantizer + G-parity gate · `a5fa402` feature-arm switch + Cell-5 shuffled-micro arm ·
`e8dcb18` multi-symbol universe loader (fold-aligned draw) · `4be2a4d` resumable atomic
checkpointing, kill-resume proven · `f021d48` determinism / attention-mode hook · `32646f1`
tripwires + abort rule · `64f728c` 5-cell × N-seed orchestrator · `b79e259` cross-sectional eval
driver · `d5c6aad` **SMOKE gate PASSED, MDE_prereg computed, `docs/m6_prereg.md` LOCKED**.

`64f728c` is also the origin of three defects (D14 seed split-brain, D15 seq_len, D16 step budget) —
all three entered in the same dataclass literal, four lines apart.

### Audit round 1 (2026-07-06)

Six items, all fixed pre-training: Cell-5 shuffle pairing (`2cac96f`), the paired moving-block
bootstrap (`374c284`), per-symbol spread deciles (`68c4081`), MDE recomputed on the pinned blocks-1-5
region (`4525543`), money mode + §3a conformance gate (`669d2d6`), micro-alignment lag-sharpness
audit (`541ac9b`).

### The canary campaign (2026-07-19 → 07-22)

The longest and most consequential phase. Fully documented in §4 (Mechanism narrative).

Sequence: toy-CUDA rehearsal (`77ca29a`, ~$5, 7 of 8 gates green, canary FULL STOP) → canary v3
(`89fcac6`) → v5 (`90157c1`, `fbc1cf2`, ~$0.70, INCONCLUSIVE) → v6 (`02c438f`, `6feb60f`, ~$0.60,
**architectural full stop**) → token-space control (`e891d8c`, `7675393`, `2c41b67`, ~$0.60, **branch
(a) DETECT**) → interface re-spec (`a1729b8`) → gate-2 rider (`e1feb64`, **the eviction measurement**)
→ calibration campaign (`d9e1b53`, `bb1e17c`, `39be8ba`) → acceptance run (`36560dc`, ~$0.90) →
μ̂ correction (`c1786f4`) → money-leg re-run (`7784df2`, ~$0.60).

### v1.5 freeze (2026-07-30, `6b2ad82`)

*"THE LAST SPECIFICATION CHANGE BEFORE THE RUN; the design is now FROZEN."* Six amendments A–F,
signed off. Detailed in §3.

### Audit round 2 — the independent audit (2026-08-01 → 08-03)

An external auditor, read-only, reported **20 confirmed findings (C-1 … C-20)** plus **4 suspected
(S-1 … S-4)** against commit `7a4d13b`. Persisted verbatim at
`docs/m6_readiness_audit_findings.md`. Full ledger in §2.

Tiering (supervisor ruling): **Tier 1** blocking (C-2, C-5, C-6, C-7) → **Tier 2** measure-only
(C-1, C-4, C-12) → **Tier 3** prereg integrity (C-8, C-9, C-13, C-17) → **Tier 4** triage (the rest).

### Re-audit (2026-08-03 → 08-04)

A second independent pass returned **NOT READY** with 12 findings (R1–R12). Its central observation:
*"everyone audited the findings, and nobody audited the fixes."* Four of the five blocking findings
were in `0b9c804` — the commit that implemented four supervisor decisions in one pass.

### Current position (2026-08-05, `0cffb4f`)

727 tests green · `ruff check` exit 0 · `ruff format --check` exit 0 · Gate-A anchor
`sha256:3f86882a63dd06c7…` byte-identical, exit 0 ×2 · mutations 15/15 with a passing negative
control · nothing rented.

**Total spend to date: $2.30 of $8** (see §6).

---

## 2. The defect ledger

Severity: **CRITICAL** (would produce a wrong published result), **HIGH** (would produce a failed or
invalid run), **MEDIUM** (degrades a claim or a record), **LOW** (documentation, hygiene).

Detection layer: **DESIGN-REVIEW**, **GATE** (an automated check), **CANARY** (a planted-signal
experiment), **AUDIT-1**, **RE-AUDIT**, **EXECUTION** (found by running the thing), **SELF** (found
by the builder or supervisor outside a formal pass).

| # | defect | sev | layer | status |
|---|---|---|---|---|
| D1 | Causal gate sampled ~2.5% of bars | CRITICAL | DESIGN-REVIEW | closed `302d651` |
| D2 | `target_valid` unverified and used to skip the target check | CRITICAL | DESIGN-REVIEW | closed `302d651` |
| D3 | `ts` outside the checked set | HIGH | DESIGN-REVIEW | closed `302d651` |
| D4 | Fine-head loss conditioned on ground-truth coarse | HIGH | DESIGN-REVIEW | closed `302d651` |
| D5 | Environment drift / no lockfile | HIGH | SELF | closed `8b0e4b7` |
| D6 | Cell-5 shuffle broke (value, mask) pairing | CRITICAL | AUDIT-1 | closed `2cac96f` |
| D7 | No paired-bootstrap machinery for ΔIR | HIGH | AUDIT-1 | closed `374c284` |
| D8 | Encoder causality leak, 41.8% | CRITICAL | CANARY | closed `85c740b` |
| D9 | v6: zero nats extracted from a 1.15-nat plant | CRITICAL | CANARY | closed `a1729b8` |
| D10 | Capacity eviction — state dim priced out | CRITICAL | CANARY | mitigated `39be8ba` |
| D11 | μ̂ mode-vs-mean decode bias | CRITICAL | EXECUTION | closed `c1786f4` |
| D12 | `w_feat_point` wiring bug; λ-search invalidated | HIGH | SELF | closed `d9e1b53` |
| D13 | Anchor claim entered a commit unverified (pipefail) | HIGH | SELF | closed `bb1e17c` |
| D14 | Seed split-brain — S=5 never reached training | CRITICAL | AUDIT-1 (C-6) | closed `4dbe27f` |
| D15 | `seq_len` 128 train vs 512 eval | HIGH | AUDIT-1 (C-6) | closed `4dbe27f` |
| D16 | Training budget was the smoke-test step count | CRITICAL | SELF | closed `0b9c804` |
| D17 | Degeneracy guard reported `armed: true` blind | CRITICAL | AUDIT-1 (C-2) | closed `9ac2943` |
| D18 | Three decision constants under no gate | CRITICAL | AUDIT-1 (C-7) | closed `d4384e6` |
| D19 | No driver ran the pre-registered money config | CRITICAL | AUDIT-1 (C-5) | closed `e76853e` |
| D20 | 49/49 records asserted bit-exactness falsely | HIGH | SELF | closed `0b9c804` |
| D21 | Micro-legibility gate passed measuring nothing | CRITICAL | AUDIT-1 (C-10) | closed `234a013` |
| D22 | Legibility gate read 1 of 200 symbols | CRITICAL | SELF | closed `234a013` |
| D23 | 12 fail-open guards | CRITICAL | SELF | closed `234a013` |
| D24 | Lake-admitting sweep vacuous on 2 of 8 surfaces | HIGH | RE-AUDIT (C-15) | closed `614200c` |
| D25 | `_betacf` transcribed from Numerical Recipes | MEDIUM | AUDIT-1 (S-4) | closed `d5aa20c` |
| D26 | Codebook gate shipped below `return bad` | CRITICAL | RE-AUDIT (R1) | closed `a974393` |
| D27 | C-3 pin and revert-KAT never shipped | HIGH | RE-AUDIT (R2) | closed `a974393` |
| D28 | v1.2 dual-spec leg was a mislabelled hybrid | HIGH | RE-AUDIT (R3) | closed `a974393` |
| D29 | Kronos verdict + BSQ disclosure discarded | HIGH | RE-AUDIT (R4) | closed `a974393` |
| D30 | Identity surface self-referential; no git sha | CRITICAL | RE-AUDIT (R6) | closed `a974393` |
| D31 | `score_cell` full return dropped both diagnostics | CRITICAL | EXECUTION (P1) | closed `dedfe4c` |
| D32 | Runbook had no lake-provisioning step | HIGH | EXECUTION (P2) | closed `bb644b7` |
| D33 | `driver_version` placeholder on a real 4090 | HIGH | EXECUTION (P2) | closed `31dbffc` |
| D34 | Offer filter had no driver constraint | MEDIUM | EXECUTION (P2) | closed `31dbffc` |
| D35 | Three clause strings named the quantity `ΔIR_info` | CRITICAL | AUDIT-3 (C-12) | closed `af3164d` |
| D36 | Invariants 4 and 8 asserted a validation never performed | HIGH | SELF | closed `0cffb4f` |

### D1 — the causal gate checked ~2.5% of bars

**What.** The causal-safety gate used sparse stratified sampling. A localized single-bar lookahead at
a non-anchor bar passed it.

**How found.** M1 adversarial review, 2026-06-18.

**If unfound.** Invariant 2 — the project's hardest constraint — would have been enforced by a check
that misses most of the surface it guards. Every downstream leak-free claim would rest on it.

**Fixed.** `exhaustive_truncation_sweep`: truncate at **every** bar boundary; 100% coverage is the
gating criterion. The stratified probe was kept as added defense, not replaced.

**Prevention.** A regression test plants a single-bar σ leak the sampler misses and the sweep
catches. This is the earliest instance of the discrimination principle: **the fix ships with proof
that the old check would have failed.**

### D2 — `target_valid` unverified, and used to skip the target check

**What.** `target_valid` — the loss-inclusion flag, a survivorship surface — was never itself
verified, and was used to *skip* checking the target.

**If unfound.** A survivorship leak in the flag would have silently excused the target from causal
checking entirely.

**Fixed.** Added to the checked outputs at horizon t+1; the skip removed; a localized
survivorship-leak regression test added.

**Note.** This surface reappeared in D24 — at production parameters the same flag was true on 0 of
569 bars, so the check that guards it could not fire. Same surface, two independent failures, seven
weeks apart.

### D3 / D4 — `ts` outside the checked set; fine-head conditioned on ground-truth coarse

D3: `ts` is a model input and was not in the causal-checked set. Added at horizon t.

D4: the main fine-head loss conditioned on ground-truth coarse; the spec mandates **sampled** coarse
during training (the exposure-bias fix). Added `coarse_sample_prob` (default 1.0, fully sampled) and
`_coarse_conditioning`; the MTP per-depth path stays on ground-truth, which is §4.1-correct. G1
stage-2 still converges (CE < 0.05).

### D5 — environment drift

**What.** No lockfile; numpy/torch versions unpinned. `[recollection — no artifact]`: the drift was
noticed when results moved without code changing. The artifact records only the fix.

**Fixed.** `8b0e4b7` — exact numpy/torch pins, committed lockfile, `setup_cloud` installs from it.
The Gate-A anchor was then re-set *under* the pinned environment (`631bcfd`), in that order.

**Prevention.** `lockfile_sha256` is one of the 16 provenance identity keys; a shard whose lockfile
differs refuses to assemble.

### D6 — the placebo shuffle broke (value, mask) pairing

**What.** Cell-5's `block_time_permute` shuffled values without carrying their mask bits with them.

**How found.** Audit round 1, item 1, 2026-07-06.

**If unfound.** The placebo — the control the entire headline rests on — would have produced values
paired with the wrong validity flags. The one cell whose job is to be a clean negative would have
been corrupt in a way no other cell was.

**Fixed.** `2cac96f` — the shuffle moves (value, mask) pairs together.

**Prevention.** `assert_perp_dims_masked` (`placebo.py:38`) fails loudly if funding/OI ever activate
without the shuffle set widening; the shuffle is verified `(symbol, seed)`-seeded and train/eval
identical.

### D7 — no paired-bootstrap machinery

**What.** The decision rule required a paired ΔIR statistic; no paired bootstrap existed.

**Fixed.** `374c284` — paired moving-block ΔIR bootstrap (B=10,000, seed 20260704, ⌈√T⌉ blocks,
percentile CI) plus Item-8 calibration KATs.

**Note.** The pairing is not cosmetic: cells 4 and 5 share market exposure, and the common factor
cancels in the paired series. `docs/m6_prereg.md` records ρ̄_raw ≈ 0.62, so effective breadth is
~1.6 symbols — the cross-section adds robustness, not power.

### D8 — the encoder causality leak, 41.8%

**What.** The tokenizer's encoder read the whole window bidirectionally. Measured **41.8%
token-lookahead leak**.

**How found.** Canary campaign, §7 v1.3.

**If unfound.** Every token would have carried future information. The entire experiment would have
been invalid and the leak would not have shown up as an error — only as unrealistically good
forecasts.

**Fixed.** `85c740b` — `encoder_causal=True` forced for all M6 cells; `PINNED_ENCODER_CAUSAL` in
conformance.

### D9 — v6: zero nats from a 1.15-nat plant

Detailed in §4. **The single most consequential finding in the project.**

### D10 — capacity eviction

Detailed in §4. Mitigated by `PINNED_MICRO_POINT_WEIGHT = 3.0` (`39be8ba`); the mechanism itself is
a permanent property of reconstruction training and is a result, not a bug.

### D11 — the μ̂ mode-vs-mean decode bias

**What.** `predict_mu` defaulted to greedy-argmax (mode) decoding. Compounded over a rollout this
produced a systematic sign bias: mean −0.097 against std 0.036 — a ×15 drift.

**How found.** The acceptance run (`36560dc`). Every noise cell returned a **bit-identical IR of
−7.56** across both quantizers. Bit-identical results across arms that should differ is a
degeneracy signature, not a result.

**If unfound.** Every cell's forecasts would have been sign-saturated. The verdict would have been
computed on an artifact.

**Fixed.** `c1786f4` — default MODE → MEAN (expectation decode). Fixture receipt: h=15 argmax mean
−0.051 / frac_neg 0.929 → expectation −0.004 / 0.534; corr(μ, state) 0.858 → 0.898. Skewed-toy
known-answer KAT (argmax = mode, expectation = true mean). Gate-A re-anchored `5eead7b6` →
`3f86882a`, bit-identical ×2 — `predict.py` is inside the anchored instrument.

**Confirmed by re-run.** `7784df2`: the saturation is gone, cells discriminate, and `planted_fires`
flipped False → True.

### D12 / D13 — the λ-search wiring bug, and the false anchor claim

D12: `loss_point` read `w_feat` rather than `w_feat_point`. The first λ-search was invalidated and
retained only as λ=1 replicates; an asymptote analysis built on it was withdrawn.

D13: `d9e1b53`'s commit message claimed the anchor was re-proven. **It was false.** An unconditional
`w_feat_point` buffer broke old-schema checkpoint loading, and a `| grep | tail` pipeline masked the
exit-1, so the claim entered the commit message unverified.

**Fixed.** `bb1e17c` — buffer registered only under `fine_pointwise`; anchor genuinely re-proven,
exit 0.

**Prevention.** This is the origin of the **pipefail rider** (§5). It recurred three more times, in
both directions, and is the most persistent single failure mode in the project.

### D14 / D15 — the seed and seq_len split-brain

**What.** `OrchestratorConfig` defaulted to `seeds=(0,1,2)` and `seq_len=128` while the pinned
surface said `(0,1,2,3,4)` and `512`.

**How found.** Audit round 1 finding C-6.

**If unfound.** **The five seeds the operator approved and funded would never have reached the
training code.** The run would have produced three seeds and reported five. Separately, models
trained at 128 would have been evaluated at 512 — a regime mismatch.

**Fixed.** `4dbe27f` — the eval side is authoritative; the orchestrator **asserts** the pins rather
than defaulting to them; the second copy of every pinned value removed; a shadowed-field sweep run.

### D16 — the training budget was the smoke-test step count

**What.** `steps_stage1 = steps_stage2 = 2000`. Traced to `64f728c` — **the same dataclass literal
as `seeds=(0,1,2)` and `seq_len=128`, four lines apart.** The design spec uses "≤ 2000 steps" as the
threshold for the **G1 overfit-a-single-batch smoke gate** (spec :1763). The prereg records that
`steps_stage2` *"shadows nothing — no prereg constant states them"* (:1215) and that every driver
sets both from a single `--steps` flag (:1489).

Measured consequence: 2,000 × 32 × 512 = **32,768,000 tokens = 1.538 tokens/param = 7.69% of
compute-optimal**.

**How found.** The builder computed tokens/param to answer a C-4 probability question; the
supervisor noticed it applies to all five cells and traced the literal's origin.

**If unfound.** All five cells would have been trained at ~1/13 of compute-optimal. A flat ΔIR(4−5)
could not have distinguished *"microstructure does not help"* from *"we stopped before it could."*

**Fixed.** `0b9c804` — 26,003 steps, matched and fixed across all units. 20.00 tokens/param.

**Corrective note (2026-08-05).** The commit and the supervisor's brief both described this as
*"1.399 passes — the spec's own 1–3 passes."* **That is wrong.** `m6_money_run.py:116` records that
the pinned 40 symbols are **84,153,600 bars**, so 425,993,216 tokens is **5.06 passes**, not 1.399.
The token budget is correct; the justification was not. Recorded here as the first correction this
document produces.

### D17 — the degeneracy guard reported `armed: true` having examined nothing

**What.** `degeneracy_guard` read `evals[(c,s)].get("mu_diag", {})` and returned a hardcoded
`"armed": True`. The only artifact-writing driver never wrote `mu_diag`. The verdict KAT suite also
ran the blind path.

**How found.** Audit round 1, C-2. The auditor reproduced **a full SURVIVES verdict with both guards
reporting armed while examining nothing.**

**If unfound.** A degenerate cell — the exact pathology this project has hit repeatedly — would have
passed into a published verdict with a guard asserting it had been checked.

**Fixed.** `9ac2943` — fixed at four layers, mutation-proven. `armed` is now computed from whether
every (cell, seed) yielded finite values; the guard fails closed; `mu_diag` is a **required** artifact
field and the writer refuses a missing/empty/None value.

**Prevention.** This defect is the origin of the phrase *"a check that cannot fail is not a check"*
and it recurred **five times**: C-2, C-10, the fail-open sweep's 12 rows, the codebook gate (R1), and
the unit test that was supposed to catch it (D21).

### D18 — three decision constants under no gate

`ECON_FLOOR_IR`, `BOOT_POWER`, `PLACEBO_DISPERSION_TRIPWIRE` were unpinned; four `PINNED_DSR` keys
were decorative — present but read by nothing.

**Fixed.** `d4384e6` — every pinned value unchanged; what changed is that they are now **read**.
Eight separate mutations, all exit 1.

### D19 — no driver ran the pre-registered money configuration

**What.** `write_cell_eval_artifact` was called from exactly three places, none of them a money
driver. The pre-registered configuration had no executable path.

**Fixed.** `e76853e` — `src/trikaal/run/matrix.py` and `scripts/m6_money_run.py`, built to an A1–A9
scope: conformance first (AST-enforced), no pin-shaped flags, `mu_diag` required, dry-run artifacts
structurally refused.

**The A9 dry run found a further blocker before any GPU saw it**: the backbone was built at default
`max_len=512` while the rollout addresses `seq_len + h`. The money run would have trained every unit
for hours and died at first eval.

### D20 — 49 records asserted bit-exactness falsely

**What.** 49 of 49 run records carried `bit_exact_claim: true` beside
`deterministic_algorithms=False`. Root cause: `attention_mode.py` computed
`claim = attention_mode == MODE_SDPA` — invariant 7's **sufficiency premise encoded in code**.
Deterministic attention is necessary, not sufficient.

**Scope, established at the time.** Only the GPU-training clause was false. The pipeline,
frozen-stats and prediction-replay clause and the Gate-A anchor were intact; no prior result was
invalidated.

**Fixed.** `0b9c804` — `deterministic_algorithms=True` on the production path. Measured penalty
**13.175%** (`docs/m6_cuda_probe_report.md`).

**Secondary finding.** Same-seed runs diverged (frac_neg 0.5662 vs 0.99883 at identical
h/cap/estimator/seed/recipe) — basin-hopping, a lower bound on across-seed variance. The
pre-registered MDE contained scoring noise only. Mitigated by `verdict.power_guard`, HALT-only.

### D21 — the legibility gate, and the test that certified the defect

**What.** `micro_legibility_gate` initialized `ok = True`; a dim below the 10,000-unmasked floor hit
`continue` **before** `ok = ok and acc >= min_acc`. With all six micro dims thin the gate returned
`pass: True` having measured nothing.

**And the test.** `test_gate_skips_masked_everywhere_dims` carried the docstring *"…SKIPPED with a
named receipt entry, **never trivially passed**"* and a body asserting only
`"skipped" in receipt["per_dim"]["8"]`, called at `min_acc=0.0`. **It never read `receipt["pass"]`.**

**This is the sharpest instance in the project: the defect was not an oversight, it was the written
contract, and its guard was itself a check that could not fail.**

**Fixed.** `234a013` — a skipped dim can never contribute to a pass; the test rewritten as
`test_a_masked_everywhere_dim_HALTS_the_gate` using `pytest.raises`, with the diagnostic preserved.

### D22 — the legibility gate read one symbol of 200

**What.** `id_legibility_sign_acc` took `values[:n]`, `b_c[:n]` with n = 150,000 — the **head** of a
symbol-ordered concatenation, with a contiguous 80/20 split.

**Measured.** `runs_manifest/m6_c10_micro_density.json`: the 150k window spans **1 of 200 symbols**
(`1000BONKUSDT`, first in sorted order), out of ~300 million rows.

**If unfound.** The hard stop deciding whether Stage-2 spend proceeds would have been reading 0.05%
of the data from a single symbol.

**Fixed.** `234a013` — a per-symbol **stratified blocked** split: every symbol contributes, and
within each symbol train is the earlier block and val the later one. Both properties preserved —
coverage and time-blocking — where the naive fix (shuffling) would have broken the second.

**Materiality, measured.** The same receipt shows the all-thin case **cannot fire** on this lake:
minimum unmasked bars per micro dim is **277,758** against a floor of 10,000, zero symbols below it.
The defect was real; leg 1 was immaterial and leg 2 was severe.

### D23 — the fail-open class, 12 rows

**What.** A systematic sweep of every M6-path guard against every degenerate input (absent, empty,
None, NaN, inf, wrong-type) found **12 fail-open rows across 14 functions / 84 cases** —
`power_guard` on both data and claim, `decode_agreement_disclosure`, `pinned_threshold_failures`,
`shard_partition_failures`.

**Two nobody had listed:** *deleting a pin deleted its own check* (the guard iterated the pin dict),
and *an empty unit matrix was certified a perfect cover* — the dangerous case in a fan-out.

**Method note, and it is the finding's method contribution.** The first run reported 21. Three were
harness bugs: `provenance_failures` reads `meta.provenance` and the probe mutated a top-level key so
the guard never saw degenerate input; two functions were called with wrong signatures so every case
died of the same `TypeError` — both arms failing alike, which is `PROBE INVALID`, not a verdict. **21
before the harness was right, 12 after. A defect count is worthless until the probe can separate its
own bugs from its findings.**

**Fixed.** `234a013` — all 12 closed, 0/84 remain, verified by the supervisor re-running the sweep
against the pre-fix commit in an isolated worktree.

### D24 — the sweep that admitted the lake could not fail on 2 of its 8 surfaces

**What.** The causal sweep gating the lake write ran on **800 bars against a 1,440-bar volatility
warm-up**, on **2 of 200 symbols** (1,600 bars of 304,625,181 — 0.00053%), and printed to stdout
without persistence. `target_valid` was true on **0 bars inside that slice**, so the surface could
not fail.

**Sharper than the audit stated:** re-running at production parameters would have made the gate
*vacuous*, not stronger — the warm-up exceeds the fixture and no bar has a valid target for a leak
to flip.

**Fixed retrospectively.** `614200c` — the production-parameter planted-leak battery run on real
BTCUSDT bars past the warm-up: **1,600 anchors, 12,798 checks, coverage 1.0, zero failures**, and
both planted leaks caught including `localized_validity_next`, the surface that could not fire
before.

**Whole-lake surface check.** `runs_manifest/m6_c15b_lake_surface_check.json`: all 200 symbols, all
304,625,181 bars, fraction 1.0. Verdict on question 1: **`DEGENERATE`** — dims 13/14/15 have stddev
exactly 0.0 and are masked on 100% of bars. This is the documented funding/OI absence (A3), not a
new defect, and the receipt states it flatly where the report softened it.

### D25 — `_betacf` transcribed from Numerical Recipes

**What.** `tdist._betacf` was the NR modified-Lentz `betacf` in structure. NR carries a restrictive
licence; a released artifact should not transcribe it.

**Fixed.** `d5aa20c` — rewritten from the published continued-fraction definition (A&S/DLMF backward
recurrence), verified against mpmath at 50 dps (worst 5.8e-12; design range ~1e-13), **MDE outputs
bit-identical** and both pinned multipliers bit-identical.

### D26 — the codebook gate shipped below `return bad`

**What.** The entire ≥0.95 codebook-utilization enforcement block sat **after** `_validate_artifact`'s
`return bad`. Born dead in `0b9c804`, the commit claiming to close the defect class.

**How found.** Re-audit R1. Reproduced by execution: a document with `utilization: 0.01`, or no
codebook key at all, loaded clean through the real write → load → assemble path. Every test fixture
supplied a healthy codebook, so no test could ever have fired.

**If unfound.** A collapsed cell-5 or BSQ tokenizer would have assembled into a verdict unflagged,
and the stated compensating control for dropping the Kronos gate would not have existed at runtime.

**Fixed.** `a974393` — block moved above the return; 8 of 9 codebook tests fail without it.

**Supervisor note.** The builder reported this gate closed. The supervisor verified it by grepping
for `PINNED_CODEBOOK_MIN_UTILIZATION` and finding it, and **never checked it was reachable** — the
same class of error, committed while enforcing it.

### D27–D30 — the re-audit's other blockers

**D27 (R2).** The C-3 amendment shipped without the pinned convention key and mutation KAT its own
signed package specified. Reverting `PRIMARY_H → h` passed all 601 tests, silently restoring the
easier bar. Fixed: the pin plus `dsr_unit_convention_failures`, which re-derives all 300 trial values
from the artifacts on the money path — so a revert fails the run, not just a unit test.

**D28 (R3).** The `v1_2_original` dual-specification leg was labelled *"reconstructs the
pre-amendment rules faithfully"* but computed `var_sr_v12` over trials already in `PRIMARY_H` units —
v1.2 de-annualized each trial at its own h. Measured: shipped hybrid var_sr 3.6497e-05 vs faithful
6.3722e-05 (**1.75×**). Smaller variance → higher dsr_v12 → the leg agreed with the primary more
readily **exactly when the primary says SURVIVES**, biasing the v1.5 disagreement safeguard toward
not firing. Fixed by rebuilding the faithful leg.

**D29 (R4).** The G-§8.C.3 gate's `evaluate()` result and the required BSQ disclosure were computed,
passed to an assertion, and **discarded** — bound to nothing, never in the manifest. Two call-site
comments still promised a HALT that no longer happens. Fixed: both persisted as required fields with
a loader that refuses their absence.

**D30 (R6).** The provenance identity tests parametrized over `PROVENANCE_IDENTITY_KEYS` — the live
tuple they were meant to pin — so deleting a key deleted its own test (49/49 still passed). Only
`image` and `lockfile_sha256` were literal-asserted. Missing: the **git commit sha** and the
**training step counts**, so a box launched on a pre-`0b9c804` checkout (2,000 steps) would have
shared all 13 identity values and assembled silently. Fixed: 16 keys including `git_commit`,
`steps_stage1`, `steps_stage2`; the expected key list pinned in a KAT independent of the live tuple;
plus `test_the_stamper_POPULATES_every_identity_key`, because a key the stamper never writes is
absent from all shards equally and protects nothing.

### D31 — the money driver could never have written an eval artifact

**What.** `score_cell` has two return sites. The `val_only` return passed both required diagnostics;
**the full return — the one the verdict artifact is built from — passed neither.** They defaulted to
`{}` and `write_cell_eval_artifact` refused every artifact.

**How found.** P1 — the first end-to-end execution of `m6_money_run.py` in the project's history.
Verified pre-existing against the right baseline (`git show HEAD~1:…/xsection.py` — the full return
never mentions either field), so not collateral from an adjacent fix.

**Why 655 tests missed it.** The end-to-end test asserted on `s.codebook` and never on these two, and
the driver had never been run to completion at this configuration.

**If unfound.** Every unit would have trained for hours on rented hardware and died at first eval.

**Fixed.** `dedfe4c` — the new test is parametrized over **both** returns, so repairing one and
leaving the other cannot pass.

**This defect is the empirical justification for the execution rule (§5).** The supervisor had
triaged "the driver has never been executed at its current configuration" as non-blocking. Running
it is what made it blocking.

### D32–D34 — what renting a box found

**D32.** The fan-out runbook's payload tarball shipped `src/`, `scripts/`, `pyproject.toml`,
`uv.lock` and two small manifests — **not the lake** — and the run command had `--lake …`, a literal
ellipsis. The run could not have executed on any rented box. Cost of the miss, corrected: the lake
gate is at `m6_money_run.py:250` and `train_matrix` at `:363`, so five boxes' **setup** (~$0.50), not
a training run.

**D33.** `driver_version` stamped `unavailable` on a real 4090 where `nvidia-smi` reported
`580.159.03` — `torch._C._cuda_getDriverVersion()` no longer behaves as assumed under torch 2.12.1.
**This is D30's class recurring inside D30's own fix.** Fixed by asking `nvidia-smi` directly, plus
`identity_placeholder_failures` refusing any CUDA unit carrying a placeholder.

**D34.** The offer filter constrained reliability and disk but nothing about the driver, so a box
with driver 565.77 installed the pinned torch and reported `CUDA_AVAILABLE False`. Fixed with
`cuda_max_good>=13.0`.

### D35 — three clause strings named the tested quantity `ΔIR_info`

**What.** Clauses 1, 2 and 4 all described the tested quantity as `ΔIR_info`. The code computes
`pb45.delta_ir` — the paired difference — which the prereg's own §1413 states carries
**(information) + (capacity handicap)**. Both clause 2 and clause 4 test the *difference*; neither
tests cell 4 alone, so the prereg's *"Cell 4 must still clear the MDE and the 0.5 economic floor on
its own"* is false in code.

**Severity.** The rule strings are **persisted into the verdict manifest**. The published artifact
would have asserted that the information effect cleared the floor.

**How found.** A third external reviewer named clause 4. The builder found all three.

**Fixed.** `af3164d` — all three now name ΔIR(4−5) and state that it carries (information) +
(capacity handicap).

### D36 — invariants 4 and 8 asserted a validation never performed

**What.** `docs/ENGINEERING.md` invariant 4 stated Cell 1 is *"externally validated against published
Kronos-small"*; invariant 8 stated the weights *"appear in exactly one place: the eval harness, as an
external validation target."* `external_validation.GATE_IS_BINDING = False` and no weights are ever
pulled.

**Why it persisted.** `docs/ENGINEERING.md` is loaded at the start of every session. A false statement there is
not believed once — it is re-read continuously. `[recollection — no artifact]`: the supervisor
carried the "externally validated" framing into its own briefings for roughly a month for exactly
this reason.

**Fixed.** `0cffb4f` — both invariants now state what is true, and both record that they previously
carried a false statement rather than being silently rewritten.

---

## 3. The decision log

| # | decision | date | commit / doc |
|---|---|---|---|
| K1 | One headline claim; v2/v3 firewall | 2026-06-18 | `026cf58`, docs/ENGINEERING.md inv. 3 |
| K2 | TFI, never OFI | 2026-06-18 | docs/ENGINEERING.md inv. 1 |
| K3 | Train-once / eval-forward | 2026-06-19 | `ada5980` |
| K4 | M5 before M4 | 2026-06-19 | `c37eacd` |
| K5 | 200 pairs, multi-year lake | 2026-06-20 | `33637fb` |
| K6 | Environment pinned before the anchor | 2026-07-04 | `8b0e4b7` → `631bcfd` |
| K7 | Paired decision rule | 2026-07-04 | `1583048` (v1.1) |
| K8 | Causal encoder forced | 2026-07-19 | `85c740b` (v1.3) |
| K9 | Interface re-spec: pointwise fine | 2026-07-20 | `a1729b8` (v1.4) |
| K10 | λ = 3.0 | 2026-07-21 | `39be8ba` (v1.4.1) |
| K11 | μ̂ estimator MODE → MEAN | 2026-07-21 | `c1786f4` (v1.4.2) |
| K12 | Canary gate closed 8/8, B1 attached | 2026-07-30 | `d0bbaeb` (v1.4.7) |
| K13 | v1.5: N 180→60, var_sr basis, S=5, two-term MDE | 2026-07-30 | `6b2ad82` |
| K14 | "Foundation model" dropped | 2026-07-31 | `8ef5568` |
| K15 | Determinism paid at 13.175% | 2026-08-03 | `0b9c804` |
| K16 | Kronos gate dropped as binding | 2026-08-03 | `90e4b64`, `0cffb4f` |
| K17 | C-3 unit fix, pre-data | 2026-08-03 | `0b9c804` |
| K18 | Training budget → 26,003 steps | 2026-08-03 | `0b9c804` |
| K19 | `embargo_flatness` deleted | 2026-08-03 | `614200c` |
| K20 | The three BSQ marginals withdrawn as claims | 2026-08-03 | v2_and_limitations B1 |
| K21 | The run-blocking bar | 2026-08-03 | v2_and_limitations |
| K22 | Cell 6 conditional on a $0.50 probe | 2026-08-05 | `0cffb4f` |

### K1 — one headline claim, and the firewall

The contribution is the microstructure-aware FSQ tokenizer. MTP heads, volatility-scaled targets and
the eval harness are secondary engineering, not competing claims. Out of scope for v1 and firewalled:
equities/cross-asset, Bybit/OKX, orderbook depth (→ true OFI), base-class (~100M) scale-up, MLA
attention, latency micro-optimization.

**Rationale.** A second claim doubles the surface a reviewer must accept and halves the clarity of
each. `[recollection — no artifact]`: this was reaffirmed repeatedly when scope pressure arose,
including against the temptation to elevate the eviction finding to a co-headline (§4).

### K2 — TFI, never OFI

Microstructure imbalance here is computed from free Binance aggTrades: signed *executed-volume*
imbalance = **trade-flow imbalance**. True order-flow imbalance requires orderbook depth and is
explicit v2 work. **A reviewer who sees "OFI" computed from trades alone rejects on the technicality.**
The name is enforced in code, configs and prose.

### K3 — train-once / eval-forward

Adopted `ada5980` from Tier-1 review. Models are trained once on the train region and evaluated
forward through purged walk-forward folds with a leading embargo. The alternative — refitting per
fold — multiplies compute and introduces per-fold researcher freedom.

### K4 — M5 before M4 (a reversal)

The roadmap originally sequenced M4 (universe) before M5 (harness). `c37eacd` reversed it: build and
validate the measuring instrument on the M3 single-symbol checkpoint *first*, because an instrument
validated after the data exists can be tuned to the data. The same commit fixed the M5/M6
Cell-1-validation ordering.

### K7 — the paired decision rule (v1.1)

`1583048`, pre-training. The headline moved to a **paired** ΔIR statistic. Cells 4 and 5 share market
exposure; the common factor cancels in the paired series, which is the only way a thin effect is
detectable at this cross-section.

### K8 / K9 / K10 — the interface chain

Each is a response to a measured failure and is documented in §4. K8 closed a 41.8% leak. K9 followed
the token-space acquittal of the AR. K10 followed the three-seed eviction measurement, with λ
re-derived on corrected instruments after D12 invalidated the first search.

### K13 — the v1.5 amendment window

Six amendments, signed off `6b2ad82`, described in the commit as **the last specification change
before the run**:

- **N 180 → 60.** Seeds are *replicates*, not configurations. A multiple-testing adjustment cannot
  legitimately get stricter because you replicated the same configuration more times. The further
  0.385× loosening to κ-only N=4 was **declined**.
- **`var_sr` basis → the cell-5 placebo.** The all-arms basis made clause 5 anti-correlated with its
  own hypothesis and likely unpassable (required cell-4 IR 3–10 against a 0.5 floor). The proposed
  cell-2 fallback was **rejected** because §5 can *claim* IR(2)−IR(1), making cell 2 a treatment arm.
- **Two-term MDE** with self-written Student-t and Welch–Satterthwaite (t, not z: 2 df at S=3).
  Direction recorded pre-data as **harder**.
- **`PINNED_SEEDS` → 5 seeds up front**, which commits the design to R3 (INCONCLUSIVE) as the
  response to an underpowered result — written out and signed in advance.
- **Outcome taxonomy** SURVIVES / NULL / INCONCLUSIVE.
- **λ slice carved from the end of the train region**, never block 0.

**A ruling premise was corrected during this window and withdrawn.** The claim that every amendment
loosens is false: clause 2 **tightens** 1.24–1.60× while clause 5 loosens 0.859×; net indeterminate.

**Safeguards made binding:** a direction-blind test per amendment, the aggregate reported as one
number, and `dual_specification` as a **required** manifest field — both specs from the same
artifacts, v1.5 primary, disagreement a first-class finding in the abstract. A manifest without it
cannot be quoted.

### K14 — "foundation model" dropped

`8ef5568`. The artifact is a 21.3M-parameter tokenizer study. Calling it a foundation model
overstates the artifact and misdescribes what the ablation measures.

### K15 — determinism paid

Measured penalty **13.175%** (1.152×) on a real 4090 at the money surface, both arms in separate
processes, `postures_actually_differed: true`. The historical 47.2k bars/s figure was itself already a
forced measurement, so the previously-banked 1.3× estimate **double-counted**; confirmed by artifact.

### K16 — the Kronos gate's rise and fall

**Rise.** G-§8.C.3 was specified as a binding M6 **entry gate**: Cell 1's RankIC must reach ≥ 0.85 ×
published Kronos-small on a pinned common slice, with a pre-committed halt-before-any-Δ /
Cell-1-only-fix / full-5-cell-same-seed-retrain protocol. Named binding in three places including
`ROADMAP.md:58`.

**Fall.** Read from the paper (Table 2, p. 6), verified by the supervisor against the PDF:

| Kronos*small* | value |
|---|---|
| RankIC, price-series forecasting | **0.0254** |
| RankIC, return forecasting | **0.0622** |
| dataset | **Shanghai Stock Exchange, 15-minute frequency** |

**Two published figures 2.4× apart — wider than the ±15% band the gate is built from — and both on
equities at 15-minute bars**, a market and frequency v1 firewalls out. Separately, the published
weights are a bare `state_dict` whose loader requires Kronos model code that invariant 8 forbids.

**Ruling.** Dropped as binding (Lakshay, 2026-08-03). The compensating control is a **required
disclosure** carried in every verdict manifest and refused if absent:

> *"we cannot exclude that our BSQ baseline is weaker than a reference BSQ implementation, which
> would inflate the FSQ-vs-BSQ comparison reported in §5."*

**A proposed substitute was rejected.** Train-to-saturation was offered as a replacement control. It
fails: Kronos-small **is a BSQ model**, so the gate was the only control anchoring our BSQ
implementation against a reference BSQ implementation, and saturation cannot see that — a poorer
token stream converges to a worse val NLL and saturation certifies "converged", which is true and
useless. The word "substitute" was withdrawn. Codebook health was then examined as an alternative and
also rejected: utilization measures **that** the bits were spent, never **what on** — it discriminates
collapsed from non-collapsed, not crippled from competent.

**Consequence:** the three 2×2 marginals with a BSQ arm are withdrawn as claims (K20).

### K18 — the training budget (a correction to spec, not a preference)

Raised 2,000 → 26,003 steps. **The decisive argument is that the design spec already specified a
budget and the code ignored it** — the spec sets 1–3 corpus passes at ≈0.5M tokens/step with
early-stop on val-NLL saturation, and `docs/ENGINEERING.md` makes the spec a source of truth. Raising the
budget **implements the blueprint**; leaving 2,000 is a standing undisclosed deviation from it.

Corroborating detail: the spec sets `eval interval = every 5000 steps` against a total budget of
2,000, **so the designed schedule's first evaluation never fires** — independent proof the number was
never reconciled with the spec.

**Cost, measured:** training is **2.6–6.1%** of spend; 13× the budget costs **+$16–24**.

**Reconciliation recorded.** The spec says early-stop; `m6_design.md:18` says all cells share the same
training draw. Resolved for v1 as **fixed matched budget, saturation measured and reported** — because
data-dependent stopping would give 25 different budgets and confound ΔIR(4−5) with "cell 4 trained
longer", which is the C-12 class inside the primary. Registered as C5 in the limitations.

**And see D16's corrective note:** the "1–3 passes" justification was arithmetically wrong; it is 5.06
passes over 84,153,600 bars. The token budget stands; the stated reason does not.

### K19 — `embargo_flatness` deleted, with evidence

A purge/embargo leak diagnostic whose only caller was its own test. **Cost of wiring it: 3× the M6
run (~$99–150)**, because the embargo binds `fold_valid_starts`, which gates which training windows
are legal — each E needs its own training.

**The premise it defends was measured instead** (control arm recovers a known AR(1) first): signed-return
ACF at lag 60 is **0.0067 mean / 0.0191 worst-symbol** across all 200 symbols, already ≈0 by lag 5,
against `L_corr = 60` — a **~24× margin**.

**What is lost, on the record:** end-to-end IR flatness in E is now asserted from the premise, not
demonstrated, and |return| ACF is 0.20 at lag 60 (volatility clustering, not a label-leakage channel).
Registered as limitation A2.

### K21 — the run-blocking bar

Pre-committed 2026-08-03, **before the re-audit's findings existed**, and not adjustable afterwards:

> A finding delays the run **if and only if** it would cause us to publish a FALSE VERDICT — SURVIVES
> when the truth is NULL, or NULL when SURVIVES — **and cannot be neutralized by disclosure.**
> Everything else goes to `docs/v2_and_limitations.md` and the run proceeds.

**Rationale.** The audit-fix-reaudit loop produces real catches, every catch feels like progress, no
spend is ever risked, and the project approaches the run without taking it. The bar exists to make
"done" decidable. `[recollection — no artifact]`: it was written after the operator named that
pattern explicitly.

**It has been applied twice.** Against the re-audit (5 of 12 findings blocked) and against C-12 —
where it initially produced the wrong answer (§7).

### K22 — cell 6, conditional

Adding a sixth cell (16 dims, micro slots **constant**) would bound the C-12 capacity handicap in IR
units via a bracket: **ΔIR(4−6) ≤ information effect ≤ ΔIR(4−5)**.

**Not adopted unconditionally**, for three reasons established by the builder against the supervisor's
ruling: (i) the bracket's ordering is an *empirical* claim about two trained models, not an identity —
if it inverts the interval is empty and the larger number flatters us; (ii) constant dims may drop
codebook utilization below the 0.95 gate (refusing every cell-6 artifact) or realized bpt outside
19.5–20.5 (making the lower end a different-capacity instrument); (iii) the *blocking* half of C-12
was the false rule strings (D35), which cost $0.

**Decided by a ~$0.50 Stage-1 probe on box 1**, with the rule in code
(`scripts/m6_cell6_stage1_probe.py`, `decide()` reading live pins, boundary-tested including that it
can say no). Both gates pass → wire it, +$16–22. Either fails → decline and report the handicap as
unquantified in IR units.

---

## 4. The mechanism narrative

This section is written to feed the paper's §3 directly. Every number cites its receipt.

### 4.1 The setup

The tokenizer is trained to reconstruct. The implicit assumption — never stated as an assumption
until it failed — is that **a representation good enough to reconstruct from is good enough to
forecast from.** Everything below is the measurement of that assumption failing.

### 4.2 v6 — a planted signal, and zero extraction

**Commit `6feb60f`, 2026-07-20, ~$0.60, box destroyed.**

Design: a sub-epoch stream of **42M bars** (0.75 epochs, so memorization is not available as an
explanation), two arms — one carrying a planted lag-2 conditional worth **~1.15 nats** **[the exact value is 1.151292546497023 = ½·ln(1+c²) at c = `C_SIGNAL = 3.0`, `scripts/m6_canary.py:100`; this line's tilde was later cited as evidence that no more exact figure existed, which was wrong — annotated 2026-08-21, the recorded wording is left as written]**, one noise —
with the arithmetic recorded and asserted, and a `train_nll_main` log for the memorization-gap check.

Result:

| quantity | measured |
|---|---|
| val−train gap, both arms | **±0.007** (sub-epoch regime achieved) |
| schedule | complete, both arms, peak 3e-4, **no spikes** |
| planted-signal correlation | **< 0.05 at all 40 evaluation points** (max 0.043) |
| val(planted) − val(noise) | **+0.013** |

**The AR extracted zero nats from a plant its tokenizer provably encoded** — dim-9 reconstruction was
non-degenerate. Recorded as **branch (b), architectural full stop.**

### 4.3 The token-space control — the AR is acquitted

The v6 result admits two explanations: **H-T**, the tokenizer's id geometry hides the rule; **H-A**,
the AR cannot learn thin conditionals at canonical scale. These are separated by planting the signal
**directly in token space**, bypassing the tokenizer.

**Step 0 (`e891d8c`, $0) blocked the first attempt and produced its own finding.** The coarse digits
are **99.2–99.8% max-entropy**, so a dense graded translation plant maxes at **0.2769 nats exact** —
below the 0.7–1.1 band the branch condition needs. The same probe measured per-bar id visibility:
**logistic 0.5135, MLP correlation 0.0142** — the v6 state is nearly invisible in per-bar ids
*despite non-degenerate window reconstruction.* **That is the first direct evidence for H-T**, and it
arrived from a gate that refused to run.

**Step 0 re-gate (`7675393`, $0).** Translation retired as info-poor by physics on uniform marginals.
Adopted: a **noisy-monotone-map** plant — the target digit redrawn from a concentrated distribution
centered monotonically on the source level, σ tuned so exact information lands in band. Receipt:
σ 1.0077 → **exact 0.9000 nats** (MC 0.9009, agreeing), KL 0.00476, ΔH −0.0054 (no stacking needed),
oracle ceiling 0.926.

**The run (`2c41b67`, ~$0.60, box destroyed): branch (a) DETECT.**

| quantity | measured |
|---|---|
| probe Spearman | crosses 0.3 at step **2,000**; final **0.9999** |
| final val | **12.7483** = H0 − 0.8496 |
| fraction of planted information extracted | **94%** of 0.9003 nats |
| schedule | complete, no spikes, sub-epoch held |

**H-A is refuted. H-T is confirmed.** The AR learns per-bar-legible token rules essentially perfectly,
at 10× speed. The v6 zero-nats finding localizes to the **tokenizer → AR interface**.

### 4.4 The mechanism — a causal encoder smears per-bar state

The window encoder is causal but *contextual*: bar t's token is a function of bars ≤ t. State
belonging to bar t is therefore distributed across the ids of every subsequent bar in the window, and
**no individual id carries its own bar's features legibly.** The window reconstructs; the per-bar
symbol does not inform.

Receipt-2, recorded as a standing mechanism: *causal encoder smears state forward across window ids;
per-bar illegible.*

### 4.5 The architectural fix

**`a1729b8` (§7 v1.4), 2026-07-20.** The **fine** subtoken becomes a per-bar **pointwise** encoding of
bar t's own features; the coarse subtoken keeps the contextual causal encoder. Both properties
preserved in one token pair. Pinned as `PINNED_FINE_POINTWISE`, with extended flip-KATs and
anti-vacuity tests.

**Gate 2 blocked immediately, with receipts.** Winner-dim arbitrariness: 10.26 fine bits ≈ 4–5
arbitrary winner dims at the 1.78 bits/dim that 0.9 sign accuracy requires; dim 9 wins ~1/3 of inits
on a 3-seed receipt; all-13 coverage would need 23 bits against 20.06 bits/token total.

### 4.6 The eviction measurement — the project's second result

**`e1feb64`, 2026-07-20, local $0.**

The fixture was first calibrated to the real lake: **0.4631 nats/dim measured on 3.9M bars** →
AR(1) ρ = 0.7993, receipt in-band. Then the standing micro-legibility gate landed. Then the gate was
run on three seeds.

**It failed 3/3, and the failure carried its mechanism.** Per-dim point-decoder correlations, near
identical across seeds:

| feature class | correlation |
|---|---|
| return (high variance) | **0.98** |
| correlated fillers | **0.82 – 0.92** |
| independent state dim (dim 9) | **0.001 – 0.014** |

**The state dimension is priced out deterministically, not by chance.** The commit states the
mechanism in one line: **reconstruction loss buys variance and covariance, never independence.** A
lottery hypothesis was replaced by deterministic exclusion.

**Why this matters beyond Trikaal.** Microstructure is low-variance and weakly covariant with price.
A tokenizer trained to reconstruct will allocate bits away from it *by construction* — not as a bug,
but as the objective working correctly against the downstream goal. This predicts the real-data TFI
failure mode and is the paper's argument for why deliberate micro-tokenization is necessary rather
than incidental.

### 4.7 The calibration and the pin

**`d9e1b53` (2026-07-21):** a wiring bug found, fixed and disclosed — `loss_point` read `w_feat`, not
`w_feat_point`. The first λ-search was invalidated (retained only as λ=1 replicates) and an asymptote
analysis withdrawn. A **detach amendment** landed: window losses see fine channels detached, which
separates the objectives and stabilizes collateral damage.

26-trial receipt: **no (λ, β) clears 0.9 on 3/3 seeds.** The legibility sits 0.84–0.93 around the
threshold at the achievable ceiling **while the state is demonstrably carried** — dim 9 at 0.77–0.96,
class 0.86–0.92, return 0.96+.

**`39be8ba` (2026-07-21):** adjudication. Three disclosures including an **unseeded-init defect** that
invalidated the λ=2 literal. λ re-derived on corrected, seeded instruments → **`PINNED_MICRO_POINT_WEIGHT = 3.0`**
(λ=2 fails the restated gate at mean 0.8517; λ=3 passes 0.9060 / 0.9142 / 0.9000). The formal
cell-path re-run passes at mean 0.9047 with direct == cell-path bit-identity proven at fixed OMP
threads.

### 4.8 The standing gate

`micro_legibility_gate` (`gates.py`): after Stage 1 and **before any Stage-2 spend**, each of the six
micro dims must be linearly recoverable from bar t's own id at logistic sign-accuracy ≥ **0.90** on
the run's real training stream. Raises `RuntimeError` — a hard stop.

Its two defects (D21, D22) are ledgered above. As it now stands: a skipped dim can never contribute
to a pass, and the sample is stratified by symbol with blocking preserved in time.

**Expectation, recorded pre-data.** Real TFI is exactly the eviction profile the fixture measured, and
the fix was validated on a synthetic plant. **If the gate fires on real data, that is the real-data
confirmation of the mechanism** — simultaneously the ablation's blocker and the mechanism paper's
strongest evidence. Its adjudication is pre-written.

### 4.9 Framing commitment (binding)

The eviction result was **measured on a synthetic fixture, not on real microstructure.** It is not to
be elevated to the headline. The outcome-independent framing stands: both "micro survives" and "micro
nulls after costs" are complete papers containing it.

---

## 5. Standing rules

Every rule below exists because of a specific failure. None is a general principle adopted in advance.

| rule | born from |
|---|---|
| **Pipefail rider** — an exit code read through a pipe is the pipe's | D13: `\| grep \| tail` masked an exit-1 and a false anchor claim entered a commit message |
| **Control-arm rule** — both arms failing is `PROBE INVALID`, never a conclusion | a probe emitted a verdict while both arms died of the same `TypeError` |
| **Fixture-discrimination rule** — prove the fixture separates the cases before relying on it | the v1.5 A.4 mutation check passed vacuously because the fixture made two dispersions numerically identical; caught only because the mutation *failed to fail* |
| **Degeneracy rule** — a perfect agreement score is as suspicious as a failing one | `micro_shuffled/pre_v1_4` scored **1.0000 sign agreement at variance_ratio 0.000** — a collapsed decode; without the flag the comparison would have read ASYMMETRIC on an artifact |
| **Self-scaling rule** — do not invent a band; scale against the claim or the statistic's own spread | a symmetry reading used a hand-chosen 0.05 band and landed at 0.0469 — the conclusion rested on an invented convention |
| **Baseline rule** — check a claim against the commit it was made about | an audit citation was disputed and the dispute withdrawn: the builder's own fix had added 64 lines, moving the string from 520 to 584 |
| **Class rule** — fix the class, not the instance; when the class needs a tool, the tool is the deliverable | the C-17 pass corrected the strings an audit named and left two siblings standing, one half a sentence from an edit |
| **A check that cannot fail is not a check** | D17, and four recurrences — including D21, where the *unit test* was the defect's written contract |
| **Mock rule** — a mock is not a subject; assert the call, not the stub's reply | the first F3 test stubbed `subprocess.run` and asserted on the stub's return, so corrupting the argv left it green. **The argv is the contract.** |
| **Execution rule** — "has never been executed" is not an operational state; it is an unmeasured risk | D26 (a symbol confirmed to exist, not a gate confirmed to fire) and D31 (a driver confirmed to be built, not to run). **Any component whose first execution is on rented hardware is untested, whatever its coverage says.** |
| **Remediation rule** — audit the fix, not only the finding | D26: the codebook gate shipped below a `return` in the commit claiming to close its defect class. A finding is closed only when its pre-fix source has been restored and the new test observed to fail there |
| **Reporting rule** — state every routine verification's exit code explicitly, pass or fail | a report silently dropped its `ruff` line and `ruff check` was exit 1. **An omitted claim is harder to catch than a false one:** a missing green line must be *noticed*, a wrong one only has to be *read* |
| **Anchor rule** — the Gate-A anchor is procedure-anchored, re-proven from an unpiped command, twice | D13, and the two legitimate re-anchorings (`5eead7b6` → `3f86882a`) when a file inside the anchored instrument changed |
| **Verify, don't relay** — the supervisor verifies every claim in source before ruling | recurring; §7 enumerates what it caught and what it missed |
| **$0 gate** — post the exact invocation and a cost estimate before any spend | the CUDA-probe tooling commit (`845c287`) records the gate *catching a defect* before money moved |
| **Teardown rule** — `vastai destroy -y`, then **re-list** | `destroy` prompts, aborts, and returns exit 0 with the box still running. Extended after P2: **re-list after `create` too** — `create` printed nothing and still created a box |
| **Cost a rental as setup + compute** | setup was 90% of a $0.14 probe; provisioning ranged 29 s to 45 min on hosts with reliability 0.992–0.9987 |
| **No silent caps** | a bounded scope that is not stated reads as complete coverage |
| **Context-stripping rule** — an artifact must remain safe when its context is stripped; assume every excerpt travels alone | paper Figs 9/10 drew placeholder IRs at plausible magnitudes in the hoped-for cell ordering, protected only by a DRAFT watermark that any crop removes. **Placeholder data must be impossible to mistake for a result with the watermark gone** — absent, absurd, or symmetric about zero. Binds captions, tables, abstracts and receipt fields, not just figures |

---

## 6. Current state

### What exists and is verified

| | |
|---|---|
| HEAD | `0cffb4f`, branch `real-data-slice`, pushed, rev-list 0 |
| tests | **727 passed, exit 0** |
| lint | `ruff check` exit 0 · `ruff format --check` exit 0 |
| Gate-A anchor | **`sha256:3f86882a63dd06c7…`**, byte-identical, exit 0 ×2 |
| mutations | **15/15 closed** + passing negative control (`m6_reaudit_mutations.json`) |
| lake | 200 symbols, **304,625,181 bars**, Merkle `5dfd667d`, on HuggingFace (private, ungated, 7,029 files, 14.72 GiB) |
| run subset | **4.08 GiB / 1,920 files / 40 pinned symbols**, sha256 per file committed pre-transfer |
| P1 | end-to-end dry run **exit 0** — 25 trained, 25 checkpoints reloaded, 25 artifacts + index, manifest produced; loader correctly refused the dry-run set |
| identity surface | **16 keys**, all stamped, one distinct stamp across 25 units |

### The pinned surface

seeds (0,1,2,3,4) · seq_len 512 · batch 32 · steps 26,003 · h=15 · DSR n_trials 60 (cells × horizons
× κ) · var_sr basis = cell 5 · λ = 3.0 · encoder_causal True · fine_pointwise True · predict_mu
"expectation" · headline cost 0.30% · κ ∈ (1.0, 1.5, 2.0, 3.0) · ECON_FLOOR_IR 0.5 · BOOT_POWER 0.8 ·
PLACEBO_DISPERSION_TRIPWIRE 1.5 · MICRO_LEGIBILITY_MIN 0.9 · codebook utilization ≥ 0.95 · bpt band
19.5–20.5 (realized 20.0578) · bootstrap B=10,000 seed 20260704 · `dual_specification` required ·
`mu_diag`, `ohlcv_recon`, `decode_agreement`, `codebook` all required per (cell, seed).

Every pin carries a mutation KAT proving the gate rejects its predecessor.

### Spend, from receipts

| phase | amount |
|---|---|
| toy-CUDA rehearsal (`77ca29a`) | ~$5.00 `[recollection — figure appears in the commit message, not a receipt]` |
| canary v5 (`fbc1cf2`) | ~$0.70 |
| canary v6 (`6feb60f`) | ~$0.60 |
| token control (`2c41b67`) | ~$0.60 |
| acceptance run (`36560dc`) | ~$0.90 (incl. $0.15 on two bad hosts) |
| money-leg re-run (`7784df2`) | ~$0.60 |
| CUDA probe (`2a320a0`) | **$0.80** |
| eval-throughput probe (`0b9c804`) | **$0.14** |
| P2 attempt (`2559c5c`) | **$0.426** (three stalled boxes) |

**Current balance recorded at HEAD: $2.30 of $8 spent, ~$5.70 remaining.** The earlier phases were
funded separately; `[recollection — no artifact]`: the $2.30 figure is the current wallet accounting
and does not include the pre-July rentals above.

### The measured price of the run

| | GPU-h | @$0.29 | @$0.40 |
|---|---:|---:|---:|
| eval (measured: 50.0 dec/s, 1,402,560 decisions × 25 units × 0.020009 s) | **194.9** | $56.52 | $77.96 |
| training @26,003 steps, forced determinism | **75.5** | $21.90 | $30.20 |
| **total** | **270.4** | **$78** | **$108** |

Top-up target **$150**; **$175** if the cell-6 probe returns WIRE.

### Position relative to the run

Everything buildable without spend is built. The launch sequence (`docs/m6_fanout_runbook.md` §1a/§2b)
is one executable block: rent box 1 → payload + pinned toolchain → lake pull, sha256-verify per file,
scrub the token and prove the scrub → **P2** (`--dry-run --shard 0/25 --device cuda`) → **cell-6
Stage-1 probe + the pre-committed decision** → real shard 0 → **P3** (16 identity keys real) → boxes
2–5.

**The only outstanding item is the operator top-up.**

---

## 7. Honest self-assessment

### 7.1 Withdrawn claims — builder

| claim | withdrawn because |
|---|---|
| `d9e1b53`'s "anchor re-proven" | false; a `\| grep \| tail` masked exit-1 (`bb1e17c`) |
| the first λ-search and its asymptote analysis | `loss_point` read the wrong weight (`d9e1b53`) |
| "the lazy dry run will also be killed" | it was not killed — macOS **swapped**, the more dangerous failure mode (`4dd3d0a`) |
| a `verdict.py:519-521` provenance correction | withdrawn under the baseline rule; `git show` proved the auditor right |
| a Z1 half-width | 5× too strict |
| the C-4 retrain contingency at "~$66–100, roughly doubling" | assumed training was ~half of spend; measured 2.6–6.1%. True figure **+$1.51–2.32** |
| "up to 2.5× the banked figure" (eval cost) | mixes rate endpoints; at matched rates **1.60–1.64×** |
| "the lake step would have hit after training spend" | the gate is 113 lines before `train_matrix`; cost was setup only |
| the F3 test asserting on a stub's return | a mock is not a subject; rewritten to assert the argv |

### 7.2 Withdrawn claims — supervisor

| claim | withdrawn because |
|---|---|
| an MDE "floor" of 4.35 | combined a ν→4 multiplier with an SE from the mutually exclusive `se_train=0` limit |
| "run eval on the Mac, 12 hours, free" | divided a σ-scaled numerator by an unscaled denominator; true figure **28,589 hours** |
| "both HALT guards are blind" | repeated an auditor's phrasing without checking; `power_guard` reads `headline_series` |
| a leg-(ii) asymmetry test | specified against the wrong pair; cells 4 and 5 are the same width |
| a symmetry reading using the power guard's refusal form | `\|diff\| / one arm's sd` for an inference question; correct is `\|diff\| / SE(diff)` |
| "the build is done" (×2) | it was not |
| "the pinned DSR basis is the easier bar — the deciding fact" | direction is **data-dependent and unknown on the real cells**; the builder's own receipt said so twice and the qualifier was dropped |
| "saturation substitutes for the Kronos gate" | Kronos-small is a BSQ model; saturation cannot see a weak BSQ implementation |
| "codebook health controls BSQ competence" | it discriminates collapsed from non-collapsed only |
| "ΔIR(4−6) ≤ ΔIR(4−5) **always**" | an empirical claim about two trained models, not an identity |
| "all six ruff errors are in one file" | an anchored grep matched 5 of 6 while the same output printed "Found 6" |
| "the manifest is not on disk anywhere" | a repo-scoped `find` against an "anywhere" claim; it was in a scratch path |
| the C-12 finding blocks the run | the *blocking* half was the false rule strings, which cost $0; cell 6 is additive |

### 7.3 The pattern, stated plainly

**Builder failures cluster on probes that manufacture findings from their own bugs** — three in one
week, each caught by the builder itself: a leak planted at bar 1640 on a 1600-bar slice reported
"MISSED" with a ★FINDING verdict; a registry that mutated the wrong key so guards never saw
degenerate input; a marginal-seconds field computed from `runs[:2]`.

**Supervisor failures cluster on two shapes.** First, **a formula used outside the regime it was
derived for** — the MDE floor, the Mac timing, the symmetry statistic. Second, and more persistently,
**an instrument narrower than the sentence written with it** — the anchored grep, the repo-scoped
find, a glob matching filenames when the key was in the directory name, and three exit codes read
through pipes. The reasoning has generally held; the verification has not.

**The single most instructive pair.** D26: the supervisor verified a gate existed by finding its
symbol, and never checked it was reachable. D31: the supervisor verified a driver was built, and
triaged "never executed" as scheduling. **Both are confirming existence instead of behaviour** — the
exact defect class being enforced against everyone else.

### 7.4 What caught what

| detection layer | findings attributable |
|---|---|
| design review (M0, M1) | D1–D4 and the spec-stage corrections |
| automated gates | D6 (audit), the causal sweeps, the conformance pins |
| planted-signal canaries | D8, D9, D10 — the mechanism findings, none of which any gate could have produced |
| external audit 1 | 20 confirmed + 4 suspected, in a repo two parties had reviewed for weeks |
| external re-audit | 12 more, four of them inside the commit that fixed the first twenty |
| **execution** | D31 (657 tests could not see it), D32, D33, D34 — none reachable by any static check |
| mutation harness | D33's first test, and its own author twice |

**What is proven is not that the internal loop catches things — it demonstrably did not. What is
proven is that commissioning independent review works, and that executing a thing finds what
inspecting it cannot.**

### 7.5 What remains unproven or at risk going in

1. **Eval under forced determinism on CUDA has never completed.** The single uncovered leg. P2
   stopped at the lake gate. Training under forced determinism on CUDA *is* measured, and eval is
   forward-only, so the risk is low — but it is not measured.
2. **Statistical power.** Reference MDE at h=15 is **3.518 annualized IR** against a 0.5 economic
   floor; the paired MDE governs and will sit below it to the extent cells 4/5 share exposure, by an
   amount not knowable pre-data. Against an M2 prior of |RankIC| ≈ 0.027, **NULL or INCONCLUSIVE is
   the most likely honest outcome.** Nothing in the pinned design manufactures power it does not have.
3. **The C-12 capacity handicap is unquantified in IR units** unless the cell-6 probe returns WIRE.
   The disclosure is a reconstruction-MAE ratio and no reconstruction-to-IR mapping exists.
4. **The BSQ baseline has no external anchor** (K16). The three affected marginals are withdrawn as
   claims, but the FSQ leg remains the only claim in the design with no internal control.
5. **One calendar year, one regime, 40 symbols.** No cross-year replication.
6. **The embargo is justified on the signed-return channel only** (A2).
7. **5.06 training passes over 84.2M bars**, above the spec's 1–3 band (D16 corrective note).
8. **The 512-bar context is 8.5 hours at 1-minute resolution.** Inherited from a configuration
   designed for 15-minute equity bars, where it spans ~5 days. Partly compensated by explicit
   temporal embeddings. **This was never examined until 2026-08-05** and is newly registered here.
9. **The cell-6 probe script has never executed.** Disclosed by the builder; placed after P2 and
   before any production shard, where failure costs minutes and no science.

---

## 8. Surfaced during reconstruction

Items this reconstruction produced that were not previously in any ledger.

**8.1 — The "1–3 passes" justification is arithmetically wrong.** Recorded at D16 and K18. The
training budget of 26,003 steps is **5.06 passes over 84,153,600 bars**, not 1.399 over 304,625,181.
The token budget (20.0 tokens/param, compute-optimal) is correct and unaffected; the stated
justification is not. **Now registered as risk 7.**

**8.2 — The 512-bar context was never examined for this data.** 512 bars is 8.5 hours at 1-minute
resolution, against ~5 days at the 15-minute resolution the configuration was designed for. The model
cannot see the previous day. Registered as risk 8; belongs in the paper's limitations.

**8.3 — Two items named in the reconstruction request could not be located in artifacts.** The
request listed *"the swapped-cells spec table"* and *"the contradicting checkpoints (7.8)"*. Searching
the prereg, design doc and commit log surfaces neither; the only "7.8" matches are the 27.8 GiB
memory-strategy figures. **Flagged rather than reconstructed from memory.** If these are real, they
exist only in conversation and should be re-supplied so they can be ledgered properly.

**8.4 — The rollout μ̂ one-bar lookahead could not be isolated as a distinct artifact.** The M1 review
(D1) records a localized single-bar lookahead the sparse sampler missed, and M5 records the harness as
leak-free on the M3 checkpoint. Whether a separate rollout-stage μ̂ lookahead was found and fixed is
**not determinable from artifacts**; `[recollection — no artifact]`. Marked here rather than merged
into D1.

**8.5 — Pre-July spend is not reconciled against the current wallet figure.** The commit messages
record ~$5.00 + ~$0.70 + ~$0.60 + ~$0.60 + ~$0.90 + ~$0.60 ≈ **$8.40** of rentals before 2026-07-31,
while the current accounting reads $2.30 of $8. These are different funding periods. **No receipt
reconciles them**; the totals in §6 are presented as two separate ledgers rather than summed.

**8.6 — `q_over_adv = 1e-3`** remains hardcoded at `harness.py:83`, unpinned and unreceipted. It feeds
the modelled-cost secondary and never the headline, and `harness.py` is frozen. Registered as C7,
non-blocking by construction — recorded here so it is not rediscovered as new.

**8.7 — The `student_t_ppf` root-finder residual** (~1.1e-8 at (0.99, df=30)) is pre-existing and was
unmoved by the `_betacf` rewrite. The design operates at Welch df 2–4 where agreement is ~1e-12.
Registered as C8.

---

*End of record. Reconstructed 2026-08-05 at `0cffb4f`. Every numeric claim above is traceable to a
commit, a milestone document, a receipt under `runs_manifest/`, or a line of source; statements
without such backing carry `[recollection — no artifact]`.*

---

# 9. Pre-publication audit remediation — 2026-08-17

An independent auditor completed a full pre-publication audit and returned **PUBLISH WITH NAMED
CHANGES**. It praised the measurement layer specifically — the Gate-A anchor re-run end to end and
reproduced bit-for-bit with zero git diff on the tracked manifest, 864 tests with no xfails and no
unconditional skips, five gates attacked and all five biting, the codebook gate re-killed in a
throwaway worktree, all six pinned parameter counts recomputed exact, ~90 paper values traced to
the digit, the demo acceptance gate 9/9 and `demo/index.html` rebuilding byte-identical. **Every
defect it found was text, figures or repository hygiene. None needed a GPU-hour.**

What follows is what was changed, what was refused, and what was routed. Items under `paper/` are
the writer's and are reported rather than edited.

## 9.1 — The paper's sole §6.6 artifact was never committed

`runs_manifest/m6_lambda_sweep.json` is the only artifact behind §6.6, the once-only λ
re-derivation. Not gitignored — never added. A fresh clone could not verify the subsection, every
local check passed, and nothing in the repository could tell the difference. **This is the
pre-declaration defect repeating, in a new costume: an artifact whose existence is asserted rather
than checked.**

The commit is the instance. **[NOT LANDED — RELABELLED 2026-08-20 AS PROPOSED. `tests/test_paper_artifact_citations.py` is not tracked and never was; nothing described below fails CI. It was created and then deleted on 2026-08-19 when `paper/` left the repository, at which point one of its five passing cases had become vacuous — and with `paper/` gone it would now sweep an EMPTY DIRECTORY, so it is not being rewritten. The sentence 'it reproduced the auditor's finding independently' describes a run that happened; the test that performed it no longer exists.]** The fix is `tests/test_paper_artifact_citations.py`, which sweeps every
`% artifact:` citation under `paper/` and requires each path to EXIST and to be TRACKED. "Tracked"
is the operative word: "exists on the builder's disk" is precisely the state that makes an
unverifiable claim look verified, and it is the only place the two predicates disagreed. Run
against the pre-fix tree it reproduced the auditor's finding independently and named exactly one
file; it was also watched failing against a synthetic paper citing a ghost receipt and a
present-but-uncommitted one, so it cannot pass by doing nothing.

`runs_cloud/` and `processed/` are gitignored by design and are allow-listed — but each allow-list
entry must name a **tracked mirror**, asserted in the same file, so the exception cannot grow into
a hole. Eleven other untracked load-bearing files landed with it, disposition COMMITTED in every
case.

## 9.2 — Clause 2 shipped the superseded MDE formula

`verdict.py`'s clause-2 rule string read `MDE_paired = (z0.95+z0.80)·SE_boot` — the pre-v1.5
recipe — while `paired_bootstrap` computed t-quantiles at the Welch–Satterthwaite ν over SE_total,
the signed-off v1.5 item-B amendment. The string ships inside every emitted verdict manifest. No
manifest was ever emitted, so nothing false reached a reader; the string was in the released
artifact and asserted a recipe the code does not run.

**This is the class rule failing two lines from where it is written down.** Clause 5's own comment
says *"a rule string that cannot disagree with the recipe is the fix"*, and clause 5 was rebuilt
from its pins. Clause 2's formula half was left restated. Sweeping the class found a **third**
site the audit did not name: `paired_bootstrap`'s module docstring stated the z form as the rule,
directly above the code computing the t form.

The fix is structural. `mde_terms()` is the single site deciding the recipe; both the number and
the prose read it. The string carries the multiplier and the SE it multiplies at 12 significant
figures in a pinned `:: A x B = C` tail, and `tests/eval/test_mde_rule_string.py` parses them back
out and requires `A·B == C == mde_paired`. Verified a bit-exact no-op: 18 configurations × 9 float
fields, `.hex()` equality against the pre-change module, with a control proving the comparison can
report a difference.

**Routed, not edited:** `docs/m6_prereg.md:75` states clause 2 in the same superseded form.
`runs_manifest/m6_toy_verdict_manifest.json` carries the stale string as a record of what was
actually emitted at the time; regenerating it would falsify the record, so it stands and is
disclosed. `scripts/m6_prereg.py:15` uses z correctly — a different quantity (the tabled temporal
MDE).

## 9.3 — The release the paper promises did not exist

No `LICENSE`, no model card, no weights manifest, no release path of any kind — while §8.9 promises
"the trained weights" and the M8 gate requires Apache-2.0. "Specified but not enforced", at the
release boundary.

`LICENSE` (Apache-2.0, extracted from a vendored copy and diffed word-for-word against a second
independent copy, 11,358 bytes = the canonical length); `docs/MODEL_CARD.md`, which leads with what
the weights are **NOT**, because that gap is the result; `runs_manifest/m6_weights_release.json`
with per-file SHA-256, roles, and parameter arithmetic **recomputed from the tensors** (31,725,568 /
10,493,952 / 21,231,616), `--verify` re-hashing and refusing on drift. Deliberately **not** an
uploader: the HuggingFace push stays an operator action.

`runs_manifest/m6_cell1_eval_mirror.json` mirrors the `runs_cloud/` artifacts §7.5 and §7.10 cite,
verbatim including `headline_series` — a summary would leave a reader able to read our information
ratios and unable to recompute them. §7.5's activity figures reproduce to the digit off the mirror.

## 9.4 — ★ Three false external-validation claims neither audit named

`docs/ROADMAP.md` asserted in **four places** that Cell 1 is *"externally validated against
published Kronos-small"*, that *"the public weights appear only inside the eval harness as a
validation target"*, and that the metric cross-check was *"deferred to M6 (weights load there
anyway)"*. **No Kronos weights were ever pulled and none ever will be.** The gate was dropped as
binding on 2026-08-03; `external_validation.GATE_IS_BINDING = False`. docs/ENGINEERING.md has called that
wording *"a false statement of what validation we performed"* for two weeks — in the file every
session reads first — and the roadmap, which is the file a newcomer reads second, kept saying it.
Corrected at all four sites, plus `docs/m6_design.md`'s cell table and the two occurrences in the
blueprint spec (annotated rather than rewritten: the spec records what was SPECIFIED).

The lesson is not "the roadmap was stale". It is that **a correction applied to the canonical
statement of an invariant does not propagate to the documents that restate it**, and nothing in the
repository was looking.

## 9.5 — The verification tooling was the least verified code in the project

The tooling rule, sixth instance, and the auditor found it rather than the builder.

**`m6_verify_extracted_facts.py` carried four false-pass vectors underneath a green mutation
control.** The control proved the checker could refute a *wrong* value; it never asked whether the
checker declines an *undecidable* one. All four were demonstrated live against the shipped code and
all four returned CONFIRMED: a claim and its own negation confirming each other via bidirectional
substring containment (`"supported"` inside `"premise NOT supported"`); `"true"` inside `"untrue"`;
a multi-number claim confirming against **the number it was contrasting with**; and a claim's number
appearing anywhere inside a container — unconditional in a 35,000-element series. `_values_agree`
now returns three values rather than two, because the missing one was "I cannot decide".

**The confirmation rate fell from 424 to 147 of 626.** That is a property of the checker, not of the
facts: quarantine says the tool never had grounds, not that the fact is wrong. The receipt carries
the superseded totals and a `READ_THIS_FIRST` so a count cannot be quoted without them. The tool was
also **not re-runnable at all** — its documented input was a session scratch file that no longer
exists — so the input is now reconstructed from the receipt's own rows and committed beside it.

**`m6_receipt_provenance_sweep.py` had two independent vectors, and I had the causal story
backwards on the second.** Prose washing is real and measured: 6,701 of a 17,058-token vocabulary
entered only via docstrings and sentence-shaped literals, 39% of the evidence the tool weighs. But
that is **not** why `m6_demo_seed_sign_disagreement.json` — committed with no producer anywhere in
the repository — passed. I asserted washing, then measured it: excluding prose moves that receipt
from 11 to 14 untraceable key paths and still would not flag it. The real reason is that the
detector is **block-shaped**, needing a nested object of ≥3 keys before it says anything, and that
receipt is flat top-level scalars plus one list. Both are fixed; `hand_authored_flat` is new.
**Closing one vector while believing it was the other is the failure this entry exists to record.**
Flagged receipts move 4/78 → 16/98, the correct direction for a tool that was under-reporting.

The missing producer is landed as `scripts/m6_demo_seed_sign_disagreement.py`, and its `--check`
reproduces the committed receipt.

## 9.6 — The demo's lake pin was doing less work than it looked like

`LAKE_ROOT` fixed **where** we read, after the shared-input defect. Nothing asserted **what** we
read: the only check was a fail-loud "no bars", which fires for an absent symbol and stays silent
for a wrong lake at the right path — so the acceptance gate would have agreed with itself again,
for exactly the same reason as the first time. `verify_lake_identity` pins 200 symbols /
304,625,181 bars / Merkle `5dfd667d` in tracked code and cross-checks the partition tree against
the lake's own manifest, so neither is trusted alone. Filesystem only: the equivalent SQL takes
16 s — as long as a whole forecast — and a check that has to be skipped in the interactive path is
not a check. Five negative cases prove it bites.

## 9.7 — Appendix D's pins are transcribed, and now they are checked

`appendix_repro.tex` states its values are *"imported live … none is transcribed"*. No generator
exists; they are hand-typed, and they are all currently correct. Rather than write into `paper/`,
**[NOT LANDED — RELABELLED 2026-08-20 AS PROPOSED. `tests/test_paper_pins.py` is not tracked; no drifted pin fails CI, and the table it would read lives in `paper/appendix_repro.tex`, which left the repository on 2026-08-19.]** `tests/test_paper_pins.py` reads the table and asserts every printed pin equals its live constant —
which makes the claim operationally true, since a drifted pin now fails CI. The header's wording is
the writer's to keep or soften.

## 9.8 — Reported, not edited (`paper/**` is the writer's)

- `make_fig8_10_stubs.py` documents a `--real <verdict-manifest.json>` flag and Appendix F.5
  describes the swap. The script has **no argv handling at all**. "Specified but not enforced", in
  released figure tooling.
- Figures 4, 8, 9 and 10 say "awaiting the run", and fig 8's caption says the run "has not been
  executed". Under item E's final disposition **nothing is awaited** — the honest text is "the gate
  fired; never run".
- `main_full.txt` and the `review_s3–s7.pdf` exports predate the gate firing and still carry the
  superseded "21,301,248-parameter measurement instrument" and the 0.327% gap.

## 9.9 — Verification performed for this pass

`ruff check src tests scripts docs` exit 0; `ruff format --check` exit 0. **Repository-wide
`ruff check .` is exit 1**, on two errors in `paper/figures/make_fig1_architecture.py` (E501) and
`make_fig4_legibility.py` (F401) — the writer's files, being edited in parallel, deliberately not
touched. Gate-A anchor re-run twice after the `verdict.py` change: exit 0 both times,
`results_hash sha256:3f86882a63dd06c7…` both times.

*The first attempt to compare the two anchor runs reported a difference. It was comparing whole
stdout, which includes wall-clock timings. The instrument was wrong and the work was right — the
tooling rule, again, in the same session that fixed three other instruments.*
