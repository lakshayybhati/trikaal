# The staged CUDA probe — results (§7 v1.6)

**Run 2026-07-31. RTX 4090, instance 46391088, DESTROYED after 9/9 sha256 verification.**
Receipt: `runs_manifest/m6_cuda_probe.json` · teardown: `runs_manifest/m6_teardown_verification.json`
Cell manifest: `runs_manifest/m6_cuda_probe_cell_manifest.json` · raw set: `runs_cloud/cuda_probe/`

---

## JOB 1 — the determinism penalty, and the double-counting question SETTLED

| arm | requested | **actual `deterministic_algorithms`** | steps/s | **bars/s** |
|---|---|---|---|---|
| unforced (control) | False | **False** | 3.419563 | **56,026.1** |
| forced | True | **True** | 2.969033 | **48,644.6** |

**Penalty = 13.175 %** — forced/unforced ratio **0.8682**, i.e. **1.152×** slower.
`probe_valid: true`, `postures_actually_differed: true`, both arms `stable`, each in its own process.

The `actual_deterministic_algorithms` field is read by `determinism_record()` from
`torch.are_deterministic_algorithms_enabled()` — the **live torch flag**, not our intent. The arm is
therefore self-identifying and this question is closed by the artifact rather than by argument.

### The double-counting catch is CONFIRMED

| quantity | bars/s | gap vs historical |
|---|---|---|
| historical bench (**posture never recorded**) | 47,218.5 | — |
| **measured FORCED** | **48,644.6** | **+3.02 %** |
| measured UNFORCED | 56,026.1 | +18.65 % |

The historical 47.2k lands **within 3 % of the forced arm** and **19 % below the unforced arm**. It
was already a forced-determinism measurement, exactly as suspected — `bench_mode` called
`set_determinism(0)`, whose default is `deterministic_algorithms=True`, while production
(`orchestrator.py:159`) runs unforced.

**Consequences.** Applying a penalty on top of 47.2k *inflates* rather than corrects. The measured
penalty is **1.152×**, and the illustrative **1.300×** was **1.13× too large** — so the
"$43–65 with forced determinism" figure was overstated on two independent counts: a penalty applied
to a number that already contained it, and a penalty larger than the real one.

---

## JOB 2 — end-to-end throughput, per stage

| stage | steps | wall (s) | steps/s | bars/s |
|---|---|---|---|---|
| Stage-1 tokenizer | 120 | 7.15 | 16.7859 | 275,020.2 |
| Stage-2 AR | 120 | 36.40 | 3.2967 | **54,013.3** |
| **pooled (the JOB-2 headline)** | 240 | **43.55** | — | **90,293.3** |

`bars_per_s_into_gpu = 90293.3` · `train_wall_s = 43.55` · `measured: true` · `is_costable: true` ·
basis `end_to_end_training`. Cell `cell4_fsq_micro`, seed 0, **5 symbols**, 3,732,698 bars,
21,301,248-param gate enforced, canonical geometry (batch 32 × seq 512).

**Fixed overheads, reported separately:** lake load **4.0 s**, window build **4.7 s**,
tokenize + checkpoint **6.7 s** — 15.4 s per (cell, seed).

### Two honesty notes on these numbers

1. **The pooled 90,293 must not be quoted as "the" throughput.** It averages two stages whose rates
   differ 5×. It is the right aggregate *only* because the run does equal step counts per stage; for
   any other mix it is wrong.
2. **Finding 0's "compute-only is an upper bound" was directionally right but overstated in
   magnitude for Stage-2.** End-to-end Stage-2 (54,013) sits **3.6 % below** the compute-only bench
   (56,026), not far below: sampling is from in-memory arrays, so the loader adds little per step.
   The framing is corrected here rather than left to stand.

---

## Provenance

RTX 4090 (capability 8.9, 24,564 MiB) · driver **580.159.03** · CUDA build **13.0** · cuDNN
**92000** · torch **2.12.1+cu130** · numpy **2.4.6** · Python 3.11.9 · 128 cores / 251 GB RAM ·
env built by `uv sync --locked` from the committed lockfile, so it is the **exact pinned
environment**. `driver_version` records `"unavailable: AttributeError"` — see below; that is the
guard working, not a failure.

---

## THE MOST CONSEQUENTIAL FINDING — a production-blocking crash, found only because we ran on CUDA

The **first** CUDA attempt failed **both JOB-1 arms and JOB-2 identically**:

```
AttributeError: module 'torch._C' has no attribute '_cuda_getDriverVersion'
```

Per the control-arm rule the probe emitted **PROBE INVALID**, not a determinism verdict.

**Cause.** In `attention_mode._cuda_provenance`, the probe table listed
`("driver_version", torch._C._cuda_getDriverVersion)` — passing the bound function **resolves the
attribute while the tuple is being built**, i.e. *before* the `try/except` that the comment claimed
guarded it. The pinned torch 2.12.1 does not define that symbol.

**Why it blocks the run.** `orchestrator.run_cell` calls `determinism_record()` **before training
starts**. On a CUDA box with the pinned torch, **the real M6 run would have died at cell 1, seed 0**,
before a single training step — 5 cells × 5 seeds, all of them.

**Why nothing local could catch it.** Every CPU/MPS path returns at the `torch.cuda.is_available()`
check above and never reaches the tuple. 333 green tests, on hardware that cannot execute the line.

**The fix** defers the lookup into the lambda so it falls inside the guard. The receipt now records
`driver_version: "unavailable: AttributeError"` — proof the guard fires instead of raising. *A guard
that cannot fire is not a guard* — the same family as the pipefail rider, the control-arm rule and
the fixture-discrimination rule.

---

## EXTRAPOLATION — labelled, with its assumptions

**`EXTRAPOLATED_total_hours` 50.51 h · `EXTRAPOLATED_usd_at_observed_dph` $16.45 ·
`EXTRAPOLATED_usd_if_determinism_forced` $18.94** at $0.3256/hr.

Every field name carries `EXTRAPOLATED`; the block refuses to emit anything when `is_costable()` is
false. Six assumptions travel with it, of which three are load-bearing:

- **EVAL IS EXCLUDED ENTIRELY.** No scoring pass was measured. This is a **training-only floor**,
  not a run cost.
- **The lake-load term is understated** — measured on 5 symbols; the real run loads 40.
- The determinism variant applies JOB 1's **compute-only** ratio to an **end-to-end** rate; the two
  need not share a penalty.

**A seventh assumption, found while writing this up and disclosed here:** the extrapolation's
**20,000 steps/stage** is **not a pre-registered pin**. I could not find a step budget pinned
anywhere in the prereg. It matches prior practice (the v6 Stage-1 recipe used 20,000) but the
dominant input to this projection is itself unbacked — which is precisely the Finding-0 defect,
applied to my own arithmetic. It is labelled, not quietly used.

---

## Process

- **$0 gate passed before any spend**, and caught a real defect: `--warmup` was accepted by the
  probe and never forwarded, so both arms returned NaN. Fixed, plus a guard that refuses a JOB-1
  verdict unless both arms report `throughput.measured == true`.
- **Pre-teardown gate: PASS 9/9**, fail-closed, proven to fail closed beforehand (corrupt → exit 1,
  missing → exit 1). It **did fail 8/9 on the first attempt** — `probe.log` was digested by the
  script while that same script was still appending to it, so the recorded digest was stale two
  lines later. No corruption: the box's live digest matched local byte-for-byte (1,968 bytes). The
  box script now excludes the open log, and the digests were regenerated after every writer exited.
- **Box DESTROYED** and confirmed absent. `destroy instance` without `-y` **aborted at a
  confirmation prompt while reporting exit 0** — caught only by re-listing instances afterwards.
- **Gate-A re-proven** (`src/` changed): exit-0 **×2**, manifest byte-identical to the pre-change
  baseline both times, `results_hash 3f86882a`, causal sweep 420/420. Suite **333 green**, ruff clean.

**ACTUAL SPEND: $0.80** GPU-time (2.457 h × $0.3256/hr), against the ~$1–2 authorized. Dominated by
transfer wait, not compute — the probe itself ran in under two minutes.
