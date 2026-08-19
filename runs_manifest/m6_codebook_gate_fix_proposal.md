# PROPOSAL FOR REVIEW — ONE COMMIT, THREE CHANGES, ONE §7 TAG

**NOT A COMMIT. HEAD `95fc017` unmoved, no repo file modified.**
**Change A is a gate RELAXATION and lands only after §7 v1.6.30 from the writer, as v1.6.29 did.
Changes B and C move no gate value and mint no tag — described in the commit message and
BUILD_RECORD, tagged: no.**

| # | change | class | tag |
|---|---|---|---|
| **B** | pre-flight identity gate compares the 16 keys to a pinned reference | **bug fix** (a gate that never compared) | none |
| **A** | codebook threshold scoped to FSQ; required-and-reported for BSQ | **amendment** (relaxes a gate) | **§7 v1.6.30** |
| **C** | eval progress instrumentation (finding 12), stdout only | **addition** | none |

Ordered by consequence: **B first.** A mismatched box costs the whole matrix; the missing
progress line costs only readability.

---

## CHANGE B — THE PRE-FLIGHT IDENTITY GATE DOES NOT COMPARE ANYTHING

### The defect

`provenance.py:154-176`, `identity_placeholder_failures`, is the only pre-spend identity check.
It rejects **placeholder values** on CUDA:

```python
bad = [k for k in PROVENANCE_IDENTITY_KEYS
       if str(prov.get(k)) in (UNAVAILABLE, "unknown", "None", "")]
```

Its own docstring says so: *"A placeholder is not a refusal across shards."* **A box with driver
`580.x`, a different glibc, or a different 4090 SKU carries all-real values, passes pre-flight,
computes its unit, and is caught only at assembly by `provenance_failures`** — after every unit
has been paid for.

Survivable at three boxes in one driver-filtered batch, which I hand-verified. **Not survivable at
25 units across many boxes over ~5 days**, where each new box is a silent chance to poison the
matrix and the failure surfaces at the end.

### The fix

```diff
+REFERENCE = Path("runs_manifest/m6_identity_reference.json")   # COMMITTED, written at unit 1
+
+def identity_reference_failures(prov: dict, reference: dict) -> list[str]:
+    """§7-untagged bug fix — REFUSE AT SECOND ZERO, not at assembly.
+
+    Compares EXACTLY the keys `verdict.provenance_failures` compares — no more, no fewer, or the
+    pre-flight gate and the assembly gate can disagree and the run passes one while failing the
+    other. `platform` is RECORDED, NEVER COMPARED (§7 v1.6.29): its kernel component differs
+    legitimately across hosts (r0/r1 5.15.0-52 vs r2 6.8.0-101) and comparing it would have
+    refused a healthy pool. `platform_abi` is the identity key.
+
+    The reference is a COMMITTED FILE written at unit 1 — never "the first box I happen to talk
+    to", which would make the instrument whatever machine answered first."""
+    return [
+        f"identity key {k!r} = {prov.get(k)!r} but the pinned reference is {reference.get(k)!r} "
+        "— this box is a different instrument and its unit could never join the matrix"
+        for k in PROVENANCE_IDENTITY_KEYS
+        if prov.get(k) != reference.get(k)
+    ]
```

wired into `m6_money_run.py` immediately after the existing placeholder check (`:287-290`), before
any compute, and **skipped only when the reference file does not yet exist** — i.e. at unit 1,
which is what writes it.

### Discrimination proof — required, per the supervisor

1. reference driver `590.48.01`, box reports `580.159.03` → **REFUSES**, naming the key.
2. reference kernel `5.15.0-52`, box reports `6.8.0-101`, **`platform_abi` identical** → **PASSES**
   (this is r2's real situation; a gate that refused here would have killed a healthy pool).
3. all 16 identical → passes.
4. `gpu_name` differs → refuses.
5. **Key-set equality test:** `set(compared_keys) == set(PROVENANCE_IDENTITY_KEYS)`, asserted
   directly, so the pre-flight and assembly gates can never drift apart.
6. **Negative control:** make the comparison `return []` and tests 1 and 4 must FAIL.

---

## CHANGE A — CODEBOOK THRESHOLD SCOPED TO FSQ (§7 v1.6.30)

### The measurement that picked this shape

`runs_manifest/m6_cell2_stage1_probe.json` — Stage-1 only, 40 symbols, 26,003 steps, seed 0.

| | CELL 1 · BSQ + ohlcv | CELL 2 · FSQ + ohlcv |
|---|---|---|
| tokens per leg | 84,153,600 | **84,153,600** |
| symbols / seed / steps | 40 / 0 / 26,003 | **40 / 0 / 26,003** |
| coarse utilization | 1.0000 (1024/1024) | 1.0000 (891/891) |
| **fine utilization** | **0.8672 (888/1024)** | **0.9951 (1219/1225)** |
| fine dead codes | **136 — 13.3%** | **6 — 0.5%** |
| effective bits/token | 17.0935 | 19.1844 |
| **vs the pinned 0.95** | **FAIL** | **PASS** |

Identical basis; the quantizer is the only difference. It reproduces design `:1154`'s own
prediction — *"the dead-code rate that plagues BSQ … expected to be 0 for FSQ by construction"*.

**The amendment does not depend on this outcome.** Had BSQ returned 0.96, restoring the spec's
scope would still have been correct: `:1859` sets the target for **FSQ** codes, `:986` calls usage
*"a health diagnostic, not a failure mode"*, `:1154` says the BSQ dead-code rate *"is reported"*,
and a 37-hit sweep of design + prereg + m6_design found **zero BSQ-scoped minimums**.

### The diff

```diff
     cb = doc.get("codebook")
     if not isinstance(cb, dict) or not cb:
         bad.append(f"{name}: codebook diagnostic is missing or empty (§7 v1.6.22; spec :1859)")
     else:
+        # §7 v1.6.30 — THRESHOLD IS FSQ-SCOPED; FOR BSQ THE NUMBER IS REPORTED, NOT GATED.
+        # FAIL CLOSED: THE QUANTIZER COMES FROM THE PINNED CELL REGISTRY, NEVER FROM THE
+        # ARTIFACT'S SELF-DECLARATION. An artifact must not be able to relabel itself out of
+        # the threshold, and a self-declared field is exactly that lever.
+        spec = _CELL_BY_ID.get(int(doc.get("cell_id", -1)))
+        if spec is None:
+            bad.append(f"{name}: cell_id {doc.get('cell_id')!r} is not in the pinned registry")
+        elif doc.get("quantizer") != spec.quantizer:
+            bad.append(
+                f"{name}: quantizer {doc.get('quantizer')!r} disagrees with the registry "
+                f"({spec.quantizer!r}) — an artifact may not relabel its own quantizer"
+            )
+        thresholded = spec is not None and spec.quantizer == "fsq"
         for sub in ("coarse", "fine"):
             leg = cb.get(sub)
             if not isinstance(leg, dict):
                 bad.append(f"{name}: codebook.{sub} missing (§7 v1.6.22)")
                 continue
             u = leg.get("utilization")
+            # REQUIRED AND FINITE FOR EVERY CELL — v1.6.22's requirement is untouched.
             if not isinstance(u, (int, float)) or not np.isfinite(float(u)):
                 bad.append(f"{name}: codebook.{sub}.utilization missing/non-finite")
-            elif float(u) < PINNED_CODEBOOK_MIN_UTILIZATION:
+            elif thresholded and float(u) < PINNED_CODEBOOK_MIN_UTILIZATION:
                 bad.append(...)
```

Plus `codebook_report(evals)` in the verdict manifest carrying every cell's utilization, n_used,
vocab, entropy, perplexity, **dead-code count**, quantizer and `thresholded` flag — because
":1154 reported for the BSQ arm" is not satisfied by a number sitting in 25 files nobody reads.
`load_verdict_manifest` refuses a manifest without it, as it does the BSQ disclosure.

### ALSO IN THE MANIFEST — BOTH DISPERSION STATISTICS

> *"Refusing on the wrong statistic is survivable if the right one is printed beside it."*

`power_guard` HALTs on the **within-cell IR level range**. `_per_seed_delta(4,5)` (`verdict.py:942`)
already computes the **paired per-seed contrast** for `pb45`. Emit both:

```diff
+        "dispersion": {
+            "level_range_by_cell": {c: per_cell[str(c)]["ir_range_across_seeds"] for c in (1,2,3,4,5)},
+            "paired_delta_by_seed": {"4-5": _per_seed_delta(4,5).tolist(),
+                                     "4-2": _per_seed_delta(4,2).tolist()},
+            "paired_delta_range":   {"4-5": float(np.ptp(_per_seed_delta(4,5))), ...},
+            "why_both": "power_guard HALTs on the LEVEL range. Whether levels or paired "
+                        "differences are the right basis is an OPEN DESIGN QUESTION (§7 v1.6.30 "
+                        "note): if the seed effect is common-mode, the paired dIR can be stable "
+                        "while both levels swing, and the guard halts on a well-measured effect. "
+                        "NOT RESOLVED — two seeds of one cell and zero of the headline pair "
+                        "cannot distinguish them. Both printed so the question is answerable "
+                        "from the manifest if the run HALTs.",
+        },
```

### Tests

| # | case | expected |
|---|---|---|
| 1 | FSQ cell, fine 0.9400 | **REJECTED** — the threshold still bites where the spec puts it |
| 2 | FSQ cell, fine 0.9600 | clean |
| 3 | BSQ cell, fine **0.8672** (real measured value) | **clean** |
| 4 | BSQ cell, utilization absent | **REJECTED** — v1.6.22 survives |
| 5 | BSQ cell, utilization NaN | **REJECTED** |
| 6 | `cell_id`=1 but `quantizer:"fsq"` | **REJECTED** — cannot relabel |
| 7 | `cell_id`=99 | **REJECTED** — unknown cell fails closed |
| 8 | manifest without `codebook_report` / `dispersion` | **REJECTED** on read-back |

**Negative controls** (`scripts/m6_reaudit_mutations.py`): restore the pre-fix `elif` → **test 3
must FAIL**. Force `thresholded = True` → **test 3 must FAIL**. Force `thresholded = False` →
**test 1 must FAIL** — the mutation that catches a "fix" which quietly disables the gate everywhere.

---

## CHANGE C — EVAL PROGRESS, STDOUT ONLY, EXCEPTION-PROOF

Per ruling: **the W&B leg is dropped entirely.** stdout is what makes 14 hours readable, and I
could not establish whether the W&B run is still open at eval time — not a thing to discover at
hour 12.

Design as specified in `m6_finding12_progress_instrumentation.md` (per-symbol, 160 lines/unit,
one every ~5.5 min, `flush=True` because the shard log is a redirected file), with one change:

```diff
+            if progress is not None:
+                # ★ AN OBSERVER MUST NEVER BE ABLE TO KILL THE MEASUREMENT IT WATCHES.
+                # Swallowing is CORRECT HERE AND WRONG IN A GATE, and the distinction is the
+                # whole point: a GATE that swallows its own failure is the project's oldest
+                # defect ("a check that cannot fail is not a check"); an OBSERVER that throws
+                # kills a 14.65-hour eval at hour 12 to protect a progress line. Do not
+                # "fix" this into a bare call — that turns a readability feature into a
+                # matrix-killer. This block computes NOTHING the artifact depends on.
+                try:
+                    progress({...})
+                except Exception:  # noqa: BLE001 — deliberate, see above
+                    pass
```

**P1 (local `--dry-run` at the new HEAD, $0) and P2 (one box, reduced scale) are MANDATORY**, not
optional: C is the only one of the three changes that executes inside the money path, and the
execution rule says code whose first run is on rented hardware is untested code.

---

## PRE-DECLARED READING — THE DETERMINISM TEST, DECLARED BEFORE THE NUMBER EXISTS

Re-running seeds 0 and 2 under the new HEAD, with the pre-fix artifacts banked, is **the first
same-seed reproducibility test under FORCED determinism on DISTINCT physical hardware.** The
artifacts claim `deterministic_algorithms: true`, `cudnn_deterministic: true`,
`cublas_workspace_config ":4096:8"`, `bit_exact_claim: true`; v1.4.7's divergence evidence was
gathered with determinism **OFF**, and no seed has ever been run twice under forced determinism on
two different physical 4090s. **r0's machine is destroyed, so the re-run necessarily lands on a
different box — which is what makes it the test.**

**THE QUANTITY, FIXED IN ADVANCE:** headline IR at `PRIMARY_H=15`, computed by
`metrics.information_ratio` from `headline_series`, compared against the banked pre-fix value
**IR(cell1, seed0) = −67.0160** (recomputed from the series; the log line reads −67.02).

**THE SCALE, MEASURED NOT CHOSEN** — moving-block bootstrap on that same series under the pinned
recipe (B=10,000, seed 20260704, L=⌈√35064⌉=188, 187 blocks):

    SE_scoring   = 2.0378          95% interval [-71.0020, -63.1127]      = 3.04% of |IR|

**THE READING:**

1. **`headline_series` byte-identical** → invariant 7's GPU-training clause holds in the strict
   bit-exact sense it actually claims. This is the real claim and it is binary.
2. **Not byte-identical but |ΔIR| ≤ SE_scoring = 2.0378** → the strict claim is FALSE; the runs
   are statistically indistinguishable and nothing downstream is threatened.
3. **|ΔIR| > 2.0378** → the clause is false even under forced determinism **AND the across-seed
   power problem is worse than the 79.47 range suggests, because part of that range is not seed
   variance at all.**

**The re-run box must carry `driver_version 590.48.01` + `RTX 4090` + `platform_abi
x86_64-with-glibc2.35`**, or the comparison is confounded by the instrument rather than testing it
— which is exactly what Change B now enforces before any compute.

★ **SCALE CHECK, RECORDED NOW:** the measured across-seed IR range on cell 1 (seed 0 vs seed 4) is
**79.47 = 39.0 × SE_scoring**. The seed-to-seed wobble is thirty-nine times the scoring noise the
tabled MDE was built from. That is the v1.4.7 power risk with a number on it.

---

## WHAT NONE OF THIS DOES

- Does not change `PINNED_CODEBOOK_MIN_UTILIZATION` (0.95, still binding on FSQ).
- Does not touch `power_guard`, any §3 clause, the DSR basis, κ, or any pin.
- Does not weaken v1.6.22's requirement that the diagnostic exist and be finite.
- Does not re-score anything: under Change A, cell1_seed0 and cell1_seed4 are conforming **as they
  stand**. They are re-run only because `git_commit` is identity key #2 — **for a string, not for
  a number.** Say it that way in the record; overstating what the $24 buys would be its own defect.
