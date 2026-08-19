# Invariant 7 — a decision for Lakshay (drafted 2026-07-30, §7 v1.4.7)

**This is not a builder decision and not a supervisor decision.** docs/ENGINEERING.md invariant 7 embeds a
false premise **in its own text**, so amending it needs the person who owns that file. Two candidate
amendments are drafted below with the measured facts attached. **Neither is recommended yet** — the
CUDA feasibility measurement is PENDING and it is the input that separates them.

---

## What was found

Invariant 7 currently reads (docs/ENGINEERING.md, emphasis added):

> **GPU training is bit-exact only under the deterministic-attention fallback** (FlashAttention-2 is
> otherwise non-deterministic) and every run records its mode.

That sentence makes deterministic attention **sufficient**. It is only **necessary**. Bit-exact CUDA
training additionally requires `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG`,
and `cudnn.deterministic` — none of which are set, because **every** M6 path calls
`set_determinism(..., deterministic_algorithms=False)`:

| path | call site | role |
|---|---|---|
| `orchestrator.py:159` | `set_determinism(seed, deterministic_algorithms=False)` | **the production 15-day run path** |
| `m6_canary.py:482,557` | `set_determinism(args.seed, deterministic_algorithms=False)` | the fixture path |

`False` overrides a default of `True` on a function whose own docstring calls it *"required for the
G2 determinism gate and any published run"*.

### The measured consequences

- **49 of 49** determinism records claim `bit_exact_claim: true` beside `deterministic_algorithms:
  false` on `device: cuda` — a 100% rate, not a sample. (46 run records over **16 distinct training
  runs**; the record count over-counts runs ~3× because each run appears in a per-run manifest and
  in a rollup duplicated across `runs_cloud/` and `runs_manifest/`. Receipt:
  `runs_manifest/m6_bit_exact_claim_correction.json`.)
- **The observable effect is not float jitter.** Two runs of a byte-identical recipe at the **same
  seed** produced cell4 models with `frac_negative` 0.5662 vs 0.99883 and IR +15.008 vs −3.4258.
  That is basin-hopping.
- **Forced determinism completes on CPU.** The `$0` feasibility probe runs the real Stage-1 +
  Stage-2 objectives under `use_deterministic_algorithms(True, warn_only=False)` and it completes —
  no op in the M6 training path lacks a deterministic kernel there. So on CPU this is purely a
  throughput question. **CUDA is not yet measured.** Receipt:
  `runs_manifest/m6_determinism_feasibility_probe.json`.

### Scope — neither broader nor narrower than this

- **FALSE:** the GPU-training bit-exactness clause, for the enumerated runs.
- **INTACT:** the data-pipeline / frozen-stats / prediction-replay clause (independently evidenced:
  M4a `dataset_hash` reproduction, token-stream content hashing, fixed-chunk rollout replay KATs).
- **INTACT:** the Gate-A anchor. It anchors the **CPU eval-harness replay** on a frozen M3
  checkpoint, is genuinely bit-identical (`results_hash 3f86882a`, re-proven with a byte-identical
  run manifest), and does not depend on GPU training determinism.
- **No prior result is invalidated.** No number changes. What changes is that a run stamping
  `bit_exact_claim: true` may not be **quoted** as a bit-exact GPU training run.

---

## Amendment A — make the claim true

Force deterministic algorithms in the training path, so the invariant's existing promise holds.

- **Text change:** none required. Invariant 7 keeps its current wording and becomes true.
- **Code change:** `deterministic_algorithms=True` at `orchestrator.py:159` (and the canary, for
  consistency), plus `CUBLAS_WORKSPACE_CONFIG` set before the first CUDA matmul.
- **Cost:** **PENDING.** Deterministic CUDA kernels are slower; the 2–3 GPU-day / \$20–30 budget was
  measured at 47.2k bars/s **without** them. The delta is unmeasured, so the new budget is unknown.
- **Risk:** if any CUDA op in the path lacks a deterministic kernel, A degrades into "warn_only=True
  plus an explicit allowlist of ops that stay non-deterministic" — which weakens the claim anyway,
  **independently of cost**. The CPU probe passed; the CUDA probe is what settles this.
- **Buys:** a genuine reproducibility deliverable, and it removes the *same-seed* component of the
  divergence.
- **Does NOT buy:** across-seed variance. Different seeds still land in different basins. The
  statistical-power problem below survives amendment A untouched.

## Amendment B — scope the deliverable honestly

Amend the invariant to state that GPU training is **not** bit-reproducible, scope the bit-exactness
deliverable to pipeline / frozen-stats / prediction-replay (where it is already true and evidenced),
and report **across-seed spread** as the reproducibility statistic instead.

- **Text change:** replace the GPU-training clause with something like: *"GPU training is NOT
  bit-reproducible: CUDA reduction order is unconstrained and deterministic kernels are not forced
  (a measured throughput decision). The bit-exactness deliverable is scoped to the data pipeline,
  frozen stats, and prediction replay, each independently evidenced. Training reproducibility is
  reported as across-seed spread of the headline statistic, not as bit-identity."*
- **Code change:** none required beyond what already landed (`bit_exact_preconditions_met` +
  caveat).
- **Cost:** \$0.
- **Buys:** the claim matches reality today, and it forces the paper to report seed spread — which is
  the honest statistic and the one a referee should want.
- **Costs:** gives up a headline reproducibility claim that Kronos-style work often asserts. Being
  the artifact that says "our GPU training is not bit-reproducible, here is the seed spread instead"
  is defensible and arguably stronger, but it is a positioning choice.

---

## What is true under either amendment

**The statistical-power risk is independent of this decision.** The pre-registered MDE was derived
from **scoring** noise only (a paired moving-block bootstrap over time) and contains **no
training-variance term**. Same-seed divergence is a *lower bound* on across-seed variance, so the
tabled MDE understates the true detection threshold and the 3-seed design may be underpowered
against its own effect size. Amendment A removes the same-seed component and **not** the across-seed
component.

Mitigation already landed, at \$0 and without pre-committing to either amendment: a **HALT-only power
guard** (`verdict.power_guard`) persists per-cell across-seed IR spread and refuses to *report* a
SURVIVES whose claimed ΔIR is smaller than the seed-to-seed wobble of its own inputs. Like the
degeneracy guard it can never flip SURVIVES↔NULL — it only declines to emit an unsupported verdict.
That measures the power question **on real data at zero cost**, which beats arguing about it now.

---

## Recommendation

**None yet, deliberately.** The CUDA feasibility probe is the discriminating input and it is staged,
not run:

```bash
PYTHONPATH=src python3 scripts/m6_determinism_probe.py --device cuda --steps 50
```

~\$1–2 of box time, < 2 minutes for the probe; a quotable forced-vs-unforced throughput pair needs
~25 minutes total on one box. If it completes, the choice is a straight cost question and A becomes
viable. If it raises, A is compromised on its own terms and B is the honest option regardless of
budget.

**No spend is authorized here and none has been taken.**
