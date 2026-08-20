# Model card — Trikaal cell-1 units (BSQ + OHLCV-only)

**Read the second section before the first.** What these weights are is less useful to you than
what they are not, and the gap between the two is the whole result.

| | |
|---|---|
| **Artifact** | 3 units × (Stage-1 tokenizer + Stage-2 AR backbone with MTP heads) |
| **Cell** | 1 — **BSQ** quantizer, **OHLCV-only** inputs. The baseline arm. |
| **Seeds** | 0, 2, 4 |
| **Realized parameters** | **31,725,568** total per predictor (10,493,952 in the MTP heads; 21,231,616 backbone excluding them) |
| **Training draw** | **40 symbols**, **84,153,600 bars**. Not a random subsample: the prereg draws the **40 deepest** symbols, and every one of them carries the full 2,103,840-bar span — so the draw is **survivorship-selected** toward coins that listed before 2021-01 and were still trading at 2025-01. Only 42 of the lake's 200 symbols qualify. Training consumed 832,096 windows × 512 = 426,033,152 bar-positions at the pinned 26,003 steps × batch 32, i.e. **5.06 passes** over the draw. Read it back from any unit: `draw.drawn_by_symbol_stage1` in `run_manifest.json` (identical across all three seeds). |
| **Source corpus** | The **200-symbol / 304,625,181-bar** universe lake, 1-minute bars, 2021-01-01 → 2025-01-01, Merkle `sha256:5dfd667d…`. The draw above is **27.6% of it** — these units never saw the other 160 symbols. |
| **Train / holdout** | `train_frac = 0.7` (`PINNED_TRAIN_FRAC`) over the pinned window ⇒ trained on **2021-01-01 → 2023-10-20**. **2023-10-20 → 2025-01-01 is held out** — so all of 2024 is out-of-sample. |
| **Licence** | Apache-2.0, same as the code (`LICENSE`). Binance source data is **not** redistributed — the pipeline and its content hashes are. |
| **Repository** | <https://github.com/lakshayybhati/trikaal> — **public**. Every `runs_manifest/…`, `scripts/…` and `docs/…` path on this card resolves there. |
| **Release manifest** | `runs_manifest/m6_weights_release.json` — per-file SHA-256, byte counts, and the recomputed parameter arithmetic |
| **Byte verification** | `runs_manifest/m6_rescue_inventory.json` — on-box vs post-transfer SHA-256 for every file, all matching |

## ★ What these weights are NOT

**They are not the artifact of the ablation the paper is about, and they are not a
microstructure-aware tokenizer.** Trikaal's single research claim is a microstructure-aware FSQ
tokenizer, tested by a 2×2 design {BSQ, FSQ} × {OHLCV-only, +microstructure} plus a
shuffled-microstructure placebo. **That design did not run.** A pre-registered micro-legibility
gate — written 2026-07-30, thirteen days before it fired — refused the microstructure arms on real
data on 2026-08-12, and §7 v1.5 item E took effect: gate fires → stop; the primary result becomes
the **mechanism finding**. **Stage 2 was never entered for cells 2–5**, and no cell 2–5 unit produced
a scored artifact (`artifacts_produced: 0`). Cells 4 and 5 *did* run through **Stage 1** on the money
path — that run is what produced the legibility receipt this whole finding rests on
(`runs_manifest/m6_micro_legibility_stop.json`), so the numbers below are measured, not inferred.
Cell 2 has a separate Stage-1 probe that is explicitly not a unit (`runs_manifest/m6_cell2_stage1_probe.json`).
**There is no microstructure MODEL to release, and there will not be.** Stage 2 was never entered,
so no predictor exists for cells 2–5 and nothing from those arms can forecast anything.

**What does ship is the evidence.** The cell-4 and cell-5 **Stage-1** artifacts are published beside
the three cell-1 units, at `cell4_seed0/` and `cell5_seed0/`: each a tokenizer, its optimizer
state, and the run's log — six files, 139.4 MiB, with their SHA-256s in
`m6_legibility_evidence_manifest.json` and cross-checkable against
`runs_manifest/m6_micro_legibility_stop.json` in the code repository, which was committed before
the bundle existed.

**★ THEY ARE THE STAGE-1 ARTIFACTS OF AN ARM THE GATE REFUSED, and publishing them is not a claim
that the arm worked.** It did not — that refusal *is* the finding. They are here so the 97.3%
can be checked against the thing it was measured on, which otherwise existed nowhere public.
Nothing in that bundle is a predictor and nothing in it can forecast; the three cell-1 units
remain the only trained forecasters this project released.

So cell 1 is the only arm with scored artifacts, and it is the arm the claim was supposed to be
measured *against*. These weights demonstrate the **measurement vehicle** — the held-fixed backbone
— not the contribution.

**They are also not a trading model.** See "Intended use" and "Out of scope" below, where the
measured economics are stated in full: they are negative by a factor of 4.8–43×.

## What the mechanism finding is

The tokenizer keeps the microstructure that **duplicates OHLCV** and **drops the signed channels**.
Which channels survive is the finding, and they are not the ones price would predict: the four
**magnitude** dims (`trade_count`, `mean_trade_size`, `trade_size_dispersion`, `large_trade_share`)
co-vary with **volume** — which the OHLCV arm already carries as `log_volume` and `log_amount` —
and they essentially clear the gate. The two **signed** dims (`TFI`, `signed_count_imbalance`)
carry **97.3% of *cell 4's* shortfall** — one cell, one seed, one run — and they are exactly the
channels invariant 1 scopes the claim to. **Cell 4's weights are not published** (see above), so
that number is recomputed below from the receipt rather than asserted.

Measured three ways that do not share an input: a synthetic fixture (return dim reconstructs at
0.98, the correlated filler dims 0.82–0.92, an independent low-variance dim 0.001–0.014), a canary
(**1.151** nats planted in feature space → **zero** extracted; **0.900** nats planted in token
space → 94.4% extracted), and real data (40 symbols, n=150k/dim, cell 4 seed 0).
**Both canary figures are exact, and each is exact for a different reason.** The token-space half
is *measured*: `runs_manifest/m6_token_control_run_manifest.json` carries
`i_planted_full_stream = 0.9002715667652758` and `final_val_minus_H0 = −0.8496`, and
0.8496 / 0.90027 = 0.943715 — the 94.4% to the digit. The feature-space half is *derived*: the plant
is a Gaussian channel, so the information it injects is
I(s_t ; r_{t+2}) = ½·ln(1 + c²) with c = `C_SIGNAL = 3.0` (`scripts/m6_canary.py:100`), giving
½·ln 10 = **1.151292546497023**, i.e. **1.151** to four significant figures. Nothing is rounded and
nothing is estimated.

**[CORRECTED 2026-08-21, AND THE WITHDRAWN NOTE IS SHOWN RATHER THAN DELETED.** From 2026-08-20
this paragraph read *"the feature-space half is recorded only as an approximation"* and published
**~1.15**, justified by *"no receipt in `runs_manifest/` holds a more exact value"*. That
justification was **false**, and the search behind it was the wrong shape: the figure is not a
stored measurement at all, so no sweep of `runs_manifest/` could ever have found it. It is a closed
form fixed by a constant in `scripts/` — which none of the three people who reviewed the change
searched. A figure called approximate because you looked in one place is not an approximate
figure; it is an unfinished search.**]**
The cause is that an MSE reconstruction objective allocates capacity by variance **and
covariance**, which makes an independent low-variance channel the worst per-bit investment
available. A 512-bar windowed read recovers nothing beyond the per-bar read, so this is eviction
rather than smearing.

### The 97.3%, so you can recompute it

Every number here is in `runs_manifest/m6_micro_legibility_stop.json` under
`legibility_receipt.cell4_fsq_micro_seed0.per_dim`. Shortfall is `max(0, 0.90 − sign_acc)` against
the gate's pinned `min_acc = 0.90`; each dim is measured at n = 150,000 rows over 40 symbols.

| dim | channel | kind | sign_acc | shortfall vs 0.90 |
|---|---|---|---|---|
| 7 | `TFI` | **signed** | 0.8223 | 0.0777 |
| 8 | `signed_count_imbalance` | **signed** | 0.7528 | 0.1472 |
| 9 | `trade_count` | magnitude | 0.8975 | 0.0025 |
| 10 | `mean_trade_size` | magnitude | 0.9076 | — clears |
| 11 | `trade_size_dispersion` | magnitude | 0.8962 | 0.0038 |
| 12 | `large_trade_share` | magnitude | 0.9320 | — clears |

Total shortfall 0.2312; the two signed dims contribute 0.0777 + 0.1472 = 0.2249. 0.2249 / 0.2312 =
**0.9728**. The magnitude dims span **0.8962–0.9320** — note that two of the four (9 and 11) sit
*just* under 0.90, which is what "essentially clear" means here and is why the word is hedged.

**How stable is 97.3%?** It is a single seed of a single cell (cell 4, seed 0). The 18 stored calibration replicates
from the spent λ re-derivation (`runs_manifest/m6_lambda_sweep.json`, λ ∈ {5, 8, 12} × 3 seeds × 2
arms) put the same signed share between **77% and 93%** (cell 4 alone: 88–93%). That file declares
itself `NOT_COMPARABLE_TO_THE_GATE_FIRING_RUN` and it is not being compared to it — it is quoted only
as the spread of the statistic across replicates. **The direction is robust across all 18; the exact
percentage is not.**

## Verifying what you downloaded

**A partial download must not look like a clean one.** Run this from the directory holding
`m6_weights_release.json`; it hashes every file in the inference bundle, **counts what it checked
against the manifest's own expected count**, and exits non-zero on any missing or mismatched file.

```python
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
man = json.loads((root / "m6_weights_release.json").read_text())

ok = missing = bad = 0
for seed, unit in sorted(man["units"].items()):
    for name, meta in sorted(unit["files"].items()):
        if not meta["needed_for_inference"]:
            continue                      # optimizer state: not part of the inference bundle
        p = root / f"seed{seed}" / name
        if not p.is_file():
            print(f"MISSING   seed{seed}/{name}"); missing += 1; continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got == meta["sha256"]:
            print(f"MATCH     seed{seed}/{name}"); ok += 1
        else:
            print(f"MISMATCH  seed{seed}/{name}"); bad += 1

expected = sum(1 for u in man["units"].values()
               for f in u["files"].values() if f["needed_for_inference"])
print(f"\n{ok}/{expected} inference files verified, {missing} missing, {bad} mismatched")
sys.exit(0 if (ok == expected and not missing and not bad) else 1)
```

It verifies the **9 inference files** (3 units × `tokenizer.pt`, `predictor.pt`,
`run_manifest.json`) and deliberately skips the 6 optimizer-state files, which are resume-only. A
verifier that prints per-file `MATCH` lines without a count cannot tell a complete download from a
partial one — that exact gap produced a false finding during the audit of this release, which then
had to be retracted.

## Loading them

`TrikaalAR()`'s defaults are the **FSQ** vocabulary (`v_c=891`, `v_f=1225`, `max_len=512`) — the arm
that was never trained. These checkpoints are **BSQ** (`v_c=v_f=1024`, `max_len=572`), so the
obvious load raises a shape error. Build from the checkpoint's own config instead:

```python
import torch
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE

# These checkpoints stamp torch's version as a TorchVersion OBJECT, which weights_only=True
# refuses by default. Allow that one global and the safe loader works — do NOT reach for
# weights_only=False, which executes arbitrary pickle from a 127 MB download.
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

ckpt = torch.load("seed0/predictor.pt", map_location="cpu", weights_only=True)
model = TrikaalAR(**ckpt["config"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

tok = torch.load("seed0/tokenizer.pt", map_location="cpu", weights_only=True)
tokenizer = TokenizerAE(**tok["config"])
tokenizer.load_state_dict(tok["state_dict"])
tokenizer.eval()
```

Summing `numel()` over the predictor's `state_dict` reproduces **31,725,568**, the
`n_params_realized_total` in `runs_manifest/m6_weights_release.json`. That is the check to run
first: it proves you loaded the artifact the manifest describes.

**The `add_safe_globals` line is needed only for these already-published files.** The stamp is a
plain string from now on (`provenance.py`), so future checkpoints load under `weights_only=True`
with nothing registered. The line is kept here rather than the files being re-uploaded, because
re-pickling a published artifact changes its SHA-256 and every hash on this page with it.

## Running them

The scripts take `--units`, pointing at wherever you downloaded the bundle. Units are found by
**content** — any folder holding `run_manifest.json`, `predictor.pt` and `tokenizer.pt` — so the
layout of your download does not matter, and an incomplete bundle is refused rather than
half-loaded:

```bash
uv run python scripts/m6_csv_dashboard.py --units /path/to/download
```

Without `--units` the scripts look under the in-repo `runs_cloud/` paths of the original run,
which a fresh clone does not have.

## Intended use

Research replication and inspection: reproducing the reported reconstruction and forecast
diagnostics, and re-running the demo. That is the whole of it.

## Out of scope — explicitly

- **Trading, in any form.** The measured break-even is the mean gross return per active period:
  0.00232% / 0.00538% / 0.02082% across the three units, against a realistic ~0.10% round trip —
  **4.8× to 43× short**, and the shortfall survives sampling error. No tokenizer change closes
  that gap. The reported net information ratios (−28 / −67 / −146) are reproduced to within 8–17%
  by a **zero-skill** model that pays the same costs, i.e. they are cost drag and not evidence of
  negative skill.
- **Any horizon or market other than the one trained**: 1-minute crypto perpetuals, forecast
  horizon h=15 minutes. Equities, other frequencies and other venues are firewalled out of v1.
- **Treating the three seeds as an ensemble.** They disagree about direction. P(μ̂>0) is
  0.3500 / 0.3250 / 0.8625; seeds 0 and 4 take opposite directions more often than they agree
  (`runs_manifest/m6_demo_seed_sign_disagreement.json`). Averaging them hides the finding.

## Known limitations, stated as measured

- **Three estimators of the same quantity disagree on majority sign.** Expectation gives
  +0.0113 / −0.0036 / +0.0173 (2 of 3 positive), mc_mean@32 gives +0.0403 / −0.0004 / −0.0195
  (1 of 3), the trajectory median gives +0.0353 / −0.0122 / −0.0223 (1 of 3). Seed 4 flips
  outright. The directional signal is weak enough that the choice of estimator changes its sign
  (`runs_manifest/m6_estimator_sign_disagreement.json`).
- **μ̂ is 25–446× over-dispersed** relative to the returns it forecasts, and the execution filter's
  threshold is absolute rather than relative — so decision activity is not comparable across units.
- **The return dimension is the worst-reconstructed channel** (6th or 7th of 7 depending on basis),
  and the basis matters: one 1INCHUSDT window gives an artifact fraction of 0.50 where 8×512
  BTCUSDT bars give 0.10.
- **Training is not bit-reproducible** unless the deterministic-attention fallback is enabled, and
  deterministic attention is *necessary but not sufficient* — reduction order elsewhere in the
  backward pass is unconstrained. Every run records its mode. The data pipeline, frozen statistics
  and prediction replay **are** bit-exact.
- **The BSQ baseline is our own and is not externally validated.** The pre-registered external
  check against published Kronos-small was found unexecutable and dropped as binding (two
  published RankICs 2.4× apart, both on Shanghai 15-minute equity bars, and running the weights
  would need model code this project forbids). The compensating disclosure is carried in every
  verdict manifest: *we cannot exclude that our BSQ baseline is weaker than a reference BSQ
  implementation, which would inflate the FSQ-vs-BSQ comparison.* **No Kronos weights or code are
  part of Trikaal.**

## Provenance and reproduction

Every unit carries a `run_manifest.json` stamping its environment. Measured against the 16 keys in
`PROVENANCE_IDENTITY_KEYS` (`src/trikaal/utils/provenance.py`), these three units carry **12**:
GPU name, CUDA build, driver version, torch / numpy / python versions, both step budgets, attention
mode, and the three determinism flags.

**Four identity keys were never captured on these runs, and we are not back-filling them:**
`image`, `git_commit`, `lockfile_sha256`, `platform_abi`. `driver_version` holds the placeholder
string `"unavailable: AttributeError"` rather than a version — the lookup failed on the box and the
code records a placeholder instead of a plausible default, by design. Stamping any of this in
after the fact would be manufacturing a receipt for a run that never took the measurement, so the
gap is stated instead. The full 16-key surface is enforced for future runs, where a mismatch
refuses rather than passing quietly.

The demo asserts checkpoint identity on load by recomputing both content hashes from the live modules,
and refuses to serve if either differs. `scripts/m6_demo_acceptance.py` re-proves that the demo's
single-decision assembly lands on the same float64 bits as the production whole-symbol path, and
is itself proven capable of failing by a negative control that injects a one-bar off-by-one.

## Citation

The paper is the citable object; these weights are its evidence. Cite the paper, and cite the
lake Merkle root `sha256:5dfd667d…` and the unit hashes in `m6_weights_release.json` for the
artifacts.
