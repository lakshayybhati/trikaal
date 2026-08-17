# Model card — Trikaal cell-1 units (BSQ + OHLCV-only)

**Read the second section before the first.** What these weights are is less useful to you than
what they are not, and the gap between the two is the whole result.

| | |
|---|---|
| **Artifact** | 3 units × (Stage-1 tokenizer + Stage-2 AR backbone with MTP heads) |
| **Cell** | 1 — **BSQ** quantizer, **OHLCV-only** inputs. The baseline arm. |
| **Seeds** | 0, 2, 4 |
| **Realized parameters** | **31,725,568** total per predictor (10,493,952 in the MTP heads; 21,231,616 backbone excluding them) |
| **Training data** | 200 USDT-perpetual symbols, 1-minute bars, 2021-01-01 → 2025-01-01, 304,625,181 bars, lake Merkle `sha256:5dfd667d…` |
| **Licence** | Apache-2.0, same as the code (`LICENSE`). Binance source data is **not** redistributed — the pipeline and its content hashes are. |
| **Release manifest** | `runs_manifest/m6_weights_release.json` — per-file SHA-256, byte counts, and the recomputed parameter arithmetic |
| **Byte verification** | `runs_manifest/m6_rescue_inventory.json` — on-box vs post-transfer SHA-256 for every file, all matching |

## ★ What these weights are NOT

**They are not the artifact of the ablation the paper is about, and they are not a
microstructure-aware tokenizer.** Trikaal's single research claim is a microstructure-aware FSQ
tokenizer, tested by a 2×2 design {BSQ, FSQ} × {OHLCV-only, +microstructure} plus a
shuffled-microstructure placebo. **That design did not run.** A pre-registered micro-legibility
gate — written 2026-07-30, thirteen days before it fired — refused the microstructure arms on real
data on 2026-08-12, and §7 v1.5 item E took effect: gate fires → stop; the primary result becomes
the **mechanism finding**. Cells 2, 3, 4 and 5 were never trained. Stage 2 was never entered for
them. There are no microstructure weights to release, and there will not be.

So cell 1 is the only arm with scored artifacts, and it is the arm the claim was supposed to be
measured *against*. These weights demonstrate the **measurement vehicle** — the held-fixed backbone
— not the contribution.

**They are also not a trading model.** See "Intended use" below; the measured economics are in the
paper and they are negative by a factor of 4.8–43×.

## What the mechanism finding is

The tokenizer keeps microstructure that duplicates price and evicts what is independent of it.
Measured three ways that do not share an input: a synthetic fixture (return dim reconstructs at
0.98, price-correlated dims 0.82–0.92, an independent low-variance dim 0.001–0.014), a canary
(1.151 nats planted in feature space → **zero** extracted; 0.900 nats planted in token space →
94.4% extracted), and real data (40 symbols, n=150k/dim: TFI 0.8223 and signed-count 0.7528
against magnitude dims 0.8975–0.9320, with 97.3% of the shortfall in the two *signed* channels).
The cause is that an MSE reconstruction objective allocates capacity by variance **and
covariance**, which makes an independent low-variance channel the worst per-bit investment
available. A 512-bar windowed read recovers nothing beyond the per-bar read, so this is eviction
rather than smearing.

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

Every unit carries a `run_manifest.json` stamping its environment (image, git commit, lockfile
SHA-256, GPU, CUDA build, driver, torch/numpy/python versions, platform ABI, attention mode). The
demo asserts checkpoint identity on load by recomputing both content hashes from the live modules,
and refuses to serve if either differs. `scripts/m6_demo_acceptance.py` re-proves that the demo's
single-decision assembly lands on the same float64 bits as the production whole-symbol path, and
is itself proven capable of failing by a negative control that injects a one-bar off-by-one.

## Citation

The paper is the citable object; these weights are its evidence. Cite the paper, and cite the
lake Merkle root `sha256:5dfd667d…` and the unit hashes in `m6_weights_release.json` for the
artifacts.
