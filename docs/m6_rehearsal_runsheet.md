# M6 Toy-CUDA Rehearsal — On-Box Runsheet (pre-flight Items 2+canary / 3-CUDA / 4 / 5)

**Scope:** ONE short 4090 rental (~hours, ~$5–20 ceiling; offers ≈ $0.26–0.40/hr). Everything
below runs ON the rented box unless marked LOCAL. Fail-closed: any manual fix lands as a commit,
then that stage re-runs clean. Governing docs: `docs/m6_preflight.md`, `docs/m6_prereg.md` §3a,
`docs/cloud_runbook.md`.

**Human gates before stage 0** (from the runbook's "Blocked on Lakshay" list — the rest is
already satisfied: vast key configured, SSH key registered, spend scoped by the instruction):
- **W&B API key** — `wandb login` on the box (or `export WANDB_API_KEY=…` in the ssh session);
  never committed, never in this file. Required by Item 2's "W&B ONLINE" evidence.
- **Snapshot home** — where Item 5 pushes the lake + one checkpoint (suggested: an HF private
  dataset repo → needs `hf auth login` on the box; or any S3/B2 bucket via `vastai cloud copy`).
  Required for the round-trip restore proof — and 23G `processed/universe_bars_orig` on the Mac
  is deletable ONLY after this proof.
- (GH_TOKEN is NOT needed: the repo is transferred by `vastai copy` with its `.git`;
  `setup_cloud.sh`'s tokenless fetch warns and proceeds on local refs at `TRIKAAL_COMMIT`.)

## 0. Provision + integrity (fail-closed before any training)
```bash
# LOCAL — rent (≈$0.26–0.40/hr 4090, ≥100GB disk), then STOP for the $0-GPU upload window
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 cuda_vers>=12.0 disk_space>=100 inet_down>=200 rentable=true verified=true' -o 'dph_total' --limit 20
vastai create instance <ID> --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel --disk 100 --ssh --direct --label trikaal-m6-toy
vastai stop instance <IID>          # confirm GPU $/hr = 0 on the dashboard (runbook §3 Method A)
vastai copy local:. <IID>:/workspace/trikaal                       # repo incl. .git (~15 MB)
vastai copy local:processed/universe_bars <IID>:/workspace/trikaal/processed/universe_bars   # 15G
vastai copy local:processed/universe <IID>:/workspace/trikaal/processed/universe             # ledger+manifest
vastai start instance <IID> && vastai ssh-url <IID>

# ON BOX
export TRIKAAL_COMMIT=<sha of the pushed rehearsal-prep commit>
bash /workspace/trikaal/scripts/setup_cloud.sh          # lockfile install; asserts torch==2.12.1+CUDA+bf16; fast suite green
wandb login                                             # ← Lakshay's key, runtime only
PYTHONPATH=src python3 scripts/m6_conformance.py                                  # §3a surface PASS
PYTHONPATH=src python3 scripts/compact_lake.py --skip-compact --dst processed/universe_bars \
    --expected-merkle "sha256:5dfd667d…full-hash…"       # lake integrity: Merkle + 7,024 files re-derived
```

## 1. Item 2 ★ — the continuous 5-cell×3-seed toy run (real dims, real lake, W&B online)
```bash
PYTHONPATH=src python3 scripts/m6_toy_rehearsal.py --device cuda --steps 300 --wandb online
```
Produces (all in `runs/m6_toy/` + `rehearsal_manifest.json`, content-hashed bundle): nvidia-smi
+ per-run determinism mode; 15 trained+reloaded checkpoint hashes (5 cells × seeds {0,1,2} —
the 3-seed shape is deliberate: the verdict dry run needs the full 15-artifact matrix); thin-coin
(FRONTUSDT) draw counts under ^0.5; the Cell-5 permuted-micro training-input assert; money-CODE-
PATH scoring (money=True, no cap, per-symbol deciles, continuous blocks-1–5 grid of the TOY
window) → 15 verdict artifacts via `write_cell_eval_artifact` with the codebook diagnostic;
W&B group `m6_toy_rehearsal`; measured steps/s + bars/s + GPU-util samples (Item 4 inputs).

## 2. The verdict dry run (zero first-time events extends to the DECISION)
```bash
PYTHONPATH=src python3 scripts/m6_verdict.py --artifacts runs/m6_toy/eval \
    --out runs/m6_toy/verdict_manifest.json --allow-toy-grid
```
Expected: conformance gate PASS first, 15 artifacts content-verified, all five §3 clauses
evaluated on garbage toy numbers WITHOUT crashing, manifest written with `grid_pinned: false`
(loud — a toy manifest is never quotable as the M6 outcome).

## 3. Item 2 canary — the machine MEASURES (supervisor ruling 2026-07-19, option (i))
```bash
PYTHONPATH=src python3 scripts/m6_canary.py --device cuda
```
REAL canonical dims (d512 backbone, canonical vocabs), ≥200k-bar stream (the 26k toy's
median-1 joint-id sparsity was a measured confound), CONVERGENCE-BASED stopping (val-loss
improvement < ε=0.005 nats over K=5 consecutive evals, evals every 200 steps, hard cap
20,000; overridable flags — the values used are stated in the log + manifest), toy lr
(TOY-ONLY; the real run's schedule stays the orchestrator default, identical across cells).
Every eval logs the committed probes (teacher-forced corr + h=2 rollout corr) — the
TRAJECTORY discriminates under-convergence from inability. PASS = planted fires (probe corr
rises materially + ΔIR ordering + paired CI clears) + placebo neutral both arms + recon/
codebook non-degenerate. **IF DETECTION FAILS WITH CONVERGED CURVES AT REAL DIMS: FULL STOP —
ship the probe trajectories; nothing further runs (pre-spend architectural finding; the
15-day run cannot be authorized over it).** Commit `runs/m6_canary/canary_manifest.json`.

## 4. The §3a attention-mode decision
```bash
PYTHONPATH=src python3 scripts/m6_attention_bench.py --device cuda --steps 60
```
Mechanical rule (in the script): flash2 iff stable AND ≥1.15× faster AND final-loss gap ≤5% of
scale, else sdpa_deterministic. Append the dated decision + evidence to `docs/m6_prereg.md` §7
(pre-authorized) and wire it as the recorded default for the real run.

## 5. Item 3 on CUDA — kill-resume
```bash
TRIKAAL_KILL_RESUME_DEVICE=cuda PYTHONPATH=src python3 -m pytest \
    tests/train/test_train_state.py::test_kill_resume_matches_uninterrupted_bit_exact -q
```
Under sdpa_deterministic (+ forced deterministic algorithms, wired into the test): bit-exact
REQUIRED. If the bench chose flash2: additionally document the realized resume tolerance with a
separate on-box comparison — never by weakening the test.

## 6. Item 4 — G3 throughput arithmetic
From `rehearsal_manifest.json`: bars/s into-GPU + util (>80 % or fix the feed and re-measure).
Arithmetic to show: 15 runs × (§4 budget ≈ 270–300M effective bars/run ÷ measured bars/s), both
stages, ≤ ~15 GPU-days on THIS 4090 — else STOP and report (A100 hour-test / budget trim are
supervisor decisions).

## 7. Item 5 — disk + snapshot round-trip
- Footprint: measured per-run dir size × 15 + logs + eval artifacts vs the box disk, with headroom.
- Push the lake + ONE real checkpoint to the snapshot home; **restore both and hash-verify**
  (lake: re-run the `--skip-compact` Merkle check on the restored copy; checkpoint: sha256).
- Only after the round-trip is proven: (a) pull `runs/` back (`vastai copy <IID>:… ./runs_cloud`,
  confirm non-empty), (b) `vastai destroy instance <IID> -y` (never re-ingest), and (c) REPORT —
  do not delete — that the Mac's 23G `processed/universe_bars_orig` is now safe to remove
  (Lakshay's call, on his machine).

## Evidence checklist for the report
run-log hash · 15 checkpoint hashes (5 seed-0 quoted, all in the bundle) · W&B group URL ·
toy verdict manifest content-hash (`grid_pinned:false`) · canary verdicts (planted/noise/recon) ·
attention §7 entry · CUDA kill-resume comparison · bars/s + util + the ≤15-day arithmetic ·
snapshot round-trip hashes · total $ spent.
