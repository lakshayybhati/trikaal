# M6 fan-out runbook — executable, costed, credential-free

**§7 v1.6.23. Every number here is measured, not estimated.** Sources:
`runs_manifest/m6_eval_throughput_probe.json` (eval, real 4090), `m6_cuda_probe_cell_manifest.json`
(training, real 4090 at the money surface), `m6_cuda_probe_report.md` (determinism 13.175 %), and
the 2026-08-03 probe rental for the setup timings.

---

## 0. The shape, and why fan-out at all

25 (cell, seed) units, independent. **270.4 GPU-h on one box is ~54 h wall on five at the same
dollars, and a preemption costs 1/25th instead of everything.**

| | 1 box | **5 boxes** | 25 boxes |
|---|---:|---:|---:|
| wall-clock (compute) | 270.4 h ≈ 11.3 days | **54.1 h ≈ 2.3 days** | 10.8 h |
| setup overhead (**measured 0.28 h/box**, ×boxes) | 0.28 h | **1.4 h** | 7.0 h |
| **billed GPU-h** | 270.7 | **271.8** | 277.4 |
| **$ @0.29 / @0.40** | $79 / $108 | **$79 / $109** | $80 / $111 |
| a preemption costs | everything | 1/5 | 1/25 |

**Five is the choice — and the reason is the refusal, not the tax.** Setup overhead is **0.5 % of
the bill at 5 boxes and 2.5 % at 25**; that is small enough at both ends that it does not by itself
decide anything. What decides it is that **25 boxes is 25 chances for an instrument mismatch, and
one mismatch refuses all 25 units.**

> **CORRECTION (2026-08-03), recorded because the row said "measured".** This table previously
> carried **0.6 h/box** labelled *(measured)*, which is **double the measurement stated two
> sections below** (17 min = 0.28 GPU-h) and supported by nothing. It fed the billed-GPU-h row, so
> **271.0 / 273.4 / 285.4 all inherited it**; the corrected values are above. The direction was
> conservative, so **the $150 top-up is untouched**. It also **halves the setup tax** — 5 boxes is
> 0.5 %, not 1 % — which strengthens rather than changes the five-box conclusion, for the
> identity-mismatch reason stated above. **A self-contradicting artifact carrying a "measured"
> label is the class this project has closed four times.**

**The blocking constraint:** identical GPU model, driver, image, interpreter and lockfile across
every shard. Verified by `PROVENANCE_IDENTITY_KEYS` (13 keys) and enforced by
`load_cell_evals` → `VerdictInputError`. **A mismatch on any one key refuses all 25.** Proven:
`tests/run/test_fanout_refusal.py` (18 tests, every key mutation-proven) and
`runs_manifest/m6_fanout_dry_run.json` (13/13 refuse, uniform assembles).

---

## 1. Setup cost — measured 2026-08-03, not estimated

This is the number that was **90 % of the probe's cost** and must be in the plan rather than
discovered again.

| step | measured | note |
|---|---:|---|
| create → `running` (image pull) | **~10 min** | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` |
| **pinned-torch install** | **6 min 41 s** | the image ships **torch 2.5.1**; we pin **2.12.1**. Non-negotiable — a timing/numerics mismatch across shards is an identity-key refusal. |
| payload scp (629 KB) | < 5 s | |
| **total per box** | **≈ 17 min ≈ 0.28 GPU-h** | ≈ **$0.08–0.11** per box |

**At 5 boxes: ~1.4 GPU-h ≈ $0.41–0.56.** Trivial against the bill — *provided it is planned*. It
was 90 % of a 2-minute probe's cost because the probe was short; the run is not.

> **The lesson, recorded: cost a rental as `setup + compute`, never `compute`.**

---

## 2. Per-box procedure — copy-paste, in order

```bash
set -Eeuo pipefail
SHARD=$1; N_SHARDS=5          # 0..4
IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# --- 1. select + launch (reliability first, price second) -----------------------------------
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.98 disk_space>40' \
  -o 'dph+' --raw | head
vastai create instance <ID> --image "$IMAGE" --disk 40 --ssh --direct --label "trikaal-m6-s${SHARD}"

# --- 2. payload: NO CREDENTIALS EVER LEAVE THIS MACHINE ---------------------------------------
# The repo is PRIVATE. Do NOT clone on the box and do NOT ship a token — scp a tarball.
tar czf /tmp/m6_payload.tgz --exclude='__pycache__' src/trikaal scripts pyproject.toml uv.lock \
    runs_manifest/m6_mde_inputs.json runs_manifest/m6_spread_deciles.json
shasum -a 256 /tmp/m6_payload.tgz            # record; verify again on the box
scp -P $PORT /tmp/m6_payload.tgz root@$HOST:/root/

# --- 3. pin the toolchain. The image's torch is NOT ours. --------------------------------------
ssh -p $PORT root@$HOST 'set -Eeuo pipefail
  cd /root && mkdir -p trikaal && tar xzf m6_payload.tgz -C trikaal && cd trikaal
  sha256sum ../m6_payload.tgz                       # must match the local value
  pip install -q "torch==2.12.1" "numpy==2.4.6"
  python3 -c "import torch,numpy,sys;print(torch.__version__,numpy.__version__,sys.version)"'

# --- 4. run the shard. TRIKAAL_IMAGE is an IDENTITY KEY — set it or the verdict refuses. -------
ssh -p $PORT root@$HOST "set -Eeuo pipefail
  cd /root/trikaal && export PYTHONPATH=/root/trikaal/src
  export TRIKAAL_IMAGE='$IMAGE'
  python3 scripts/m6_money_run.py --shard ${SHARD}/${N_SHARDS} --lake ..."

# --- 5. PULL AND VERIFY BEFORE TEARDOWN. No exceptions. ---------------------------------------
ssh -p $PORT root@$HOST 'cd /root/trikaal && sha256sum runs/m6/**/*.json' > /tmp/shard${SHARD}.sha
scp -P $PORT -r root@$HOST:/root/trikaal/runs/m6/ "runs_cloud/shard${SHARD}/"
shasum -a 256 runs_cloud/shard${SHARD}/**/*.json   # must match /tmp/shard${SHARD}.sha

# --- 6. destroy, THEN RE-LIST. `destroy` without -y prompts, aborts, and returns 0. -----------
vastai destroy instance <ID> -y
vastai show instances --raw                        # MUST show the instance gone
```

**Step 6 is not pedantry.** On 2026-08-03 `vastai destroy instance <id>` prompted, aborted, and
returned while the box was still billing. **The mandatory re-list is the only reason it was
caught.**

---

## 3. The rules that are not negotiable

| # | rule | why |
|---|---|---|
| R1 | **No credential ever reaches a rented box.** The repo is private; scp a tarball. | a write-capable token on a preemptible box is unrecoverable |
| R2 | **`TRIKAAL_IMAGE` must be exported on every shard.** | it is an identity key; if *some* shards set it and others do not, that is a refusal — which is the correct outcome and better than silence |
| R3 | **Pin torch/numpy on every box.** | the image ships 2.5.1; ours is 2.12.1. A version split across shards refuses — correctly, and after paying for the compute |
| R4 | **Pull and sha256-verify every scientific artifact before `destroy`.** | bulk checkpoints from a run declared INVALID are exempt, exemption recorded |
| R5 | **`destroy -y`, then RE-LIST.** | see above |
| R6 | **One re-launch on the same shard is pre-authorised for an instrument defect.** Anything else is a STOP and report. | |

---

## 4. Failure modes and the response, pre-committed

| failure | response |
|---|---|
| a shard is preempted mid-run | relaunch that shard only; artifacts are per-unit and resume binds to checkpoint hashes (C-14) |
| a shard's provenance diverges | **the verdict refuses all 25.** Re-run the divergent shard on a matching box. Do NOT hand-edit the stamp. |
| `chunk=512` OOMs on a box | it did **not** on a 4090 (peak 8.91 GiB of 24). If it happens, that box is not a 4090-class card — destroy and relaunch, do not lower the chunk (it is pinned) |
| a legibility/codebook gate fires | **STOP.** Pre-written adjudication (`§7 v1.4.1`, `v1.6.22`); it is a named checkpoint, not a threshold to move |
| the bill approaches $150 | **STOP and report.** Do not continue into an unfunded run |

---

## 5. What this runbook does NOT establish

It proves the **assembly** contract and the **procedure**. It does not prove the launcher stamps
provenance correctly on a real box — **that is verified by the first shard's artifact, before the
other four launch.** Launch shard 0 alone, pull its artifact, confirm all 13 identity keys are
present and populated, *then* launch the remaining four.
