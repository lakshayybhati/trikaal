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
every shard, **and the same code and step budget** (§7 v1.6.25 R6). Verified by
`PROVENANCE_IDENTITY_KEYS` (**16 keys** — `git_commit`, `steps_stage1`, `steps_stage2` added) and
enforced by
`load_cell_evals` → `VerdictInputError`. **A mismatch on any one key refuses all 25.** Proven:
`tests/run/test_fanout_refusal.py` (24 tests, every key mutation-proven against a **literal** key
list — parametrizing over the live tuple meant deleting a key deleted its own test, §7 v1.6.25 R6)
and
`runs_manifest/m6_fanout_dry_run.json` (16/16 refuse, uniform assembles).

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

## 1a. ★ THE BOX-1 SEQUENCE — P2 is not a phase, it is the first command (§7 v1.6.26)

**P2 no longer gets its own rental, and that was the supervisor's correction: the probe was the
wrong shape.** Its entire content is one command that runs *before* training. A dedicated box pays
a full setup for minutes of work — and the market tax only bites when the rental exists ONLY for
the probe (three boxes stalled in `loading` on 2026-08-04 for $0.188 and produced nothing). On a
box we are renting anyway, the marginal cost of the probe is **minutes**.

| | gate | when | cost |
|---|---|---|---|
| **P1** | `m6_money_run.py --dry-run` locally at the shipped HEAD | ✅ **DONE** — exit 0, `runs_manifest/m6_p1_dry_run_completion.json` | $0 |
| **P2** | `--dry-run --shard 0/25 --device cuda` — **step 3 of box 1**, zero training steps, minutes of eval | first production box, before any training is paid for | minutes |
| **P3** | all 16 identity keys REAL in shard 0's artifact | after the real shard 0, before the other four launch | $0 |

**The residual P2 buys down is smaller than it was priced at, and the reason is worth stating:**
`m6_cuda_probe_cell_manifest.json` shows training COMPLETED under `deterministic_algorithms=True`
on CUDA (2.969 steps/s, both arms stable). **Training runs forward AND backward, and
non-deterministic CUDA kernels are overwhelmingly BACKWARD; eval is forward-only.** So the
remaining risk is genuinely small — but step 3 still catches it, before a shard's training is paid
for, at a cost of minutes.

### The box-1 sequence, in order — executable form in §2b

1. rent box 1 (**which we are renting anyway**)
2. **pull the lake** — 4.08 GiB, route B, §2a — verify by sha256, scrub the token, prove the scrub
3. **`--dry-run --shard 0/25 --device cuda`** ← **THIS IS P2**
4. **the cell-6 Stage-1 probe** (~1.5 GPU-h ≈ $0.45–0.60) → the **pre-committed** WIRE / DECLINE
   decision, taken **before the shard structure is fixed** so the matrix is never re-cut 25 ↔ 30
   mid-flight
5. green → **run the real shard 0 on the SAME box, no new setup**
6. verify the artifact, all 16 keys real → **THEN** launch the other four

## 2. Per-box procedure — copy-paste, in order

```bash
set -Eeuo pipefail
SHARD=$1; N_SHARDS=5          # 0..4
IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# --- 1. select + launch. THE TWO FILTERS BELOW ARE NOT OPTIONAL (P2, §7 v1.6.26) --------------
#   cuda_max_good>=13.0 : we pin torch 2.12.1+cu130. On driver 565.77 it installs FINE and then
#                         reports cuda.is_available() == False. P2 paid a full setup to learn it.
#   reliability>=0.995  : setup is a HOST PROPERTY WITH A LONG TAIL, not a constant. Measured on
#                         the same day: 45 min at rel 0.992 (produced nothing, $0.254) vs 29 s at
#                         rel 0.997. A 90x spread.
#   inet_down>1000      : setup is dominated by a ~7 GB image pull, so host DOWNLINK is the one
#                         searchable field plausibly predictive of a stall — RELIABILITY
#                         DEMONSTRABLY IS NOT (0.9987 and 0.9976 both stalled past 15 min on
#                         2026-08-04). VERIFIED to be a real, discriminating filter rather than an
#                         ignored keyword, by watching the count move: >100 -> 52, >1000 -> 29,
#                         >5000 -> 20, >99999 -> 0. A field that is silently dropped would return
#                         52 every time. THIS IS A HYPOTHESIS WITH A MECHANISM, NOT A MEASURED
#                         PREDICTOR: n=5 boxes cannot establish it, and the KILL below is what
#                         actually bounds the loss (~$0.09 each vs $0.254 when I waited).
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.995 \
  cuda_max_good>=13.0 disk_space>60 inet_down>1000' -o 'dph+' --raw | head
#              disk 60G, not 40: the lake subset is 4.08 GiB on top of the image + torch wheels.

vastai create instance <ID> --image "$IMAGE" --disk 60 --ssh --direct --label "trikaal-m6-s${SHARD}"
vastai show instances --raw    # ★ RE-LIST AFTER create, NOT ONLY AFTER destroy — see R5b

# --- 1b. KILL A SLOW BOX RATHER THAN WAITING FOR IT ------------------------------------------
# Pre-commit a provisioning cutoff and hold to it. A box stuck in `loading` bills the whole time.
START=$(date +%s)
until vastai show instances --raw | grep -q '"actual_status": "running"'; do
  [ $(( $(date +%s) - START )) -gt 900 ] && { echo "STALLED >15min — destroying"; break; }
  sleep 15
done

# --- 2. payload: NO CREDENTIALS EVER LEAVE THIS MACHINE ---------------------------------------
# ★ THE CODE TARBALL IS NOT THE LAKE (P2 finding F2, §7 v1.6.26). This ships 550 KB of source. The
# run ALSO needs `processed/universe_bars`, and until P2 nothing in this runbook said so — the
# money driver stops at m6_money_run.py:250 with "LAKE MISSING ... refusing to invent data",
# BEFORE train_matrix (:363), so the cost is five boxes' setup (~$0.50), not a training run.
# See §2a. Do not launch a shard until the lake path is settled.
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

# --- 4. run the shard. TRIKAAL_IMAGE and TRIKAAL_GIT_COMMIT are IDENTITY KEYS — export BOTH.
# There is no .git on the box (we ship a tarball, never a credential), so the commit is stamped
# from the machine that BUILT the payload. Unset on some shards and set on others = a refusal.
GIT_SHA=$(git rev-parse HEAD)              # on THIS machine, at tarball-build time
ssh -p $PORT root@$HOST "set -Eeuo pipefail
  cd /root/trikaal && export PYTHONPATH=/root/trikaal/src
  export TRIKAAL_IMAGE='$IMAGE'
  export TRIKAAL_GIT_COMMIT='$GIT_SHA'
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

## 2a. ★ THE LAKE — executable, content-verified, credential-scrubbed (P2 finding F2)

**The money run cannot execute on any rented box without `processed/universe_bars`, and until
2026-08-04 no step here provisioned it.** P2 reached `LAKE MISSING at processed/universe_bars —
refusing to invent data` on a real 4090 with everything else working.

**Only the 40 pinned symbols are needed — MEASURED, not estimated:** **4.08 GiB / 1,920 files of
the 14.72 GiB repo = 27.7 %. 72 % of the lake never moves.** Five boxes = 20.4 GiB.
Ground truth: `runs_manifest/m6_lake_subset_manifest.json`, generated by
`scripts/m6_lake_subset_manifest.py` and committed **before any transfer**, carrying a **sha256 per
file** (the HF `lfs.oid`; 1,920 of 1,920 have one).

**ROUTE B — the repo is PRIVATE, not absent.** Unauthenticated: HTTP 401 on both
`api/datasets/…` and `api/models/…`, account lists zero public repos. Authenticated: **HTTP 200,
`private=True`, `gated=False`, 7,029 files**. Route A (public pull) is off; route C (rsync) is the
fallback and is not needed.

> **THE TOKEN NEVER TOUCHES THE BOX'S DISK.** It lives at `~/.trikaal_hf_token` (mode 0600, outside
> the repo), is read at the moment of the pull, passed through `ssh` into the **environment of one
> command**, and scrubbed immediately after — **and the scrub is VERIFIED, because a scrub nobody
> checks is a scrub that did not happen.** Never the existing write-scoped token: the risk is
> INTEGRITY, not confidentiality — a write token on a box rented by the hour from an anonymous host
> can **overwrite or delete the Merkle-`5dfd667d` anchor the whole reproducibility claim rests on.**

**`LAKE MISSING` CATCHES ABSENCE AND IS BLIND TO TRUNCATION**, which is what a 4.08 GiB / 1,920-file
transfer actually produces. A short lake opens, queries and answers with **fewer bars**, and the box
scores a silently different cross-section. So the pull is verified file-by-file against the
committed manifest by sha256 — proven to catch missing, short, right-length-wrong-bytes, and
unpinned-extra files (`tests/run/test_lake_provisioning.py`, healthy case asserted first).

```bash
# ON THE OPERATOR MACHINE — the token is read here and never stored on the box.
ssh -p $PORT root@$HOST 'pip install -q huggingface_hub'   # AFTER pinned torch; it touches neither

ssh -p $PORT root@$HOST "cd /root/trikaal && HF_TOKEN='$(cat ~/.trikaal_hf_token)' \
  python3 scripts/m6_fetch_lake.py --dest processed/universe_bars"   # pull + sha256 verify
# ^ the token exists only in this one command's environment on the box, never on its disk

# SCRUB, THEN PROVE THE SCRUB — exit 1 if any hf_ credential survives anywhere
ssh -p $PORT root@$HOST 'rm -f ~/.cache/huggingface/token ~/.huggingface/token; \
  history -c 2>/dev/null; cd /root/trikaal && python3 scripts/m6_fetch_lake.py --assert-no-token'
echo "SCRUB_VERIFIED_EXIT=$?"    # MUST be 0 before anything else runs

# and an independent re-verify with no token present at all
ssh -p $PORT root@$HOST 'cd /root/trikaal && python3 scripts/m6_fetch_lake.py --verify-only'
```

---

## 2b. ★ THE BOX-1 LAUNCH BLOCK — one executable sequence, hold for the top-up

**This is the next rental. There is no other.** Everything before it is $0 and done.

```bash
set -Eeuo pipefail
IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
GIT_SHA=$(git rev-parse HEAD)
# HOST/PORT/ID come from §2 step 1: filters reliability>=0.995 + cuda_max_good>=13.0 +
# inet_down>1000 + disk_space>60, RE-LIST after create, KILL if not `running` inside 15 min.

# 1 ── payload + pinned toolchain (§2 steps 2-3), then the transfer client
ssh -p $PORT root@$HOST 'pip install -q huggingface_hub'   # AFTER pinned torch; touches neither

# 2 ── THE LAKE: pull → sha256-verify every file → scrub the token → PROVE the scrub (§2a)
ssh -p $PORT root@$HOST "cd /root/trikaal && HF_TOKEN='$(cat ~/.trikaal_hf_token)' \
  python3 scripts/m6_fetch_lake.py --dest processed/universe_bars"
ssh -p $PORT root@$HOST 'rm -f ~/.cache/huggingface/token ~/.huggingface/token; history -c 2>/dev/null; \
  cd /root/trikaal && python3 scripts/m6_fetch_lake.py --assert-no-token'
echo "SCRUB_VERIFIED_EXIT=$?"          # MUST be 0 before anything else runs

# 3 ── ★ P2. Zero training steps, minutes of eval, BEFORE any training is paid for.
ssh -p $PORT root@$HOST "set -Eeuo pipefail
  cd /root/trikaal && export PYTHONPATH=/root/trikaal/src
  export TRIKAAL_IMAGE='$IMAGE' TRIKAAL_GIT_COMMIT='$GIT_SHA'
  python3 scripts/m6_money_run.py --dry-run --shard 0/25 --device cuda --out /root/p2
  echo P2_EXIT=\$?"                     # read the exit from THE PROCESS, never from a compound

# 4 ── ★ THE ~$0.50 MEASUREMENT THAT DECIDES THE $16-22 (C-12 bracket). Stage 1 ONLY, one seed,
#      ~1.5 GPU-h. It runs HERE — before the shard structure is fixed — so the matrix is never
#      re-cut 25 <-> 30 mid-flight.
ssh -p $PORT root@$HOST "set -Eeuo pipefail
  cd /root/trikaal && export PYTHONPATH=/root/trikaal/src
  python3 scripts/m6_cell6_stage1_probe.py --device cuda --lake processed/universe_bars
  echo CELL6_PROBE_EXIT=\$?"
scp -P $PORT root@$HOST:/root/trikaal/runs_manifest/m6_cell6_stage1_probe.json ./runs_cloud/

# 5 ── the REAL shard 0 on the SAME box, no new setup. --shard N depends on step 4's decision:
#        WIRE_CELL_6      -> 30 units, --shard 0/6   (5 boxes x 6 units)
#        DO_NOT_BUY_CELL_6 -> 25 units, --shard 0/5
ssh -p $PORT root@$HOST "set -Eeuo pipefail
  cd /root/trikaal && export PYTHONPATH=/root/trikaal/src
  export TRIKAAL_IMAGE='$IMAGE' TRIKAAL_GIT_COMMIT='$GIT_SHA'
  python3 scripts/m6_money_run.py --shard 0/5 --device cuda --out /root/m6
  echo SHARD0_EXIT=\$?"

# 6 ── P3: pull shard 0's artifact, sha256-verify, confirm all 16 identity keys are REAL
#      ("unavailable" or "unknown" in ANY key ⇒ do NOT fan out). Then launch boxes 2-5.
```

### The cell-6 decision rule — PRE-COMMITTED, in code, not a judgement made on the box

`scripts/m6_cell6_stage1_probe.py::decide()` reads two **Stage-1** numbers and returns a word.
Boundary-tested in `tests/train/test_cell6_decision_rule.py`, including that it **can say no**.

| | gate | source |
|---|---|---|
| codebook utilization | **≥ 0.95** (`PINNED_CODEBOOK_MIN_UTILIZATION`) | below it, `_validate_artifact` **refuses every cell-6 artifact** — the arm cannot enter a verdict at all |
| realized bits-per-token | **inside [19.5, 20.5]** (`FSQ_BPT_BAND`) | outside it the bracket's lower end sits on a **different-capacity instrument** — the matched-bits control invariant 6 protects |

**BOTH pass → `WIRE_CELL_6`.** Buy 5 seeds (54.1 GPU-h, $15.68–21.63; run becomes 324.5 GPU-h,
$94–130). Report **BOTH ENDS of the bracket whichever way they fall** — **ΔIR(4−6) ≤ ΔIR(4−5) is an
empirical claim about two trained models, NOT an identity**; if it inverts the interval is empty
and the larger number flatters us. Disclose that the two ends are **confounded in opposite
directions**, and that the lower end is capacity-matched only because bpt held.

**EITHER fails → `DO_NOT_BUY_CELL_6`.** Report the capacity handicap as **unquantified in IR
units** — which is exactly what the clause rule strings now say — and record the **measured** reason
the bracket was not constructed. **That is an honest, publishable outcome and it costs nothing.**

> **If P2 raises for want of a deterministic CUDA kernel, that is a RESULT, not a failure** — the
> finding the gate exists to produce, arriving before four more boxes have trained.
>
> **Step 4's script has never executed.** `decide()` is unit-tested at its boundaries; the I/O path
> around it runs for the first time on box 1. By the execution rule that makes it an untested
> component — which is why it sits *after* P2 and *before* any production shard, where its failure
> costs minutes and no science.

---

## 3. The rules that are not negotiable

| # | rule | why |
|---|---|---|
| R1 | **No credential ever reaches a rented box.** The repo is private; scp a tarball. | a write-capable token on a preemptible box is unrecoverable |
| R2 | **`TRIKAAL_IMAGE` AND `TRIKAAL_GIT_COMMIT` must be exported on every shard.** | both are identity keys; if *some* shards set them and others do not, that is a refusal — the correct outcome, and better than silence. The commit key exists because the surface recorded *which machine* and said nothing about *which code* (R6): a stale-payload shard training at the old 2,000-step budget would otherwise have assembled silently |
| R3 | **Pin torch/numpy on every box.** | the image ships 2.5.1; ours is 2.12.1. A version split across shards refuses — correctly, and after paying for the compute |
| R4 | **Pull and sha256-verify every scientific artifact before `destroy`.** | bulk checkpoints from a run declared INVALID are exempt, exemption recorded |
| R5 | **`destroy -y`, then RE-LIST.** | see above |
| R5b | **RE-LIST AFTER `create` TOO — the exact mirror, opposite sign.** | P2, 2026-08-04: `vastai create` **printed nothing and created the box anyway**. Retrying on the silence would have meant paying for two. `destroy` lies by returning 0 having done nothing; `create` lies by saying nothing having done everything. **The re-list is the only statement about instance state either way.** |
| R7 | **Host selection is a COST decision, not a preference.** Filter `reliability>=0.995` and `cuda_max_good>=13.0`, and **destroy any box not `running` inside 15 min** rather than waiting on it. | P2 measured a **90× provisioning spread on one day** — 45 min at rel 0.992 (produced nothing, $0.254) vs 29 s at rel 0.997 — and a driver-565.77 box on which the pinned torch cannot see the GPU. Both were full-price setups that bought nothing. **Cost setup with a TAIL, not a mean:** the P2 estimate missed by 1.4× and this was the entire cause. |
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
other four launch (§1a P3).** Launch shard 0 alone, pull its artifact, confirm all **16** identity
keys are present and populated, *then* launch the remaining four.

Nor does it prove the money driver runs at this configuration, or that the eval survives forced
determinism on CUDA. Those are **P1** and **P2** in §1a, and they are ordered ahead of the fan-out
for that reason.
