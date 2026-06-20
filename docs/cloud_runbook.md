# Trikaal — M6 Cloud Runbook (vast.ai)

The Mac has no CUDA GPU, so the **M6 5-cell ablation** ({BSQ,FSQ}×{OHLCV-only,+micro} + shuffled-micro
placebo, ~27M params, bf16) trains on a **rented vast.ai NVIDIA GPU** (single A100/H100, 40–80 GB).
This runbook is the one-clean-sequence: **provision → transfer the lake → smoke → train → destroy**.

> **Cost discipline.** GPU time is billed per second while an instance is *running*. The rules:
> (1) avoid paying GPU $/hr during the ~27 GB lake upload (see §3 — verify the chosen method in Phase B);
> (2) **`destroy`** the instance the moment training + artifact-download finish — a *stopped* instance
> still bills **storage**, so destroy (not just stop) when done for good;
> (3) get flash-attn as a **prebuilt wheel** (§4) — compiling it on a rented GPU wastes ~30–60 min of $/hr.

Verified against **vastai CLI 1.0.1** (key at `~/.config/vastai/vast_api_key`, already configured).

---

## 0. Phase A vs Phase B

- **Phase A (done now, zero GPU spend):** this runbook + `scripts/setup_cloud.sh`, repo pushed private,
  SSH key ready, vast CLI verified. **No instance rented.**
- **Phase B (tomorrow, with spend approval):** everything below the line. Rents a GPU.

**Blocked on Lakshay before Phase B:** (a) paste the SSH public key into vast.ai → Account → Keys;
(b) the W&B API key (pasted at `wandb login`, never committed); (c) explicit spend approval; (d) a
short-lived **read-only GitHub token** to clone the private repo onto the box; (e) confirm the **full
M4b lake** has finished building locally (§3 precondition).

---

## 1. One-time account setup (no spend)

**SSH key** — add the Mac's public key so you can `ssh` into instances. Paste it into vast.ai →
**Account → Keys → SSH Keys**, or via CLI (always pass your existing `.pub` explicitly — a bare
`create ssh-key` with no argument generates a *new* server-side keypair):
```bash
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
vastai show ssh-keys           # confirm it's listed
```

**GitHub read-only token** — the repo is private and contains code only (the lake/raw are gitignored
and never pushed). Create a *fine-grained* PAT (Contents: Read-only, scoped to `lakshayybhati/trikaal`)
at github.com → Settings → Developer settings. You pass it as `GH_TOKEN` at provision time — it is
**never** committed, written here, or left in `.git/config` (the setup script scrubs it on every run).

---

## 2. Find + rent a GPU (Phase B)

Search on-demand A100/H100 offers, CUDA 12.x, ≥40 GB VRAM, ≥100 GB disk, cheapest first:
```bash
vastai search offers 'gpu_name in [A100_SXM4,A100_PCIE,H100_SXM,H100_PCIE] \
  gpu_ram>=40 cuda_vers>=12.0 disk_space>=100 inet_down>=500 rentable=true verified=true' \
  -o 'dph_total' --limit 20
# fields verified valid in 1.0.1 (cuda_vers is a CLI alias for cuda_max_good; inet_down is MB/s)
```
Pick an `ID` (first column). Create it with a **Python-3.11 CUDA-12.x PyTorch** image, launched as an
SSH instance (the `--ssh` flag selects the ssh launch type — there is no `--runtype`). ~27 GB lake +
repo + torch + flash-attn + checkpoints ⇒ **≥100 GB disk**:
```bash
vastai create instance <ID> \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  --disk 100 --ssh --direct --label trikaal-m6
vastai show instances                 # wait until status = running
vastai ssh-url <IID>                  # → ssh://root@<host>:<port>
```
> **IMAGE — Python version is load-bearing.** The official `pytorch/pytorch` **2.0–2.4** tags ship
> **Python 3.10** (conda base), which `setup_cloud.sh` will *refuse* (project requires ≥3.11). Use a
> tag that ships **3.11** — `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` (or newer), or vast's own
> `vastai/pytorch` template. **Verify before relying on it:** the create's onstart or a quick ssh +
> `python3 --version`. If you must use a 3.10 image, `setup_cloud.sh` tells you to `apt-get install
> python3.11 python3.11-venv` and re-run with `INSTALL_TORCH=1` (it then installs CUDA torch into a
> fresh 3.11 venv).

---

## 3. Transfer the ~27 GB lake — minimize GPU $/hr during upload

**Precondition:** the lake at `processed/universe_bars/` must be the **COMPLETE** M4b universe ingest
(expected ~27 GB; its content-hash/Merkle root is the reproducibility gate in §6). A partial lake
invalidates M6. Check first:
```bash
du -sh processed/universe_bars        # expect ~27 GB once M4b finishes; don't transfer a partial lake
```

Destination on the box is **`/workspace/trikaal/processed/universe_bars`** — `/workspace` is the
conventional persistent data dir on vast images (and on many templates the larger volume); `/root`
works for SSH but isn't the persistence convention.

**Method A — stop → copy → start (no GPU billing during upload; VERIFY in Phase B).**
Stopping an instance halts compute/GPU billing while only storage keeps billing; `vastai copy` to a
stopped instance is documented as supported. **Confirm both on the dashboard the first time** before
trusting it for a multi-hour upload:
```bash
vastai stop instance <IID>                       # → status=stopped; confirm GPU $/hr shows 0, storage only
vastai copy local:processed/universe_bars <IID>:/workspace/trikaal/processed/universe_bars
vastai start instance <IID>                       # resume only when ready to provision + train
```
> Use the explicit `local:` prefix on the source (the bare `./path` legacy form still works but is
> ambiguous). The upload is bounded by **your home upload bandwidth**, not GPU time.

**Method B — Cloud Sync (vast's documented GPU-free path).** If you have an S3/GDrive/Dropbox/B2
bucket, `vastai cloud copy` transfers server-side without compute spend — push the lake to the bucket
once, then cloud-copy it onto the (stopped or running) instance. See `vastai cloud copy --help`.

**Method C — network volume (advanced).** Create a volume, fill it with **no GPU attached**, then
`create instance --link-volume <id>`. See `vastai create network volume --help` and `vastai search
volumes`.

**Honest fallback.** If, in Phase B, `vastai copy` to a *stopped* instance does not work on your
machine type, the copy needs a **running** instance and therefore **bills GPU $/hr for the whole
upload window**. In that case pick the *cheapest acceptable* card, copy straight to `/workspace/...`,
and `destroy` immediately after training. `rsync` over the ssh endpoint is resumable if the link drops:
```bash
# vastai scp-url <IID> → scp://root@<host>:<port>
rsync -avP -e "ssh -p <port>" ./processed/universe_bars/ root@<host>:/workspace/trikaal/processed/universe_bars/
```

---

## 4. flash-attn: prefer a prebuilt wheel (skip the on-GPU compile)

Compiling flash-attn from source on the rented box runs **15–45 min (sometimes >1 h)** of CPU work
while the GPU idles and bills — and is **RAM-OOM-prone**. Pick the matching prebuilt wheel instead.
On the box (after `setup_cloud.sh` step 4, or any torch env) print the three coordinates:
```bash
python -c "import torch,sys; print('torch',torch.__version__); print('cuda',torch.version.cuda); \
print('abi', torch._C._GLIBCXX_USE_CXX11_ABI); print('py cp%d%d'%sys.version_info[:2])"
```
Then choose a wheel from **https://github.com/Dao-AILab/flash-attention/releases**. Filename pattern:
```
flash_attn-<ver>+cu12torch<MINOR>cxx11abi<TRUE|FALSE>-cp311-cp311-linux_x86_64.whl
#  cuda = MAJOR only (cu12) · torch = MINOR (torch2.5) · py = cp311
#  abi  = TRUE  for conda-built pytorch/pytorch images (the _GLIBCXX_USE_CXX11_ABI print is True)
#         FALSE for pip wheels from download.pytorch.org
```
Pass it to the provisioner so it skips the compile entirely:
```bash
export FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/<ver>/<file>.whl
```
If you let it compile anyway, `setup_cloud.sh` caps `MAX_JOBS` to RAM (≈1 job / 6 GB) to avoid the OOM
kill; override with `MAX_JOBS=<n>` if needed. The torch **minor** (2.5 vs 2.4) must match the wheel's
`torchX.Y` tag exactly or the import fails.

---

## 5. Provision the box (one idempotent script)

```bash
ssh -p <port> root@<host>                       # from `vastai ssh-url <IID>`
export GH_TOKEN=<fine-grained-read-only-token>
export TRIKAAL_COMMIT=<exact-sha>               # pin (local: git rev-parse HEAD)
export FLASH_ATTN_WHEEL=<wheel-url-from-§4>     # optional but recommended
git clone https://$GH_TOKEN@github.com/lakshayybhati/trikaal.git /workspace/trikaal
bash /workspace/trikaal/scripts/setup_cloud.sh
```
A green run ends in a **`TRIKAAL CLOUD READY`** banner (GPU name + VRAM, torch+CUDA, flash-attn). The
script is idempotent — re-run it freely (it re-fetches with the token then re-scrubs it).

---

## 6. W&B, then smoke + train (M6 entrypoint finalized at M6)

Log in **on the box at runtime only** — the key is never in git, this runbook, or the setup script:
```bash
wandb login                                     # paste key when prompted (or: export WANDB_API_KEY=...)
cd /workspace/trikaal && source .venv/bin/activate
# smoke: confirm the lake reads + a few GPU training steps before the full run.
# (The M6 training entrypoint + the 5-cell ablation config are authored in the M6 task — NOT here.)
```
The lake is read from `/workspace/trikaal/processed/universe_bars` via DuckDB; the realized-universe
Merkle root gates reproducibility. **Do not start M6 in Phase A.**

---

## 7. Tear down (stop billing — destroy is irreversible)

**Pull artifacts back FIRST and confirm they landed before destroying.** Destroy is irreversible and a
stopped instance still bills storage:
```bash
vastai copy <IID>:/workspace/trikaal/runs ./runs_cloud && ls -la runs_cloud   # MUST succeed + be non-empty
# verify W&B run synced AND ./runs_cloud is populated, THEN:
vastai destroy instance <IID> -y
vastai show instances                                                         # confirm it's gone → $0/hr
```
> Don't assume W&B sync alone — the local `runs_cloud` copy is the safety net. If the artifact copy
> failed or is empty, do **not** destroy; debug while the instance is still running.

---

## 8. Secret + data hygiene (non-negotiable)

- **Never** commit or print: the W&B key, the GitHub token, the vast API key, or the SSH **private** key.
- The lake (`processed/`) and raw caches (`raw/`) are **gitignored** — uploaded by `vastai copy`,
  **never** pushed to GitHub. Confirm with `git check-ignore processed raw`.
- `setup_cloud.sh` injects `GH_TOKEN` only into transient clone/fetch URLs and scrubs it from
  `.git/config` immediately after each.
- Destroy the instance when done; don't leave a stopped instance accruing storage charges.

---

## Quick command reference (vastai 1.0.1)

| Action | Command |
|---|---|
| Find GPUs | `vastai search offers '<query>' -o dph_total --limit 20` |
| Create | `vastai create instance <ID> --image <py3.11-img> --disk 100 --ssh --direct --label trikaal-m6` |
| List / status | `vastai show instances` · `vastai show instance <IID>` |
| SSH / SCP endpoint | `vastai ssh-url <IID>` · `vastai scp-url <IID>` |
| Upload lake | `vastai copy local:processed/universe_bars <IID>:/workspace/trikaal/processed/universe_bars` |
| Stop (storage-only) | `vastai stop instance <IID>` |
| Start | `vastai start instance <IID>` |
| Destroy ($0/hr) | `vastai destroy instance <IID> -y` |
