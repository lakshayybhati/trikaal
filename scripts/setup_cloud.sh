#!/usr/bin/env bash
# Trikaal — M6 cloud provisioning for a fresh vast.ai CUDA-12.x PyTorch instance.
# Phase-A artifact (no GPU rented by this file; it runs ON the instance once one exists).
#
# Idempotent: safe to re-run. Brings a bare GPU box → verified-ready:
#   clone repo @ pinned commit → uv sync --locked from the COMMITTED uv.lock (exact pinned env:
#   numpy==2.4.6, torch==2.12.1 CUDA wheel; uv auto-provisions Python 3.11 — the image's
#   python/torch no longer matter) → wandb + flash-attn → VERIFY (nvidia-smi, torch CUDA+bf16,
#   flash_attn, ruff, pytest -m "not slow") → READY banner (GPU name + VRAM).
#
# Usage (on the instance — see docs/cloud_runbook.md for the full sequence).
# ★ NO CREDENTIAL IS NEEDED. The repo is PUBLIC as of 2026-08-18. This block used to read "the repo
# is PRIVATE, so pass a short-lived read-only GitHub token via env" — that requirement is gone, and
# its removal serves credential rule R1 (no credential ever reaches a rented box) more directly
# than the token-scrubbing it replaces. The GH_TOKEN path below is OPTIONAL and retained only for a
# private fork; the script already treated it as optional, so no logic changed here.
#   export TRIKAAL_COMMIT=<exact-sha>            # pin for reproducibility
#   git clone https://github.com/lakshayybhati/trikaal.git /workspace/trikaal \
#     && bash /workspace/trikaal/scripts/setup_cloud.sh
# (The script re-clones/refreshes idempotently, so running the in-repo copy again is fine.)
#
# To skip the slow on-GPU flash-attn compile (recommended — see runbook §4), pre-pick a wheel:
#   export FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/<...>.whl
set -Eeuo pipefail

# ----------------------------------------------------------------- config (env-overridable)
REPO_SLUG="${REPO_SLUG:-lakshayybhati/trikaal}"
REPO_URL="${REPO_URL:-https://github.com/${REPO_SLUG}.git}"
TRIKAAL_BRANCH="${TRIKAAL_BRANCH:-real-data-slice}"
TRIKAAL_COMMIT="${TRIKAAL_COMMIT:-}"          # exact sha to pin; empty → branch tip (warns)
WORKDIR="${WORKDIR:-/workspace/trikaal}"      # /workspace = conventional persistent data dir (not /root)
# (INSTALL_TORCH/TORCH_CUDA knobs removed: torch now comes ONLY from the committed uv.lock —
#  installing "latest" or floor-resolved torch is the exact drift failure mode this script had.)
FLASH_ATTN="${FLASH_ATTN:-1}"                 # 0 = skip flash-attn (e.g. a quick smoke)
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-}"      # prebuilt wheel URL → skips the ~30-60 min on-GPU compile
MAX_JOBS="${MAX_JOBS:-}"                       # flash-attn build parallelism; empty → derived from RAM (OOM-safe)
NVCC_THREADS="${NVCC_THREADS:-2}"

log() { printf '\n\033[1;36m[setup_cloud]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[setup_cloud FATAL]\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------- 0. GPU present?
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found — this is not a CUDA GPU box."
log "GPU(s):"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || log "WARN: nvidia-smi query failed (driver/GPU may be wedged) — continuing to the torch check"

# ----------------------------------------------------------------- 1. clone/refresh @ pinned commit
# GH_TOKEN is OPTIONAL and is unset for the public repo — the plain URL clones and fetches fine.
# When it IS set (a private fork), it is injected ONLY into a transient URL and scrubbed from
# .git/config immediately after, on BOTH the clone and the fetch path (a private repo cannot fetch
# from the scrubbed token-free origin).
mkdir -p "$(dirname "$WORKDIR")"
if [ -d "$WORKDIR/.git" ]; then
  log "repo present → fetch"
  [ -n "${GH_TOKEN:-}" ] && git -C "$WORKDIR" remote set-url origin "https://${GH_TOKEN}@github.com/${REPO_SLUG}.git"
  git -C "$WORKDIR" fetch --all --tags --prune --quiet || log "WARN: fetch failed — using local refs"
  git -C "$WORKDIR" remote set-url origin "$REPO_URL"   # re-scrub the token
else
  clone_url="$REPO_URL"
  [ -n "${GH_TOKEN:-}" ] && clone_url="https://${GH_TOKEN}@github.com/${REPO_SLUG}.git"
  log "cloning ${REPO_URL}"
  git clone --quiet "$clone_url" "$WORKDIR"
  git -C "$WORKDIR" remote set-url origin "$REPO_URL"   # scrub any embedded token
fi
cd "$WORKDIR"
if [ -n "$TRIKAAL_COMMIT" ]; then
  log "checkout pinned commit ${TRIKAAL_COMMIT}"
  git checkout --quiet "$TRIKAAL_COMMIT" 2>/dev/null \
    || die "commit ${TRIKAAL_COMMIT} not found — confirm it's pushed and the sha is correct."
else
  log "WARNING: TRIKAAL_COMMIT unset → using branch tip ${TRIKAAL_BRANCH} (pin a sha for repro)"
  git checkout --quiet "$TRIKAAL_BRANCH"
  git pull --quiet --ff-only origin "$TRIKAAL_BRANCH" || true
fi
log "HEAD = $(git rev-parse HEAD)"

# ------------------------------------------- 2-4. pinned env from the COMMITTED LOCKFILE (uv.lock)
# The env drifted TWICE under install-latest + floor pins (the Gate-A anchor incident). The box
# must run the EXACT env the anchor was established under: uv installs the lockfile-resolved
# versions (numpy==2.4.6, torch==2.12.1 — the PyPI linux torch wheel bundles CUDA 12.x, so the
# image's preinstalled torch/python no longer matter; uv auto-provisions Python 3.11 if absent).
command -v uv >/dev/null 2>&1 || {
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}
log "uv $(uv --version 2>/dev/null | head -1) — syncing the committed lockfile (--locked)"
[ -f uv.lock ] || die "uv.lock missing — the committed lockfile is required (never install floors)."
uv sync --locked --extra data --extra dev || die "uv sync --locked failed — lockfile/pyproject mismatch."
# shellcheck disable=SC1091
source .venv/bin/activate

python - <<'PY' || die "CUDA unavailable or torch != lockfile pin — wrong box or a drifted env."
import torch
assert torch.__version__.split("+")[0] == "2.12.1", f"torch {torch.__version__} != pinned 2.12.1"
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
p = torch.cuda.get_device_properties(0)
bf16 = torch.cuda.is_bf16_supported()
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | {p.name} | {p.total_memory/1e9:.0f} GB | bf16={bf16}")
if not bf16:
    print("WARN: bf16 not reported on device 0 — A100/H100 support it; check the GPU / training config.")
PY

# ----------------------------------------------------------------- 5. wandb + flash-attn
log "installing wandb (experiment tracking; outside the eval-replay pin surface)"
uv pip install -q wandb
if [ "$FLASH_ATTN" = "1" ]; then
  if ! python -c 'import flash_attn' 2>/dev/null; then
    # --no-build-isolation reuses the live venv, so flash-attn's setup.py needs these at build time:
    uv pip install -q ninja packaging psutil
    ninja --version >/dev/null 2>&1 || { uv pip uninstall -q ninja; uv pip install -q ninja; }
    if [ -n "$FLASH_ATTN_WHEEL" ]; then
      log "installing flash-attn from prebuilt wheel (fast, no compile)"
      uv pip install -q "$FLASH_ATTN_WHEEL" \
        || die "prebuilt wheel failed — torch/cuda/py/abi must match (see runbook §4 detection one-liner)."
    else
      # Cap build parallelism to RAM: each nvcc/cicc worker eats ~5-8 GB; default MAX_JOBS=nproc OOM-kills.
      MEM_GB="$(awk '/MemTotal/{printf "%d",$2/1024/1024}' /proc/meminfo 2>/dev/null || echo 16)"
      [ -n "$MAX_JOBS" ] || MAX_JOBS=$(( MEM_GB/6 > 0 ? MEM_GB/6 : 1 ))
      export MAX_JOBS NVCC_THREADS
      log "compiling flash-attn (--no-build-isolation, MAX_JOBS=${MAX_JOBS}) — typically 15-45 min, can"
      log "exceed 1h on low-vCPU boxes; this is NORMAL, do not interrupt. Prefer FLASH_ATTN_WHEEL (runbook §4)."
      uv pip install -q flash-attn --no-build-isolation \
        || die "flash-attn build failed (usually RAM OOM) — set FLASH_ATTN_WHEEL (runbook §4) or lower MAX_JOBS."
    fi
  fi
  python -c 'import flash_attn; print("flash_attn", flash_attn.__version__)'
fi

# ----------------------------------------------------------------- 6. env: PYTHONPATH (persist)
export PYTHONPATH="${WORKDIR}/src${PYTHONPATH:+:$PYTHONPATH}"
RC_LINE="cd ${WORKDIR} && source .venv/bin/activate && export PYTHONPATH=${WORKDIR}/src"
grep -qsF "$RC_LINE" "${HOME}/.bashrc" || echo "$RC_LINE" >> "${HOME}/.bashrc"

# ----------------------------------------------------------------- 7. VERIFY: lint + fast tests
log "ruff check ."; ruff check .
log "pytest -m 'not slow' (synthetic suite — no real lake needed)"; pytest -m "not slow" -q

# ----------------------------------------------------------------- 8. READY banner
GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
LAKE="${WORKDIR}/processed/universe_bars"
LAKE_STATE="absent — copy the lake in (see runbook §3)"; [ -d "$LAKE" ] && LAKE_STATE="present ($(du -sh "$LAKE" 2>/dev/null | cut -f1))"
cat <<BANNER

================== TRIKAAL CLOUD READY ==================
 repo    : ${WORKDIR} @ $(git rev-parse --short HEAD)
 python  : $(python -V 2>&1)   venv .venv   (PYTHONPATH=${WORKDIR}/src)
 gpu     : ${GPU}
 torch   : $(python -c 'import torch;print(torch.__version__,"cuda",torch.version.cuda)')
 flash   : $([ "$FLASH_ATTN" = 1 ] && python -c 'import flash_attn;print(flash_attn.__version__)' || echo skipped)
 lake    : ${LAKE} → ${LAKE_STATE}
 next    : wandb login   →   M6 5-cell ablation training (see docs/cloud_runbook.md)
========================================================
BANNER
