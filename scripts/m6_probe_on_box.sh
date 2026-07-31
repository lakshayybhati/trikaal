#!/usr/bin/env bash
# §7 v1.6 — run the staged CUDA probe ON the rented box, then compute the pre-teardown digests.
#
# Runs from the repo root on the instance. The repo + a 5-symbol lake slice are pushed by rsync
# (the private-repo clone path needs a GitHub token; rsync needs none, and this probe only needs
# code + five symbols, not the 27 GB lake).
#
#   bash scripts/m6_probe_on_box.sh 2>&1 | tee runs_cloud_out/probe.log
#
# EXIT CODES ARE LOAD-BEARING (pipefail rider): any step failing aborts before the digests are
# written, so a partial run can never present itself as a complete one.
set -Eeuo pipefail

OUTDIR="${OUTDIR:-/workspace/probe_out}"
STEPS="${STEPS:-60}"
WARMUP="${WARMUP:-10}"
CELL_STEPS="${CELL_STEPS:-120}"
mkdir -p "$OUTDIR"

echo "=== [1/5] host provenance ==="
nvidia-smi | head -12 | tee "$OUTDIR/nvidia_smi.txt"

echo "=== [2/5] uv + the PINNED env from the committed lockfile ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version
# --locked: resolve NOTHING. The exact pinned env (numpy 2.4.6 / torch 2.12.1) or a hard failure —
# a drifted env would make the probe's numbers incomparable with every prior receipt.
uv sync --locked --extra data --extra dev

echo "=== [3/5] CUDA sanity (fail closed BEFORE any measurement) ==="
uv run python - <<'PY'
import sys
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA NOT AVAILABLE — abort. A CPU-resolved torch makes this probe meaningless.")
print("gpu:", torch.cuda.get_device_name(0))
import numpy
print("numpy", numpy.__version__)
PY

echo "=== [4/5] THE PROBE (job 1 determinism pair + job 2 end-to-end) ==="
PYTHONPATH=src uv run python scripts/m6_cuda_probe.py \
  --device cuda --steps "$STEPS" --warmup "$WARMUP" --cell-steps "$CELL_STEPS" \
  --out "$OUTDIR/cell" --receipt "$OUTDIR/m6_cuda_probe.json"

echo "=== [5/5] pre-teardown digests (the box computes them; the Mac re-verifies) ==="
cp -f runs_manifest/m6_determinism_feasibility_probe.json "$OUTDIR/" 2>/dev/null || true
cp -f runs/m6_attention_bench.json "$OUTDIR/m6_attention_bench_last_arm.json" 2>/dev/null || true
cd "$OUTDIR"
find . -type f ! -name 'SHA256SUMS.box' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.box
wc -l < SHA256SUMS.box | xargs echo "files digested:"
echo "=== BOX RUN COMPLETE ==="
