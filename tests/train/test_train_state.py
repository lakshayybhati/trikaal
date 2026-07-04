"""Resumable + atomic checkpointing KATs (m6_design §6 item 9) — incl. the KILL-RESUME PROOF.

The load-bearing test: train a tiny tokenizer to 2N uninterrupted → reference weights hash;
train the same config in a SUBPROCESS, SIGKILL it (hard, not graceful) mid-run at ~step N;
resume from its last atomic train-state and continue to 2N → the resumed weights hash must equal
the uninterrupted reference BIT-EXACTLY (CPU deterministic path). A match proves optimizer state,
LR-schedule position, RNG streams, and the step counter were ALL restored — any miss diverges.
This is the local twin of pre-flight Item 3."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from trikaal.train.checkpoint import state_dict_hash
from trikaal.train.train_state import (
    autocast_ctx,
    load_train_state,
    save_train_state,
)
from trikaal.train.train_tokenizer import train_stage1

# small-but-real config shared by the reference, the killed run, and the resume. Sized so the
# subprocess trains for SECONDS (not ms) — the parent's poll must reliably observe a mid-flight
# step; a too-fast run finishes before the kill lands (a race caught during test development).
N_BARS, SEQ_LEN, BATCH, N_EVAL = 1200, 32, 32, 8
TWO_N = 200  # total steps; the kill lands in [N, 2N)
TOK_KW = dict(n_features=16, d_model=64, n_layers=1, n_heads=2, d_ff=128)
DATA_SEED, RUN_SEED = 123, 0


def _data():
    rng = np.random.default_rng(DATA_SEED)
    x = rng.standard_normal((N_BARS, 16)).astype(np.float32)
    m = np.zeros((N_BARS, 16), dtype=np.uint8)
    return x, m, np.zeros(N_BARS, dtype=np.int64)


def _train(max_steps: int, state_path=None, resume=False):
    x, m, seg = _data()
    return train_stage1(
        x,
        m,
        seg,
        seq_len=SEQ_LEN,
        batch_size=BATCH,
        max_steps=max_steps,
        eval_every=10,
        n_eval_windows=N_EVAL,
        device="cpu",
        seed=RUN_SEED,
        model_kwargs=TOK_KW,
        state_path=state_path,
        resume=resume,
        state_every=1,
    )


def test_atomic_write_and_roundtrip(tmp_path):
    """A half-written tmp file can never corrupt the live state; the roundtrip restores
    step + RNG + weights exactly."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters())
    rng = np.random.default_rng(5)
    _ = rng.standard_normal(3)  # advance the stream
    p = tmp_path / "state.pt"
    save_train_state(p, model=model, optimizer=opt, step=7, np_rng=rng, sampler_state={"k": 1})
    next_draw = rng.standard_normal(4)  # what the stream yields AFTER the snapshot
    # crash simulation: garbage lands in the tmp path — the live file must be untouched
    (tmp_path / "state.pt.tmp").write_bytes(b"garbage from a crashed writer")
    model2 = torch.nn.Linear(4, 4)
    opt2 = torch.optim.AdamW(model2.parameters())
    rng2 = np.random.default_rng(99)  # wrong stream on purpose — restore must overwrite it
    st = load_train_state(p, model=model2, optimizer=opt2, np_rng=rng2)
    assert st["step"] == 7 and st["sampler_state"] == {"k": 1}
    assert state_dict_hash(model2.state_dict()) == state_dict_hash(model.state_dict())
    np.testing.assert_array_equal(rng2.standard_normal(4), next_draw)  # RNG stream continues


def _spawn_and_kill_midflight(state_path: Path) -> int | None:
    """Launch the training subprocess and SIGKILL it once a mid-flight step is observed.

    Returns the step observed at kill time, or None if the subprocess out-raced the poll (the
    caller retries — SIGKILL timing is inherently racy; the retry makes the test deterministic)."""
    code = f"""
import numpy as np
from trikaal.train.train_tokenizer import train_stage1
rng = np.random.default_rng({DATA_SEED})
x = rng.standard_normal(({N_BARS}, 16)).astype(np.float32)
m = np.zeros(({N_BARS}, 16), dtype=np.uint8)
seg = np.zeros({N_BARS}, dtype=np.int64)
train_stage1(x, m, seg, seq_len={SEQ_LEN}, batch_size={BATCH}, max_steps={TWO_N},
             eval_every=50, n_eval_windows={N_EVAL}, device="cpu", seed={RUN_SEED},
             model_kwargs={TOK_KW!r}, state_path={str(state_path)!r}, state_every=1)
print("FINISHED_UNKILLED")
"""
    env = {**os.environ, "PYTHONPATH": "src"}
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    killed_at = None
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            break  # finished before the kill → race lost, caller retries
        if state_path.exists():
            try:
                st = torch.load(state_path, map_location="cpu", weights_only=False)
                if TWO_N // 2 <= st["step"] < TWO_N - 10:
                    proc.send_signal(signal.SIGKILL)  # the disaster rehearsal — no cleanup runs
                    killed_at = int(st["step"])
                    break
            except Exception:
                pass  # os.replace is atomic so a torn read can't happen; stay robust anyway
        time.sleep(0.01)
    proc.wait(timeout=30)
    if killed_at is not None:
        assert proc.returncode == -signal.SIGKILL  # genuinely SIGKILLed, not a graceful exit
    return killed_at


@pytest.mark.slow
def test_kill_resume_matches_uninterrupted_bit_exact(tmp_path):
    """THE PROOF (§6 item 9): SIGKILL mid-training → resume → bit-exact match at 2N."""
    # 1) uninterrupted reference to 2N
    ref = _train(TWO_N)
    ref_hash = state_dict_hash(ref.model.state_dict())

    # 2) the same run in a subprocess, state saved EVERY step; hard-kill it mid-flight.
    #    Up to 3 attempts — the kill must land while training is genuinely in progress.
    killed_at = None
    state_path = tmp_path / "state.pt"
    for _attempt in range(3):
        state_path.unlink(missing_ok=True)
        killed_at = _spawn_and_kill_midflight(state_path)
        if killed_at is not None:
            break
    assert killed_at is not None, "subprocess out-raced the kill 3x — enlarge TWO_N"

    # 3) resume from the last atomic state and continue to 2N
    saved = torch.load(state_path, map_location="cpu", weights_only=False)
    assert TWO_N // 2 <= saved["step"] < TWO_N  # it really was interrupted mid-flight
    res = _train(TWO_N, state_path=state_path, resume=True)

    # 4) BIT-EXACT: the resumed trajectory lands on the identical weights
    assert state_dict_hash(res.model.state_dict()) == ref_hash


def test_bf16_autocast_knob_runs_on_cpu():
    """The precision knob executes (CUDA exercises it for real at the toy run): enabled bf16
    autocast forward returns a finite loss; disabled is the default no-op."""
    from trikaal.tokenizer.model import TokenizerAE

    tok = TokenizerAE(**TOK_KW)
    x = torch.randn(2, 16, 16)
    m = torch.zeros(2, 16, 16)
    with autocast_ctx("cpu", bf16=True):
        out = tok(x, m)
    assert torch.isfinite(out["loss"])
    with autocast_ctx("cpu", bf16=False):
        out2 = tok(x, m)
    assert torch.isfinite(out2["loss"])
