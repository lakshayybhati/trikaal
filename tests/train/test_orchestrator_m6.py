"""KATs for the 5-cell orchestrator (§6 item 5) + determinism hook (item 7) + tripwire (item 8).

The integration test runs ALL FIVE cells × 1 seed at tiny dims on synthetic 2-symbol data — a
mini-smoke proving: per-cell content-hashed artifacts + run manifests, identical draw across
cells, the determinism record in manifest AND checkpoints, and the tested NaN-injection halt."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from trikaal.data.universe_loader import build_symbol_windows
from trikaal.model.attention_mode import (
    MODE_SDPA,
    determinism_record,
    resolve_attention_mode,
    set_attention_backend,
)
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import ARMS
from trikaal.train.cells import CELLS, build_cell_backbone, build_cell_tokenizer
from trikaal.train.checkpoint import load_checkpoint
from trikaal.train.orchestrator import OrchestratorConfig, run_all_cells, run_cell
from trikaal.train.tripwire import TripwireAbort, TripwireConfig, TripwireMonitor

BAR_MS = 60_000
TOK_KW = dict(d_model=16, n_layers=1, n_heads=2, d_ff=32, max_len=64)
BB_KW = dict(
    n_layers=1,
    d_model=16,
    d_ff=32,
    n_heads=2,
    mtp_depths=0,
    max_len=64,
    ffn_dropout=0.0,
    resid_dropout=0.0,
    attn_dropout=0.0,
    token_dropout=0.0,
)


def _per_symbol_by_arm(n=400, seed=0):
    """Two synthetic symbols, per-arm SymbolWindows (funding/OI masked, one segment)."""
    out: dict[str, list] = {a: [] for a in ARMS}
    for i, sym in enumerate(("AAAUSDT", "BBBUSDT")):
        rng = np.random.default_rng(100 + i)
        x = rng.standard_normal((n, 16)).astype(np.float32)
        m = np.zeros((n, 16), dtype=np.uint8)
        m[:, 13:16] = 1
        ts = (1_600_000_000_000 + np.arange(n) * BAR_MS).astype(np.int64)
        seg = np.zeros(n, dtype=np.int64)
        boundary = int(ts[-1]) + BAR_MS  # all bars are train-legal in this toy
        for arm in ARMS:
            out[arm].append(
                build_symbol_windows(
                    sym,
                    x,
                    m,
                    seg,
                    ts,
                    arm=arm,
                    seq_len=16,
                    boundary_ms=boundary,
                    embargo_bars=0,
                    seed=seed,
                )
            )
    return out


def _cfg(tmp_path, steps=6) -> OrchestratorConfig:
    return OrchestratorConfig(
        seeds=(0,),
        seq_len=16,
        batch_size=4,
        steps_stage1=steps,
        steps_stage2=steps,
        tok_kwargs=TOK_KW,
        backbone_kwargs=BB_KW,
        expect_backbone_params=None,
        enforce_parity=False,  # parity is a canonical-dims property (KAT'd in test_bsq)
        out_dir=tmp_path / "m6",
        state_every=3,
        wandb_mode="disabled",
    )


@pytest.mark.slow
def test_all_five_cells_run_and_produce_hashed_artifacts(tmp_path):
    """The mini-smoke: every cell trains both stages, checkpoints reload, manifests exist."""
    cfg = _cfg(tmp_path)
    top = run_all_cells(_per_symbol_by_arm(), cfg)
    assert top["n_runs"] == 5 and len(top["artifacts"]) == 5
    for spec in CELLS:
        run_dir = tmp_path / "m6" / f"{spec.name}_seed0"
        man = json.loads((run_dir / "run_manifest.json").read_text())
        # determinism record present (item 7) — in the manifest AND both checkpoints
        assert man["determinism"]["attention_mode"] == MODE_SDPA
        assert man["determinism"]["bit_exact_claim"] is True
        tok_l = load_checkpoint(run_dir / "tokenizer.pt", TokenizerAE)
        pred_l = load_checkpoint(run_dir / "predictor.pt", TrikaalAR)
        assert tok_l.meta["attention_mode"] == MODE_SDPA
        assert pred_l.meta["attention_mode"] == MODE_SDPA
        assert tok_l.hash == man["artifacts"]["tokenizer_hash"]
        assert pred_l.hash == man["artifacts"]["predictor_hash"]
        # the cell's arm drives the tokenizer input width; the quantizer drives the vocab
        assert tok_l.config["n_features"] == (7 if spec.arm == "ohlcv" else 16)
        assert tok_l.config["quantizer"] == spec.quantizer
        assert (run_dir / "stage1_state.pt").exists()  # atomic resumable state was written
        # both symbols were drawn (the thin-coin-presence evidence pattern for the real smoke)
        drawn = man["draw"]["drawn_by_symbol_stage1"]
        assert set(drawn) == {"AAAUSDT", "BBBUSDT"} and min(drawn.values()) > 0


@pytest.mark.slow
def test_identical_draw_across_cells(tmp_path):
    """m6_design §4: the (symbol, start) draw sequence must be IDENTICAL across cells — the only
    varied factor is quantizer × arm. Proven via the per-symbol draw counts of two different
    cells at the same seed (the sequence is a pure function of (seed, starts))."""
    cfg = _cfg(tmp_path, steps=4)
    per = _per_symbol_by_arm()
    m1 = run_cell(CELLS[0], 0, per, cfg)  # bsq × ohlcv
    m4 = run_cell(CELLS[3], 0, per, cfg)  # fsq × micro
    assert m1["draw"]["drawn_by_symbol_stage1"] == m4["draw"]["drawn_by_symbol_stage1"]
    assert m1["draw"]["drawn_by_symbol_stage2"] == m4["draw"]["drawn_by_symbol_stage2"]


def test_attention_mode_resolution_and_backend_walk():
    assert resolve_attention_mode("cpu") == MODE_SDPA
    assert resolve_attention_mode("mps", prefer_flash=True) == MODE_SDPA  # flash needs CUDA
    tok = TokenizerAE(**TOK_KW)
    n = set_attention_backend(tok, MODE_SDPA)
    assert n >= 2  # encoder + decoder attention blocks stamped
    with pytest.raises(RuntimeError, match="flash2 requested but"):
        set_attention_backend(tok, "flash2")  # unavailable here — must fail LOUDLY, not fall back
    rec = determinism_record(seed=3, device="cpu", attention_mode=MODE_SDPA)
    assert rec["bit_exact_claim"] and rec["seed"] == 3
    assert {"python", "torch", "numpy"} <= set(rec["env"])


# ------------------------------------------------------------- item 8: the TESTED halt
def test_tripwire_definitions_fire():
    cfg = TripwireConfig(k_first_steps=5, flat_loss_tol=0.02, ir_abs_bound=100.0)
    mon = TripwireMonitor(cfg)
    with pytest.raises(TripwireAbort, match="NON-FINITE loss"):
        mon.check_step("c", 1, float("nan"))
    mon2 = TripwireMonitor(cfg)
    mon2.check_step("c", 1, 1.0)
    with pytest.raises(TripwireAbort, match="NOT DROPPING"):
        mon2.check_step("c", 5, 1.05)  # rising beyond the 2% tolerance at K
    mon2b = TripwireMonitor(cfg)
    mon2b.check_step("c", 1, 1.0)
    mon2b.check_step("c", 5, 1.01)  # within tolerance → NO trip (the threshold is committed)
    mon3 = TripwireMonitor(cfg)
    mon3.check_step("c", 1, 1.0)
    mon3.check_step("c", 5, 0.5)  # dropping → no trip
    with pytest.raises(TripwireAbort, match="beyond the sane bound"):
        mon3.check_eval("c", ir=1e6)
    with pytest.raises(TripwireAbort, match="non-finite"):
        mon3.check_eval("c", rank_ic=float("inf"))
    assert TripwireConfig.load().k_first_steps == 200  # the COMMITTED thresholds parse


@pytest.mark.slow
def test_nan_injection_halts_the_whole_run(tmp_path, monkeypatch):
    """Inject a NaN into one cell's loss mid-training → the monitor trips → the WHOLE
    orchestrator run aborts (pre-flight Item 7's tested halt)."""
    cfg = _cfg(tmp_path, steps=6)
    real_forward = TokenizerAE.forward
    calls = {"n": 0}

    def poisoned(self, x, mask):
        out = real_forward(self, x, mask)
        calls["n"] += 1
        if calls["n"] >= 3:  # the NaN cell appears at step 3
            out["loss"] = out["loss"] * torch.nan
        return out

    monkeypatch.setattr(TokenizerAE, "forward", poisoned)
    with pytest.raises(TripwireAbort, match="NON-FINITE loss"):
        run_all_cells(_per_symbol_by_arm(), cfg)
    # the halt happened inside cell 1 of 5 — nothing later ran (no cell-2+ dirs)
    made = sorted(p.name for p in (tmp_path / "m6").glob("cell*"))
    assert made == ["cell1_bsq_ohlcv_seed0"]


def test_cell_construction_gates():
    """The 21,301,248 pin at canonical dims (FSQ vocab exact; BSQ vocab within ±2%) + teeth."""
    bb = build_cell_backbone(891, 1225, mtp_depths=0)
    assert bb.num_backbone_params() == 21_301_248
    bb2 = build_cell_backbone(1024, 1024, mtp_depths=0)  # BSQ vocab → the ±2% band (0.33%)
    assert abs(bb2.num_backbone_params() - 21_301_248) / 21_301_248 <= 0.02
    with pytest.raises(AssertionError, match="outside ±2%"):
        build_cell_backbone(64, 64, n_layers=1, d_model=16, d_ff=32, n_heads=2, mtp_depths=0)
    tok = build_cell_tokenizer(CELLS[4], **TOK_KW)  # cell 5 = fsq × micro_shuffled
    assert tok.quantizer == "fsq" and tok.n_features == 16
