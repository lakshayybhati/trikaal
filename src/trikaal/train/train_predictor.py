"""Stage-2 AR training on the frozen token stream (§7.3) — single-symbol M3 de-risk.

Trains the decoder-only backbone + MTP on the materialized ``(coarse, fine)`` stream with the
hierarchical sampled-coarse objective (§6.3) and the MTP causal chain (§6). Reports a temporal-tail
val convergence read and — the M3 exit gate — whether the model beats two trivial baselines per
sub-token: the **train-marginal** predictor (context must help → lower CE) and **last-token
persistence** (``b_t = b_{t-1}`` → the model's next-token accuracy must exceed just repeating).

This is a convergence smoke on one coin-year, NOT the full-scale schedule and NOT a forecast-skill
claim (no RankIC/IR here — that needs the M5 eval harness). Temporal-tail val, not purged WF.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from trikaal.model.predictor import TrikaalAR
from trikaal.train.checkpoint import save_checkpoint
from trikaal.train.gates import cosine_warmup_lr
from trikaal.train.train_state import autocast_ctx, load_train_state, save_train_state
from trikaal.train.train_tokenizer import _gather, build_window_index
from trikaal.utils.seeding import set_determinism


def marginal_ce_and_acc(
    train_ids: np.ndarray, val_targets: np.ndarray, vocab: int, *, alpha: float = 1.0
) -> tuple[float, float]:
    """CE (nats) and top-1 accuracy of the train-marginal-frequency predictor on val targets.

    Uses additive (Laplace/Lidstone, ``alpha``) smoothing so the marginal is a *valid* distribution
    that sums to 1 and never charges an unseen-in-train token more than ignorance: an unseen token
    costs ``-log(alpha/(total + alpha·vocab)) < log(vocab)``. A raw-empirical floor would instead
    assign sub-uniform mass to unseen tokens, *inflating* the marginal CE and making this baseline
    (which the model must beat) artificially easy — loosening the gate in the too-lenient direction.
    """
    counts = np.bincount(train_ids.ravel(), minlength=vocab).astype(np.float64)
    p = (counts + alpha) / (counts.sum() + alpha * vocab)
    ce = float(-np.log(p[val_targets.ravel()]).mean())
    acc = float((val_targets.ravel() == int(counts.argmax())).mean())  # mode predictor
    return ce, acc


def persistence_accuracy(windows_ids: np.ndarray) -> float:
    """Next-token accuracy of repeating the previous token (``b_t = b_{t-1}``) over [nb, L]."""
    return float((windows_ids[:, 1:] == windows_ids[:, :-1]).mean())


@torch.no_grad()
def accuracy_batch(
    model: TrikaalAR, b_c: torch.Tensor, b_f: torch.Tensor, ts: torch.Tensor
) -> tuple[float, float]:
    """Teacher-forced next-token top-1 accuracy (coarse, fine) on a batch of windows."""
    model.eval()
    h = model.trunk(b_c, b_f, ts)[:, :-1]
    tgt_c, tgt_f = b_c[:, 1:], b_f[:, 1:]
    acc_c = (model.w_c(h).argmax(-1) == tgt_c).float().mean().item()
    acc_f = (model.fine_logits(h, tgt_c).argmax(-1) == tgt_f).float().mean().item()
    return acc_c, acc_f


@dataclass
class Stage2Result:
    converged_step: int | None
    history: list[dict] = field(default_factory=list)
    baselines: dict[str, float] = field(default_factory=dict)
    final: dict[str, float] = field(default_factory=dict)
    model: TrikaalAR | None = None
    ckpt_hash: str | None = None  # content hash of the best-val checkpoint saved during training


def train_stage2(
    b_c_np: np.ndarray,
    b_f_np: np.ndarray,
    ts_np: np.ndarray,
    segment_id: np.ndarray,
    *,
    seq_len: int = 512,
    stride: int = 64,
    batch_size: int = 16,
    val_frac: float = 0.1,
    peak_lr: float = 3e-4,
    max_steps: int = 4000,
    warmup_frac: float = 0.05,
    eval_every: int = 100,
    n_eval_windows: int = 96,
    converge_rel: float = 0.005,
    converge_patience: int = 4,
    mtp_depths: int = 4,
    mtp_mu: float = 0.3,
    model_max_len: int = 512,
    model_kwargs: dict | None = None,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
    ckpt_path: str | Path | None = None,
    state_path: str | Path | None = None,
    resume: bool = False,
    state_every: int | None = None,
    autocast_bf16: bool = False,
) -> Stage2Result:
    """Train the AR backbone on the token stream; return convergence history + baseline margins.

    The model keeps its full ``model_max_len`` context (default 512, Kronos_small) regardless of the
    training-window ``seq_len`` — so a short-window convergence smoke still produces a 512-context
    checkpoint. When ``ckpt_path`` is set, the best-val model is saved there on every improvement
    (crash-safe: a killed run keeps its best artifact); ``verbose`` prints a live per-eval line.

    Resumable (§6 item 9): ``state_path`` + ``resume`` as in ``train_stage1`` — the full train
    state (model+opt+step+RNG) is saved atomically every ``state_every`` steps and a resumed run
    continues the exact trajectory. ``autocast_bf16`` = the CUDA bf16 precision knob.
    """
    set_determinism(seed, deterministic_algorithms=False)  # MPS-friendly convergence run, not G2
    assert seq_len <= model_max_len, f"seq_len {seq_len} exceeds model_max_len {model_max_len}"
    dev = torch.device(device)
    n = b_c_np.shape[0]
    split_bar = int(n * (1.0 - val_frac))

    starts = build_window_index(segment_id, seq_len, stride=stride)
    train_starts = starts[starts + seq_len <= split_bar]
    val_starts = starts[starts >= split_bar]
    if train_starts.size < batch_size or val_starts.size < 1:
        raise ValueError(f"too few windows (train {train_starts.size}, val {val_starts.size})")

    b_c = torch.from_numpy(b_c_np.astype(np.int64))
    b_f = torch.from_numpy(b_f_np.astype(np.int64))
    ts = torch.from_numpy(ts_np.astype(np.int64))
    rng = np.random.default_rng(seed)
    vb = rng.choice(val_starts, size=min(n_eval_windows, val_starts.size), replace=False)
    vc = _gather(b_c, vb, seq_len).to(dev)
    vf = _gather(b_f, vb, seq_len).to(dev)
    vts = _gather(ts, vb, seq_len).to(dev)

    mk = dict(mtp_depths=mtp_depths, mtp_mu=mtp_mu, max_len=model_max_len, **(model_kwargs or {}))
    model = TrikaalAR(**mk).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01, betas=(0.9, 0.95))
    warmup = int(warmup_frac * max_steps)
    state_every = state_every or eval_every

    start_step = 0
    if resume and state_path is not None and Path(state_path).exists():
        st = load_train_state(state_path, model=model, optimizer=opt, np_rng=rng)
        start_step = st["step"]

    history: list[dict] = []
    converged_step: int | None = None
    ckpt_hash: str | None = None
    best, stalls = float("inf"), 0
    t0 = time.time()
    for step in range(start_step + 1, max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_warmup_lr(step, peak_lr, warmup, max_steps)
        sb = rng.choice(train_starts, size=batch_size, replace=False)
        xb_c = _gather(b_c, sb, seq_len).to(dev)
        xb_f = _gather(b_f, sb, seq_len).to(dev)
        xb_ts = _gather(ts, sb, seq_len).to(dev)
        model.train()
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device, bf16=autocast_bf16):
            out = model.compute_loss(xb_c, xb_f, xb_ts)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if state_path is not None and step % state_every == 0:
            save_train_state(state_path, model=model, optimizer=opt, step=step, np_rng=rng)

        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                vo = model.compute_loss(vc, vf, vts)
            acc_c, acc_f = accuracy_batch(model, vc, vf, vts)
            rec = {
                "step": step,
                "train_nll": float(out["nll_main"]),
                "val_nll": float(vo["nll_main"]),
                "val_nll_coarse": float(vo["nll_coarse"]),
                "val_nll_fine": float(vo["nll_fine"]),
                "val_acc_coarse": acc_c,
                "val_acc_fine": acc_f,
            }
            for k in sorted(vo):
                if k.startswith("mtp_depth"):
                    rec[f"val_{k}"] = float(vo[k])
            history.append(rec)
            v = rec["val_nll"]
            if v < best and ckpt_path is not None:  # save best-val artifact (crash-safe)
                ckpt_hash = save_checkpoint(ckpt_path, model, model.get_config())
            if verbose:
                print(
                    f"  step {step:>5}  train {rec['train_nll']:.4f}  val {v:.4f} "
                    f"(c {rec['val_nll_coarse']:.4f}/f {rec['val_nll_fine']:.4f})  "
                    f"acc c{rec['val_acc_coarse']:.3f}/f{rec['val_acc_fine']:.3f}  "
                    f"[{time.time() - t0:.0f}s]",
                    flush=True,
                )
            if v > best * (1 - converge_rel):
                stalls += 1
                if stalls >= converge_patience and converged_step is None:
                    converged_step = step
            else:
                stalls = 0
            best = min(best, v)

    # baselines on the same val batch (apples-to-apples with the model's val targets)
    vc_np, vf_np = vc.cpu().numpy(), vf.cpu().numpy()
    mce_c, macc_c = marginal_ce_and_acc(b_c_np[:split_bar], vc_np[:, 1:], model.v_c)
    mce_f, macc_f = marginal_ce_and_acc(b_f_np[:split_bar], vf_np[:, 1:], model.v_f)
    baselines = {
        "marginal_ce_coarse": mce_c,
        "marginal_ce_fine": mce_f,
        "marginal_acc_coarse": macc_c,
        "marginal_acc_fine": macc_f,
        "persistence_acc_coarse": persistence_accuracy(vc_np),
        "persistence_acc_fine": persistence_accuracy(vf_np),
    }
    final = dict(history[-1]) if history else {}
    if ckpt_path is not None and ckpt_hash is None:  # no eval improved → persist the last model
        ckpt_hash = save_checkpoint(ckpt_path, model, model.get_config())
    return Stage2Result(
        converged_step=converged_step,
        history=history,
        baselines=baselines,
        final=final,
        model=model,
        ckpt_hash=ckpt_hash,
    )


__all__ = [
    "Stage2Result",
    "accuracy_batch",
    "marginal_ce_and_acc",
    "persistence_accuracy",
    "train_stage2",
]
