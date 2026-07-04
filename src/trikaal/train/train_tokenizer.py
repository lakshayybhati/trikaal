"""Stage-1 FSQ tokenizer training (reconstruction) — §7.2.

Trains the tokenizer on segment-respecting windows of per-bar features and reports the Stage-1
acceptance metrics: reconstruction MAE convergence, codebook usage + perplexity (collapse check),
and the per-stage coarse/fine contribution Δ_fine (the residual-decay guard). This is a real
generalizing-codebook training loop, not the single-batch overfit gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from trikaal.constants import FSQ_LEVELS
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.gates import cosine_warmup_lr
from trikaal.train.train_state import autocast_ctx, load_train_state, save_train_state
from trikaal.utils.seeding import set_determinism


def build_window_index(segment_id: np.ndarray, length: int, stride: int = 1) -> np.ndarray:
    """Start indices of every length-``L`` window that lies within a single segment."""
    starts: list[int] = []
    n = segment_id.shape[0]
    i = 0
    while i + length <= n:
        if segment_id[i] == segment_id[i + length - 1]:
            starts.append(i)
            i += stride
        else:
            # jump to the next segment start
            i += 1
    return np.asarray(starts, dtype=np.int64)


def _gather(t: torch.Tensor, starts: np.ndarray, length: int) -> torch.Tensor:
    idx = starts[:, None] + np.arange(length)[None, :]
    return t[torch.from_numpy(idx)]


def codebook_health(model: TokenizerAE, x: torch.Tensor, m: torch.Tensor) -> dict[str, float]:
    """Codebook usage + perplexity over an eval batch (collapse check, §f)."""
    model.eval()
    with torch.no_grad():
        cidx, fidx = model.encode_tokens(x, m)

    def stats(codes: torch.Tensor, vocab: int) -> tuple[float, float]:
        a = codes.flatten().cpu().numpy()
        counts = np.bincount(a, minlength=vocab).astype(np.float64)
        p = counts / counts.sum()
        nz = p[p > 0]
        ppl = float(np.exp(-(nz * np.log(nz)).sum()))
        usage = float(np.unique(a).size) / vocab
        return ppl, usage

    ppl_c, use_c = stats(cidx, model.v_c)
    ppl_f, use_f = stats(fidx, model.v_f)
    return {"ppl_coarse": ppl_c, "ppl_fine": ppl_f, "usage_coarse": use_c, "usage_fine": use_f}


@dataclass
class Stage1Result:
    converged_step: int | None
    final_recon_mae: float
    history: list[dict] = field(default_factory=list)
    model: TokenizerAE | None = None  # the trained tokenizer (for freezing/checkpointing)


def train_stage1(
    x_np: np.ndarray,
    m_np: np.ndarray,
    segment_id: np.ndarray,
    *,
    levels: tuple[int, ...] = FSQ_LEVELS,
    seq_len: int = 128,
    batch_size: int = 64,
    peak_lr: float = 1e-3,
    max_steps: int = 4000,
    warmup_frac: float = 0.1,
    eval_every: int = 200,
    n_eval_windows: int = 256,
    converge_rel: float = 0.01,
    converge_patience: int = 3,
    device: str = "cpu",
    seed: int = 0,
    model_kwargs: dict | None = None,
    state_path: str | Path | None = None,
    resume: bool = False,
    state_every: int | None = None,
    autocast_bf16: bool = False,
) -> Stage1Result:
    """Train the tokenizer to reconstruction convergence; return metrics history.

    ``converged_step`` = the first eval at which val recon-MAE improved by < ``converge_rel`` for
    ``converge_patience`` consecutive evals (the plateau).

    Resumable (§6 item 9): with ``state_path`` set, the FULL train state (model+opt+step+RNG) is
    atomically saved every ``state_every`` (default ``eval_every``) steps; ``resume=True`` restores
    it and continues the exact trajectory. ``autocast_bf16`` is the CUDA-target precision knob
    (off on MPS/CPU by default).
    """
    set_determinism(seed, deterministic_algorithms=False)  # MPS-friendly; convergence run, not G2
    dev = torch.device(device)
    state_every = state_every or eval_every
    starts = build_window_index(segment_id, seq_len)
    if starts.size < batch_size + n_eval_windows:
        raise ValueError(f"too few windows ({starts.size}) for seq_len={seq_len}")
    n_val = max(n_eval_windows, int(0.1 * starts.size))
    train_starts, val_starts = starts[:-n_val], starts[-n_val:]  # temporal split (val = tail)

    x = torch.from_numpy(x_np.astype(np.float32))
    m = torch.from_numpy(m_np.astype(np.float32))
    rng = np.random.default_rng(seed)
    vb = rng.choice(val_starts, size=min(n_eval_windows, val_starts.size), replace=False)
    xv, mv = _gather(x, vb, seq_len).to(dev), _gather(m, vb, seq_len).to(dev)

    model = TokenizerAE(levels=levels, dropout=0.0, **(model_kwargs or {})).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01, betas=(0.9, 0.95))
    warmup = int(warmup_frac * max_steps)

    start_step = 0
    if resume and state_path is not None and Path(state_path).exists():
        st = load_train_state(state_path, model=model, optimizer=opt, np_rng=rng)
        start_step = st["step"]

    history: list[dict] = []
    converged_step: int | None = None
    best = float("inf")
    stalls = 0
    for step in range(start_step + 1, max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_warmup_lr(step, peak_lr, warmup, max_steps)
        sb = rng.choice(train_starts, size=batch_size, replace=False)
        xb, mb = _gather(x, sb, seq_len).to(dev), _gather(m, sb, seq_len).to(dev)
        model.train()
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device, bf16=autocast_bf16):
            out = model(xb, mb)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if state_path is not None and step % state_every == 0:
            save_train_state(state_path, model=model, optimizer=opt, step=step, np_rng=rng)

        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                vo = model(xv, mv)
            val_mae = float(vo["recon_mae"])
            health = codebook_health(model, xv, mv)
            rec = {
                "step": step,
                "train_mae": float(out["recon_mae"]),
                "val_mae": val_mae,
                "coarse_mae": float(vo["coarse_mae"]),
                "delta_fine": float(vo["coarse_mae"]) - val_mae,
                **health,
            }
            history.append(rec)
            if val_mae > best * (1 - converge_rel):
                stalls += 1
                if stalls >= converge_patience and converged_step is None:
                    converged_step = step
            else:
                stalls = 0
            best = min(best, val_mae)

    final = history[-1]["val_mae"] if history else float("nan")
    return Stage1Result(
        converged_step=converged_step, final_recon_mae=final, history=history, model=model
    )
