"""TrikaalAR — the decoder-only AR backbone (exact Kronos_small) + optional MTP heads (§5/§6).

Autoregresses over the per-bar ``(coarse, fine)`` token stream, factorizing each bar as
``p(b_t|b_{<t}) = p(b^c_t|b_{<t})·p(b^f_t|b_{<t}, b^c_t)``. The fine head conditions on the coarse
token via intra-block cross-attention. The backbone config is locked to Kronos_small (8 layers,
d_model 512, d_ff 1024, 8 heads) so the 2×2 ablation isolates the tokenizer claim. Realized base
param count lands ~21.30M; MTP depths are an additive, separately-reported configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from trikaal.constants import FSQ_V_C, FSQ_V_F
from trikaal.model.attention import RMSNorm
from trikaal.model.block import TransformerBlock
from trikaal.model.cross_attn import CoarseToFineCrossAttn
from trikaal.model.embeddings import TemporalEmbedding, TokenEmbedding
from trikaal.model.mtp import MTPHeads


@dataclass
class BackboneOutput:
    h_final: Tensor  # [B, L, d_model] post-final-RMSNorm trunk state
    logits_c: Tensor  # [B, L, V_c]
    logits_f: Tensor  # [B, L, V_f]
    coarse_cond: Tensor  # [B, L] coarse ids the fine head conditioned on


class TrikaalAR(nn.Module):
    def __init__(
        self,
        *,
        v_c: int = FSQ_V_C,
        v_f: int = FSQ_V_F,
        n_layers: int = 8,
        d_model: int = 512,
        d_ff: int = 1024,
        n_heads: int = 8,
        ffn_dropout: float = 0.25,
        resid_dropout: float = 0.25,
        attn_dropout: float = 0.1,
        token_dropout: float = 0.1,
        mtp_depths: int = 4,
        mtp_mu: float = 0.3,
        coarse_sample_prob: float = 1.0,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self._config = {
            "v_c": v_c,
            "v_f": v_f,
            "n_layers": n_layers,
            "d_model": d_model,
            "d_ff": d_ff,
            "n_heads": n_heads,
            "ffn_dropout": ffn_dropout,
            "resid_dropout": resid_dropout,
            "attn_dropout": attn_dropout,
            "token_dropout": token_dropout,
            "mtp_depths": mtp_depths,
            "mtp_mu": mtp_mu,
            "coarse_sample_prob": coarse_sample_prob,
            "max_len": max_len,
        }
        self.v_c, self.v_f, self.d_model = v_c, v_f, d_model
        self.mtp_mu = mtp_mu
        # §6.3: fraction of the time the fine head is conditioned on the SAMPLED coarse (vs
        # teacher-forced) during training. Default 1.0 = fully sampled from step 0 (Kronos parity).
        self.coarse_sample_prob = coarse_sample_prob
        self.token_embed = TokenEmbedding(v_c, v_f, d_model, token_dropout)
        self.temporal = TemporalEmbedding(d_model)
        self.resid_drop = nn.Dropout(resid_dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model,
                n_heads,
                d_ff,
                causal=True,
                ffn_kind="silu",
                resid_dropout=resid_dropout,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
                max_len=max_len,
            )
            for _ in range(n_layers)
        )
        self.norm_final = RMSNorm(d_model)
        self.w_c = nn.Linear(d_model, v_c, bias=False)
        self.cross = CoarseToFineCrossAttn(d_model)
        self.w_f = nn.Linear(d_model, v_f, bias=False)
        self.mtp = (
            MTPHeads(d_model, d_ff, n_heads, mtp_depths, max_len=max_len)
            if mtp_depths > 0
            else None
        )

    def get_config(self) -> dict:
        """The complete constructor kwargs — pass to ``save_checkpoint`` for a faithful reload."""
        return dict(self._config)

    # ---- trunk + heads -------------------------------------------------------------------
    def trunk(self, b_c: Tensor, b_f: Tensor, ts: Tensor) -> Tensor:
        e = self.token_embed(b_c, b_f) + self.temporal(ts)
        x = self.resid_drop(e)
        for blk in self.blocks:
            x = blk(x)
        return self.norm_final(x)

    def fine_logits(self, h: Tensor, coarse_ids: Tensor) -> Tensor:
        # query embedding REUSES CoarseEmbed (zero marginal params, §2)
        q = self.token_embed.coarse(coarse_ids)
        return self.w_f(self.cross(h, q))

    # ---- KV-cached single-step generation (the M5/M6 rollout primitive) -------------------
    def init_caches(self) -> list[dict]:
        """One persistent KV dict per backbone block; reset per generated sequence."""
        return [{"k": None, "v": None} for _ in self.blocks]

    @torch.no_grad()
    def trunk_step(
        self, b_c_t: Tensor, b_f_t: Tensor, ts_t: Tensor, caches: list[dict], pos: int
    ) -> Tensor:
        """Cached one-bar trunk: mirrors ``trunk`` at position ``pos`` with dropout off.

        Inputs are the single-bar ids/timestamp ``[B, 1]``. ``caches`` carries the per-block KV
        state across calls; ``pos`` is the absolute bar index (drives RoPE)."""
        x = self.token_embed(b_c_t, b_f_t) + self.temporal(ts_t)  # token-drop is identity in eval
        for blk, cache in zip(self.blocks, caches, strict=True):
            x = blk.step(x, cache, pos)
        return self.norm_final(x)

    @torch.no_grad()
    def step(
        self,
        b_c_t: Tensor,
        b_f_t: Tensor,
        ts_t: Tensor,
        caches: list[dict],
        pos: int,
        *,
        sample: bool = False,
    ) -> BackboneOutput:
        """One cached AR step: feed bar ``pos``'s tokens, get its next-bar logits (coarse+fine).

        AGREEMENT WITH THE PARALLEL ``forward``, STATED AS WHAT IS ACTUALLY VERIFIED (§7 v1.6):
        the DISCRETE chain is EXACT — ``tests/model/test_rollout.py:53`` asserts ``torch.equal``
        on ``coarse_cond`` — while the CONTINUOUS values agree to **1e-4**, asserted at
        ``:54-56``, not bit-exactly. This docstring previously said "Bit-equivalent to the
        parallel ``forward`` (verified in tests)", which the cited test does not establish: prose
        over-claiming its own artifact, in the one place a later performance question turned on.
        ``sample`` draws the conditioning coarse from the softmax; else argmax (matches forward)."""
        h = self.trunk_step(b_c_t, b_f_t, ts_t, caches, pos)
        logits_c = self.w_c(h)
        if sample:
            probs = F.softmax(logits_c, dim=-1).reshape(-1, self.v_c)
            coarse_cond = torch.multinomial(probs, 1).reshape(b_c_t.shape)
        else:
            coarse_cond = logits_c.argmax(dim=-1)
        logits_f = self.fine_logits(h, coarse_cond)
        return BackboneOutput(h, logits_c, logits_f, coarse_cond)

    def _coarse_conditioning(self, logits_c: Tensor, tgt_c: Tensor) -> Tensor:
        """§6.3: condition the fine head on the SAMPLED coarse prediction during training (so it
        learns to be robust to its own coarse mistakes — the exposure-bias fix), not the
        ground-truth coarse. Gradient never flows through the discrete sample (stop-grad)."""
        if not self.training or self.coarse_sample_prob <= 0.0:
            return tgt_c
        with torch.no_grad():
            probs = F.softmax(logits_c, dim=-1).reshape(-1, self.v_c)
            sampled = torch.multinomial(probs, 1).reshape(tgt_c.shape)
            if self.coarse_sample_prob < 1.0:
                use = torch.rand(tgt_c.shape, device=tgt_c.device) < self.coarse_sample_prob
                sampled = torch.where(use, sampled, tgt_c)
        return sampled

    def forward(
        self,
        b_c: Tensor,
        b_f: Tensor,
        ts: Tensor,
        *,
        coarse_cond: Tensor | None = None,
        sample: bool = False,
    ) -> BackboneOutput:
        h = self.trunk(b_c, b_f, ts)
        logits_c = self.w_c(h)
        if coarse_cond is None:
            with torch.no_grad():
                if sample:
                    probs = F.softmax(logits_c, dim=-1).reshape(-1, self.v_c)
                    coarse_cond = torch.multinomial(probs, 1).reshape(b_c.shape)
                else:
                    coarse_cond = logits_c.argmax(dim=-1)
        logits_f = self.fine_logits(h, coarse_cond)
        return BackboneOutput(h, logits_c, logits_f, coarse_cond)

    # ---- losses --------------------------------------------------------------------------
    def compute_loss(self, b_c: Tensor, b_f: Tensor, ts: Tensor) -> dict[str, Tensor]:
        """Next-token NLL (coarse+fine) + the MTP causal-chain loss. Teacher-forced coarse."""
        h = self.trunk(b_c, b_f, ts)
        h_pred = h[:, :-1]
        tgt_c, tgt_f = b_c[:, 1:], b_f[:, 1:]
        logits_c = self.w_c(h_pred)
        cond = self._coarse_conditioning(logits_c, tgt_c)  # SAMPLED coarse during training (§6.3)
        logits_f = self.fine_logits(h_pred, cond)
        nll_coarse = F.cross_entropy(logits_c.reshape(-1, self.v_c), tgt_c.reshape(-1))
        nll_fine = F.cross_entropy(logits_f.reshape(-1, self.v_f), tgt_f.reshape(-1))
        nll_main = nll_coarse + nll_fine
        loss = nll_main

        per_depth: list[Tensor] = []
        if self.mtp is not None:
            emb = self.token_embed.fuse(b_c, b_f)
            states = self.mtp(h, emb)
            length = b_c.shape[1]
            for k, h_k in enumerate(states, start=1):
                valid = length - (k + 1)
                if valid <= 0:
                    continue
                h_kv = h_k[:, :valid]
                tc = b_c[:, k + 1 : k + 1 + valid]
                tf = b_f[:, k + 1 : k + 1 + valid]
                lc = self.w_c(h_kv)
                lf = self.fine_logits(h_kv, tc)
                d_nll = F.cross_entropy(lc.reshape(-1, self.v_c), tc.reshape(-1)) + F.cross_entropy(
                    lf.reshape(-1, self.v_f), tf.reshape(-1)
                )
                per_depth.append(d_nll)
            if per_depth:
                loss = nll_main + self.mtp_mu * (torch.stack(per_depth).mean())

        out = {
            "loss": loss,
            "nll_main": nll_main.detach(),
            "nll_coarse": nll_coarse.detach(),
            "nll_fine": nll_fine.detach(),
        }
        for i, d in enumerate(per_depth, start=1):
            out[f"mtp_depth{i}"] = d.detach()
        return out

    # ---- param accounting ----------------------------------------------------------------
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_backbone_params(self) -> int:
        """Realized base count excluding MTP depths (the ablation's reported figure)."""
        total = self.num_params()
        if self.mtp is not None:
            total -= sum(p.numel() for p in self.mtp.parameters())
        return total
