"""TokenizerAE — encoder → W_in → FSQ → W_out → decoder, with the hierarchical loss (§4).

Trained stage-1 standalone (reconstruction only). The token stream ``(cidx, fidx)`` is the sole
interface to the AR backbone; the backbone never sees ``x`` or ``z``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from trikaal.constants import FSQ_LEVELS, MICRO_DIMS_IDX
from trikaal.tokenizer.bsq import BSQQuantizer
from trikaal.tokenizer.decoder import TokenizerDecoder
from trikaal.tokenizer.encoder import PointwiseEncoder, TokenizerEncoder
from trikaal.tokenizer.hierarchy import FSQQuantizer
from trikaal.tokenizer.losses import default_feature_weights, masked_mae, weighted_masked_huber


class TokenizerAE(nn.Module):
    """Tokenizer autoencoder — FSQ (the contribution) or BSQ (the Cells-1/3 baseline) bottleneck.

    The quantizer is the ONLY swap (spec §1190's symmetry guarantee): identical encoder/decoder
    shells and ``d_model→D`` / ``D→d_model`` projections; the loss extra is arm-specific via
    ``quant.aux_loss(z)`` — ``gamma_reg`` (NO commitment) for FSQ, ``λ·L_quant`` for BSQ.
    ``n_features=7`` is the OHLCV-only ablation arm (input dims REMOVED, never zero-filled).
    """

    def __init__(
        self,
        levels: tuple[int, ...] | list[int] = FSQ_LEVELS,
        *,
        quantizer: str = "fsq",
        bsq_bits: tuple[int, int] | list[int] = (10, 10),
        bsq_lambda: float = 0.25,
        n_features: int = 16,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.0,
        per_stage_scale: bool = True,
        encoder_causal: bool = False,
        fine_pointwise: bool = False,
        micro_point_weight: float = 1.0,
        point_loss_coef: float = 1.0,
        huber_delta: float = 1.0,
        finance_weighted: bool = False,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        if quantizer not in ("fsq", "bsq"):
            raise ValueError(f"quantizer must be 'fsq' or 'bsq', got {quantizer!r}")
        self._config = {
            "levels": list(levels),
            "quantizer": quantizer,
            "bsq_bits": list(bsq_bits),
            "bsq_lambda": bsq_lambda,
            "n_features": n_features,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "dropout": dropout,
            "per_stage_scale": per_stage_scale,
            "encoder_causal": encoder_causal,
            "fine_pointwise": fine_pointwise,
            "micro_point_weight": micro_point_weight,
            "point_loss_coef": point_loss_coef,
            "huber_delta": huber_delta,
            "finance_weighted": finance_weighted,
            "max_len": max_len,
        }
        self.encoder = TokenizerEncoder(
            n_features, d_model, n_layers, n_heads, d_ff, dropout, encoder_causal, max_len
        )
        if quantizer == "bsq":
            self.quant: nn.Module = BSQQuantizer(bsq_bits[0], bsq_bits[1], bsq_lambda)
        else:
            self.quant = FSQQuantizer(levels, per_stage_scale)
        self.w_in = nn.Linear(d_model, self.quant.dim, bias=True)
        self.fine_pointwise = fine_pointwise
        if fine_pointwise:
            # §7 v1.4: the fine subtoken is a PER-BAR encoding of bar t's own features —
            # a pointwise branch feeds the fine latent dims; the contextual (causal) encoder
            # keeps the coarse dims. Identical for both quantizers (bpt parity untouched).
            self.point_encoder = PointwiseEncoder(n_features, d_model, d_ff, 2, dropout)
            self.w_in_fine = nn.Linear(d_model, self.quant.dim, bias=True)
            # The per-bar BOTTLENECK leg: bar t's fine code alone must reconstruct bar t's
            # features (pointwise decode head — zero cross-bar). Without it the optimizer
            # routes per-bar state through the contextual coarse smear and leaves the fine
            # bits on other structure (measured: pointwise-encoder-only legibility 0.5202
            # ~= chance while window recon of the state was strong). Train-time only; the
            # AR interface and the main decode path are unchanged.
            n_fine = len(self.quant.fine_idx)
            self.point_decoder = nn.Sequential(
                nn.Linear(n_fine, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, n_features),
            )
        self.w_out = nn.Linear(self.quant.dim, d_model, bias=True)
        self.decoder = TokenizerDecoder(
            n_features, d_model, n_layers, n_heads, d_ff, dropout, False, max_len
        )
        self.huber_delta = huber_delta
        self.quantizer = quantizer
        self.n_features = n_features
        self.v_c, self.v_f = self.quant.v_c, self.quant.v_f
        self.register_buffer("w_feat", default_feature_weights(finance_weighted, n_features))
        # §7 v1.4.1 (gate-2 final ruling): the per-bar BOTTLENECK leg weights THE SIX MICRO
        # DIMS as a class by micro_point_weight (lambda) — the mechanism receipt showed the
        # unweighted recon objective buys variance and covariance, never independence, so the
        # state-like micro dims are priced out of the fine budget deterministically. Identical
        # across arms and quantizers; OHLCV arms (n_features=7) carry no micro dims, so their
        # weights stay uniform. Lambda is CALIBRATED (smallest value clearing the 3-seed 0.9
        # legibility gate), pinned in conformance, and applies ONLY to the bottleneck leg —
        # the main coarse/fine losses and the decision rule are untouched.
        w_point = default_feature_weights(finance_weighted, n_features).clone()
        for i in MICRO_DIMS_IDX:
            if i < n_features:
                w_point[i] = w_point[i] * micro_point_weight
        self.register_buffer("w_feat_point", w_point)
        # The bottleneck leg's coefficient beta (weighted_masked_huber self-normalizes, so
        # in-leg weights cannot raise the LEG's own gradient share — this can). Default 1.0
        # preserves the landed v1.4 behavior exactly.
        self.point_loss_coef = point_loss_coef

    def get_config(self) -> dict:
        """The complete constructor kwargs — pass to ``save_checkpoint`` for a faithful reload."""
        return dict(self._config)

    def latent(self, x: Tensor, mask: Tensor) -> Tensor:
        z = self.w_in(self.encoder(x, mask))
        if self.fine_pointwise:
            z_f = self.w_in_fine(self.point_encoder(x, mask))
            fine = list(self.quant.fine_idx)
            z = z.clone()
            z[..., fine] = z_f[..., fine]
        return z

    def encode_tokens(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """``encode(x:[B,L,16], mask) -> (b_c:[B,L], b_f:[B,L])`` — the AR-backbone interface."""
        _z_hat, _codes, cidx, fidx = self.quant(self.latent(x, mask))
        return cidx, fidx

    def decode_latent(self, z_hat: Tensor) -> Tensor:
        return self.decoder(self.w_out(z_hat))

    def _group_radix(self, group: list[int]) -> list[int]:
        """Per-dim radices for a coarse/fine group: FSQ level counts, or 2 for every BSQ bit."""
        if self.quantizer == "bsq":
            return [2] * len(group)
        return [self.quant.levels[i] for i in group]

    def decode_tokens(self, cidx: Tensor, fidx: Tensor) -> Tensor:
        """``(b_c, b_f) -> x_hat`` — inverse of ``encode_tokens`` (the AR rollout→features path).

        Unflatten the mixed-radix coarse/fine ids back to per-dim codes, dequantize to the exact
        STE forward value the decoder consumes — FSQ grid value ``code − (L_i−1)/2``; BSQ corner
        ``(2·bit − 1)/√k`` — and decode. ``decode_tokens(*encode_tokens(x, m))`` reproduces
        ``forward(x, m)["x_hat"]``.
        """
        q = self.quant
        codes = torch.zeros((*cidx.shape, q.dim), dtype=torch.long, device=cidx.device)
        for ids, group in ((cidx, q.coarse_idx), (fidx, q.fine_idx)):
            radix = self._group_radix(list(group))
            rem = ids.clone()
            for pos in reversed(range(len(group))):  # big-endian mixed-radix unflatten
                lvl = radix[pos]
                codes[..., group[pos]] = rem % lvl
                rem = rem // lvl
        if self.quantizer == "bsq":
            z_hat = (2.0 * codes.to(torch.float32) - 1.0) / (float(q.dim) ** 0.5)
        else:
            half = (q.levels_t - 1.0) / 2.0
            z_hat = codes.to(q.levels_t.dtype) - half
        return self.decode_latent(z_hat)

    def forward(self, x: Tensor, mask: Tensor) -> dict[str, Tensor]:
        z = self.latent(x, mask)
        z_hat, codes, cidx, fidx = self.quant(z)
        if self.fine_pointwise:
            # §7 v1.4.1 objective separation: the WINDOW losses see the fine channels
            # DETACHED — the decoder still trains on (and consumes) them, but the fine
            # ENCODER is shaped exclusively by the per-bar bottleneck leg. Measured basis:
            # with the window pull attached, the window objective forces all four fine
            # channels into shared-component duty and the in-leg micro weighting is inert
            # at every lambda (per-dim receipts identical from lambda=2 to lambda=1e6);
            # detached, lambda regains monotone effect. Values are unchanged (detach only
            # stops gradient), so eval and decode paths are identical.
            fine = list(self.quant.fine_idx)
            z_win = z_hat.clone()
            z_win[..., fine] = z_hat[..., fine].detach()
        else:
            z_win = z_hat
        x_hat = self.decode_latent(z_win)
        x_hat_coarse = self.decode_latent(self.quant.coarse_only(z_win))

        keep = 1.0 - mask.to(x.dtype)
        loss_fine = weighted_masked_huber(x_hat, x, keep, self.w_feat, self.huber_delta)
        loss_coarse = weighted_masked_huber(x_hat_coarse, x, keep, self.w_feat, self.huber_delta)
        # arm-specific extra: FSQ → gamma_reg ONLY (NO commitment term — the headline
        # simplification); BSQ → λ·L_quant (spec §1190)
        loss = loss_coarse + loss_fine + self.quant.aux_loss(z)
        if self.fine_pointwise:
            # §7 v1.4 per-bar bottleneck: reconstruct bar t from its OWN fine code alone.
            x_hat_point = self.point_decoder(z_hat[..., list(self.quant.fine_idx)])
            loss_point = weighted_masked_huber(
                x_hat_point, x, keep, self.w_feat_point, self.huber_delta
            )
            loss = loss + self.point_loss_coef * loss_point

        with torch.no_grad():
            recon_mae = masked_mae(x_hat, x, keep)
            coarse_mae = masked_mae(x_hat_coarse, x, keep)
        return {
            "loss": loss,
            "loss_coarse": loss_coarse,
            "loss_fine": loss_fine,
            "x_hat": x_hat,
            "x_hat_coarse": x_hat_coarse,
            "codes": codes,
            "cidx": cidx,
            "fidx": fidx,
            "z": z,
            "z_hat": z_hat,
            "recon_mae": recon_mae,
            "coarse_mae": coarse_mae,
        }
