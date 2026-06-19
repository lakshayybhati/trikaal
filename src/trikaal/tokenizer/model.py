"""TokenizerAE — encoder → W_in → FSQ → W_out → decoder, with the hierarchical loss (§4).

Trained stage-1 standalone (reconstruction only). The token stream ``(cidx, fidx)`` is the sole
interface to the AR backbone; the backbone never sees ``x`` or ``z``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from trikaal.constants import FSQ_LEVELS
from trikaal.tokenizer.decoder import TokenizerDecoder
from trikaal.tokenizer.encoder import TokenizerEncoder
from trikaal.tokenizer.hierarchy import FSQQuantizer
from trikaal.tokenizer.losses import default_feature_weights, masked_mae, weighted_masked_huber


class TokenizerAE(nn.Module):
    """Microstructure-aware FSQ tokenizer autoencoder."""

    def __init__(
        self,
        levels: tuple[int, ...] | list[int] = FSQ_LEVELS,
        *,
        n_features: int = 16,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.0,
        per_stage_scale: bool = True,
        encoder_causal: bool = False,
        huber_delta: float = 1.0,
        finance_weighted: bool = False,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self._config = {
            "levels": list(levels),
            "n_features": n_features,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "dropout": dropout,
            "per_stage_scale": per_stage_scale,
            "encoder_causal": encoder_causal,
            "huber_delta": huber_delta,
            "finance_weighted": finance_weighted,
            "max_len": max_len,
        }
        self.encoder = TokenizerEncoder(
            n_features, d_model, n_layers, n_heads, d_ff, dropout, encoder_causal, max_len
        )
        self.quant = FSQQuantizer(levels, per_stage_scale)
        self.w_in = nn.Linear(d_model, self.quant.dim, bias=True)
        self.w_out = nn.Linear(self.quant.dim, d_model, bias=True)
        self.decoder = TokenizerDecoder(
            n_features, d_model, n_layers, n_heads, d_ff, dropout, False, max_len
        )
        self.huber_delta = huber_delta
        self.v_c, self.v_f = self.quant.v_c, self.quant.v_f
        self.register_buffer("w_feat", default_feature_weights(finance_weighted))

    def get_config(self) -> dict:
        """The complete constructor kwargs — pass to ``save_checkpoint`` for a faithful reload."""
        return dict(self._config)

    def latent(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.w_in(self.encoder(x, mask))

    def encode_tokens(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """``encode(x:[B,L,16], mask) -> (b_c:[B,L], b_f:[B,L])`` — the AR-backbone interface."""
        _z_hat, _codes, cidx, fidx = self.quant(self.latent(x, mask))
        return cidx, fidx

    def decode_latent(self, z_hat: Tensor) -> Tensor:
        return self.decoder(self.w_out(z_hat))

    def forward(self, x: Tensor, mask: Tensor) -> dict[str, Tensor]:
        z = self.latent(x, mask)
        z_hat, codes, cidx, fidx = self.quant(z)
        x_hat = self.decode_latent(z_hat)
        x_hat_coarse = self.decode_latent(self.quant.coarse_only(z_hat))

        keep = 1.0 - mask.to(x.dtype)
        loss_fine = weighted_masked_huber(x_hat, x, keep, self.w_feat, self.huber_delta)
        loss_coarse = weighted_masked_huber(x_hat_coarse, x, keep, self.w_feat, self.huber_delta)
        loss = loss_coarse + loss_fine + self.quant.gamma_reg()  # NO commitment term (FSQ)

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
