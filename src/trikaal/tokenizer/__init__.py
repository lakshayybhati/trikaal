"""===== THE HEADLINE SUBSYSTEM ===== the microstructure-aware FSQ tokenizer (§4).

Finite Scalar Quantization replaces Kronos's BSQ: no commitment loss, no codebook-collapse
mode, bounded quantization error. A small Transformer autoencoder maps each 16-dim bar vector
to a single fused token via a per-dimension FSQ grid, with a derived coarse→fine hierarchy.
"""

from trikaal.tokenizer.model import TokenizerAE

__all__ = ["TokenizerAE"]
