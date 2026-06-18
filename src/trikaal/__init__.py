"""Trikaal — a microstructure-aware FSQ tokenizer for crypto K-line foundation models.

This package is built against the v1 blueprint spec
(``docs/superpowers/specs/2026-06-18-trikaal-v1-design.md``), which is the single
source of truth. The current milestone is the **synthetic vertical slice**: prove
the model + harness architecture trains and is leak-free on synthetic data with the
pre-flight gates G0 (unit/causal-safety), G1 (overfit-a-single-batch), and G2
(determinism smoke) all green — before any real Binance ingest.
"""

from trikaal._version import __version__

__all__ = ["__version__"]
