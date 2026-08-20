"""Trikaal — a tokenizer study: microstructure-aware FSQ quantization for crypto K-lines.

STATUS: the study is complete and the experiment it designed did not run. M1-M5 closed; M6's
five-cell ablation was stopped on 2026-08-12 by its own pre-registered micro-legibility gate, and
the primary result is the mechanism finding. Cell 1 (BSQ, OHLCV-only, seeds 0/2/4) is the only arm
with scored artifacts and is what was published. See ``docs/MODEL_CARD.md``.

★ THIS DOCSTRING PREVIOUSLY SAID the current milestone was "the synthetic vertical slice ...
before any real Binance ingest", and that the blueprint spec is "the single source of truth".
Both were false and both were load-bearing: an agent reads the package docstring first, and the
first claim would have it rebuild a lake that exists while the second points it at a frozen design
document that still authorises pulling third-party Kronos weights. The invariants in
``docs/ENGINEERING.md`` outrank the spec; the spec is a frozen record of what was intended.
"""

from trikaal._version import __version__

__all__ = ["__version__"]
