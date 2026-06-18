"""Model-batch assembly + the loader QA-only assertion (data-pipeline §7.2, gate G0).

Exactly four column groups may reach the model: ``x`` (input vector), ``mask`` (= the
strictly-causal ``m_t``), ``ts`` (temporal-embedding key), and ``raw`` (vol-scaling hook).
Everything else — ``is_stale``, ``dq_flags``, ``vol_recon_err``, ``segment_id``, ``symbol`` —
is provenance/QA and is NEVER materialised into a model-visible tensor or OR-ed into the mask.
This module makes that guarantee a runtime invariant, not just a documentation claim.
"""

from __future__ import annotations

import numpy as np

from trikaal.data.features import PerBarOutput

_QA_ONLY_KEYS = ("is_stale", "dq_flags", "vol_recon_err", "segment_id", "symbol")
_MODEL_KEYS = ("x", "mask", "ts", "raw")


def build_model_batch(out: PerBarOutput) -> dict[str, np.ndarray]:
    """Select ONLY the model-visible tensors; QA-only columns are structurally excluded."""
    batch = {
        "x": out.x,  # [T,16] float32
        "mask": out.m,  # [T,16] uint8 — strictly-causal m_t
        "ts": out.ts,  # [T] int64
        "raw": out.raw.astype(np.float32),  # [T,2] vol-hook
    }
    assert_qa_columns_excluded(batch)
    return batch


def assert_qa_columns_excluded(batch: dict[str, np.ndarray]) -> None:
    """Hard assertion: no QA-only column reaches the model; only model keys are present."""
    for key in _QA_ONLY_KEYS:
        assert key not in batch, f"QA-only column '{key}' must never reach the model (§7.2)"
    extra = set(batch) - set(_MODEL_KEYS)
    assert not extra, f"unexpected non-model tensors in batch: {sorted(extra)}"
    assert batch["mask"].dtype == np.uint8, "mask must be the feature-spec m_t (uint8)"
