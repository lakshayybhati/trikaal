"""Shared fixtures: the tiny seeded synthetic stream and feature windows.

All tests run on synthetic data (no real Parquet), so the suite finishes in CI minutes.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from trikaal.data.config import synthetic_test_config
from trikaal.data.features import compute_features
from trikaal.data.synthetic import make_synthetic_stream
from trikaal.utils.provenance import _git_commit


@pytest.fixture(scope="session", autouse=True)
def _stable_git_commit_for_the_whole_run() -> str:
    """Pin ``git_commit`` for the duration of one suite run, so a COMMIT cannot turn it red.

    THE DEFECT THIS CLOSES, and it is a real one that happened. ``provenance._git_commit`` shells
    out to ``git rev-parse HEAD`` at the moment each artifact is stamped. ``git_commit`` is an
    IDENTITY key, so a verdict assembled from artifacts stamped at two different HEADs is REFUSED
    — correctly, because on the money path that means two shards ran different code. But inside a
    10-minute test run on a branch two seats share, a commit landing mid-suite splits one
    session-scoped fixture's artifacts across two HEADs and the refusal fires on a defect that
    does not exist. The suite went red exactly this way.

    THE GUARD IS RIGHT AND IS NOT WEAKENED. What changes is only that the SUITE stops using the
    live repository as an input: ``_git_commit`` already prefers ``TRIKAAL_GIT_COMMIT`` (the money
    path sets it, because a rented box has no ``.git``), so pinning it here resolves the revision
    ONCE and every stamp in the run agrees. Tests that need a different value still monkeypatch
    the env var, which continues to win.

    Resolved through ``_git_commit`` rather than by shelling out again, so a checkout-less tree
    pins ``"unavailable"`` and is equally stable.
    """
    pinned = _git_commit()
    before = os.environ.get("TRIKAAL_GIT_COMMIT")
    os.environ["TRIKAAL_GIT_COMMIT"] = pinned
    try:
        yield pinned
    finally:
        if before is None:
            os.environ.pop("TRIKAAL_GIT_COMMIT", None)
        else:
            os.environ["TRIKAAL_GIT_COMMIT"] = before


@pytest.fixture(scope="session")
def causal_cfg():
    """Small-window feature config for the fast causal-safety gate."""
    return synthetic_test_config(
        half_life_fast=48,
        half_life_slow=96,
        n_warm=12,
        w_tau=20,
        vol_window=24,
        n_warm_vol=12,
    )


@pytest.fixture(scope="session")
def causal_data(causal_cfg):
    """(stream, anom, cfg) — a small fixture sized so the EXHAUSTIVE truncation sweep stays fast."""
    stream, anom = make_synthetic_stream(seed=7, n_bars=600)
    return stream, anom, causal_cfg


@pytest.fixture(scope="session")
def feature_window():
    """A dense [L, 16] feature tensor + mask drawn from one clean segment (for model gates).

    Returns ``(x, mask)`` as float32 / uint8 numpy arrays of shape ``[L, 16]`` / ``[L, 16]``,
    taken from the longest contiguous segment past warm-up so the vectors carry real signal.
    """
    cfg = synthetic_test_config(half_life_fast=48, half_life_slow=96, n_warm=12, w_tau=20)
    stream, _ = make_synthetic_stream(seed=11, n_bars=2400, inject_anomalies=True)
    out = compute_features(stream, cfg)
    # longest segment
    seg_ids, counts = np.unique(out.segment_id, return_counts=True)
    best = seg_ids[int(np.argmax(counts))]
    idx = np.nonzero(out.segment_id == best)[0]
    warm = 100  # skip z-score warm-up
    idx = idx[warm:]
    L = 256
    assert idx.size >= L, f"need >= {L} post-warm-up bars, got {idx.size}"
    sel = idx[:L]
    return out.x[sel].copy(), out.m[sel].copy()
