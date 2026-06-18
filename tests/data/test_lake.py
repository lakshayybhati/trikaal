"""Parquet lake + DuckDB round-trip (§3/§4) — on the synthetic fixture, no network."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from trikaal.data.config import synthetic_test_config
from trikaal.data.features import compute_features
from trikaal.data.lake import assemble_window, connect, write_lake
from trikaal.data.synthetic import make_synthetic_stream


def test_parquet_duckdb_round_trip(tmp_path):
    stream, _ = make_synthetic_stream(seed=3, n_bars=1500)
    out = compute_features(stream, synthetic_test_config())

    root = tmp_path / "bars"
    write_lake(out, "SYNTH", root=root)
    assert list(root.rglob("*.parquet")), "no parquet partitions written"

    con = connect(root)
    lo, hi = int(out.ts.min()), int(out.ts.max()) + 1
    data = assemble_window(con, "SYNTH", lo, hi)

    # exact round-trip of the model-visible + provenance columns, in bar-open order
    assert np.array_equal(data["x"], out.x)
    assert np.array_equal(data["mask"], out.m)
    assert np.array_equal(data["ts"], out.ts)
    assert np.array_equal(data["segment_id"], out.segment_id)
    assert bool(np.all(np.diff(data["ts"]) > 0))


def test_assemble_window_is_split_bounded(tmp_path):
    """The assembly query must return only bars within [lo, hi) — the temporal firewall."""
    stream, _ = make_synthetic_stream(seed=4, n_bars=1500)
    out = compute_features(stream, synthetic_test_config())
    write_lake(out, "SYNTH", root=tmp_path / "bars")
    con = connect(tmp_path / "bars")
    mid = int(out.ts[len(out) // 2])
    left = assemble_window(con, "SYNTH", int(out.ts.min()), mid)
    assert left["ts"].max() < mid
    assert left["ts"].shape[0] == int((out.ts < mid).sum())
