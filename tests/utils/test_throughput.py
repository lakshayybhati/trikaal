"""KATs for the throughput record (prereg §7 v1.6 Finding 0).

The defect these guard against is specific: a cost basis that exists only as a prose string, whose
cited receipt is NaN, and which conflates a compute-only micro-benchmark with end-to-end training
throughput. Each test below is one of those failure modes made impossible.
"""

from __future__ import annotations

import math

import pytest

from trikaal.utils.throughput import (
    BASIS_COMPUTE_ONLY,
    BASIS_END_TO_END,
    BASIS_UNMEASURED,
    is_costable,
    throughput_record,
    unmeasured_record,
)


def test_rates_are_numbers_not_strings():
    """The whole point: bars/s is a float on a manifest, never a formatted label."""
    r = throughput_record(steps=50, batch=32, seq_len=512, wall_s=17.35, basis=BASIS_COMPUTE_ONLY)
    assert isinstance(r["steps_per_s"], float)
    assert isinstance(r["bars_per_s"], float)
    assert r["measured"] is True


def test_bars_per_s_arithmetic_matches_the_47k_derivation():
    """The exact numbers from m6_attention_bench.json, which is where 47.2k really came from."""
    r = throughput_record(
        steps=50,
        batch=32,
        seq_len=512,
        wall_s=50 / 2.8819876814331926,
        basis=BASIS_COMPUTE_ONLY,
    )
    assert r["steps_per_s"] == pytest.approx(2.8819876814331926, rel=1e-12)
    assert r["bars_per_s"] == pytest.approx(47218.48617260143, rel=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0, "batch": 32, "seq_len": 512, "wall_s": 10.0},
        {"steps": 50, "batch": 32, "seq_len": 512, "wall_s": 0.0},
        {"steps": 50, "batch": 32, "seq_len": 512, "wall_s": None},
        {"steps": None, "batch": 32, "seq_len": 512, "wall_s": 10.0},
        {"steps": 50, "batch": 32, "seq_len": 512, "wall_s": float("nan")},
    ],
)
def test_absent_inputs_give_nan_and_measured_false_never_zero(kwargs):
    """``train_wall_s = 0`` is what made the rehearsal manifest read as a measurement of nothing.

    An unmeasurable rate must come back NaN with ``measured=False`` — never 0.0, which a reader
    (or a downstream division) can mistake for a datum.
    """
    r = throughput_record(basis=BASIS_END_TO_END, **kwargs)
    assert r["measured"] is False
    assert math.isnan(r["bars_per_s"])
    assert math.isnan(r["steps_per_s"])


def test_unknown_basis_is_rejected():
    """A closed vocabulary is what stops a new caller inventing an unlabelled basis."""
    with pytest.raises(ValueError, match="basis must be one of"):
        throughput_record(steps=1, batch=1, seq_len=1, wall_s=1.0, basis="fast_enough")


def test_compute_only_is_not_costable_even_when_measured():
    """THE conflation behind Finding 0: an upper bound quoted as the cost basis."""
    compute = throughput_record(
        steps=50, batch=32, seq_len=512, wall_s=17.35, basis=BASIS_COMPUTE_ONLY
    )
    end2end = throughput_record(
        steps=3000, batch=32, seq_len=512, wall_s=900.0, basis=BASIS_END_TO_END
    )
    assert compute["measured"] is True  # it IS a real measurement...
    assert is_costable(compute) is False  # ...of something that cannot price a run
    assert is_costable(end2end) is True


def test_unmeasured_record_is_structurally_present_and_numerically_absent():
    r = unmeasured_record(reason="eval-only invocation — no training to time")
    assert r["basis"] == BASIS_UNMEASURED
    assert r["measured"] is False
    assert math.isnan(r["bars_per_s"])
    assert is_costable(r) is False
    assert "eval-only" in r["note"]


def test_excludes_travels_with_every_record():
    """A reader must not be able to lose what the basis leaves out."""
    for basis in (BASIS_COMPUTE_ONLY, BASIS_END_TO_END, BASIS_UNMEASURED):
        r = throughput_record(steps=10, batch=2, seq_len=4, wall_s=1.0, basis=basis)
        assert r["excludes"]
        assert isinstance(r["excludes"], str)


def test_is_costable_rejects_none_and_empty():
    assert is_costable(None) is False
    assert is_costable({}) is False
