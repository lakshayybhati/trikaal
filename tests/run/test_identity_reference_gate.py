"""THE PRE-FLIGHT IDENTITY GATE — refuse a mismatched instrument BEFORE compute, not at assembly.

**WHY THIS FILE EXISTS.** Until now the only pre-spend identity check was
``identity_placeholder_failures``, which rejects PLACEHOLDER VALUES and nothing else — its own
docstring says "a placeholder is not a refusal *across* shards". A box carrying driver ``580.x``,
a different glibc, or a different 4090 SKU has all-REAL values, passes the pre-flight gate,
computes its unit, and is caught only at assembly by ``verdict.provenance_failures`` — after every
unit has been paid for. Survivable at three boxes rented in one driver-filtered batch and
hand-verified. NOT survivable at 25 units across many boxes over days.

**THE CASE THAT MUST PASS IS AS IMPORTANT AS THE ONES THAT MUST FAIL.** The live pool ran kernel
5.15.0-52 on two boxes beside 6.8.0-101 on a third, with identical ``platform_abi``. A gate that
refused there would have killed a healthy matrix — which is exactly why §7 v1.6.29 split the key.
A gate is only as good as its ability to tell those two situations apart.
"""

from __future__ import annotations

from trikaal.utils.provenance import PROVENANCE_IDENTITY_KEYS, identity_reference_failures

# The live pool's actual stamp, so the fixture is the thing the gate will really see.
REFERENCE = {
    "image": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
    "git_commit": "95fc0177b9b3cab4f38044431018a15f176e178c",
    "steps_stage1": 26003,
    "steps_stage2": 26003,
    "lockfile_sha256": "bd1297e1ba67afafc5c3edb3559df2fcdd17f42f1b715b914c3aa69f0fe2dff4",
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "cuda_build": "13.0",
    "driver_version": "590.48.01",
    "torch": "2.12.1+cu130",
    "numpy": "2.4.6",
    "python": "3.11.10",
    "platform_abi": "x86_64-with-glibc2.35",
    "attention_mode": "sdpa_deterministic",
    "deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    # RECORDED, NEVER COMPARED (§7 v1.6.29) — deliberately present in the fixture so the test
    # proves the gate ignores it rather than proving it was never given the chance.
    "platform": "Linux-5.15.0-52-generic-x86_64-with-glibc2.35",
    "device": "cuda",
}


def test_DISCRIMINATION_an_identical_box_passes():
    """If this failed every refusal below would be vacuous."""
    assert identity_reference_failures(dict(REFERENCE), REFERENCE) == []


def test_a_different_DRIVER_is_refused():
    """The concrete fan-out hazard: a box that is real in every field and simply not ours."""
    box = {**REFERENCE, "driver_version": "580.159.03"}
    fails = identity_reference_failures(box, REFERENCE)
    assert len(fails) == 1
    assert "driver_version" in fails[0] and "580.159.03" in fails[0] and "590.48.01" in fails[0]


def test_a_different_GPU_is_refused():
    box = {**REFERENCE, "gpu_name": "NVIDIA A100-SXM4-40GB"}
    fails = identity_reference_failures(box, REFERENCE)
    assert len(fails) == 1 and "gpu_name" in fails[0]


def test_THE_KERNEL_MAY_DIFFER_AND_MUST_PASS():
    """★ r2's REAL situation, and the reason §7 v1.6.29 split the key.

    r0/r1 ran ``Linux-5.15.0-52-...`` and r2 ran ``Linux-6.8.0-101-...``; ``platform_abi`` was
    ``x86_64-with-glibc2.35`` on all three. Comparing the full ``platform`` string would have
    refused a healthy pool on a kernel patch level that changes no ABI. This is the case a naive
    "compare everything you recorded" gate gets wrong."""
    box = {**REFERENCE, "platform": "Linux-6.8.0-101-generic-x86_64-with-glibc2.35"}
    assert identity_reference_failures(box, REFERENCE) == []


def test_a_different_ABI_IS_refused():
    """The other half of the split: the part of `platform` that DOES bear on identity."""
    box = {**REFERENCE, "platform_abi": "x86_64-with-glibc2.31"}
    fails = identity_reference_failures(box, REFERENCE)
    assert len(fails) == 1 and "platform_abi" in fails[0]


def test_EVERY_identity_key_is_compared_and_NOTHING_ELSE_IS():
    """The gate compares EXACTLY ``PROVENANCE_IDENTITY_KEYS`` — no more, no fewer.

    NO FEWER: differ on each key in turn and require exactly one failure naming it. A key silently
    omitted from the comparison is a key that protects nothing, which is the F3 defect in a new
    costume.
    NO MORE: differ on a recorded-but-not-identity key (`platform`, `device`) and require silence.
    Over-comparing is not "safer" — it refuses healthy boxes, which is how a fan-out dies.

    This is asserted directly rather than by trusting that this gate and
    ``verdict.provenance_failures`` were kept in step by hand."""
    for key in PROVENANCE_IDENTITY_KEYS:
        box = {**REFERENCE, key: "DEFINITELY-NOT-THE-REFERENCE-VALUE"}
        fails = identity_reference_failures(box, REFERENCE)
        assert len(fails) == 1, f"{key} produced {len(fails)} failures, expected exactly 1"
        assert key in fails[0], f"the failure for {key} does not name it: {fails[0]}"

    for key in ("platform", "device"):
        box = {**REFERENCE, key: "SOMETHING-ELSE-ENTIRELY"}
        assert identity_reference_failures(box, REFERENCE) == [], (
            f"{key!r} is RECORDED, NEVER COMPARED — comparing it refuses healthy boxes"
        )


def test_a_MISSING_key_on_the_box_is_refused():
    """An absent key is not a matching key. ``None != "590.48.01"`` must refuse, or a stamp that
    failed to resolve slips through as agreement."""
    box = {k: v for k, v in REFERENCE.items() if k != "driver_version"}
    fails = identity_reference_failures(box, REFERENCE)
    assert len(fails) == 1 and "driver_version" in fails[0]


def test_the_gate_compares_the_SAME_KEY_SET_as_the_assembly_gate():
    """If the pre-flight gate and ``provenance_failures`` compared different sets, a run could pass
    the cheap gate and still be doomed at the expensive one — which is the whole failure this
    change exists to prevent, merely moved."""
    import inspect

    from trikaal.eval.verdict import provenance_failures

    assert "PROVENANCE_IDENTITY_KEYS" in inspect.getsource(provenance_failures)
    assert "PROVENANCE_IDENTITY_KEYS" in inspect.getsource(identity_reference_failures)
