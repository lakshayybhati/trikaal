"""The `platform` split: refuse on arch and libc, tolerate the kernel, RECORD everything.

WHAT CHANGED AND WHAT DID NOT. ``platform.platform()`` on Linux is
``Linux-<kernel>-<arch>-with-<libc>``. It was ONE identity key, so any kernel difference refused
the whole 25-unit verdict — and vast exposes no way to filter on kernel, so the only way to find a
matched pool was to rent boxes and look. Measured: one driver cohort split three ways on kernel,
and eight boxes taken from the cheap end of the market produced eight distinct kernels. The gate
priced a multi-box pool out of the market while defending nothing, because **a kernel patch
release does not participate in GPU arithmetic**.

So the COMPARISON moved, not the RECORD:
  * ``platform_abi`` (arch + libc) is the identity key and still REFUSES on a mismatch
  * ``platform`` (the full string, kernel included) is stamped on every artifact, never compared

Every test here is written so that a relaxation which accidentally neutered the key would FAIL.
That is the fail-open class and it looks exactly like success from the inside.
"""

from __future__ import annotations

import pytest

from trikaal.utils.provenance import PROVENANCE_IDENTITY_KEYS, platform_abi, run_provenance

LINUX = "Linux-5.15.0-179-generic-x86_64-with-glibc2.35"
LINUX_REL = "5.15.0-179-generic"


# ------------------------------------------------------------------ what it strips, and does not
def test_the_kernel_is_stripped_and_nothing_else_is():
    assert platform_abi(LINUX, "Linux", LINUX_REL) == "x86_64-with-glibc2.35"


@pytest.mark.parametrize(
    ("full", "release"),
    [
        ("Linux-5.15.0-179-generic-x86_64-with-glibc2.35", "5.15.0-179-generic"),
        ("Linux-5.15.0-186-generic-x86_64-with-glibc2.35", "5.15.0-186-generic"),
        ("Linux-5.15.0-187-generic-x86_64-with-glibc2.35", "5.15.0-187-generic"),
        ("Linux-6.5.0-21-generic-x86_64-with-glibc2.35", "6.5.0-21-generic"),
        ("Linux-6.8.0-106-generic-x86_64-with-glibc2.35", "6.8.0-106-generic"),
        ("Linux-6.8.0-111-generic-x86_64-with-glibc2.35", "6.8.0-111-generic"),
        ("Linux-6.8.0-124-generic-x86_64-with-glibc2.35", "6.8.0-124-generic"),
    ],
)
def test_every_kernel_we_actually_rented_collapses_to_one_abi(full, release):
    """These are the LITERAL strings measured on rented 4090s across two batches and a probe.
    All seven differ only in kernel, so all seven must now agree — that is the whole point."""
    assert platform_abi(full, "Linux", release) == "x86_64-with-glibc2.35"


# ------------------------------------------------------------------ 1. DISCRIMINATION
def test_a_libc_difference_still_separates():
    """glibc can bear on numerics through libm. It must still refuse."""
    a = platform_abi(LINUX, "Linux", LINUX_REL)
    b = platform_abi("Linux-5.15.0-179-generic-x86_64-with-glibc2.31", "Linux", LINUX_REL)
    assert a != b, "a glibc difference collapsed to the same ABI — the key is neutered"


def test_an_arch_difference_still_separates():
    a = platform_abi(LINUX, "Linux", LINUX_REL)
    b = platform_abi("Linux-5.15.0-179-generic-aarch64-with-glibc2.35", "Linux", LINUX_REL)
    assert a != b, "an arch difference collapsed to the same ABI — the key is neutered"


# ------------------------------------------------------------------ FAILS CLOSED
def test_an_unrecognised_format_keeps_comparing_everything():
    """macOS reports platform() as macOS-26.5.1-... while system()/release() say Darwin/25.5.0,
    so the prefix does not match. The full string must come back UNCHANGED: when we cannot
    confidently identify the kernel we go on comparing what we compared before."""
    mac = "macOS-26.5.1-arm64-arm-64bit"
    assert platform_abi(mac, "Darwin", "25.5.0") == mac
    assert platform_abi("", "Linux", LINUX_REL) == ""
    # a string that IS exactly the prefix has no ABI part left; returning "" would silently
    # compare nothing, so it returns the original.
    assert platform_abi("Linux-x-", "Linux", "x") == "Linux-x-"


# ------------------------------------------------------------------ 3. STILL RECORDED
def test_the_key_set_swapped_platform_for_platform_abi():
    assert "platform_abi" in PROVENANCE_IDENTITY_KEYS
    assert "platform" not in PROVENANCE_IDENTITY_KEYS, (
        "the full string must NOT be compared — that is the change"
    )
    assert len(PROVENANCE_IDENTITY_KEYS) == 16, "the surface must stay 16 keys wide"


def test_the_full_kernel_string_is_still_stamped():
    """The relaxation must not cost us the audit trail. A reviewer has to be able to read which
    kernels ran, so the full string stays in every stamp as a NON-compared field."""
    prov = run_provenance("cpu")
    assert prov.get("platform"), "the kernel string vanished from the stamp"
    assert prov.get("platform_abi")
    assert (
        prov["platform"].endswith(prov["platform_abi"]) or prov["platform"] == prov["platform_abi"]
    ), "the ABI must be a suffix of the recorded string (or the fail-closed whole)"
