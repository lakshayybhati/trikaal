"""The PRE-FLIGHT identity gate must not refuse itself on CUDA.

BUG FIX, NOT A SPEC CHANGE, so no §7 tag is minted here: no gate value, pin or threshold moves —
this makes an already-specified gate REACHABLE. The prereg log is the writer's, and a builder
minting a number they have not blessed is the same defect one seat over.

THE DEFECT THIS EXISTS FOR. ``m6_money_run.py`` stamped its pre-flight provenance with
``run_provenance(args.device)``, whose ``attention_mode`` defaults to ``"unknown"``, and the very
next line refuses any identity key reading ``"unknown"`` on a CUDA device. ``--device cuda`` could
therefore never clear the gate — **the money run was unrunnable on a GPU** — and 727 tests could
not see it, because not one of them executes the driver on a CUDA device. It was found by the P2
dry run on a rented 4090 (exit 1), before a training step was paid for.

WHY THIS TEST CAN RUN IN CI AT ALL. ``resolve_attention_mode`` returns the SDPA fallback without
consulting CUDA whenever ``prefer_flash`` is false, so ``preflight_provenance("cuda")`` produces a
real ``attention_mode`` on a CPU-only runner. The pre-fix source returns ``"unknown"`` here and
fails — which is the only reason this counts as a check (remediation rule: a gate is closed only
when something has watched it fail).

WHAT IT DELIBERATELY DOES NOT ASSERT. The other CUDA-only keys (``gpu_name``, ``cuda_build``,
``driver_version``) are legitimately ``"unavailable"`` on a CPU runner, so this asserts on the
``attention_mode`` failure specifically rather than on an empty failure list — an assertion that
demanded zero failures would fail in CI for an unrelated and correct reason, and would then be
"fixed" by weakening it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from trikaal.model.attention_mode import MODE_FLASH2, MODE_SDPA
from trikaal.utils.provenance import UNAVAILABLE, identity_placeholder_failures

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "scripts/m6_money_run.py"


def _driver():
    spec = importlib.util.spec_from_file_location("m6_money_run", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _attention_failures(prov: dict) -> list[str]:
    return [m for m in identity_placeholder_failures(prov) if "'attention_mode'" in m]


def test_preflight_stamp_carries_a_real_attention_mode_on_cuda():
    """The regression itself: the pre-fix source returns "unknown" and this fails."""
    prov = _driver().preflight_provenance("cuda")
    assert prov["attention_mode"] in (MODE_SDPA, MODE_FLASH2), prov["attention_mode"]
    assert prov["attention_mode"] not in (UNAVAILABLE, "unknown", "None", "")
    assert _attention_failures(prov) == []


def test_the_gate_still_refuses_a_placeholder_attention_mode():
    """FIXTURE DISCRIMINATION — the fix must not have neutered the thing it was protecting.

    Without this, the test above would pass just as well against a gate that had been taught to
    ignore ``attention_mode`` entirely, which is the cheap wrong fix.
    """
    prov = _driver().preflight_provenance("cuda") | {"attention_mode": "unknown"}
    assert _attention_failures(prov), "the gate no longer refuses a placeholder attention_mode"


def test_cpu_is_exempt():
    """On CPU the CUDA-only keys are honestly unresolvable; refusing there would be wrong."""
    assert identity_placeholder_failures(_driver().preflight_provenance("cpu")) == []
