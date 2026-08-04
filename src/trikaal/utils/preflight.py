"""§7 v1.6 C-5 — RESOURCE PRECONDITIONS, checked before any compute begins.

THE ARGUMENT IS A MEASURED ONE, NOT A HYPOTHETICAL. The A9 dry run needed ~27.8 GiB on a 16.0 GiB
machine. It did **not** die: macOS absorbed the overflow in swap (11.96 GiB in use, 16.0 + 11.96 ≈
27.96 GiB against the predicted 27.8), and one unit with **zero training steps** took 2,581 s. On
rented hardware that is strictly worse than an OOM — **an OOM stops the meter; swap-thrash keeps
billing while looking like ordinary slowness**. So the requirement is asserted up front, against
the host, with both numbers in the message.

SAME PRINCIPLE AS CONFORMANCE-FIRST: every resource the run needs is checked before any compute,
not discovered thirty minutes in. Physical RAM is the right comparison rather than "available"
RAM: if the requirement exceeds what the machine physically has, the run WILL swap regardless of
what is free at that instant.

Memory strategy is a property of the HOST, not of the specification. Both strategies compute
byte-identical results — ``shuffle_micro`` is seeded on ``(symbol, seed)``, so a rebuilt window is
the same window — they differ only in how much is resident at once and how often it is rebuilt.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from trikaal.constants import N_FEATURES
from trikaal.train.arms import ARMS, arm_n_features

#: One bar of raw arrays: x float32 + mask uint8 + ts/segment/sigma/raw_ret int64/float64.
RAW_BYTES_PER_BAR = N_FEATURES * 4 + N_FEATURES * 1 + 8 * 4

STRATEGY_FAST = "fast"  # all arms for a seed resident — fewest rebuilds, most memory
STRATEGY_LOW = "low"  # one arm resident — 3x/5x rebuilds, ~2.5x less window memory
STRATEGIES = (STRATEGY_FAST, STRATEGY_LOW)


def _arm_bytes_per_bar(arm: str) -> int:
    """One bar of ONE arm's SymbolWindows: arm-selected x/mask (a COPY — ``select_arm`` returns
    ``np.ascontiguousarray``, verified, not assumed) plus segment_id and ts."""
    f = arm_n_features(arm)
    return f * 4 + f * 1 + 8 + 8


def matrix_arms() -> tuple[str, ...]:
    """The arms the CELL MATRIX actually materializes — derived from ``CELLS``, never restated.

    **§7 v1.6.27.** This read ``ARMS``, the registry of every arm that EXISTS. Adding the
    constant-micro arm for the C-12 bracket therefore inflated the fast-strategy memory
    requirement of every run by an arm no 5-cell run ever builds (84.15M bars: 27.82 -> 35.35
    GiB), and that figure chooses the memory strategy on the box. Deriving it from the cell
    registry means the number tracks the experiment instead of the vocabulary: add cell 6 to
    ``CELLS`` and the requirement rises because a run really does materialize a fourth arm.

    Imported inside the call so ``utils`` keeps no module-level edge into ``train`` (which pulls
    torch); this is preflight arithmetic, called a handful of times per run."""
    from trikaal.train.cells import CELLS

    seen = {c.arm for c in CELLS}
    return tuple(a for a in ARMS if a in seen)


def window_bytes_per_bar(strategy: str) -> int:
    arms = matrix_arms()
    if strategy == STRATEGY_FAST:
        return sum(_arm_bytes_per_bar(a) for a in arms)
    if strategy == STRATEGY_LOW:
        return max(_arm_bytes_per_bar(a) for a in arms)  # the worst single arm must fit
    raise ValueError(f"unknown memory strategy {strategy!r}; expected one of {STRATEGIES}")


def memory_requirement_bytes(n_bars: int, strategy: str) -> dict:
    """Peak resident bytes for ``n_bars`` bars under ``strategy`` — arithmetic, not a guess."""
    raw = int(n_bars) * RAW_BYTES_PER_BAR
    win = int(n_bars) * window_bytes_per_bar(strategy)
    return {
        "n_bars": int(n_bars),
        "strategy": strategy,
        "raw_bytes": raw,
        "windows_bytes": win,
        "peak_bytes": raw + win,
        "peak_gib": (raw + win) / 2**30,
    }


def physical_ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):  # pragma: no cover — non-POSIX
        return None


def free_disk_bytes(path: Path) -> int | None:
    try:
        p = Path(path)
        while not p.exists() and p != p.parent:
            p = p.parent
        return shutil.disk_usage(p).free
    except OSError:  # pragma: no cover
        return None


def vram_bytes() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:  # pragma: no cover — no CUDA in CI
        return None


def resource_failures(
    *,
    n_bars: int,
    strategy: str,
    out_dir: Path,
    disk_needed_bytes: int = 2 * 2**30,
    headroom: float = 1.0,
) -> list[str]:
    """Every unmet precondition (empty = safe to start). Fail-fast inputs, with both numbers.

    ``headroom`` = 1.0 means "the requirement must not exceed physical RAM". Anything above that
    ratio is swap territory, which is the hazard this exists to stop."""
    fails: list[str] = []
    req = memory_requirement_bytes(n_bars, strategy)
    ram = physical_ram_bytes()
    if ram is None:
        fails.append(
            "could not read physical RAM — refusing to start blind on the one resource "
            "whose exhaustion bills silently instead of stopping"
        )
    elif req["peak_bytes"] > ram * headroom:
        fails.append(
            f"MEMORY: strategy {strategy!r} needs {req['peak_gib']:.2f} GiB "
            f"({req['n_bars']:,} bars) but the host has {ram / 2**30:.2f} GiB of physical RAM. "
            f"This does not fail cleanly — it SWAPS: the measured precedent is one unit with zero "
            f"training steps taking 2,581 s. An OOM stops the meter; swap-thrash keeps billing. "
            f"Use strategy 'low' ({memory_requirement_bytes(n_bars, STRATEGY_LOW)['peak_gib']:.2f} "
            f"GiB) or a host with more RAM."
        )
    free = free_disk_bytes(out_dir)
    if free is None:
        fails.append(f"DISK: could not stat {out_dir}")
    elif free < disk_needed_bytes:
        fails.append(
            f"DISK: {free / 2**30:.2f} GiB free at {out_dir}, need >= "
            f"{disk_needed_bytes / 2**30:.2f} GiB for checkpoints and artifacts"
        )
    return fails


def resource_record(*, n_bars: int, strategy: str, out_dir: Path) -> dict:
    """What was checked and what the host had — recorded in the manifest, never inferred later."""
    ram, disk, vram = physical_ram_bytes(), free_disk_bytes(out_dir), vram_bytes()
    return {
        "requirement": memory_requirement_bytes(n_bars, strategy),
        "host_physical_ram_gib": (ram / 2**30) if ram else None,
        "host_free_disk_gib": (disk / 2**30) if disk else None,
        "host_vram_gib": (vram / 2**30) if vram else None,
        "strategy_is_a_host_property": (
            "both strategies compute BYTE-IDENTICAL results (shuffle_micro is seeded on "
            "(symbol, seed), so a rebuilt window is the same window); they differ only in "
            "residency and rebuild count. 'low' costs 3x-5x window rebuilds depending on unit "
            "order and is for small hosts; 'fast' is correct on a large-RAM box, where paying "
            "that recompute to solve a problem the box does not have would be a net loss."
        ),
    }
