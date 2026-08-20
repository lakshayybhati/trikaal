"""One gate every receipt writer passes through, so a false record cannot enter the tree.

★ THE SYSTEMIC DEFECT THIS CLOSES. Six scripts wrote confident receipts from nothing:

  * ``m6_h_sweep.py`` skipped all three cells and still wrote ``L1_all_reproduced: true`` from
    **zero** cells, exiting 0 while deleting 317 lines of a tracked receipt — and printed
    ``worst |delta| = None`` and ``all_reproduced=True`` on the same line;
  * ``m6_pull_verify.py`` printed ``PASS ... verified 0/0`` and exited 0, rewriting 9/9 to 0/0,
    under a docstring that says *"Exit 0 ONLY if every file verifies"*;
  * ``m6_determinism_probe.py`` printed ``PROBE INVALID — this measured nothing`` and then
    overwrote a VALID receipt, exiting 0;
  * ``m6_c3_dsr_units.py`` printed ``PROBE INVALID`` and wrote anyway, deleting 103 lines of
    CONFIRMED audit findings;
  * ``m6_bit_exact_correction.py`` silently rewrote a disclosure receipt from 46 records to 19;
  * ``m4b_universe_ingest.py`` made an unprompted LIVE NETWORK CALL on a no-argument run and
    overwrote committed ``config/universe_full.yaml``.

Individually each looks like a rough edge. Together they are the mechanism by which a repository
starts asserting things that were never measured, and the norm forbidding it — *a probe must
REFUSE to emit a verdict when its own control arm fails* — was already written down and had been
implemented for exactly one receipt.

THREE REFUSALS, and the third is the one that protects existing evidence:

1. **Nothing measured** → refuse. An empty result set is not a passing result set.
2. **The run says it is invalid** → refuse. A probe that reports ``PROBE INVALID`` may not also
   emit a verdict; that is the standing norm, now enforced at the write.
3. **The target is TRACKED and ``force`` was not passed** → refuse. Every one of the six
   overwrote committed evidence as a side effect of being run. Overwriting a receipt that is
   under version control is now a deliberate act, spelled ``--force``.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sized
from pathlib import Path

__all__ = ["ReceiptRefused", "is_tracked", "write_receipt"]


class ReceiptRefused(RuntimeError):
    """Raised instead of writing. The message states which of the three rules refused."""


def is_tracked(path: Path, *, repo: Path | None = None) -> bool:
    """Whether ``path`` is under version control, via ``git ls-files --error-unmatch``."""
    root = repo or Path(__file__).resolve().parents[3]
    r = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def _count(measured: Sized | int | None) -> int:
    if measured is None:
        return 0
    if isinstance(measured, bool):  # a bool is a Sized-less int; treat True as one measurement
        return int(measured)
    if isinstance(measured, int):
        return measured
    return len(measured)


def write_receipt(
    path: Path,
    doc: dict,
    *,
    measured: Sized | int | None,
    valid: bool = True,
    invalid_reason: str = "",
    force: bool = False,
    repo: Path | None = None,
) -> Path:
    """Write ``doc`` as JSON to ``path``, or raise ``ReceiptRefused``.

    ``measured`` is whatever the run actually measured — a list of cells, a count of files, a dict
    of results. Its emptiness is the question, not its type.
    """
    n = _count(measured)
    if n == 0:
        raise ReceiptRefused(
            f"REFUSING TO WRITE {path.name}: the run measured NOTHING (measured set is empty). "
            "An empty result set is not a passing result set — a receipt written here would "
            "assert a verdict no measurement supports. Fix the inputs and re-run."
        )
    if not valid:
        raise ReceiptRefused(
            f"REFUSING TO WRITE {path.name}: the run reports itself INVALID"
            + (f" ({invalid_reason})" if invalid_reason else "")
            + ". A probe whose own control arm failed may not emit a verdict — it reports PROBE "
            "INVALID and writes nothing. Writing here would overwrite a valid record with a "
            "measurement that did not happen."
        )
    if is_tracked(path, repo=repo) and not force:
        raise ReceiptRefused(
            f"REFUSING TO OVERWRITE {path.name}: it is TRACKED, and this run did not pass "
            "force=True (--force). Every one of the six writers this rule exists for destroyed "
            "committed evidence as a side effect of merely being run — 317 lines here, 103 there, "
            "46 records reduced to 19. Overwriting version-controlled evidence is a deliberate "
            "act. Re-run with --force if that is what you mean."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return path
