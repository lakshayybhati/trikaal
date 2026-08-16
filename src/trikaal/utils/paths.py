"""Display-path helper — because ``Path.relative_to`` RAISES, and we kept using it to print.

★ THE DEFECT THIS EXISTS TO KILL, STATED PLAINLY. ``p.relative_to(root)`` throws ``ValueError``
when ``p`` is not under ``root``. That is correct behaviour for a path COMPUTATION and a
catastrophe for a path being PRINTED: the work has already succeeded, the file is already on
disk, and the run dies inside its own success message. The operator sees a stack trace and
concludes the tool failed while a perfectly good artifact sits beside them.

It happened twice in one day, in two scripts, to the same author, and BOTH times the artifact had
been written before the crash:

  * ``m6_horizon_break_even.py`` — caught locally by a negative control, whose whole job was to
    prove a guard could fail and which instead fell over on this;
  * ``m6_demo_build.py`` — caught by the supervisor doing the one thing no test did: passing
    ``--out`` a path outside the repository, which is where every real user would point it
    (a Desktop, a ``/tmp`` scratch dir, a web server root).

``m6_skill_vs_bias.py`` carried the same line, unreported, and is fixed here by the same sweep.

WHY A SHARED HELPER RATHER THAN THREE TERNARIES. The class rule: fixing the sites an audit points
at leaves the siblings standing and marks the finding closed, which is worse than not having
looked. With one helper the next script cannot reintroduce it by writing the obvious thing,
because the obvious thing is now importing this.
"""

from __future__ import annotations

from pathlib import Path


def display_path(p: Path | str, root: Path | str) -> str:
    """``p`` relative to ``root`` for display, or its absolute form when it is not under ``root``.

    NEVER RAISES. This is for human-facing output only — a provenance string, a log line, a
    receipt field. Anything that needs a genuine relative path for a COMPUTATION should call
    ``Path.relative_to`` directly and handle the error, because there the exception is the point.
    """
    path = Path(p)
    base = Path(root)
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


__all__ = ["display_path"]
