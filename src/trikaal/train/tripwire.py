"""Garbage-detector tripwires + the abort rule (pre-flight Item 7 / m6_design §6 item 8).

An obviously-broken run must halt EARLY — a NaN at hour 2 must not become 15 GPU-days of nonsense.
The committed "broken" definitions live in ``config/m6_tripwires.yaml`` (thresholds are part of
the pre-registration discipline, not runtime knobs); the orchestrator calls
:meth:`TripwireMonitor.check_step` every training step and :meth:`check_eval` on every eval
metric, and a trip raises :class:`TripwireAbort` — which the orchestrator lets PROPAGATE,
aborting the whole run (a broken cell invalidates the matched-cell comparison, so there is
nothing worth continuing)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

TRIPWIRE_CONFIG_DEFAULT = Path("config/m6_tripwires.yaml")


class TripwireAbort(RuntimeError):
    """A committed broken-run definition fired — the run must stop, not continue."""


@dataclass(frozen=True)
class TripwireConfig:
    k_first_steps: int = 200
    flat_loss_tol: float = 0.02
    nan_check: bool = True
    ir_abs_bound: float = 10_000.0
    heartbeat_timeout_s: float = 1800.0

    @classmethod
    def load(cls, path: str | Path = TRIPWIRE_CONFIG_DEFAULT) -> TripwireConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            k_first_steps=int(raw["k_first_steps"]),
            flat_loss_tol=float(raw["flat_loss_tol"]),
            nan_check=bool(raw["nan_check"]),
            ir_abs_bound=float(raw["ir_abs_bound"]),
            heartbeat_timeout_s=float(raw["heartbeat_timeout_s"]),
        )


class TripwireMonitor:
    """Per-(cell, stage) trip state. One instance per training run; keyed by a run label."""

    def __init__(self, cfg: TripwireConfig | None = None):
        self.cfg = cfg or TripwireConfig()
        self._first_loss: dict[str, float] = {}
        self._last_beat: dict[str, float] = {}

    def check_step(
        self, label: str, step: int, loss: float, grad_norm: float | None = None
    ) -> None:
        """Every training step: NaN/Inf in loss/grads → trip; flat/rising loss at K → trip."""
        self._last_beat[label] = time.time()
        if self.cfg.nan_check and not math.isfinite(loss):
            raise TripwireAbort(f"[{label}] step {step}: NON-FINITE loss ({loss}) — run aborted")
        if grad_norm is not None and self.cfg.nan_check and not math.isfinite(grad_norm):
            raise TripwireAbort(
                f"[{label}] step {step}: NON-FINITE grad norm ({grad_norm}) — run aborted"
            )
        if label not in self._first_loss:
            self._first_loss[label] = loss
        elif step >= self.cfg.k_first_steps:
            first = self._first_loss[label]
            ceiling = first * (1.0 + self.cfg.flat_loss_tol)
            if step == self.cfg.k_first_steps and loss >= ceiling:
                raise TripwireAbort(
                    f"[{label}] loss NOT DROPPING over the first {self.cfg.k_first_steps} steps "
                    f"({first:.4f} → {loss:.4f}, tol {self.cfg.flat_loss_tol:.0%}) — run aborted"
                )

    def check_eval(self, label: str, *, ir: float | None = None, **metrics: float) -> None:
        """Every eval: NaN metric or |net-IR| beyond the sane bound → trip."""
        vals = dict(metrics)
        if ir is not None:
            vals["ir"] = ir
        for name, v in vals.items():
            if self.cfg.nan_check and not math.isfinite(v):
                raise TripwireAbort(f"[{label}] eval metric {name} is non-finite ({v}) — aborted")
        if ir is not None and abs(ir) > self.cfg.ir_abs_bound:
            raise TripwireAbort(
                f"[{label}] |net-IR| = {abs(ir):.1f} beyond the sane bound "
                f"{self.cfg.ir_abs_bound:.0f} — degenerate eval, run aborted"
            )

    def check_heartbeat(self, label: str) -> None:
        """A stalled cell (no step within the timeout) trips — GPU-util collapse's local twin."""
        beat = self._last_beat.get(label)
        if beat is not None and (time.time() - beat) > self.cfg.heartbeat_timeout_s:
            raise TripwireAbort(
                f"[{label}] no training step for {self.cfg.heartbeat_timeout_s:.0f}s — stalled"
            )


__all__ = ["TRIPWIRE_CONFIG_DEFAULT", "TripwireAbort", "TripwireConfig", "TripwireMonitor"]
