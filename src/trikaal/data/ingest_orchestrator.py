"""Governed parallel ingest orchestrator (M4b) — raw is a BOUNDED CACHE, not a guarantee.

Producer–consumer over the ``(symbol, month)`` shard plan, designed for the measured reality that
the **download link is the throughput wall** (≈ 8 MB/s here; parallel streams do not beat it), so
concurrency exists to *overlap* download with reduce and to keep the link busy — not to multiply
bandwidth. Downloads run in a thread pool; the per-symbol reduce→features→lake build and all
eviction run single-threaded in the consumer, so the disk governor and the evict-after-record
invariant have no races.

**Disk model (measured).** The reduced-shard npz still retains every trade's size pool, so it is
~60 % of the raw zip — *not* a cheap intermediate. The genuinely small kept artifact is the
**feature lake** (~3.9 MB/month/symbol vs ~hundreds of MB of raw). So M4b reduces a symbol's months
**in memory** (no npz) straight into the lake, then evicts that symbol's raw. The bounded cache is
therefore the raw of the *in-flight* symbols (downloaded-but-not-yet-built), not the whole universe.

**The load-bearing invariant (adversarially reviewed): a raw shard is unlinked ONLY after its
symbol's lake partition is built AND recorded** — `download → verify → (all months staged) →
features → write-lake → ledger.record_lake_built → (only then) evict that symbol's raw`. The ledger
(append-only JSONL) is the commit marker; :func:`evict_raw` hard-asserts ``is_lake_built`` so no
path can delete an unrecorded symbol's raw. Resume skips lake-built symbols; a non-built symbol's
raw is re-fetched if evicted (idempotent verified download) or re-used if still cached.
Reproducibility anchor = the content-hashed **lake** (kept); the raw Binance SHA-256 is recorded for
**provenance only** (re-pullable), so reproducibility survives raw eviction.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

GB = 1_000_000_000


class EvictBeforeRecordError(RuntimeError):
    """Raised if eviction is attempted on a symbol not yet recorded lake-built (invariant)."""


class DiskGovernorAbort(RuntimeError):
    """A single symbol's raw alone cannot fit the cache cap / free floor (pathological)."""


@dataclass(frozen=True)
class GovernorKnobs:
    raw_cache_cap_gb: float = 380.0  # pause downloads when in-flight raw exceeds this
    min_free_disk_gb: float = 50.0  # hard free-disk floor
    max_concurrent_downloads: int = 4  # link-capped; >1 only overlaps, never speeds the download


@dataclass
class DownloadResult:
    symbol: str
    period: str
    raw_paths: list[Path]
    raw_bytes: int
    checksums: dict[str, str]  # {filename: sha256}  — provenance only


class Ledger:
    """Append-only JSONL commit log. ``record_lake_built`` is the durable per-symbol commit mark."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._built: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("event") == "lake_built":
                        self._built[rec["symbol"]] = rec

    def is_lake_built(self, symbol: str) -> bool:
        return symbol in self._built

    def built_symbols(self) -> list[str]:
        return sorted(self._built)

    def record(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()

    def record_lake_built(
        self, symbol: str, dataset_hash: str, checksums: dict, n_bars: int
    ) -> None:
        rec = {
            "event": "lake_built",
            "symbol": symbol,
            "dataset_hash": dataset_hash,
            "checksums": checksums,
            "n_bars": n_bars,
        }
        self._built[symbol] = rec
        self.record(rec)


class DiskGovernor:
    """Decide when to evict a lake-built symbol's raw to stay under the cap / above the free floor.

    Pure decision logic over a ground-truth ``disk_free_fn`` + a tracked ``raw_cache_bytes`` + the
    set of EVICTABLE (already lake-built) symbols — unit-testable with a fake disk.
    """

    def __init__(self, knobs: GovernorKnobs, disk_free_fn: Callable[[], int]):
        self.knobs = knobs
        self._disk_free = disk_free_fn
        self.raw_cache_bytes = 0
        self.lock = threading.Lock()

    @property
    def cap_bytes(self) -> int:
        return int(self.knobs.raw_cache_cap_gb * GB)

    @property
    def floor_bytes(self) -> int:
        return int(self.knobs.min_free_disk_gb * GB)

    def room_ok(self, incoming: int) -> bool:
        return (self.raw_cache_bytes + incoming <= self.cap_bytes) and (
            self._disk_free() - incoming >= self.floor_bytes
        )

    def plan_evictions(
        self, incoming: int, evictable: list[tuple[str, int]]
    ) -> tuple[str, list[str]]:
        """Return ``(status, symbols_to_evict)`` where status ∈ {"ok","wait","abort"}.

        Evict lake-built symbols' raw (largest-first) to fit ``incoming``. "wait" means evicting all
        currently-evictable raw is still not enough but in-flight work may free more; "abort" means
        even a fully-empty cache cannot fit ``incoming`` (a single symbol bigger than the budget).
        """
        if self.room_ok(incoming):
            return "ok", []
        plan: list[str] = []
        freed = 0
        for sym, nbytes in sorted(evictable, key=lambda e: -e[1]):
            plan.append(sym)
            freed += nbytes
            if (self.raw_cache_bytes - freed + incoming <= self.cap_bytes) and (
                self._disk_free() - incoming >= self.floor_bytes
            ):
                return "ok", plan
        # nothing (more) evictable. If the empty-cache budget still can't fit one symbol → abort.
        if (
            incoming > self.cap_bytes
            or self._disk_free() + self.raw_cache_bytes - incoming < self.floor_bytes
        ):
            raise DiskGovernorAbort(
                f"cannot fit {incoming / GB:.1f} GB: free={self._disk_free() / GB:.1f} GB, "
                f"cap={self.knobs.raw_cache_cap_gb} GB, cache={self.raw_cache_bytes / GB:.1f} GB"
            )
        return "wait", plan


def evict_raw(symbol: str, raw_paths: list[Path], ledger: Ledger) -> int:
    """Delete a symbol's raw zips — ONLY if it is recorded lake-built. Returns freed bytes.

    The single chokepoint for raw deletion; hard-asserts the evict-after-record invariant so no path
    can delete an unrecorded symbol's raw.
    """
    if not ledger.is_lake_built(symbol):
        raise EvictBeforeRecordError(f"refuse to evict raw of un-built symbol {symbol}")
    freed = 0
    for p in raw_paths:
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
    return freed


def default_disk_free(path: Path) -> Callable[[], int]:
    return lambda: shutil.disk_usage(path).free


@dataclass
class OrchestratorReport:
    n_symbols_total: int
    n_symbols_built: int
    total_bars: int
    peak_raw_cache_bytes: int
    n_symbols_evicted: int
    raw_bytes_evicted: int
    aborted: bool = False
    error: str | None = None
    failed_symbols: list[str] = field(default_factory=list)


@dataclass
class IngestOrchestrator:
    """Wire the governor + ledger + injectable download/finalize into a resumable run.

    ``download_fn(symbol, period) -> DownloadResult`` — SHA-256-verified fetch into the raw store.
    ``finalize_symbol_fn(symbol, periods) -> (dataset_hash, n_bars, checksums)`` — in-memory build →
    features → Parquet lake + manifest record. Both injectable so the orchestration is tested with
    fakes (no network/zips).
    """

    plan: dict[str, list[str]]
    knobs: GovernorKnobs
    ledger: Ledger
    download_fn: Callable[[str, str], DownloadResult]
    finalize_symbol_fn: Callable[[str, list[str]], tuple[str, int, dict]]
    est_bytes_fn: Callable[[str, str], int]
    disk_free_fn: Callable[[], int]
    _gov: DiskGovernor = field(init=False)

    def __post_init__(self) -> None:
        self.plan = {s: sorted(ms) for s, ms in self.plan.items()}
        self._gov = DiskGovernor(self.knobs, self.disk_free_fn)
        self._raw_on_disk: dict[str, list[Path]] = defaultdict(list)  # symbol -> raw paths present
        self._raw_bytes: dict[str, int] = defaultdict(int)
        self._evictable: dict[str, int] = {}  # lake-built symbols whose raw is still on disk
        self._peak = 0
        self._n_evicted = 0
        self._bytes_evicted = 0

    def _evict_built(self, symbols: list[str]) -> None:
        for sym in symbols:
            freed = evict_raw(sym, self._raw_on_disk.get(sym, []), self.ledger)  # asserts built
            self._gov.raw_cache_bytes -= self._raw_bytes.get(sym, 0)
            self._evictable.pop(sym, None)
            self._raw_on_disk.pop(sym, None)
            self._raw_bytes.pop(sym, None)
            self._n_evicted += 1
            self._bytes_evicted += freed

    def run(self, *, progress: Callable[[str], None] | None = None) -> OrchestratorReport:
        log = progress or (lambda _m: None)
        pending = [s for s in self.plan if not self.ledger.is_lake_built(s)]
        todo: list[tuple[str, str]] = [(s, m) for s in pending for m in self.plan[s]]
        staged: dict[str, set[str]] = defaultdict(set)  # months downloaded this run
        failed: set[str] = set()  # symbols whose download/build raised — isolated, not fatal
        total_bars = 0
        aborted = False
        err: str | None = None

        with ThreadPoolExecutor(max_workers=self.knobs.max_concurrent_downloads) as pool:
            inflight: dict = {}

            def submit_more() -> None:
                while todo and len(inflight) < self.knobs.max_concurrent_downloads:
                    sym, per = todo[0]
                    est = self.est_bytes_fn(sym, per)
                    status, evict = self._gov.plan_evictions(
                        est, [(s, b) for s, b in self._evictable.items()]
                    )
                    if evict:
                        self._evict_built(evict)
                    if status == "wait":
                        return  # no room now; let in-flight finalize free space
                    todo.pop(0)
                    self._gov.raw_cache_bytes += est
                    self._peak = max(self._peak, self._gov.raw_cache_bytes)
                    inflight[pool.submit(self.download_fn, sym, per)] = (sym, per, est)

            def check_not_stuck() -> None:
                # stuck only if work remains, nothing is in flight, AND nothing is still buildable
                if todo and not inflight and any(s not in failed for s, _ in todo):
                    raise DiskGovernorAbort(
                        "no download could be issued — a symbol's raw exceeds raw_cache_cap_gb; "
                        "raise the cap above the largest single-symbol raw footprint"
                    )

            try:
                submit_more()
                check_not_stuck()
                while inflight:
                    done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        sym, per, est = inflight.pop(fut)
                        try:
                            dr = fut.result()  # raises on download/verify failure
                            if sym in failed:
                                # an EARLIER month of this symbol already failed; this late-arriving
                                # download is garbage — release its reservation + delete it, never
                                # fold it into state (else a phantom counter / leaked zip returns).
                                self._gov.raw_cache_bytes -= est
                                for p in dr.raw_paths:
                                    if p.exists():
                                        p.unlink()
                                continue
                            self._gov.raw_cache_bytes += dr.raw_bytes - est
                            self._peak = max(self._peak, self._gov.raw_cache_bytes)
                            self._raw_on_disk[sym].extend(dr.raw_paths)
                            self._raw_bytes[sym] += dr.raw_bytes
                            staged[sym].add(per)
                            done_sym = staged[sym] == set(self.plan[sym])
                            if done_sym and not self.ledger.is_lake_built(sym):
                                dhash, nbars, checks = self.finalize_symbol_fn(sym, self.plan[sym])
                                self.ledger.record_lake_built(sym, dhash, checks, nbars)
                                total_bars += nbars
                                self._evictable[sym] = self._raw_bytes[sym]  # only AFTER record
                                self._evict_built([sym])  # reclaim raw immediately (lake is tiny)
                                cache_gb = self._gov.raw_cache_bytes / GB
                                log(f"built {sym}: {nbars:,} bars (raw cache {cache_gb:.1f} GB)")
                        except Exception as e:  # isolate one symbol's failure — never sink the run
                            failed.add(sym)
                            # Release the symbol's FULL real footprint plus this month's live
                            # reservation. `est` is still live ONLY on the download leg (this month
                            # never reached line 283); if finalize raised, est was already folded
                            # into _raw_bytes at line 279 → do not subtract it twice (would go
                            # negative). Then physically delete the symbol's staged raw — NOT via
                            # evict_raw (it asserts is_lake_built and would refuse a failed symbol);
                            # a failed symbol has no lake partition, so its raw is garbage,
                            # re-fetched idempotently on resume.
                            live_est = est if per not in staged.get(sym, set()) else 0
                            self._gov.raw_cache_bytes -= self._raw_bytes.get(sym, 0) + live_est
                            for p in self._raw_on_disk.get(sym, []):
                                if p.exists():
                                    p.unlink()
                            self._raw_on_disk.pop(sym, None)
                            self._raw_bytes.pop(sym, None)
                            staged.pop(sym, None)
                            self._evictable.pop(sym, None)
                            todo[:] = [(s, p) for s, p in todo if s != sym]  # skip its other months
                            log(f"FAILED {sym} {per}: {type(e).__name__}: {e}")
                    submit_more()
                    check_not_stuck()
            except DiskGovernorAbort as e:
                aborted, err = True, str(e)

        return OrchestratorReport(
            n_symbols_total=len(self.plan),
            n_symbols_built=len(self.ledger.built_symbols()),
            total_bars=total_bars,
            peak_raw_cache_bytes=self._peak,
            n_symbols_evicted=self._n_evicted,
            raw_bytes_evicted=self._bytes_evicted,
            aborted=aborted,
            error=err,
            failed_symbols=sorted(failed),
        )


__all__ = [
    "DiskGovernor",
    "DiskGovernorAbort",
    "DownloadResult",
    "EvictBeforeRecordError",
    "GovernorKnobs",
    "IngestOrchestrator",
    "Ledger",
    "OrchestratorReport",
    "default_disk_free",
    "evict_raw",
]
