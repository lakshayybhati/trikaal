"""§7 v1.6.26 (P2 finding F2) — THE LAKE VERIFIER MUST CATCH TRUNCATION, NOT JUST ABSENCE.

The money driver's only lake check is `LAKE MISSING at processed/universe_bars`. **That catches
ABSENCE and is blind to TRUNCATION**, which is the failure mode a 4.08 GiB / 1,920-file network
transfer actually produces: a lake short by 200 parquets opens, queries, and answers with fewer
bars, so the box scores a silently different cross-section and nothing refuses it.

So the verifier is checked here against the four ways a pull goes wrong — missing file, short file,
right-length-wrong-bytes, and extra symbols outside the pinned 40 — with the healthy case asserted
FIRST, or none of the four would prove anything.

The scrub is tested too, because **a scrub nobody checks is a scrub that did not happen.**
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _mod():
    spec = importlib.util.spec_from_file_location("fetch_lake", REPO / "scripts/m6_fetch_lake.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _lake(tmp_path: Path, blobs: dict[str, bytes]) -> tuple[Path, dict]:
    dest = tmp_path / "universe_bars"
    files = {}
    for rel, data in blobs.items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        files[rel] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return dest, {"files": files, "total_gib": 0.0}


BLOBS = {
    "symbol=AAAUSDT/frequency=1m/month=2024-01/data_0.parquet": b"alpha-payload-0123456789",
    "symbol=AAAUSDT/frequency=1m/month=2024-02/data_0.parquet": b"alpha-payload-abcdefghij",
    "symbol=BBBUSDT/frequency=1m/month=2024-01/data_0.parquet": b"beta-payload-9876543210",
}


# ------------------------------------------------------------------ DISCRIMINATION FIRST
def test_a_complete_lake_verifies_clean(tmp_path):
    """If this failed, all four rejections below would be vacuous."""
    dest, man = _lake(tmp_path, BLOBS)
    assert _mod().verify(dest, man) == []


# ------------------------------------------------------------------ the four ways a pull fails
def test_a_MISSING_file_is_caught(tmp_path):
    dest, man = _lake(tmp_path, BLOBS)
    (dest / "symbol=BBBUSDT/frequency=1m/month=2024-01/data_0.parquet").unlink()
    bad = _mod().verify(dest, man)
    assert bad and any("MISSING" in b for b in bad)


def test_a_TRUNCATED_file_is_caught(tmp_path):
    """The named failure mode: the file exists, so `LAKE MISSING` is silent."""
    dest, man = _lake(tmp_path, BLOBS)
    p = dest / "symbol=AAAUSDT/frequency=1m/month=2024-01/data_0.parquet"
    p.write_bytes(p.read_bytes()[:5])
    bad = _mod().verify(dest, man)
    assert bad and any("TRUNCATED" in b for b in bad)


def test_RIGHT_LENGTH_WRONG_BYTES_is_caught(tmp_path):
    """Why the check is sha256 and not size. A size-only verifier passes this."""
    dest, man = _lake(tmp_path, BLOBS)
    p = dest / "symbol=AAAUSDT/frequency=1m/month=2024-02/data_0.parquet"
    data = bytearray(p.read_bytes())
    data[0] ^= 0xFF
    p.write_bytes(bytes(data))
    assert p.stat().st_size == man["files"][str(p.relative_to(dest))]["size"]  # size still matches
    bad = _mod().verify(dest, man)
    assert bad and any("CONTENT MISMATCH" in b for b in bad)


def test_an_UNPINNED_symbol_in_the_lake_is_caught(tmp_path):
    """An unrestricted pull is not merely wasteful — an unpinned symbol in the lake is an
    unpinned cross-section, and the 40 are a pre-registered pin."""
    dest, man = _lake(tmp_path, BLOBS)
    extra = dest / "symbol=ZZZUSDT/frequency=1m/month=2024-01/data_0.parquet"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"not in the pinned 40")
    bad = _mod().verify(dest, man)
    assert bad and any("NOT in the pinned" in b for b in bad)


# ------------------------------------------------------------------ the scrub
def test_the_scrub_check_FINDS_a_token_in_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_" + "a" * 30)
    found = _mod().assert_no_token(tmp_path)
    assert found and any("environment variable" in f for f in found)


def test_the_scrub_check_is_CLEAN_when_nothing_is_left(tmp_path, monkeypatch):
    """Discrimination on the scrub: it must be able to say CLEAN, or its LEAK verdict is noise."""
    m = _mod()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(m, "HF_PATTERN", __import__("re").compile(r"hf_ZZZNEVERMATCHES"))
    monkeypatch.setattr(m.Path, "home", staticmethod(lambda: tmp_path))
    assert m.assert_no_token(tmp_path) == []


# ------------------------------------------------------------------ the committed manifest
def test_the_committed_manifest_covers_the_pinned_40_with_hashes():
    man = json.loads((REPO / "runs_manifest/m6_lake_subset_manifest.json").read_text())
    pins = json.loads((REPO / "runs_manifest/m6_mde_inputs.json").read_text())["symbols_sampled"]
    assert man["n_symbols"] == len(pins) == 40
    assert man["symbols_missing_from_repo"] == []
    assert man["files_without_a_content_hash"] == []
    assert man["n_files"] == len(man["files"]) == 1920
    got = {p.split("/")[0].removeprefix("symbol=") for p in man["files"]}
    assert got == set(pins), "the manifest's symbols are not exactly the pinned cross-section"


def test_symbol_is_read_from_a_PATH_COMPONENT_not_a_substring():
    """The sizing pass reported '0.0 % of lake' by globbing the symbol name against FILENAMES. The
    name only ever appears in a DIRECTORY component, so the substring method attributes nothing —
    and it fails SILENTLY, returning a plausible-looking zero rather than an error."""
    spec = importlib.util.spec_from_file_location(
        "subset_manifest", REPO / "scripts/m6_lake_subset_manifest.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    real = "symbol=BTCUSDT/frequency=1m/month=2024-01/data_0.parquet"
    assert m.symbol_of(real) == "BTCUSDT"
    # the name is in the FILE here, not a directory — the substring method would claim a match
    assert m.symbol_of("rehearsal/BTCUSDT_predictor.pt") is None
    assert m.symbol_of(".gitattributes") is None
    assert "BTCUSDT" in "rehearsal/BTCUSDT_predictor.pt"  # …which is exactly why it fooled a glob
