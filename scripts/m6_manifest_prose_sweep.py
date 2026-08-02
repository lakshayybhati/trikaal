"""OPEN RULING (a) — SWEEP EVERY PROSE STRING PERSISTED INTO THE VERDICT MANIFEST.

Two shipped artifacts carried a false account of their own recipe, in sibling clauses, six commits
apart (§7 v1.6.2 and v1.6.12). That is a class, not two accidents. This sweeps the **emitted
artifact** rather than the source, so a string that never reaches the manifest cannot pad the count
and a string that reaches it cannot hide behind the source line that produced it.

Method: assemble a real verdict from the test fixture builder, walk every string in the manifest,
extract every number appearing in it, and check each number against the live pinned surface. A
number that matches no pin and no value present in the manifest itself is reported as UNVERIFIED —
it may still be legitimate prose, but it must be read by a human.

CONTROL ARM (standing norm): before trusting a clean sweep, a KNOWN-STALE string is injected and
the extractor must flag it. If the control does not fire the sweep emits PROBE INVALID.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from eval import test_verdict as T
from eval.test_verdict import _assemble, _build_fixture
from trikaal.eval import conformance as C
from trikaal.eval import verdict as V

OUT = Path("runs_manifest/m6_manifest_prose_sweep.json")
# NOTE (builder defect, disclosed): the first version of this pattern ended in ``(?![\w.])``, which
# silently skipped every number glued to a letter — ``3x`` (the micro weight) and ``1.5x`` (the
# dispersion tripwire) both escaped, i.e. the two recipe numbers most worth checking. A sweep that
# cannot see the strings it exists to check is the "a check that cannot fail" shape again. The
# trailing guard now excludes only a following DIGIT or decimal point, so ``3x`` is caught while
# ``v1.5`` (leading ``\w``) and ``A.4.3`` (leading ``.``) stay excluded as version/section refs.
NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\d.])")


def _pinned_numbers() -> dict[float, list[str]]:
    """Every number the manifest is ALLOWED to state, with where it comes from."""
    src: dict[float, list[str]] = {}

    def add(v, name):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        src.setdefault(f, []).append(name)

    for k, v in C.PINNED_DSR.items():
        if isinstance(v, (int, float)):
            add(v, f"PINNED_DSR[{k!r}]")
        elif isinstance(v, (tuple, list)):
            for x in v:
                add(x, f"PINNED_DSR[{k!r}]")
    for name in dir(C):
        if name.isupper():
            val = getattr(C, name)
            if isinstance(val, (int, float)):
                add(val, f"conformance.{name}")
            elif isinstance(val, (tuple, list)):
                for x in val:
                    add(x, f"conformance.{name}")
    for name in dir(V):
        if name.isupper():
            val = getattr(V, name)
            if isinstance(val, (int, float)):
                add(val, f"verdict.{name}")
            elif isinstance(val, (tuple, list)):
                for x in val:
                    add(x, f"verdict.{name}")
    from trikaal.train.arms import ARM_MICRO, ARM_OHLCV, arm_n_features

    add(arm_n_features(ARM_OHLCV), "arms.arm_n_features(ohlcv)")
    add(arm_n_features(ARM_MICRO), "arms.arm_n_features(micro)")
    add(C.PINNED_HEADLINE_COST * 100.0, "conformance.PINNED_HEADLINE_COST as a percentage")
    add(len(V.CELLS) * len(V.DSR_SEEDS), "derived: n_artifacts = cells x seeds")
    add(
        len(V.CELLS) * len(V.DSR_SEEDS) * len(V.DSR_HORIZONS) * len(V.DSR_KAPPAS),
        "derived: full trial enumeration",
    )
    add(
        len(V.DSR_SEEDS) * len(V.DSR_HORIZONS) * len(V.DSR_KAPPAS),
        "derived: var_sr basis size (one cell)",
    )
    return src


def _strings(node, path="") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out += _strings(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += _strings(v, f"{path}[{i}]")
    elif isinstance(node, str):
        out.append((path, node))
    return out


HEXISH = re.compile(r"^(sha256:)?[0-9a-f]{32,}$")


def _check(text: str, pins: dict[float, list[str]]) -> list[dict]:
    bad = []
    if HEXISH.match(text.strip()):  # content hashes are values, not prose
        return bad
    for m in NUM.finditer(text):
        tok = m.group(1)
        val = float(tok)
        if val in pins:
            continue
        # §N section refs and C-N/M-N finding ids are citations, not recipe numbers. Classified
        # rather than dropped: they still appear in the receipt, they just do not demand a pin.
        pre = text[max(0, m.start() - 2) : m.start()]
        kind = "reference" if pre.endswith(("§", "C-", "M-", "S-", "A-")) else "prose_number"
        bad.append({"number": tok, "value": val, "kind": kind})
    return bad


# Prose numbers that are NOT recipe values and therefore cannot derive from a pin: measured
# history and worked examples. Each was read against its source before being listed here, and the
# binding above re-opens the finding if the string's numbers ever change.
ADJUDICATED = {
    "/decode_agreement_disclosure/why_a_two_sample_test": {
        "numbers": ["1.10", "1.82", "2.62"],
        "record": {
            "kind": "measured history, not a recipe value",
            "source": "prereg §7 v1.6.11 (docs/m6_prereg.md:470-473)",
            "verified": (
                "1.10 = |diff| 0.1071 / one arm's sd 0.0973; 1.82 = t 1.821 to 3sf; "
                "2.62 = the Welch df-2.385 critical value 2.6226 to 3sf"
            ),
        },
    },
    "/degeneracy_guard/rule": {
        "numbers": ["0.55", "0.55", "0.700", "0.999"],
        "record": {
            "kind": "worked example of the v1.4.4 guard hole, not a recipe value",
            "source": "prereg §7 v1.4.6",
            "verified": "(0.999 + 0.55 + 0.55) / 3 = 0.69967, i.e. 0.700 to 3dp",
        },
    },
}


def main() -> int:
    import tempfile

    rep: dict = {"what": "OPEN RULING (a) — prose strings persisted into the verdict manifest"}
    pins = _pinned_numbers()
    rep["n_pinned_numbers"] = len(pins)

    # EVERY BRANCH, not one. A string emitted only on the NULL / placebo-harmed / fallback-claimed
    # path is still a shipped string, and sweeping a single healthy fixture would never see it.
    branches = {
        "planted_survives": (T.MU_PLANTED, 101),
        "null": (T.MU_NULL, 202),
        "placebo_harmed": (T.MU_PLACEBO_HARMED, 303),
        "fallback_claimed": (T.MU_FALLBACK_CLAIMED, 404),
    }
    seen: dict[str, str] = {}
    per_branch: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as td:
        for name, (mu, fs) in branches.items():
            man = _assemble(_build_fixture(Path(td) / name, mu, fixture_seed=fs))
            got = _strings(man)
            per_branch[name] = len(got)
            for path, text in got:
                seen.setdefault(f"{path}", text)
                if text != seen[f"{path}"]:  # branch-specific wording at the same path
                    seen[f"{path}|{name}"] = text
    strings = sorted(seen.items())
    rep["branches_swept"] = per_branch
    rep["n_strings_in_manifest"] = len(strings)

    # ---- CONTROL: a known-stale string must be flagged, and a clean one must not be.
    stale = "SR0 from N=180 enumerated trials; var_sr over the 180 de-annualized VAL IRs"
    clean = f"SR0 from N={V.DSR_N_TRIALS} enumerated trials, threshold {V.DSR_THRESHOLD}"
    ctl_stale = _check(stale, pins)
    ctl_clean = _check(clean, pins)
    rep["control"] = {
        "stale_string": stale,
        "stale_flagged": bool(ctl_stale),
        "stale_hits": ctl_stale,
        "clean_string": clean,
        "clean_flagged": bool(ctl_clean),
        "discriminates": bool(ctl_stale) and not ctl_clean,
    }
    if not rep["control"]["discriminates"]:
        rep["verdict"] = "PROBE INVALID — the extractor cannot separate stale from current"
        OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
        print(json.dumps(rep["control"], indent=2))
        return 1

    findings, refs_only, adjudicated = [], [], []
    for path, text in strings:
        hits = _check(text, pins)
        if not hits:
            continue
        prose = [h for h in hits if h["kind"] == "prose_number"]
        if prose and path.split("|")[0] in ADJUDICATED:
            entry = ADJUDICATED[path.split("|")[0]]
            got = sorted(h["number"] for h in prose)
            # A check that CAN fail: the adjudication is bound to the exact number set that was
            # read. A new or changed number in an adjudicated string re-surfaces it as a finding.
            if got == sorted(entry["numbers"]):
                adjudicated.append({"path": path, "numbers": got, **entry["record"]})
                continue
            findings.append(
                {
                    "path": path,
                    "text": text,
                    "unverified_numbers": hits,
                    "note": (
                        f"ADJUDICATION STALE — was adjudicated for {sorted(entry['numbers'])}, "
                        f"now states {got}; re-read it"
                    ),
                }
            )
            continue
        (findings if prose else refs_only).append(
            {"path": path, "text": text, "unverified_numbers": hits}
        )
    rep["adjudicated_historical"] = adjudicated
    rep["unverified_prose_numbers"] = findings
    rep["citations_only"] = [r["path"] for r in refs_only]
    rep["n_unverified_strings"] = len(findings)
    rep["verdict"] = "CLEAN" if not findings else "REVIEW REQUIRED"
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in rep.items()
                if k not in ("unverified_prose_numbers", "citations_only")
            },
            indent=2,
            sort_keys=True,
        )
    )
    for f in rep["unverified_prose_numbers"]:
        nums = [h["number"] for h in f["unverified_numbers"]]
        print(f"\n  {f['path']}\n    {f['text']}\n    -> {nums}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
