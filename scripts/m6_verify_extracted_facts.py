"""Check extracted §4 facts against the artifacts they cite.

Six readers extracted 626 candidate facts for the paper's experimental-design section, each with a
citation of the form ``path :: key.path = value`` or ``path:LINE``. Adversarial verification of
five of the six topics did not complete. Rather than write a paper section from unverified
extraction, this resolves every citation mechanically:

  JSON citation   the key path must exist in the named receipt AND the stated value must match
                  the value actually stored there (numeric comparison with tolerance, else string
                  containment)
  source citation the named file must exist, and the claimed value or symbol must appear within a
                  window around the cited line -- line numbers drift, so a miss at the exact line
                  is re-checked file-wide and reported as DRIFTED rather than REFUTED
  unresolvable    the citation does not name something machine-checkable; reported as MANUAL, not
                  as a pass

A fact is USABLE only if its citation resolves and its value matches. Everything else is
quarantined.

MUTATION CONTROL. --self-test corrupts a known-good fact's value and a known-good key path and
requires the checker to report REFUTED for both; if it does not, the checker is reporting
confirmations it cannot support and exits 1 without producing a report.

★ AND THAT CONTROL WAS NOT ENOUGH, WHICH IS THE POINT WORTH READING. It proved the checker could
refute a WRONG value. It never asked whether the checker DECLINES an undecidable one, and four
distinct false-pass vectors lived underneath a green mutation control:

  negation        `str(actual) in claimed or claimed in str(actual)` — "supported" is a substring
                  of "premise NOT supported", so a claim and its own negation confirmed each other
  substring       "true" is a substring of "untrue"
  any-number      a claim carrying several numbers confirmed if ANY of them matched the stored
                  value — including the one it was contrasting AGAINST
  container-any   a claim's number appearing anywhere inside a container confirmed it; in a
                  35,000-element series that is unconditional

All four are now named cases in ``_false_pass_controls`` and must come back REFUTED or QUARANTINED.
``_values_agree`` returns three values rather than two: the missing one was "I cannot decide", and
returning True in its place is how a checker manufactures confirmations. Quarantine is not a
softer pass — it is the honest state, and the migration moves counts OUT of CONFIRMED only.

    .venv/bin/python scripts/m6_verify_extracted_facts.py --self-test
    .venv/bin/python scripts/m6_verify_extracted_facts.py
    .venv/bin/python scripts/m6_verify_extracted_facts.py --rebuild-facts-from-receipt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "runs_manifest" / "m6_s4_fact_verification.json"
# The checker's INPUT, committed beside its output so the run is reproducible from a clone.
FACTS_IN = ROOT / "runs_manifest" / "m6_s4_facts_input.json"

NUM = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def _dig(doc, path: str):
    """Resolve a dotted / bracketed key path; return (found, value)."""
    cur = doc
    for part in re.findall(r"[^.\[\]]+", path):
        if isinstance(cur, dict):
            if part not in cur:
                return False, None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return False, None
        else:
            return False, None
    return True, cur


def _nums(s: str) -> list[float]:
    out = []
    for m in NUM.findall(str(s).replace(",", "")):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


# ★ THE NEGATION VECTOR. The shipped string comparison was bidirectional substring containment:
# `str(actual) in claimed or claimed in str(actual)`. "supported" is a substring of
# "premise NOT supported", so a claim and its own negation CONFIRMED each other; so did "true"
# inside "untrue". Polarity is counted on both sides and must match before any containment is
# believed. This does not make containment safe in general — it makes it not silently invertible.
_NEG = re.compile(
    r"\b(not|no|never|none|without|cannot|can't|false|fails?|failed|absent|excluded|un\w+)\b"
)


def _polarity(s: str) -> int:
    return len(_NEG.findall(s.lower())) % 2


def _num_close(c: float, actual: float) -> bool:
    if actual == 0:
        return abs(c) < 1e-12
    return abs(c - actual) <= 5e-3 * max(abs(actual), 1e-9) or abs(c - actual) < 5e-4


def _values_agree(claimed: str, actual) -> bool | None:
    """True = confirms, False = refutes, None = NOT MACHINE-DECIDABLE (quarantine, never a pass).

    The third return value is the whole point. Every vector fixed here was a case where the
    checker had no basis to decide and returned True anyway, and a confirmation the checker cannot
    support is worse than no checker: it is a green light with a receipt behind it.
    """
    if isinstance(actual, bool) or isinstance(actual, str):
        a, c = str(actual).strip().lower(), str(claimed).strip().lower()
        if a == c or a == c.strip("'\"`"):
            return True
        if _polarity(a) != _polarity(c):
            return False  # a claim and its negation must never confirm each other
        # whole-token containment only: "true" must not be found inside "untrue"
        if re.search(rf"(?<![\w-]){re.escape(a)}(?![\w-])", c):
            return True
        return None if len(a) > 24 else False
    if isinstance(actual, (int, float)):
        # ★ THE ANY-NUMBER VECTOR. `for c in _nums(claimed): if match: return True` meant a claim
        # carrying several numbers CONFIRMED whenever ANY of them matched — "TFI 0.8223 vs
        # magnitude 0.9320" confirmed against 0.9320, i.e. against the number it was contrasting
        # with. A multi-number claim does not name a single value, so it is not machine-decidable
        # and goes to quarantine; the direction of that change is confirmations OUT, never in.
        nums = _nums(claimed)
        if not nums:
            return False
        if len({round(n, 12) for n in nums}) > 1:
            return True if all(_num_close(c, actual) for c in nums) else None
        return _num_close(nums[0], actual)
    if isinstance(actual, (list, dict)):
        # ★ THE CONTAINMENT VECTOR. `all(any(...))` confirmed a claim whose numbers appeared
        # ANYWHERE inside a container of any size — in a 35,000-element series, essentially always.
        # A claim about a container is confirmed only when it accounts for the container: the two
        # number SETS must match. Anything else is quarantined for a human.
        want = {round(w, 9) for w in _nums(claimed)}
        have = {round(h, 9) for h in _nums(json.dumps(actual))}
        if not want:
            return False
        if want == have:
            return True
        return None if want <= have else False
    return False


def check(fact: dict) -> dict:
    art = (fact.get("artifact") or "").strip()
    claimed = str(fact.get("value", ""))
    res = {"claim": fact.get("claim"), "value": claimed, "artifact": art}

    m = re.search(r"([\w/\-.]+\.json)\s*(?:::|->|→|\s)\s*([\w.\[\]]+)", art)
    if m:
        path, key = ROOT / m.group(1), m.group(2)
        if not path.exists():
            return {**res, "status": "REFUTED", "why": "receipt does not exist"}
        doc = json.loads(path.read_text())
        found, actual = _dig(doc, key)
        if not found:
            for alt in (key.split(".", 1)[-1], key.rsplit(".", 1)[-1]):
                found, actual = _dig(doc, alt)
                if found:
                    return {**res, "status": "DRIFTED", "why": f"key resolves only as {alt!r}"}
            return {**res, "status": "REFUTED", "why": "key path absent from receipt"}
        ok = _values_agree(claimed, actual)
        status = {True: "CONFIRMED", False: "REFUTED", None: "MANUAL"}[ok]
        return {
            **res,
            "status": status,
            "actual": actual if not isinstance(actual, (dict, list)) else "<container>",
            "why": {
                True: "",
                False: "value at that key differs from the claim",
                None: "the claim does not name a single machine-comparable value — quarantined "
                "rather than confirmed",
            }[ok],
        }

    m = re.search(r"([\w/\-.]+\.(?:py|md|toml|lock))(?::(\d+))?", art)
    if m:
        path = ROOT / m.group(1)
        if not path.exists():
            return {**res, "status": "REFUTED", "why": "file does not exist"}
        text = path.read_text()
        lines = text.splitlines()
        needle = claimed.strip().strip("'\"")
        if not needle:
            return {**res, "status": "MANUAL", "why": "no value to check"}
        if m.group(2):
            i = int(m.group(2)) - 1
            window = "\n".join(lines[max(0, i - 6) : i + 7])
            # ★ THE WINDOW VECTOR, and the loosest of the four. This read: "CONFIRMED if ANY
            # number anywhere in a 13-line window matches ANY number in the claim". Source windows
            # are dense with constants — indices, tolerances, shapes — so a claim naming any
            # ordinary value was confirmed by coincidence. Now the claim's literal text must be in
            # the window, or the claim must name exactly ONE number and that number must be there.
            claim_nums = _nums(claimed)
            single = len({round(n, 12) for n in claim_nums}) == 1
            if needle in window or (
                single and any(_num_close(claim_nums[0], v) for v in _nums(window))
            ):
                return {**res, "status": "CONFIRMED"}
        if needle in text:
            return {**res, "status": "DRIFTED", "why": "present in file, not at the cited line"}
        # A "value" that is a CODE SNIPPET cannot be string-matched: extractors reformat, elide
        # and truncate. Reporting those as REFUTED is the probe inventing findings from its own
        # limitation -- the failure this project has hit repeatedly. Snippets are quarantined for
        # human reading; only a simple token or number that is genuinely absent is a refutation.
        is_snippet = len(needle) > 24 or any(c in needle for c in "()[]{}=<>,:")
        if is_snippet:
            distinctive = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{4,}", needle)]
            if distinctive and any(t in text for t in distinctive):
                return {
                    **res,
                    "status": "MANUAL",
                    "why": "code snippet: symbols present, exact"
                    " text differs — needs human reading",
                }
            if distinctive:
                return {
                    **res,
                    "status": "REFUTED",
                    "why": f"none of its symbols {distinctive[:3]} occur in the cited file",
                }
            return {**res, "status": "MANUAL", "why": "code snippet not machine-checkable"}
        return {**res, "status": "REFUTED", "why": "value not found in the cited file"}

    return {**res, "status": "MANUAL", "why": "citation is not machine-resolvable"}


def self_test(facts: dict) -> bool:
    """The checker must be shown able to say REFUTED before any CONFIRMED it emits counts."""
    good = None
    for topic in facts.values():
        for f in topic["facts"]:
            if check(f)["status"] == "CONFIRMED" and ".json" in (f.get("artifact") or ""):
                good = f
                break
        if good:
            break
    if good is None:
        print("  SELF-TEST INCONCLUSIVE — no confirmable JSON fact to mutate")
        return False

    # Build the key mutation from the matched receipt path rather than by substitution: an
    # re.sub on "::" is a NO-OP for citations written without it, and a mutation that does not
    # mutate reports PASS for free. The mutation must be constructed, then asserted to differ.
    jm = re.search(r"([\w/\-.]+\.json)", good["artifact"])
    mutated_artifact = f"{jm.group(1)} :: no.such.key.at.all"
    assert mutated_artifact != good["artifact"], "key mutation did not change the citation"

    bad_val = check({**good, "value": "-999999.123"})
    bad_key = check({**good, "artifact": mutated_artifact})
    ok_v = bad_val["status"] == "REFUTED"
    ok_k = bad_key["status"] == "REFUTED"
    print(f"  mutation: corrupted VALUE -> {bad_val['status']:9s} {'PASS' if ok_v else 'FAIL'}")
    print(f"  mutation: corrupted KEY   -> {bad_key['status']:9s} {'PASS' if ok_k else 'FAIL'}")
    return ok_v and ok_k and _false_pass_controls()


def _false_pass_controls() -> bool:
    """★ Each case below WAS CONFIRMED by the version that shipped. None may confirm now.

    These are not hypothetical. Every one is the shape of a real citation in the corpus, and the
    checker said CONFIRMED to all four while claiming to be mutation-controlled — because the
    mutation control only proved it could refute a WRONG value, never that it declined an
    UNDECIDABLE one. A control that tests one failure mode certifies one failure mode.
    """
    cases = [
        ("negation", "premise NOT supported for ABS returns", "supported", False),
        ("substring-inversion", "the result is untrue", "true", False),
        ("any-number", "TFI 0.8223 vs magnitude 0.9320", 0.9320, None),
        ("container-any", "0.5", [0.1, 0.5, 0.9, 1.7, 42.0], None),
    ]
    ok = True
    for name, claimed, actual, want in cases:
        got = _values_agree(claimed, actual)
        good = got is want
        ok &= good
        verdict = {True: "CONFIRMED", False: "REFUTED", None: "QUARANTINED"}[got]
        print(f"  false-pass control [{name:19s}] -> {verdict:11s} {'PASS' if good else 'FAIL'}")
    return ok


def facts_from_receipt(path: pathlib.Path) -> dict:
    """Rebuild the checker's INPUT from its own committed output.

    ★ THE TOOL WAS NOT RE-RUNNABLE AT ALL. Its documented invocation is
    ``--facts /tmp/s4_facts.json``; that file was a session scratch file and no longer exists, so
    the receipt in runs_manifest/ could not be regenerated, audited or contradicted by anyone —
    including its author. The rows it stores carry exactly the three fields ``check()`` reads
    (claim, value, artifact), so the input is recoverable from the output and is committed beside
    it as ``m6_s4_facts_input.json``. A verification tool whose input is gone is a screenshot.
    """
    doc = json.loads(path.read_text())
    return {
        topic: {"facts": [{k: r.get(k) for k in ("claim", "value", "artifact")} for r in p["rows"]]}
        for topic, p in doc["per_topic"].items()
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--facts",
        default=str(FACTS_IN),
        help="extracted-facts input; defaults to the committed one so this is re-runnable",
    )
    ap.add_argument(
        "--rebuild-facts-from-receipt",
        action="store_true",
        help=f"regenerate {FACTS_IN.name} from the committed receipt, then exit",
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.rebuild_facts_from_receipt:
        rebuilt = facts_from_receipt(OUT)
        FACTS_IN.write_text(json.dumps(rebuilt, indent=2, sort_keys=True) + "\n")
        n = sum(len(t["facts"]) for t in rebuilt.values())
        print(f"rebuilt {n} facts across {len(rebuilt)} topics -> {FACTS_IN.name}")
        return 0

    facts_path = pathlib.Path(args.facts)
    if not facts_path.exists():
        print(
            f"no facts input at {facts_path}. The original was a session scratch file; recover it "
            f"with --rebuild-facts-from-receipt.",
            file=sys.stderr,
        )
        return 1
    facts = json.loads(facts_path.read_text())

    print("MUTATION CONTROL")
    if not self_test(facts):
        print("CHECKER REFUSES TO REPORT: it has not been shown capable of refuting.")
        return 1
    if args.self_test:
        return 0

    report, tally = {}, {}
    for topic, payload in facts.items():
        rows = [check(f) for f in payload["facts"]]
        counts = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            tally[r["status"]] = tally.get(r["status"], 0) + 1
        short = topic.split("—")[0].split("(")[0].strip()[:46]
        report[short] = {"counts": counts, "rows": rows}
        n = len(rows)
        print(
            f"\n{short:48s} n={n:4d}  " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        )
        for r in rows:
            if r["status"] == "REFUTED":
                print(f"    ✗ {str(r['claim'])[:78]}")
                print(
                    f"        claimed {str(r['value'])[:40]!r} · {r['why']} · {r['artifact'][:70]}"
                )

    OUT.write_text(
        json.dumps(
            {
                "what": "mechanical resolution of extracted facts against cited artifacts",
                "control": (
                    "value and key mutations must both report REFUTED first, AND four named "
                    "false-pass vectors (negation, substring-inversion, any-number, "
                    "container-any) must each come back REFUTED or QUARANTINED"
                ),
                # ★ READ THIS BEFORE READING THE COUNTS — and it travels with them, because a
                # count quoted without it is the thing this field exists to prevent.
                "READ_THIS_FIRST": (
                    "THE CONFIRMATION RATE FELL BY 65% WHEN THE CHECKER WAS FIXED, AND THE DROP "
                    "IS A PROPERTY OF THE CHECKER, NOT OF THE FACTS. The version that produced "
                    "the previous receipt reported CONFIRMED 424 / MANUAL 148 / REFUTED 51 / "
                    "DRIFTED 3. All four of its false-pass vectors were demonstrated live before "
                    "the fix and all four confirmed: the loosest, a source-line check that "
                    "accepted ANY number anywhere in a 13-line window matching ANY number in the "
                    "claim, accounts for most of the movement. Nothing here says a quarantined "
                    "fact is wrong — it says this tool never had grounds to call it right. A fact "
                    "is USABLE only on CONFIRMED; MANUAL is unread, not refuted."
                ),
                "superseded_totals_from_the_false_passing_version": {
                    "CONFIRMED": 424,
                    "MANUAL": 148,
                    "REFUTED": 51,
                    "DRIFTED": 3,
                },
                "totals": tally,
                "per_topic": report,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"\nTOTAL {tally}\nReceipt: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
