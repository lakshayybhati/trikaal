"""Sweep runs_manifest/ for receipt keys that no code ever wrote.

WHY THIS EXISTS. The paper's §3 cited `bits_per_dim_at_rho_0.951 = 1.78` from
runs_manifest/m6_interface_respec_design_pass.json. Adjudication (2026-08-09) established that
the number was never computed: the producing script, at every revision it has ever had, does not
construct the block that contains it. It was hand-authored into the JSON and then quoted onward
into a commit message, into docs/BUILD_RECORD.md, and nearly into a submitted paper.

That is a CLASS defect, not an instance. The project's epistemology rests on runs_manifest/ being
machine-written; a receipt that mixes hand-typed numbers with computed ones, with nothing marking
which is which, is not a receipt. This sweep is the tool the class rule asks for.

METHOD. For each receipt, find the script(s) that name it, collect every string literal those
scripts (and the library modules they import from src/trikaal) can emit, and walk the JSON
reporting keys that appear in no literal. Keys that are obviously data rather than schema --
integers, symbol names, hashes, seeds -- are excluded, because those are legitimately built at
runtime. The strong signal is a whole top-level block none of whose keys appear anywhere in the
producing code: that is what a hand-edit looks like.

THE TOOL IS SUSPECT UNTIL SHOWN CAPABLE OF FAILING. --self-test runs three controls:
  POSITIVE  the known defect at git commit a1729b8 must be flagged. If it is not, the sweep is
            broken and says so rather than reporting a clean bill.
  NEGATIVE  a receipt written end-to-end by its script must come back clean, so the detector is
            not simply flagging everything.
  WASHING   prose -- docstrings, and sentence-shaped string literals -- must NOT enter the
            vocabulary. The sweep shipped word-splitting EVERY string constant, so a key occurring
            only in someone's narrative paragraph counted as "written by code". Measured over
            scripts/ + src/, 6,701 of a 17,058-token vocabulary entered ONLY via prose.

TWO FALSE-PASS VECTORS WERE FIXED HERE, AND I HAD THE CAUSAL STORY BACKWARDS ON THE SECOND ONE.
`runs_manifest/m6_demo_seed_sign_disagreement.json` was committed with NO producer anywhere in the
repository and this sweep reported it clean. I wrote that washing was the reason, then measured it
instead of leaving it asserted: excluding prose moves that receipt from 11 to 14 untraceable key
paths and STILL would not have flagged it. The actual reason is that `hand_authored_blocks` is
BLOCK-shaped -- it needs a nested object of >=3 keys before it will say anything -- and that
receipt is flat top-level scalars plus one list. So `hand_authored_flat` was added. Both vectors
are real and independent; closing one while believing it was the other is the failure this note
exists to prevent.

Exit code is 1 if any control fails, or if --strict and any finding is reported.

    python3 scripts/m6_receipt_provenance_sweep.py --self-test
    python3 scripts/m6_receipt_provenance_sweep.py
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
RM = ROOT / "runs_manifest"
OUT = RM / "m6_receipt_provenance_sweep.json"

# Keys that are data, not schema: built at runtime from loop variables, so absence from the
# source is expected and carries no signal.
DATA_KEY = re.compile(
    r"""^(
        -?\d+(\.\d+)?            # integer / float indices  (per_dim "9", levels)
      | [A-Z0-9]{2,}USDT         # instrument symbols
      | [0-9a-f]{8,}             # hashes
      | seed\d+ | cell\d+.*      # generated run names
      | \d{4}-\d{2}-\d{2}.*      # dates
      | .*[/|].*                 # file paths and prose scenario labels used as keys
      | .*\s.*                   # any key containing whitespace is a human label, not schema
    )$""",
    re.X,
)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """id() of every Constant that is a DOCSTRING — module, class or function.

    ★ THE WASHING VECTOR. The first version word-split EVERY string constant, docstrings included,
    into identifier-like tokens and put them all in the vocabulary. A docstring is prose ABOUT the
    code, not something the code emits: a receipt key that appears nowhere but in a comment or a
    narrative paragraph was being treated as traceable to a producer. Measured across scripts/ +
    src/, 6,701 of 17,058 vocabulary tokens came in that way and no other — 39% of the evidence
    this tool weighs was somebody's prose.

    NOT, HOWEVER, THE REASON THE PRODUCERLESS RECEIPT PASSED. See the module docstring: that was
    the block-shaped detector, a separate vector. The two were measured apart precisely because
    the first story was wrong.
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def literals_in(path: pathlib.Path | str, *, at_commit: str | None = None) -> set[str]:
    """Every string constant a module could EMIT — docstrings and prose deliberately excluded."""
    if at_commit:
        try:
            src = subprocess.run(
                ["git", "show", f"{at_commit}:{path}"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        except subprocess.CalledProcessError:
            return set()
    else:
        p = pathlib.Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            return set()
        src = p.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    docstrings = _docstring_nodes(tree)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue  # prose about the code is not the code emitting a key
            out.add(node.value)
            # A literal like "a.b" or an f-string fragment can still COVER a key, so it is split.
            # A literal containing WHITESPACE is a sentence — an error message, a note field, a
            # "what this proves" paragraph — and splitting it is the same washing as a docstring.
            if not any(c.isspace() for c in node.value):
                out.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", node.value))
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Name):
            out.add(node.id)
    return out


def json_keys(obj, prefix: str = "") -> list[tuple[str, str]]:
    """(full path, leaf key) for every dict key in the document."""
    found: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            found.append((path, k))
            found.extend(json_keys(v, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):  # shape repeats; three is enough to see the schema
            found.extend(json_keys(v, f"{prefix}[{i}]"))
    return found


def _grep_l(needle: str) -> list[str]:
    try:
        hits = subprocess.run(
            ["git", "grep", "-l", "--", needle, "scripts/", "src/"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.split()
    except Exception:
        hits = []
    return [h for h in hits if h.endswith(".py")]


def producing_scripts(receipt_name: str, doc: dict | None = None) -> tuple[list[str], str]:
    """(sources, how they were found).

    Receipts whose output path is built at runtime -- an ``--out`` flag, an f-string -- are NOT
    findable by filename. The first version of this sweep returned [] for those, the vocabulary
    collapsed to src/ alone, and fifteen script-built blocks were reported as hand-authored. That
    is the probe manufacturing findings from its own bug. Three fallbacks now run before the
    conservative one, and the resolution method is recorded per receipt so a reader can see which
    receipts were checked against a specific producer and which against the whole corpus.
    """
    hits = _grep_l(receipt_name)
    if hits:
        return hits, "by-filename"
    stem = receipt_name[:-5] if receipt_name.endswith(".json") else receipt_name
    hits = _grep_l(stem)
    if hits:
        return hits, "by-stem"
    if isinstance(doc, dict):
        for key in ("artifact", "probe", "canary", "control", "what"):
            v = doc.get(key)
            if isinstance(v, str) and len(v) > 8:
                hits = _grep_l(v)
                if hits:
                    return hits, f"by-{key}-field"
    # Conservative fallback: the union of every script. Under-reports rather than invents.
    return sorted(str(q.relative_to(ROOT)) for q in (ROOT / "scripts").glob("*.py")), "corpus-wide"


_CORPUS_VOCAB: set[str] | None = None


def corpus_vocab() -> set[str]:
    """Every string any script or library module in the repository could emit.

    The criterion is deliberately corpus-wide, not producer-scoped. Scoping to the resolved
    producer flagged two blocks (m6_rehearsal_manifest :: gpu_sampler and :: cell5_permuted_micro)
    whose every key is in fact written by code the resolver had not attributed. A finding that
    survives "no file in the repository contains this key" is one an auditor can check in one
    grep; a producer-scoped finding is only as good as the resolver. This under-reports -- a key
    written by one script and hand-copied into another receipt reads as traceable -- and that is
    the correct direction for a tool whose output is used to retract published numbers.
    """
    global _CORPUS_VOCAB
    if _CORPUS_VOCAB is None:
        v: set[str] = set()
        for mod in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "src").rglob("*.py")):
            if mod.name == pathlib.Path(__file__).name:
                # this sweep names the keys it reports; excluding it keeps the check honest
                continue
            v |= literals_in(mod)
        _CORPUS_VOCAB = v
    return _CORPUS_VOCAB


def audit(doc: dict, sources: list[str], *, at_commit: str | None = None) -> dict:
    if at_commit is not None:
        vocab: set[str] = set()
        for s in sources:
            vocab |= literals_in(s, at_commit=at_commit)
    else:
        vocab = corpus_vocab()

    orphans, by_block = [], {}
    for path, key in json_keys(doc):
        if DATA_KEY.match(key) or key in vocab:
            continue
        orphans.append(path)
        by_block.setdefault(path.split(".")[0], []).append(path)

    # the strong signal: a top-level block with no key traceable to any source
    hand_authored = []
    for block, paths in by_block.items():
        inner = {p for p, k in json_keys(doc.get(block, {}), block) if not DATA_KEY.match(k)}
        total = inner | {block}  # the block's own key counts on both sides
        orphans_here = {p for p in paths if p in total}
        if len(total) >= 3 and len(orphans_here) >= 0.8 * len(total):
            hand_authored.append(
                {"block": block, "orphan_keys": len(orphans_here), "total_keys": len(total)}
            )

    # ★ THE SECOND FALSE-PASS VECTOR, AND THE ONE THAT ACTUALLY LET A RECEIPT THROUGH. The check
    # above is BLOCK-shaped: it needs a nested object of >=3 keys before it will say anything. A
    # receipt hand-authored as FLAT top-level scalars plus one list -- which is exactly the shape
    # of the producerless m6_demo_seed_sign_disagreement.json -- presents no block for it to
    # measure, so it was reported clean with eleven untraceable keys sitting in plain sight. I
    # assumed the cause was vocabulary washing and measured it instead of asserting it: the fix to
    # washing moves that receipt from 11 to 14 untraceable key paths and still would not have
    # flagged it, because there was never a block. Two independent vectors, one of which I would
    # have closed while believing I had closed the other.
    flat_orphans = sorted(
        p
        for p in orphans
        if "." not in p and "[" not in p and not isinstance(doc.get(p), (dict, list))
    )
    hand_authored_flat = (
        {
            "orphan_top_level_scalars": flat_orphans,
            "n": len(flat_orphans),
            "why": "top-level scalar keys traceable to no source; a receipt with no nested block "
            "is invisible to the block heuristic above",
        }
        if len(flat_orphans) >= 3
        else None
    )
    return {
        "orphan_paths": sorted(orphans),
        "hand_authored_blocks": hand_authored,
        "hand_authored_flat": hand_authored_flat,
    }


def self_test() -> bool:
    ok = True
    print("SELF-TEST — the sweep must be shown capable of failing before its output means anything")

    # POSITIVE CONTROL: the known hand-authored block at a1729b8
    doc = json.loads(
        subprocess.run(
            ["git", "show", "a1729b8:runs_manifest/m6_interface_respec_design_pass.json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    res = audit(doc, ["scripts/m6_interface_respec.py"], at_commit="a1729b8")
    blocks = [b["block"] for b in res["hand_authored_blocks"]]
    hit = "gate2_blocked_diagnosis" in blocks
    print(
        f"  positive control  a1729b8 gate2_blocked_diagnosis flagged : {'PASS' if hit else 'FAIL'}"
        f"   (blocks flagged: {blocks})"
    )
    ok &= hit

    # NEGATIVE CONTROL: same receipt, same commit, but a block the script demonstrably builds
    clean = "g_parity_new_layout" not in blocks and "tokenizers" not in blocks
    print(
        f"  negative control  script-built blocks NOT flagged          : "
        f"{'PASS' if clean else 'FAIL'}"
    )
    ok &= clean

    # ★ WASHING CONTROL. The vector this sweep shipped with: a key that appears ONLY in prose --
    # a docstring, a sentence-shaped error message -- used to enter the vocabulary and mark a
    # hand-authored receipt as traceable. Both halves are probed here against a synthetic module,
    # because a control that only tests the docstring half would have missed the sentence half.
    src = (
        '"""A docstring that mentions washed_by_docstring and nothing else."""\n'
        'MSG = "a sentence mentioning washed_by_sentence in prose"\n'
        'D = {"genuinely_emitted": 1}\n'
        'K = "dotted.key_literal"\n'
    )
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "probe.py"
        p.write_text(src)
        vocab = literals_in(p)
    washed = [t for t in ("washed_by_docstring", "washed_by_sentence") if t in vocab]
    kept = [t for t in ("genuinely_emitted", "key_literal") if t in vocab]
    wash_ok = not washed and len(kept) == 2
    print(
        f"  washing control   prose does NOT enter the vocabulary      : "
        f"{'PASS' if wash_ok else 'FAIL'}"
        f"   (washed-in: {washed or 'none'}; emitted keys kept: {kept})"
    )
    ok &= wash_ok
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any receipt is flagged")
    args = ap.parse_args()

    if args.self_test:
        return 0 if self_test() else 1

    if not self_test():
        print(
            "\nSWEEP REFUSES TO REPORT: its own controls failed, so a clean result would be "
            "indistinguishable from a broken detector."
        )
        return 1

    print("\nSWEEP over runs_manifest/")
    report, flagged = {}, 0
    for f in sorted(RM.glob("*.json")):
        if f.name == OUT.name:
            continue
        try:
            doc = json.loads(f.read_text())
        except Exception as exc:
            report[f.name] = {"error": str(exc)}
            continue
        srcs, how = producing_scripts(f.name, doc)
        res = audit(doc, srcs)
        res["producing_scripts"] = srcs
        res["resolved"] = how
        report[f.name] = res
        if res["hand_authored_blocks"] or res["hand_authored_flat"]:
            flagged += 1
            for b in res["hand_authored_blocks"]:
                print(
                    f"  ★ {f.name} :: {b['block']}  "
                    f"({b['orphan_keys']}/{b['total_keys']} keys untraceable to any source)"
                )
            if res["hand_authored_flat"]:
                fl = res["hand_authored_flat"]
                print(
                    f"  ★ {f.name} :: <top level>  ({fl['n']} untraceable scalar keys: "
                    f"{', '.join(fl['orphan_top_level_scalars'][:5])})"
                )
        elif how == "corpus-wide":
            print(f"    {f.name}  — clean (producer not identifiable; checked corpus-wide)")

    payload = {
        "what": "receipt keys that appear in no source that could have written them",
        "method": "a key is reported only if it appears as no string literal in ANY script or "
        "library module in the repository; runtime-built data keys (indices, symbols, "
        "hashes, paths, prose labels) excluded. Corpus-wide by design: see corpus_vocab()",
        "controls": "positive = a1729b8 gate2_blocked_diagnosis must flag; "
        "negative = script-built blocks in the same file must not",
        "receipts_scanned": len(report),
        "receipts_flagged": flagged,
        "per_receipt": report,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\n{flagged} of {len(report)} receipts carry a hand-authored block. Receipt: {OUT}")
    return 1 if (args.strict and flagged) else 0


if __name__ == "__main__":
    sys.exit(main())
