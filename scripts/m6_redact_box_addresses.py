"""Replace rented-box public IPv4 addresses with stable digests, preserving the distinctness claim.

WHY THIS EXISTS AND WHY IT IS NOT A DELETION. Two tracked receipts recorded the public IP of every
rented GPU box, under the field name ``public_ipaddr``. They are there for a reason: the M6 fan-out
had to prove the shards ran on THREE DISTINCT MACHINES, and ``host_id`` cannot do that (one host_id
is a provider ACCOUNT operating many machines — the receipt says so itself). Deleting the addresses
would delete the evidence.

Three distinct digests prove three distinct machines exactly as well as three addresses do, so the
claim survives the substitution intact. The SSH ``HostPort`` is dropped outright: it proves nothing
and is the only field that was ever an access hint.

★ WHAT THIS DOES NOT CLAIM. IPv4 is a 32-bit space, so a bare SHA-256 of an address is recoverable
by exhaustive search in seconds. This is deliberately NOT presented as a cryptographic guarantee.
What it buys is real but narrow: the addresses leave the plaintext surface that automated scanners,
code search and crawlers actually read, and the provenance claim keeps verifying. The instances
themselves were destroyed in August 2026. Overstating this would be worse than leaving it alone.

    .venv/bin/python scripts/m6_redact_box_addresses.py [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import sys
from pathlib import Path

from trikaal.utils.paths import display_path

REPO = Path(__file__).resolve().parents[1]
TARGETS = (
    REPO / "runs_manifest/m6_pool_identity.json",
    REPO / "runs_manifest/m6_run_box_identity.json",
)
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

NOTE = (
    "REDACTED, NOT DELETED. Each rented box's public IPv4 was replaced by "
    "sha256(address)[:16]. THE CLAIM THIS FILE MAKES IS DISTINCTNESS -- that the shards ran on "
    "different physical machines, which host_id cannot establish because one host_id is a "
    "provider ACCOUNT operating many machines -- and three distinct digests carry it exactly as "
    "well as three addresses. NOT A CRYPTOGRAPHIC GUARANTEE: IPv4 is a 32-bit space, so the "
    "digest is recoverable by exhaustive search; what the substitution buys is removal from the "
    "plaintext surface that code search and automated scanners read. The instances were destroyed "
    "in August 2026. The SSH HostPort was dropped entirely -- it proved nothing."
)


def digest(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def routable(text: str) -> list[str]:
    out = []
    for m in IPV4.finditer(text):
        try:
            addr = ipaddress.ip_address(m.group(0))
        except ValueError:
            continue
        if addr.is_global:
            out.append(str(addr))
    return sorted(set(out))


def redact(text: str) -> tuple[str, dict[str, str]]:
    mapping = {ip: digest(ip) for ip in routable(text)}
    for ip, dig in mapping.items():
        text = text.replace(ip, f"sha256:{dig}")
    # the endpoint string also carried an SSH HostPort; it proves nothing and is removed
    text = re.sub(r"\s*\+\s*ports\['22/tcp'\]\[0\]\.HostPort\s+\d+", " + [HostPort REDACTED]", text)
    return text, mapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only; write nothing")
    args = ap.parse_args()

    rc = 0
    for path in TARGETS:
        doc = json.loads(path.read_text())
        before = routable(path.read_text())
        if args.check:
            state = "CLEAN" if not before else f"{len(before)} ROUTABLE ADDRESS(ES)"
            print(f"{display_path(path, REPO)}: {state}")
            rc |= 1 if before else 0
            continue
        if not before:
            print(f"{display_path(path, REPO)}: already clean")
            continue
        text, mapping = redact(json.dumps(doc, indent=1, sort_keys=True))
        doc = json.loads(text)
        doc["ADDRESS_REDACTION"] = NOTE
        doc["address_digests"] = {"algorithm": "sha256(address)[:16]", "n_distinct": len(mapping)}
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"{display_path(path, REPO)}: {len(mapping)} address(es) -> digests")
    return rc


if __name__ == "__main__":
    sys.exit(main())
