"""
CogniTwin Sovereign Charter — Manifest Generator
=================================================

Computes and writes the hash manifest for the current state of all
governed artefacts.  Run this script to produce a new manifest.json
that should be attached to a GitHub release tag.

Usage:
    python -m cognitwin.charter.generate_manifest [--repo-root PATH] [--out PATH]

The generated manifest is stored in cognitwin/charter/manifest.json
and should be committed + tagged as:

    git tag -a charter/v<semver> -m "Sovereign Charter v<semver>"
    git push origin charter/v<semver>

The tag name and manifest hash must match for charter verification
to pass (RULE-006 / INV-004).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


# Files and directories whose content is part of the sovereign charter.
# Any change to these paths must be reflected in a new charter release.
GOVERNED_PATHS: tuple[str, ...] = (
    "cognitwin/sdk/schema.ts",
    "cognitwin/sdk/events.ts",
    "cognitwin/sdk/client.ts",
    "cognitwin/sdk/index.ts",
    "cognitwin/governance/invariants.py",
    "cognitwin/governance/rules.py",
    "cognitwin/governance/audit.py",
    "cognitwin/governance/__init__.py",
    "cognitwin/charter/generate_manifest.py",
    "cognitwin/charter/verify.py",
)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_manifest(repo_root: Path) -> dict:
    file_hashes: dict[str, str | None] = {}
    for rel in GOVERNED_PATHS:
        abs_path = repo_root / rel
        file_hashes[rel] = sha256_file(abs_path)

    # Canonical hash = SHA-256 of all (path, hash) pairs sorted by path
    canonical_input = json.dumps(
        {k: v for k, v in sorted(file_hashes.items()) if v is not None},
        sort_keys=True,
        separators=(",", ":"),
    )
    canonical_hash = hashlib.sha256(canonical_input.encode()).hexdigest()

    return {
        "schemaVersion": "0.1.0",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "canonicalHash": canonical_hash,
        "files": file_hashes,
        "governedPaths": list(GOVERNED_PATHS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CogniTwin Sovereign Charter manifest")
    parser.add_argument("--repo-root", default=".", help="Path to repo root (default: cwd)")
    parser.add_argument("--out", default=None, help="Output path (default: cognitwin/charter/manifest.json)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_path = Path(args.out) if args.out else repo_root / "cognitwin" / "charter" / "manifest.json"

    manifest = compute_manifest(repo_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    print(f"Manifest written to: {out_path}")
    print(f"Canonical hash:      {manifest['canonicalHash']}")
    print(f"\nTo tag this charter:")
    print(f"  git tag -a charter/v0.1.0 -m 'Sovereign Charter v0.1.0'")
    print(f"  git push origin charter/v0.1.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
