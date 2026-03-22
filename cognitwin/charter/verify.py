"""
CogniTwin Sovereign Charter — Verifier
=======================================

Verifies that the currently deployed (or locally checked-out) governed
artefacts match the hash recorded in a charter manifest.

Used by:
  • The charter-verify GitHub Actions workflow (CI gate)
  • RULE-006 inference rule evaluation
  • Local developer pre-push checks

Exit codes:
  0  — Verification passed (all hashes match)
  1  — Verification failed (deviation detected)
  2  — Manifest missing or unreadable

Usage:
    python -m cognitwin.charter.verify [--manifest PATH] [--repo-root PATH] [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(manifest_path: Path, repo_root: Path) -> dict:
    """
    Compare current file hashes against the manifest.

    Returns a result dict:
      {
        "passed": bool,
        "canonicalHash": str,          # recomputed from current files
        "expectedHash": str,           # from manifest
        "deviations": [{"path": str, "expected": str, "actual": str}],
        "missing": [str],
      }
    """
    if not manifest_path.exists():
        return {"passed": False, "error": f"Manifest not found: {manifest_path}"}

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    expected_hash = manifest.get("canonicalHash", "")
    file_hashes: dict[str, str | None] = manifest.get("files", {})

    deviations: list[dict] = []
    missing: list[str] = []
    current_hashes: dict[str, str] = {}

    for rel_path, expected_file_hash in file_hashes.items():
        abs_path = repo_root / rel_path
        actual_hash = sha256_file(abs_path)
        if actual_hash is None:
            missing.append(rel_path)
        elif actual_hash != expected_file_hash:
            deviations.append({
                "path": rel_path,
                "expected": expected_file_hash,
                "actual": actual_hash,
            })
        if actual_hash:
            current_hashes[rel_path] = actual_hash

    # Recompute canonical hash from current state
    canonical_input = json.dumps(
        {k: v for k, v in sorted(current_hashes.items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    recomputed_hash = hashlib.sha256(canonical_input.encode()).hexdigest()

    passed = (not deviations) and (not missing) and (recomputed_hash == expected_hash)
    return {
        "passed": passed,
        "canonicalHash": recomputed_hash,
        "expectedHash": expected_hash,
        "deviations": deviations,
        "missing": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CogniTwin Sovereign Charter")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest.json (default: cognitwin/charter/manifest.json relative to repo-root)",
    )
    parser.add_argument("--repo-root", default=".", help="Path to repo root (default: cwd)")
    parser.add_argument("--json", action="store_true", dest="output_json", help="Output JSON result")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else repo_root / "cognitwin" / "charter" / "manifest.json"
    )

    result = verify_manifest(manifest_path, repo_root)

    if args.output_json:
        print(json.dumps(result, indent=2))
        return 0 if result.get("passed") else (2 if "error" in result else 1)

    if "error" in result:
        print(f"[CHARTER] ERROR: {result['error']}")
        return 2

    print(f"\n{'─'*52}")
    print(f"  CogniTwin Sovereign Charter Verification")
    print(f"{'─'*52}")
    print(f"  Manifest:       {manifest_path}")
    print(f"  Expected hash:  {result['expectedHash'][:16]}…")
    print(f"  Computed hash:  {result['canonicalHash'][:16]}…")

    if result["missing"]:
        print(f"\n  MISSING files ({len(result['missing'])}):")
        for p in result["missing"]:
            print(f"    - {p}")

    if result["deviations"]:
        print(f"\n  DEVIATIONS ({len(result['deviations'])}):")
        for d in result["deviations"]:
            print(f"    {d['path']}")
            print(f"      expected: {d['expected'][:16]}…")
            print(f"      actual:   {d['actual'][:16]}…")

    if result["passed"]:
        print(f"\n  RESULT: PASS — System matches Sovereign Charter")
    else:
        print(f"\n  RESULT: FAIL — Charter deviation detected (INV-004 / RULE-006)")

    print(f"{'─'*52}\n")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
