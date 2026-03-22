"""
CogniTwin Governance — Sovereign Invariant Definitions
=======================================================

Sovereign invariants are the non-negotiable logical constraints that
must hold across every deployed version of the CogniTwin system.
Violating any invariant is grounds for deployment rejection.

Any PR that modifies this file is automatically flagged by the
governance PR checker and requires an explicit governance review.

SHA-256 of this module's canonical content is recorded in the
Sovereign Charter manifest (cognitwin/charter/manifest.json).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable


class InvariantTier(str, Enum):
    """Severity tier — determines rejection vs warning behaviour."""
    CRITICAL = "critical"   # PR must be rejected on violation
    MAJOR = "major"         # PR flagged + human review required
    MINOR = "minor"         # logged only


@dataclass(frozen=True)
class SovereignInvariant:
    id: str
    name: str
    description: str
    tier: InvariantTier
    # Predicate: receives a dict of system context and returns True if invariant holds
    predicate: Callable[[dict], bool]
    # File-path globs that, when changed in a PR, trigger evaluation of this invariant
    watch_paths: tuple[str, ...] = field(default_factory=tuple)


# ─── Invariant Registry ───────────────────────────────────────────────────────

def _inv_schema_version_present(ctx: dict) -> bool:
    """Every event emitted by the SDK must carry schemaVersion '0.1.0'."""
    return ctx.get("schemaVersion") == "0.1.0"


def _inv_actor_hash_no_pii(ctx: dict) -> bool:
    """actorHash must never be a plaintext email or username (must be hex >= 32 chars or absent)."""
    actor = ctx.get("actorHash")
    if actor is None:
        return True
    return len(actor) >= 32 and all(c in "0123456789abcdef" for c in actor.lower())


def _inv_audit_log_immutable(ctx: dict) -> bool:
    """RuleEvaluation audit entries must never be deleted or updated — append-only."""
    return ctx.get("auditOperationType") in (None, "insert")


def _inv_charter_hash_matches(ctx: dict) -> bool:
    """Deployed system hash must match the hash recorded in the release tag manifest."""
    deployed_hash = ctx.get("deployedHash")
    charter_hash = ctx.get("charterHash")
    if deployed_hash is None or charter_hash is None:
        return False
    return deployed_hash == charter_hash


def _inv_no_plaintext_credentials(ctx: dict) -> bool:
    """No event payload may contain keys matching credential patterns."""
    forbidden = {"password", "secret", "token", "apikey", "api_key", "private_key"}
    payload = ctx.get("payload", {})
    if isinstance(payload, dict):
        return not any(k.lower() in forbidden for k in payload)
    return True


def _inv_commit_session_linked(ctx: dict) -> bool:
    """Every CommitEvent must carry a non-empty sessionId."""
    if ctx.get("type") != "commit":
        return True
    return bool(ctx.get("sessionId"))


SOVEREIGN_INVARIANTS: tuple[SovereignInvariant, ...] = (
    SovereignInvariant(
        id="INV-001",
        name="schema_version_present",
        description="Every SDK event must carry schemaVersion '0.1.0'.",
        tier=InvariantTier.CRITICAL,
        predicate=_inv_schema_version_present,
        watch_paths=("cognitwin/sdk/schema.ts",),
    ),
    SovereignInvariant(
        id="INV-002",
        name="actor_hash_no_pii",
        description="actorHash must be a hex digest of >= 32 chars, never plaintext identity.",
        tier=InvariantTier.CRITICAL,
        predicate=_inv_actor_hash_no_pii,
        watch_paths=("cognitwin/sdk/",),
    ),
    SovereignInvariant(
        id="INV-003",
        name="audit_log_immutable",
        description="RuleEvaluation entries are append-only; no delete or update is permitted.",
        tier=InvariantTier.CRITICAL,
        predicate=_inv_audit_log_immutable,
        watch_paths=("cognitwin/governance/audit.py",),
    ),
    SovereignInvariant(
        id="INV-004",
        name="charter_hash_integrity",
        description="Deployed system hash must match the Sovereign Charter release manifest.",
        tier=InvariantTier.CRITICAL,
        predicate=_inv_charter_hash_matches,
        watch_paths=("cognitwin/charter/",),
    ),
    SovereignInvariant(
        id="INV-005",
        name="no_plaintext_credentials",
        description="Event payloads must never include plaintext credential fields.",
        tier=InvariantTier.MAJOR,
        predicate=_inv_no_plaintext_credentials,
        watch_paths=("cognitwin/sdk/",),
    ),
    SovereignInvariant(
        id="INV-006",
        name="commit_session_linked",
        description="Every CommitEvent must carry a non-empty sessionId for twin linkage.",
        tier=InvariantTier.MAJOR,
        predicate=_inv_commit_session_linked,
        watch_paths=("cognitwin/sdk/schema.ts", "cognitwin/behavioral/"),
    ),
)


# ─── Evaluation ───────────────────────────────────────────────────────────────

@dataclass
class InvariantResult:
    invariant_id: str
    invariant_name: str
    tier: InvariantTier
    passed: bool
    rationale: str


def evaluate_all(ctx: dict) -> list[InvariantResult]:
    """Evaluate every sovereign invariant against the provided context dict."""
    results: list[InvariantResult] = []
    for inv in SOVEREIGN_INVARIANTS:
        try:
            passed = inv.predicate(ctx)
            rationale = "OK" if passed else f"Invariant {inv.id} violated."
        except Exception as exc:
            passed = False
            rationale = f"Predicate raised: {exc}"
        results.append(
            InvariantResult(
                invariant_id=inv.id,
                invariant_name=inv.name,
                tier=inv.tier,
                passed=passed,
                rationale=rationale,
            )
        )
    return results


def invariant_manifest_hash() -> str:
    """Return SHA-256 of the canonical serialised invariant registry (ids + descriptions)."""
    canonical = json.dumps(
        [
            {"id": inv.id, "name": inv.name, "description": inv.description, "tier": inv.tier.value}
            for inv in SOVEREIGN_INVARIANTS
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
