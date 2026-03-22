"""
CogniTwin Governance — RuleEvaluation Audit Logger
===================================================

Append-only audit log for RuleEvaluation entries.
Backed by a local NDJSON file; can be extended to write to any
append-only store (Postgres, Firestore, S3, etc.).

Sovereign Invariant INV-003 mandates that audit entries are never
deleted or updated — this module enforces that by opening the log
file in append mode only and offering no delete/update API.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Iterator


AUDIT_LOG_PATH_ENV = "COGNITWIN_AUDIT_LOG"
DEFAULT_AUDIT_LOG = "cognitwin_audit.ndjson"


@dataclass
class RuleEvaluationEntry:
    entry_id: str
    ts: str                   # ISO-8601 UTC
    schema_version: str       # "0.1.0"
    rule_id: str
    rule_name: str
    outcome: str              # pass | flag | reject
    triggered_by: str         # PR number, commit SHA, release tag, …
    severity: str             # info | warning | critical
    rationale: str
    actor_hash: str | None    # hex digest or None
    session_id: str | None


class AuditLogger:
    """
    Append-only writer for RuleEvaluation audit entries.

    Thread-safe for a single process; use an external lock or a
    database-backed store for multi-process deployments.
    """

    def __init__(self, log_path: str | Path | None = None) -> None:
        if log_path is None:
            log_path = os.environ.get(AUDIT_LOG_PATH_ENV, DEFAULT_AUDIT_LOG)
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Write ─────────────────────────────────────────────────────────────────

    def log(
        self,
        rule_id: str,
        rule_name: str,
        outcome: str,
        triggered_by: str,
        severity: str,
        rationale: str,
        actor_hash: str | None = None,
        session_id: str | None = None,
    ) -> RuleEvaluationEntry:
        entry = RuleEvaluationEntry(
            entry_id=str(uuid.uuid4()),
            ts=datetime.now(timezone.utc).isoformat(),
            schema_version="0.1.0",
            rule_id=rule_id,
            rule_name=rule_name,
            outcome=outcome,
            triggered_by=triggered_by,
            severity=severity,
            rationale=rationale,
            actor_hash=actor_hash,
            session_id=session_id,
        )
        # Append-only: 'a' mode — no update, no delete
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry)) + "\n")
        return entry

    # ── Read ──────────────────────────────────────────────────────────────────

    def iter_entries(self) -> Iterator[RuleEvaluationEntry]:
        """Iterate all audit entries in chronological order."""
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    yield RuleEvaluationEntry(**data)

    def filter_by_outcome(self, outcome: str) -> list[RuleEvaluationEntry]:
        return [e for e in self.iter_entries() if e.outcome == outcome]

    def filter_by_rule(self, rule_id: str) -> list[RuleEvaluationEntry]:
        return [e for e in self.iter_entries() if e.rule_id == rule_id]

    def tail(self, n: int = 20) -> list[RuleEvaluationEntry]:
        """Return the last n entries."""
        entries = list(self.iter_entries())
        return entries[-n:]

    # ── Integrity ─────────────────────────────────────────────────────────────

    def entry_count(self) -> int:
        if not self._path.exists():
            return 0
        with self._path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())


# ── Module-level singleton ────────────────────────────────────────────────────

_default_logger: AuditLogger | None = None


def get_audit_logger(log_path: str | Path | None = None) -> AuditLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = AuditLogger(log_path)
    return _default_logger


def log_rule_evaluation(
    rule_id: str,
    rule_name: str,
    outcome: str,
    triggered_by: str,
    severity: str,
    rationale: str,
    actor_hash: str | None = None,
    session_id: str | None = None,
) -> RuleEvaluationEntry:
    """Convenience wrapper around the default audit logger."""
    return get_audit_logger().log(
        rule_id=rule_id,
        rule_name=rule_name,
        outcome=outcome,
        triggered_by=triggered_by,
        severity=severity,
        rationale=rationale,
        actor_hash=actor_hash,
        session_id=session_id,
    )
