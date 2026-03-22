"""
CogniTwin Behavioral — Twin Feed
=================================

Converts CommitSignals into CogniTwin SDK events and ships them
to the ingest endpoint (or writes them to a local NDJSON file for
offline / test use).

Also applies RULE-001 through RULE-003 (commit-stream rules) and
writes the resulting RuleEvaluation audit entries.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .commit_signals import CommitSignals, RawCommit


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def signals_to_commit_events(
    signals: CommitSignals,
    commits: list[RawCommit],
    session_id: str | None = None,
    actor_hash: str | None = None,
) -> list[dict]:
    """Convert raw commits to CogniTwin CommitEvent dicts."""
    sid = session_id or str(uuid.uuid4())
    events: list[dict] = []
    sorted_commits = sorted(commits, key=lambda c: c.author_epoch)
    for i, commit in enumerate(sorted_commits):
        prev_epoch = sorted_commits[i - 1].author_epoch if i > 0 else None
        interval_min = (
            (commit.author_epoch - prev_epoch) / 60 if prev_epoch is not None else None
        )
        events.append({
            "id": str(uuid.uuid4()),
            "ts": datetime.fromtimestamp(commit.author_epoch, tz=timezone.utc).isoformat(),
            "seq": i + 1,
            "source": "repository",
            "type": "commit",
            "sessionId": sid,
            "actorHash": actor_hash,
            "schemaVersion": "0.1.0",
            "payload": {
                "sha": commit.sha,
                "hourOfDay": commit.hour_of_day,
                "dayOfWeek": commit.day_of_week,
                "messageLengthBytes": commit.message_length_bytes,
                "isAmend": commit.is_amend,
                "linesAdded": commit.lines_added,
                "linesRemoved": commit.lines_removed,
                "filesChanged": commit.files_changed,
                "interCommitIntervalMin": interval_min,
            },
        })
    return events


def apply_commit_rules(
    commits: list[RawCommit],
    session_id: str,
    triggered_by: str = "commit_feed",
) -> list[dict]:
    """
    Run commit-stream inference rules and return RuleEvaluation event dicts.
    Also writes entries to the audit log.
    """
    from cognitwin.governance.rules import RULE_INDEX, RuleOutcome
    from cognitwin.governance.audit import log_rule_evaluation

    commit_payload = {
        "commits": [
            {
                "sha": c.sha,
                "hourOfDay": c.hour_of_day,
                "isAmend": c.is_amend,
                "messageLengthBytes": c.message_length_bytes,
            }
            for c in commits
        ]
    }

    rule_events: list[dict] = []
    for rule_id in ("RULE-001", "RULE-002", "RULE-003"):
        rule = RULE_INDEX.get(rule_id)
        if rule is None:
            continue
        outcome, rationale = rule.evaluate(commit_payload)
        severity = (
            rule.severity_on_reject
            if outcome == RuleOutcome.REJECT
            else (rule.severity_on_flag if outcome == RuleOutcome.FLAG else "info")
        )
        # Persist to audit log
        entry = log_rule_evaluation(
            rule_id=rule_id,
            rule_name=rule.name,
            outcome=outcome.value,
            triggered_by=triggered_by,
            severity=severity,
            rationale=rationale,
            session_id=session_id,
        )
        rule_events.append({
            "id": str(uuid.uuid4()),
            "ts": _now_iso(),
            "seq": 0,
            "source": "system",
            "type": "rule_evaluation",
            "sessionId": session_id,
            "schemaVersion": "0.1.0",
            "payload": {
                "ruleId": rule_id,
                "ruleName": rule.name,
                "outcome": outcome.value,
                "triggeredBy": triggered_by,
                "severity": severity,
                "rationale": rationale,
            },
        })
    return rule_events


def write_events_ndjson(events: list[dict], path: str | Path) -> None:
    """Write event dicts to a NDJSON file (append mode)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for evt in events:
            fh.write(json.dumps(evt) + "\n")


async def send_events_http(events: list[dict], ingest_url: str, auth_token: str | None = None) -> None:
    """POST events to a CogniTwin ingest endpoint (async)."""
    import urllib.request
    import urllib.error

    headers = {
        "Content-Type": "application/json",
        "X-CogniTwin-Schema": "0.1.0",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    body = json.dumps({"events": events}).encode()
    req = urllib.request.Request(ingest_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 201, 202, 204):
                raise RuntimeError(f"Ingest returned HTTP {resp.status}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ingest HTTP error: {e.code} {e.reason}") from e
