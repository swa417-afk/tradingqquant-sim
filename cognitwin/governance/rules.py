"""
CogniTwin Governance — Inference Rule Sets
==========================================

Inference rules derive higher-order signals from raw CogniTwin events.
Rules operate on event streams and produce RuleEvaluationEvent outputs
that are written to the append-only audit log.

Any PR touching this file is automatically flagged by the PR checker.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence


class RuleTrigger(str, Enum):
    """What kind of input activates this rule."""
    COMMIT_STREAM = "commit_stream"
    PR_EVENT = "pr_event"
    KEYSTROKE_STREAM = "keystroke_stream"
    TEMPO_WINDOW = "tempo_window"
    RULE_EVALUATION = "rule_evaluation"
    CHARTER_CHECK = "charter_check"


class RuleOutcome(str, Enum):
    PASS = "pass"
    FLAG = "flag"
    REJECT = "reject"


@dataclass(frozen=True)
class InferenceRule:
    id: str
    name: str
    description: str
    trigger: RuleTrigger
    # Callable receives the trigger payload dict; returns (outcome, rationale)
    evaluate: Callable[[dict], tuple[RuleOutcome, str]]
    severity_on_flag: str = "warning"
    severity_on_reject: str = "critical"


# ─── Rule Implementations ─────────────────────────────────────────────────────

def _rule_amend_burst(payload: dict) -> tuple[RuleOutcome, str]:
    """
    RULE-001 — Amend burst detection.
    If >= 3 amend commits occur within a 30-minute window, flag as elevated
    cognitive-load signal.
    """
    amend_commits: list[dict] = payload.get("commits", [])
    amends = [c for c in amend_commits if c.get("isAmend")]
    if len(amends) >= 3:
        return RuleOutcome.FLAG, f"{len(amends)} amend commits detected — elevated correction cadence."
    return RuleOutcome.PASS, "Amend frequency within threshold."


def _rule_off_hours_commit(payload: dict) -> tuple[RuleOutcome, str]:
    """
    RULE-002 — Off-hours commit pattern.
    Commits between 23:00–04:00 UTC are flagged as potential fatigue signal.
    """
    commits: list[dict] = payload.get("commits", [])
    off_hours = [c for c in commits if c.get("hourOfDay", 12) in range(23, 24) or c.get("hourOfDay", 12) in range(0, 5)]
    if off_hours:
        return RuleOutcome.FLAG, f"{len(off_hours)} commit(s) in off-hours window (23–04 UTC)."
    return RuleOutcome.PASS, "All commits within normal hours."


def _rule_terse_commit_messages(payload: dict) -> tuple[RuleOutcome, str]:
    """
    RULE-003 — Terse commit messages.
    If >= 50% of commits in a session have messages < 20 bytes, flag.
    """
    commits: list[dict] = payload.get("commits", [])
    if not commits:
        return RuleOutcome.PASS, "No commits."
    terse = [c for c in commits if c.get("messageLengthBytes", 100) < 20]
    ratio = len(terse) / len(commits)
    if ratio >= 0.5:
        return RuleOutcome.FLAG, f"{ratio:.0%} of commits have terse messages (<20 bytes)."
    return RuleOutcome.PASS, "Commit message quality acceptable."


def _rule_governance_pr_invariant_touch(payload: dict) -> tuple[RuleOutcome, str]:
    """
    RULE-004 — Governance PR gate.
    A PR that touches sovereign invariants or inference rules is automatically rejected
    without an explicit governance override label.
    """
    touches_invariants: bool = payload.get("touchesInvariants", False)
    touches_rules: bool = payload.get("touchesRules", False)
    has_override: bool = payload.get("hasGovernanceOverride", False)

    if (touches_invariants or touches_rules) and not has_override:
        what = []
        if touches_invariants:
            what.append("sovereign invariants")
        if touches_rules:
            what.append("inference rule sets")
        return (
            RuleOutcome.REJECT,
            f"PR modifies {' and '.join(what)}. "
            "Apply label 'governance-override' after explicit review to proceed.",
        )
    if touches_invariants or touches_rules:
        return RuleOutcome.FLAG, "Governance override acknowledged; proceeding with audit trail."
    return RuleOutcome.PASS, "PR does not touch governed paths."


def _rule_commit_cadence_drop(payload: dict) -> tuple[RuleOutcome, str]:
    """
    RULE-005 — Commit cadence drop.
    If average inter-commit interval triples compared to the 7-day baseline, flag
    as potential decision-stability signal.
    """
    baseline_min: float = payload.get("baselineInterCommitMin", 0)
    recent_min: float = payload.get("recentInterCommitMin", 0)
    if baseline_min > 0 and recent_min > baseline_min * 3:
        return (
            RuleOutcome.FLAG,
            f"Commit cadence dropped: recent avg {recent_min:.0f} min vs baseline {baseline_min:.0f} min.",
        )
    return RuleOutcome.PASS, "Commit cadence within normal range."


def _rule_charter_deviation(payload: dict) -> tuple[RuleOutcome, str]:
    """
    RULE-006 — Sovereign Charter hash deviation.
    If the deployed system hash deviates from the charter manifest, reject.
    """
    deployed_hash: str = payload.get("deployedHash", "")
    charter_hash: str = payload.get("charterHash", "")
    if not deployed_hash or not charter_hash:
        return RuleOutcome.FLAG, "Charter hash or deployed hash missing — verification incomplete."
    if deployed_hash != charter_hash:
        return (
            RuleOutcome.REJECT,
            f"Charter deviation detected. "
            f"Deployed={deployed_hash[:12]}… Charter={charter_hash[:12]}…",
        )
    return RuleOutcome.PASS, "Deployed system matches Sovereign Charter manifest."


# ─── Rule Registry ────────────────────────────────────────────────────────────

INFERENCE_RULES: tuple[InferenceRule, ...] = (
    InferenceRule(
        id="RULE-001",
        name="amend_burst",
        description="Flag elevated correction cadence (>=3 amends in 30-min window).",
        trigger=RuleTrigger.COMMIT_STREAM,
        evaluate=_rule_amend_burst,
    ),
    InferenceRule(
        id="RULE-002",
        name="off_hours_commit",
        description="Flag commits in off-hours window (23–04 UTC) as fatigue signal.",
        trigger=RuleTrigger.COMMIT_STREAM,
        evaluate=_rule_off_hours_commit,
    ),
    InferenceRule(
        id="RULE-003",
        name="terse_commit_messages",
        description="Flag sessions where >= 50% of commit messages are under 20 bytes.",
        trigger=RuleTrigger.COMMIT_STREAM,
        evaluate=_rule_terse_commit_messages,
    ),
    InferenceRule(
        id="RULE-004",
        name="governance_pr_gate",
        description="Reject PRs modifying sovereign invariants or rule sets without override.",
        trigger=RuleTrigger.PR_EVENT,
        evaluate=_rule_governance_pr_invariant_touch,
        severity_on_reject="critical",
    ),
    InferenceRule(
        id="RULE-005",
        name="commit_cadence_drop",
        description="Flag commit cadence drops > 3× baseline as decision-stability signal.",
        trigger=RuleTrigger.COMMIT_STREAM,
        evaluate=_rule_commit_cadence_drop,
    ),
    InferenceRule(
        id="RULE-006",
        name="charter_deviation",
        description="Reject deployment if system hash deviates from Sovereign Charter manifest.",
        trigger=RuleTrigger.CHARTER_CHECK,
        evaluate=_rule_charter_deviation,
        severity_on_reject="critical",
    ),
)

RULE_INDEX: dict[str, InferenceRule] = {r.id: r for r in INFERENCE_RULES}


def evaluate_rule(rule_id: str, payload: dict) -> tuple[RuleOutcome, str]:
    rule = RULE_INDEX.get(rule_id)
    if rule is None:
        raise ValueError(f"Unknown rule id: {rule_id!r}")
    return rule.evaluate(payload)


def rule_manifest_hash() -> str:
    """SHA-256 of the canonical rule registry (ids + descriptions)."""
    canonical = json.dumps(
        [{"id": r.id, "name": r.name, "description": r.description, "trigger": r.trigger.value}
         for r in INFERENCE_RULES],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
