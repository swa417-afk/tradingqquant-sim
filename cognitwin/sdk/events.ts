/**
 * CogniTwin Capture SDK — Event Specification & Factory Helpers
 *
 * Provides typed constructors for every event variant, enforcing
 * required fields and defaulting schemaVersion / source / type.
 */

import { randomUUID } from "crypto";
import type {
  KeystrokeEvent,
  InteractionTempoEvent,
  CommitEvent,
  PREvent,
  RuleEvaluationEvent,
  RuleEvaluationOutcome,
  EventSeverity,
} from "./schema.js";

function nowIso(): string {
  return new Date().toISOString();
}

// ─── Keystroke ────────────────────────────────────────────────────────────────

export function makeKeystrokeEvent(
  sessionId: string,
  seq: number,
  intervalMs: number,
  keyCategory: KeystrokeEvent["payload"]["keyCategory"],
  targetRole: string,
  actorHash?: string
): KeystrokeEvent {
  return {
    id: randomUUID(),
    ts: nowIso(),
    seq,
    source: "browser",
    type: "keystroke",
    sessionId,
    actorHash,
    schemaVersion: "0.1.0",
    payload: { intervalMs, keyCategory, targetRole },
  };
}

// ─── Interaction Tempo ────────────────────────────────────────────────────────

export function makeInteractionTempoEvent(
  sessionId: string,
  seq: number,
  epm: number,
  windowSec: number,
  precedingIdleMs: number,
  actorHash?: string
): InteractionTempoEvent {
  return {
    id: randomUUID(),
    ts: nowIso(),
    seq,
    source: "browser",
    type: "interaction_tempo",
    sessionId,
    actorHash,
    schemaVersion: "0.1.0",
    payload: { epm, windowSec, precedingIdleMs },
  };
}

// ─── Commit ───────────────────────────────────────────────────────────────────

export function makeCommitEvent(
  sessionId: string,
  seq: number,
  params: Omit<CommitEvent["payload"], never>,
  actorHash?: string
): CommitEvent {
  return {
    id: randomUUID(),
    ts: nowIso(),
    seq,
    source: "repository",
    type: "commit",
    sessionId,
    actorHash,
    schemaVersion: "0.1.0",
    payload: params,
  };
}

// ─── Pull Request ─────────────────────────────────────────────────────────────

export function makePREvent(
  sessionId: string,
  seq: number,
  params: Omit<PREvent["payload"], never>,
  actorHash?: string
): PREvent {
  return {
    id: randomUUID(),
    ts: nowIso(),
    seq,
    source: "repository",
    type: "pull_request",
    sessionId,
    actorHash,
    schemaVersion: "0.1.0",
    payload: params,
  };
}

// ─── Rule Evaluation ──────────────────────────────────────────────────────────

export function makeRuleEvaluationEvent(
  sessionId: string,
  seq: number,
  ruleId: string,
  ruleName: string,
  outcome: RuleEvaluationOutcome,
  triggeredBy: string,
  severity: EventSeverity,
  rationale: string
): RuleEvaluationEvent {
  return {
    id: randomUUID(),
    ts: nowIso(),
    seq,
    source: "system",
    type: "rule_evaluation",
    sessionId: sessionId,
    schemaVersion: "0.1.0",
    payload: { ruleId, ruleName, outcome, triggeredBy, severity, rationale },
  };
}
