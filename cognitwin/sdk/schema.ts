/**
 * CogniTwin Capture SDK — Event Schema
 * Version: 0.1.0
 *
 * Canonical type definitions for all passive ingestion events emitted
 * by capture clients (browser layer, repository layer, interaction layer).
 */

// ─── Base ────────────────────────────────────────────────────────────────────

export type EventSource =
  | "browser"
  | "repository"
  | "interaction"
  | "system";

export type EventSeverity = "info" | "warning" | "critical";

export interface BaseEvent {
  /** Globally unique event identifier (UUIDv4) */
  id: string;
  /** ISO-8601 UTC timestamp of origination */
  ts: string;
  /** Monotonic sequence counter per session (for ordering / gap detection) */
  seq: number;
  /** Logical source domain */
  source: EventSource;
  /** Session identifier linking events across a single cognitive session */
  sessionId: string;
  /** Optional actor identifier (hashed principal, never plaintext PII) */
  actorHash?: string;
  /** Schema version for forward compatibility */
  schemaVersion: "0.1.0";
}

// ─── Interaction Events ───────────────────────────────────────────────────────

export interface KeystrokeEvent extends BaseEvent {
  source: "browser" | "interaction";
  type: "keystroke";
  payload: {
    /** Inter-keystroke interval in milliseconds */
    intervalMs: number;
    /** Key category (letter / symbol / nav / delete / modifier) — no key value */
    keyCategory: "letter" | "symbol" | "nav" | "delete" | "modifier";
    /** Target element semantic role (editor / search / form / other) */
    targetRole: string;
  };
}

export interface InteractionTempoEvent extends BaseEvent {
  source: "browser" | "interaction";
  type: "interaction_tempo";
  payload: {
    /** Events-per-minute in the last sampling window */
    epm: number;
    /** Sampling window duration in seconds */
    windowSec: number;
    /** Idle gap immediately preceding this window (ms) */
    precedingIdleMs: number;
  };
}

// ─── Repository Events ────────────────────────────────────────────────────────

export interface CommitEvent extends BaseEvent {
  source: "repository";
  type: "commit";
  payload: {
    /** Short commit SHA (first 12 chars) */
    sha: string;
    /** UTC hour-of-day (0–23) at commit creation */
    hourOfDay: number;
    /** Day of week (0 = Sunday … 6 = Saturday) */
    dayOfWeek: number;
    /** Commit message byte length */
    messageLengthBytes: number;
    /** True if this commit amended a previous one */
    isAmend: boolean;
    /** Lines added */
    linesAdded: number;
    /** Lines removed */
    linesRemoved: number;
    /** File paths touched (relative) */
    filesChanged: string[];
    /** Inter-commit interval in minutes (null for first commit in session) */
    interCommitIntervalMin: number | null;
  };
}

export interface PREvent extends BaseEvent {
  source: "repository";
  type: "pull_request";
  payload: {
    prNumber: number;
    action: "opened" | "synchronize" | "closed" | "reopened";
    /** Whether the PR touches sovereign invariant definitions */
    touchesInvariants: boolean;
    /** Whether the PR touches inference rule sets */
    touchesRules: boolean;
    /** Governance flag written by PR checker */
    governanceFlag: boolean;
  };
}

// ─── Audit Events ─────────────────────────────────────────────────────────────

export type RuleEvaluationOutcome = "pass" | "flag" | "reject";

export interface RuleEvaluationEvent extends BaseEvent {
  source: "system";
  type: "rule_evaluation";
  payload: {
    /** ID of the rule that was evaluated */
    ruleId: string;
    /** Human-readable rule name */
    ruleName: string;
    /** Evaluation outcome */
    outcome: RuleEvaluationOutcome;
    /** Triggering artifact (PR number, commit SHA, release tag, …) */
    triggeredBy: string;
    /** Severity assigned */
    severity: EventSeverity;
    /** Freeform rationale */
    rationale: string;
  };
}

// ─── Union ────────────────────────────────────────────────────────────────────

export type CogniTwinEvent =
  | KeystrokeEvent
  | InteractionTempoEvent
  | CommitEvent
  | PREvent
  | RuleEvaluationEvent;
