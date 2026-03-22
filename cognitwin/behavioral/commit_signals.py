"""
CogniTwin Behavioral — Commit-Pattern Signal Extractor
=======================================================

Extracts behavioral signals from Git commit history and emits
CogniTwin CommitEvent + RuleEvaluation events.

Behavioral dimensions captured
────────────────────────────────
  • commit_frequency_per_day   — proxy for work intensity
  • hour_of_day distribution   — circadian pattern / time-zone alignment
  • day_of_week distribution   — weekly rhythm
  • message_length_bytes       — cognitive expression / deliberation
  • amend_rate                 — correction propensity (decision instability)
  • inter_commit_interval_min  — decision pacing / flow state
  • lines_changed_per_commit   — change granularity / atomicity
  • cadence_variability        — stddev of inter-commit intervals

These are direct repository-layer analogues of keystroke tempo and
interaction rhythm — passive, non-intrusive, always-on.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


# ─── Raw Commit Record ────────────────────────────────────────────────────────

@dataclass
class RawCommit:
    sha: str
    author_epoch: int           # Unix timestamp
    message: str
    is_amend: bool
    files_changed: list[str]
    lines_added: int
    lines_removed: int

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.author_epoch, tz=timezone.utc)

    @property
    def hour_of_day(self) -> int:
        return self.dt.hour

    @property
    def day_of_week(self) -> int:
        return self.dt.weekday()  # 0=Monday … 6=Sunday

    @property
    def message_length_bytes(self) -> int:
        return len(self.message.encode())


# ─── Git Reader ───────────────────────────────────────────────────────────────

def read_commits(
    repo_path: str | Path = ".",
    max_commits: int = 500,
    branch: str = "HEAD",
) -> list[RawCommit]:
    """
    Read up to max_commits from a local git repo using git log.
    Returns commits in reverse chronological order (newest first).
    """
    repo_path = Path(repo_path).resolve()
    sep = "\x1F"  # unit separator — safe delimiter

    log_fmt = sep.join(["%H", "%at", "%s"])
    cmd = [
        "git", "-C", str(repo_path), "log",
        f"--max-count={max_commits}",
        f"--format={log_fmt}",
        "--diff-filter=ACDMRT",
        "--numstat",
        branch,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = result.stdout.strip()
    if not raw:
        return []

    commits: list[RawCommit] = []
    blocks = raw.split("\n\n")  # git separates commits with blank lines when --numstat is used

    # Fallback: simple parse without numstat for environments without it
    for block in blocks:
        lines = [l for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        header = lines[0]
        parts = header.split(sep)
        if len(parts) < 3:
            continue
        sha, epoch_str, msg = parts[0], parts[1], sep.join(parts[2:])
        try:
            epoch = int(epoch_str)
        except ValueError:
            continue

        # Detect amend: "amend" in message or "fixup!" prefix
        is_amend = msg.lower().startswith(("amend:", "fixup!", "--amend")) or "amend" in msg.lower()

        added = removed = 0
        files: list[str] = []
        for stat_line in lines[1:]:
            stat_parts = stat_line.split("\t")
            if len(stat_parts) == 3:
                try:
                    added += int(stat_parts[0]) if stat_parts[0] != "-" else 0
                    removed += int(stat_parts[1]) if stat_parts[1] != "-" else 0
                    files.append(stat_parts[2])
                except ValueError:
                    pass

        commits.append(
            RawCommit(
                sha=sha[:12],
                author_epoch=epoch,
                message=msg,
                is_amend=is_amend,
                files_changed=files,
                lines_added=added,
                lines_removed=removed,
            )
        )
    return commits


# ─── Signal Extraction ────────────────────────────────────────────────────────

@dataclass
class CommitSignals:
    """Aggregated behavioral signals derived from a commit window."""
    commit_count: int
    window_days: float
    commits_per_day: float
    amend_rate: float                      # fraction of commits that are amends
    mean_message_length_bytes: float
    median_message_length_bytes: float
    hour_distribution: dict[int, int]      # hour → count
    day_distribution: dict[int, int]       # weekday → count
    mean_inter_commit_min: float
    stddev_inter_commit_min: float         # cadence variability
    mean_lines_changed: float
    off_hours_fraction: float              # fraction in 23–04 UTC
    terse_message_fraction: float          # fraction with < 20 bytes
    # Derived composite signals
    cognitive_load_score: float            # 0–1 normalised
    decision_stability_score: float        # 0–1 normalised (higher = more stable)

    def to_dict(self) -> dict:
        return asdict(self)

    def actor_hash(self, actor_id: str) -> str:
        """Return a hex digest of the actor_id for privacy-safe attribution."""
        return hashlib.sha256(actor_id.encode()).hexdigest()


def extract_signals(commits: list[RawCommit], window_days: float = 7.0) -> CommitSignals:
    """Derive CommitSignals from a list of RawCommit objects."""
    if not commits:
        return CommitSignals(
            commit_count=0, window_days=window_days, commits_per_day=0,
            amend_rate=0, mean_message_length_bytes=0, median_message_length_bytes=0,
            hour_distribution={}, day_distribution={},
            mean_inter_commit_min=0, stddev_inter_commit_min=0,
            mean_lines_changed=0, off_hours_fraction=0, terse_message_fraction=0,
            cognitive_load_score=0, decision_stability_score=1.0,
        )

    n = len(commits)
    amend_count = sum(1 for c in commits if c.is_amend)
    msg_lengths = [c.message_length_bytes for c in commits]
    lines_changed = [c.lines_added + c.lines_removed for c in commits]

    # Hour & day distributions
    hour_dist: dict[int, int] = {}
    day_dist: dict[int, int] = {}
    off_hours_count = 0
    for c in commits:
        hour_dist[c.hour_of_day] = hour_dist.get(c.hour_of_day, 0) + 1
        day_dist[c.day_of_week] = day_dist.get(c.day_of_week, 0) + 1
        if c.hour_of_day >= 23 or c.hour_of_day < 5:
            off_hours_count += 1

    # Inter-commit intervals (newest-first, so subtract consecutive epochs)
    sorted_commits = sorted(commits, key=lambda c: c.author_epoch)
    intervals_min: list[float] = []
    for i in range(1, len(sorted_commits)):
        delta = (sorted_commits[i].author_epoch - sorted_commits[i - 1].author_epoch) / 60
        if 0 < delta < 1440:  # ignore gaps > 24h (not the same session)
            intervals_min.append(delta)

    mean_interval = statistics.mean(intervals_min) if intervals_min else 0
    stddev_interval = statistics.stdev(intervals_min) if len(intervals_min) >= 2 else 0

    terse_count = sum(1 for l in msg_lengths if l < 20)

    # Composite scores (heuristic, tunable)
    # Cognitive load: high amend rate + off hours + high cadence variability → higher load
    amend_rate = amend_count / n
    off_frac = off_hours_count / n
    cv = stddev_interval / mean_interval if mean_interval > 0 else 0
    cognitive_load_score = min(1.0, (amend_rate * 0.4 + off_frac * 0.3 + min(cv, 2) * 0.15))

    # Decision stability: low amend rate + consistent cadence + adequate message length → higher stability
    terse_frac = terse_count / n
    stability_penalty = amend_rate * 0.35 + terse_frac * 0.25 + min(cv / 2, 0.4) * 0.4
    decision_stability_score = max(0.0, 1.0 - stability_penalty)

    return CommitSignals(
        commit_count=n,
        window_days=window_days,
        commits_per_day=n / max(window_days, 1),
        amend_rate=amend_rate,
        mean_message_length_bytes=statistics.mean(msg_lengths),
        median_message_length_bytes=statistics.median(msg_lengths),
        hour_distribution=hour_dist,
        day_distribution=day_dist,
        mean_inter_commit_min=mean_interval,
        stddev_inter_commit_min=stddev_interval,
        mean_lines_changed=statistics.mean(lines_changed) if lines_changed else 0,
        off_hours_fraction=off_frac,
        terse_message_fraction=terse_frac,
        cognitive_load_score=round(cognitive_load_score, 4),
        decision_stability_score=round(decision_stability_score, 4),
    )


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main(repo_path: str = ".", window_days: float = 7.0, output_json: bool = False) -> None:
    """
    Analyse a local git repo and print CogniTwin behavioral signals.

    Usage:
        python -m cognitwin.behavioral.commit_signals [repo_path] [window_days]
    """
    commits = read_commits(repo_path, max_commits=500)
    if not commits:
        print("No commits found.")
        return

    # Filter to window
    now_epoch = datetime.now(timezone.utc).timestamp()
    cutoff = now_epoch - window_days * 86400
    window_commits = [c for c in commits if c.author_epoch >= cutoff]

    signals = extract_signals(window_commits, window_days=window_days)

    if output_json:
        print(json.dumps(signals.to_dict(), indent=2))
    else:
        print(f"\n{'─'*50}")
        print(f"  CogniTwin Commit Behavioral Signals")
        print(f"  Repo: {Path(repo_path).resolve()}")
        print(f"  Window: {window_days:.0f} days  |  Commits: {signals.commit_count}")
        print(f"{'─'*50}")
        print(f"  commits/day              : {signals.commits_per_day:.2f}")
        print(f"  amend rate               : {signals.amend_rate:.1%}")
        print(f"  off-hours fraction       : {signals.off_hours_fraction:.1%}")
        print(f"  terse message fraction   : {signals.terse_message_fraction:.1%}")
        print(f"  mean inter-commit (min)  : {signals.mean_inter_commit_min:.1f}")
        print(f"  cadence stddev (min)     : {signals.stddev_inter_commit_min:.1f}")
        print(f"  mean msg length (bytes)  : {signals.mean_message_length_bytes:.0f}")
        print(f"{'─'*50}")
        print(f"  cognitive_load_score     : {signals.cognitive_load_score:.4f}")
        print(f"  decision_stability_score : {signals.decision_stability_score:.4f}")
        print(f"{'─'*50}\n")


if __name__ == "__main__":
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    days = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0
    fmt = "--json" in sys.argv
    main(repo, days, fmt)
