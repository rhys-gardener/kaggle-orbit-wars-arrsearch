"""Action filters: strict cuts + lax tags for launch candidates.

Defaults are empirically derived from the May 24-25 leaderboard replay
analysis (``docs/replay_action_stats.md``). The strict tier rejects
candidates that essentially never appear in winning play; the lax tier
admits unusual-but-occasionally-good candidates with a feature flag so the
ranker can learn when to use them.

Ablation:
    ActionFilters(
        max_eta_strict=999,
        small_long_journey_ships=0,
        small_long_journey_eta=999,
        max_launches_per_turn=999,
    )
reproduces the unfiltered baseline.

See ``docs/plans/array-search-initiative.md`` § "Action filters".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


class _CandidateLike(Protocol):
    """Minimum surface every filter needs.

    ``LaunchCandidate`` (from ``src.graph_training.actions``) satisfies this,
    as does any dict with the same keys.
    """
    ships: int
    eta: int
    actual_hit_id: int | None


@dataclass(frozen=True)
class ActionFilters:
    """Filter / composition knobs derived from leaderboard replays.

    Strict cuts hard-reject from the candidate space. Lax tags admit the
    candidate but mark it so the ranker can learn when to use it.
    """

    # --- Strict (hard reject) ---
    max_eta_strict: int = 80
    """Reject any candidate with eta > this. Winner P95 was 57; 80 leaves margin."""

    small_long_journey_ships: int = 2
    """Combined with small_long_journey_eta: reject when ships ≤ this AND eta > the cap."""

    small_long_journey_eta: int = 40
    """Combined with small_long_journey_ships."""

    require_hits_intended: bool = True
    """Reject when the engine-solver says the launch doesn't actually hit the
    intended target (i.e. ``actual_hit_id != intended_tgt_id`` or is None).
    Filters out candidates that won't behave as the ranker expects."""

    # --- Lax (admit + flag for feature row) ---
    typical_min_ships: int = 3
    """ships ≤ this → ``is_below_typical_min_ships`` flag set. ~13% of winning launches."""

    high_eta_threshold: int = 50
    """eta > this → ``is_high_eta_launch`` flag set. ~7% of winning launches."""

    # --- Action-set composition ---
    max_launches_per_turn: int = 10
    """Hard cap on launches in any composed action set. P95 of winning non-empty turns = 11."""

    multi_source_bonus: float = 0.15
    """Soft additive bonus to a candidate's score during action-set composition
    when at least one already-chosen candidate in the set shares the same target.
    Empirical evidence: 22.4% of winning multi-launch turns coordinate this way."""


@dataclass(frozen=True)
class CandidateFlags:
    """Per-candidate feature flags emitted by ``filter_candidates``.

    These flow into the ranker's per-candidate feature row alongside the
    edge/planet features. The ranker learns to discount or trust marginal
    candidates appropriately.
    """

    is_below_typical_min_ships: bool
    is_high_eta_launch: bool


def is_strict_rejected(candidate: _CandidateLike, filters: ActionFilters) -> str | None:
    """Return the rejection reason if a candidate violates a strict filter,
    else None.

    Reason codes are stable strings so they can be counted in
    ``filter_stats`` logging.
    """
    eta = int(candidate.eta)
    ships = int(candidate.ships)

    if ships < 1:
        return "ships_lt_1"
    if eta > int(filters.max_eta_strict):
        return "eta_gt_strict"
    if ships <= int(filters.small_long_journey_ships) and eta > int(filters.small_long_journey_eta):
        return "small_fleet_long_journey"
    if filters.require_hits_intended:
        actual = getattr(candidate, "actual_hit_id", None)
        intended = getattr(candidate, "intended_tgt_id", None)
        if actual is None or (intended is not None and int(actual) != int(intended)):
            return "off_target_hit"
    return None


def compute_flags(candidate: _CandidateLike, filters: ActionFilters) -> CandidateFlags:
    return CandidateFlags(
        is_below_typical_min_ships=int(candidate.ships) <= int(filters.typical_min_ships),
        is_high_eta_launch=int(candidate.eta) > int(filters.high_eta_threshold),
    )


def filter_candidates(
    candidates: Iterable[_CandidateLike],
    filters: ActionFilters | None = None,
) -> tuple[list[_CandidateLike], list[CandidateFlags], dict[str, int]]:
    """Apply strict filters; return ``(kept, flags_per_kept, reject_stats)``.

    ``reject_stats`` maps rejection reason → count, for logging and ablation
    analysis. Total rejections = ``sum(reject_stats.values())``.
    """
    filters = filters or ActionFilters()
    kept: list[_CandidateLike] = []
    flags: list[CandidateFlags] = []
    stats: dict[str, int] = {}
    for c in candidates:
        reason = is_strict_rejected(c, filters)
        if reason is not None:
            stats[reason] = stats.get(reason, 0) + 1
            continue
        kept.append(c)
        flags.append(compute_flags(c, filters))
    return kept, flags, stats
