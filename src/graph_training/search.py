"""Production-first whole-turn action-set search."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .actions import LaunchCandidate, generate_launch_candidates
from .state import GraphState


@dataclass(frozen=True)
class ActionSet:
    launches: tuple[LaunchCandidate, ...] = field(default_factory=tuple)
    score: float = 0.0

    def actions(self) -> list[list]:
        return [launch.action() for launch in self.launches]

    @property
    def used_sources(self) -> set[int]:
        return {launch.src_id for launch in self.launches}

    @property
    def target_counts(self) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for launch in self.launches:
            if launch.actual_hit_id is not None:
                counts[int(launch.actual_hit_id)] += 1
        return dict(counts)


def _action_set_key(action_set: ActionSet) -> tuple:
    return tuple(
        sorted(
            (
                launch.src_id,
                launch.actual_hit_id,
                launch.ships,
                round(launch.angle, 8),
            )
            for launch in action_set.launches
        )
    )


def _combo_bonus(ctx: GraphState, launches: tuple[LaunchCandidate, ...]) -> float:
    committed: dict[int, int] = defaultdict(int)
    best_required: dict[int, int] = {}
    best_prod: dict[int, float] = {}
    owner_by_target: dict[int, int] = {}
    for launch in launches:
        if launch.actual_hit_id is None:
            continue
        target_id = int(launch.actual_hit_id)
        committed[target_id] += int(launch.ships)
        best_required[target_id] = min(
            best_required.get(target_id, launch.required_ships),
            launch.required_ships,
        )
        best_prod[target_id] = max(best_prod.get(target_id, 0.0), launch.target_production)
        owner_by_target[target_id] = launch.target_owner

    bonus = 0.0
    for target_id, ships in committed.items():
        owner = owner_by_target.get(target_id, -99)
        if owner == ctx.player:
            continue
        required = max(1, best_required.get(target_id, 1))
        prod = best_prod.get(target_id, 0.0)
        if ships >= required:
            bonus += prod * 3.5
            if ships <= required * 1.5:
                bonus += 1.0
        else:
            bonus -= prod * 0.6
    return bonus


def _score_launch_addition(ctx: GraphState, current: ActionSet, launch: LaunchCandidate) -> float:
    score = current.score + launch.score
    next_launches = current.launches + (launch,)
    score += _combo_bonus(ctx, next_launches) - _combo_bonus(ctx, current.launches)
    if launch.kind == "pressure" and launch.target_production < 3.0:
        score -= 2.0
    return score


def _compatible(
    ctx: GraphState,
    current: ActionSet,
    launch: LaunchCandidate,
    *,
    max_same_target: int,
) -> bool:
    if launch.src_id in current.used_sources:
        return False
    if launch.actual_hit_id is None:
        return False
    if not launch.hits_intended_target:
        return False
    target_counts = current.target_counts
    if target_counts.get(int(launch.actual_hit_id), 0) >= max_same_target:
        return False
    if launch.target_owner != ctx.player:
        committed = sum(
            item.ships
            for item in current.launches
            if item.actual_hit_id == launch.actual_hit_id
        )
        required = min(
            [item.required_ships for item in current.launches if item.actual_hit_id == launch.actual_hit_id]
            + [launch.required_ships]
        )
        if committed >= required and launch.kind in {"capture_exact", "capture_overkill", "pressure"}:
            return False
    return True


def generate_action_sets(
    ctx: GraphState,
    candidates: list[LaunchCandidate] | None = None,
    *,
    max_launches: int = 6,
    beam_width: int = 32,
    max_same_target: int = 3,
    candidate_limit: int = 80,
    top_targets_per_source: int = 3,
    include_support: bool = True,
) -> list[ActionSet]:
    """Generate a ranked beam of whole-turn action sets."""

    if candidates is None:
        candidates = generate_launch_candidates(
            ctx,
            max_candidates=candidate_limit,
            top_targets_per_source=top_targets_per_source,
            include_support=include_support,
        )
    else:
        candidates = sorted(candidates, key=lambda item: item.score, reverse=True)[:candidate_limit]

    if not candidates:
        return [ActionSet()]

    max_launches = max(1, min(max_launches, len(ctx.my_planets)))
    beam = [ActionSet()]
    for _slot in range(max_launches):
        expanded: dict[tuple, ActionSet] = {_action_set_key(item): item for item in beam}
        for current in beam:
            for launch in candidates:
                if not _compatible(ctx, current, launch, max_same_target=max_same_target):
                    continue
                next_set = ActionSet(
                    launches=current.launches + (launch,),
                    score=_score_launch_addition(ctx, current, launch),
                )
                key = _action_set_key(next_set)
                old = expanded.get(key)
                if old is None or next_set.score > old.score:
                    expanded[key] = next_set
        next_beam = sorted(expanded.values(), key=lambda item: item.score, reverse=True)[:beam_width]
        if next_beam == beam:
            break
        beam = next_beam
    return sorted(beam, key=lambda item: item.score, reverse=True)


def choose_action_set(
    ctx: GraphState,
    *,
    max_launches: int = 6,
    beam_width: int = 32,
    max_same_target: int = 3,
    candidate_limit: int = 80,
    top_targets_per_source: int = 3,
    include_support: bool = True,
) -> ActionSet:
    return generate_action_sets(
        ctx,
        max_launches=max_launches,
        beam_width=beam_width,
        max_same_target=max_same_target,
        candidate_limit=candidate_limit,
        top_targets_per_source=top_targets_per_source,
        include_support=include_support,
    )[0]
