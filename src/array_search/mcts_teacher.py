"""MCTS-style teacher labels over whole-turn action sets.

The teacher treats generated action sets as root actions.  In its cheapest
mode it evaluates each root with a passive cached rollout.  When supplied with
the scenario and geometry cache, ``depth > 1`` enables policy-response rollout:
the root action is committed, then future turns are played by the current
ranker policy before the final state is scored.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.array_search.action_filters import ActionFilters, filter_candidates
from src.array_search.labels import PRIMARY_HORIZON, attach_positive_candidate_label, score_key
from src.array_search.ranker import CandidateRanker, greedy_pack_action_set, score_candidates
from src.array_search.records import build_record_dict
from src.array_search.scenarios import InitialScenario, active_planet_mask
from src.array_search.state_adapter import arrays_to_obs
from src.graph_training.actions import generate_cached_launch_candidates
from src.graph_training.array_env import (
    PendingLaunch,
    RolloutScore,
    ScheduledFleet,
    rollout_cached_action_set,
    step_multi_seat,
)
from src.graph_training.search import generate_action_sets
from src.graph_training.state import build_graph_state


@dataclass(frozen=True)
class MCTSTeacherConfig:
    root_action_sets: int = 32
    opponent_samples: int = 4
    depth: int = 1
    rollout_horizon: int = 30
    prior_weight: float = 0.25
    softmax_temperature: float = 1.0


def _action_set_prior(record: dict[str, Any], scores: np.ndarray | None, action_set: dict[str, Any]) -> float:
    if scores is None or len(scores) == 0:
        return float(action_set.get("score", 0.0))
    indices = [int(i) for i in action_set.get("candidate_indices", []) if 0 <= int(i) < len(scores)]
    if not indices:
        return 0.0
    return float(sum(float(scores[i]) for i in indices))


def _full_arrays_from_record(
    scenario: InitialScenario,
    record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, list[ScheduledFleet]]]:
    """Rehydrate a full scenario-sized state from one active-view record."""

    owners = scenario.owners.copy()
    ships = scenario.ships.copy()
    production = scenario.production.copy()
    id_to_index = {int(pid): i for i, pid in enumerate(scenario.planet_ids.tolist())}
    for planet in record.get("observation", {}).get("planets", []):
        idx = id_to_index.get(int(planet[0]))
        if idx is None:
            continue
        owners[idx] = int(planet[1])
        ships[idx] = float(planet[5])
    schedule: dict[int, list[ScheduledFleet]] = {}
    current_turn = int(record.get("source", {}).get("rel_turn", record.get("graph", {}).get("step", 0)))
    for event in record.get("inbound_events", []):
        target_idx = id_to_index.get(int(event["target_id"]))
        if target_idx is None:
            continue
        source_idx = id_to_index.get(int(event.get("source_id", -1)), -1)
        eta = max(1, int(event["eta"]))
        schedule.setdefault(current_turn + eta, []).append(
            ScheduledFleet(
                source_idx=int(source_idx),
                target_idx=int(target_idx),
                owner=int(event["owner"]),
                ships=int(event["ships"]),
                launch_rel_turn=current_turn - eta,
            )
        )
    return owners, ships, production, schedule


def _pending_from_record_action_set(
    scenario: InitialScenario,
    record: dict[str, Any],
    action_set: dict[str, Any],
    *,
    owner: int,
) -> list[PendingLaunch]:
    id_to_index = {int(pid): i for i, pid in enumerate(scenario.planet_ids.tolist())}
    pending: list[PendingLaunch] = []
    for idx in action_set.get("candidate_indices", []):
        candidate = record["candidates"][int(idx)]
        src_idx = id_to_index.get(int(candidate["src_id"]))
        hit_id = candidate.get("actual_hit_id")
        tgt_idx = None if hit_id is None else id_to_index.get(int(hit_id))
        if src_idx is None or tgt_idx is None:
            continue
        pending.append(
            PendingLaunch(
                source_idx=int(src_idx),
                target_idx=int(tgt_idx),
                owner=int(owner),
                ships=int(candidate["ships"]),
                eta=max(1, int(candidate["eta"])),
            )
        )
    return pending


def _score_arrays_for_player(
    scenario: InitialScenario,
    owners: np.ndarray,
    ships: np.ndarray,
    production: np.ndarray,
    *,
    player: int,
    rel_turn: int,
) -> RolloutScore:
    active = active_planet_mask(scenario, int(rel_turn))
    num_players = int(scenario.num_players)
    prod_by_player = np.zeros(num_players, dtype=np.float32)
    ships_by_player = np.zeros(num_players, dtype=np.float32)
    planets_by_player = np.zeros(num_players, dtype=np.int16)
    for idx, owner in enumerate(owners.tolist()):
        if idx >= len(active) or not bool(active[idx]) or owner < 0 or owner >= num_players:
            continue
        prod_by_player[owner] += production[idx]
        ships_by_player[owner] += ships[idx]
        planets_by_player[owner] += 1
    my_prod = float(prod_by_player[player])
    my_ships = float(ships_by_player[player])
    enemy_prod = float(max([prod_by_player[i] for i in range(num_players) if i != player] or [0.0]))
    enemy_ships = float(max([ships_by_player[i] for i in range(num_players) if i != player] or [0.0]))
    return RolloutScore(
        production=my_prod,
        production_lead=my_prod - enemy_prod,
        ship_total=my_ships,
        ship_lead=my_ships - enemy_ships,
        planet_count=int(planets_by_player[player]),
        alive=bool(planets_by_player[player] > 0 or my_ships > 0),
    )


def _policy_record_for_state(
    *,
    scenario: InitialScenario,
    geometry_cache: Any,
    owners: np.ndarray,
    ships: np.ndarray,
    schedule: dict[int, list[ScheduledFleet]],
    rel_turn: int,
    seat: int,
    model: CandidateRanker | None,
    device: str,
    candidate_limit: int,
    max_launches: int,
    beam_width: int,
    max_same_target: int,
    include_support: bool,
    filters: ActionFilters | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    obs = arrays_to_obs(
        scenario,
        seat=seat,
        rel_step=rel_turn,
        owners=owners,
        ships=ships,
        schedule=schedule,
    )
    ctx = build_graph_state(obs)
    raw_candidates = generate_cached_launch_candidates(
        ctx,
        geometry_cache,
        max_candidates=candidate_limit,
        include_support=include_support,
    )
    candidates, flags, reject_counts = filter_candidates(raw_candidates, filters)
    action_sets = generate_action_sets(
        ctx,
        candidates,
        max_launches=max_launches,
        beam_width=beam_width,
        max_same_target=max_same_target,
        candidate_limit=candidate_limit,
        include_support=include_support,
    )
    record = build_record_dict(
        scenario_id=f"teacher_seed_{scenario.seed}",
        seed=int(scenario.seed),
        rel_turn=int(rel_turn),
        seat=int(seat),
        policy_id="teacher_policy",
        obs=obs,
        ctx=ctx,
        candidates=candidates,
        candidate_flags=flags,
        action_sets=action_sets,
        reject_counts=reject_counts,
    )
    action_set_records = record.get("action_sets", [])
    if not action_set_records:
        return record, {"score": 0.0, "candidate_indices": [], "actions": []}
    if model is None:
        chosen = dict(max(action_set_records, key=lambda item: float(item.get("score", 0.0))))
    else:
        scores = score_candidates(model, record, device=device)
        chosen = greedy_pack_action_set(record, scores, max_launches=max_launches)
    return record, chosen


def _policy_response_rollout_score(
    record: dict[str, Any],
    action_set: dict[str, Any],
    *,
    scenario: InitialScenario,
    geometry_cache: Any,
    model: CandidateRanker | None,
    device: str,
    config: MCTSTeacherConfig,
    candidate_limit: int,
    max_launches: int,
    beam_width: int,
    max_same_target: int,
    include_support: bool,
    filters: ActionFilters | None,
) -> RolloutScore:
    owners, ships, production, schedule = _full_arrays_from_record(scenario, record)
    current_turn = int(record.get("source", {}).get("rel_turn", record.get("graph", {}).get("step", 0)))
    root_player = int(record["graph"]["player"])
    for depth_idx in range(max(1, int(config.depth))):
        pending: list[PendingLaunch] = []
        for seat in range(int(scenario.num_players)):
            if depth_idx == 0 and seat == root_player:
                pending.extend(_pending_from_record_action_set(scenario, record, action_set, owner=seat))
                continue
            seat_record, chosen = _policy_record_for_state(
                scenario=scenario,
                geometry_cache=geometry_cache,
                owners=owners,
                ships=ships,
                schedule=schedule,
                rel_turn=current_turn,
                seat=seat,
                model=model,
                device=device,
                candidate_limit=candidate_limit,
                max_launches=max_launches,
                beam_width=beam_width,
                max_same_target=max_same_target,
                include_support=include_support,
                filters=filters,
            )
            pending.extend(_pending_from_record_action_set(scenario, seat_record, chosen, owner=seat))
        current_turn = step_multi_seat(
            owners,
            ships,
            production,
            schedule,
            pending,
            current_rel_turn=current_turn,
            active_mask=active_planet_mask(scenario, current_turn),
            arrival_active_mask=active_planet_mask(scenario, current_turn + 1),
        )

    final_turn = int(record.get("source", {}).get("rel_turn", record.get("graph", {}).get("step", 0))) + max(
        1,
        int(config.rollout_horizon),
    )
    while current_turn < final_turn:
        current_turn = step_multi_seat(
            owners,
            ships,
            production,
            schedule,
            [],
            current_rel_turn=current_turn,
            active_mask=active_planet_mask(scenario, current_turn),
            arrival_active_mask=active_planet_mask(scenario, current_turn + 1),
        )
    return _score_arrays_for_player(
        scenario,
        owners,
        ships,
        production,
        player=root_player,
        rel_turn=current_turn,
    )


def teach_record_with_mcts(
    record: dict[str, Any],
    *,
    model: CandidateRanker | None = None,
    device: str = "cpu",
    config: MCTSTeacherConfig | None = None,
    scenario: InitialScenario | None = None,
    geometry_cache: Any | None = None,
    candidate_limit: int = 40,
    max_launches: int = 6,
    beam_width: int = 12,
    max_same_target: int = 3,
    include_support: bool = False,
    filters: ActionFilters | None = None,
) -> dict[str, Any]:
    """Attach teacher labels to one record and return it."""

    config = config or MCTSTeacherConfig()
    action_sets = record.get("action_sets", [])
    if not action_sets:
        attach_positive_candidate_label(record, [], source="mcts", horizon=PRIMARY_HORIZON)
        return record

    candidate_scores = score_candidates(model, record, device=device) if model is not None else None
    root_order = sorted(
        range(len(action_sets)),
        key=lambda i: _action_set_prior(record, candidate_scores, action_sets[i]),
        reverse=True,
    )[: max(1, int(config.root_action_sets))]

    evaluated: list[tuple[int, tuple[float, ...], float]] = []
    use_policy_response = scenario is not None and geometry_cache is not None and int(config.depth) > 1
    for action_idx in root_order:
        if use_policy_response:
            score = _policy_response_rollout_score(
                record,
                action_sets[int(action_idx)],
                scenario=scenario,
                geometry_cache=geometry_cache,
                model=model,
                device=device,
                config=config,
                candidate_limit=candidate_limit,
                max_launches=max_launches,
                beam_width=beam_width,
                max_same_target=max_same_target,
                include_support=include_support,
                filters=filters,
            )
        else:
            _owners, _ships, score = rollout_cached_action_set(
                record,
                int(action_idx),
                horizon=max(1, int(config.rollout_horizon)),
            )
        prior = _action_set_prior(record, candidate_scores, action_sets[action_idx])
        key = score_key(score)
        scalar = (
            key[0] * 1000.0
            + key[1] * 100.0
            + key[2]
            + key[3] * 0.1
            + float(config.prior_weight) * prior
        )
        evaluated.append((int(action_idx), tuple(float(x) for x in key), float(scalar)))

    evaluated.sort(key=lambda item: (item[1], item[2]), reverse=True)
    best_idx = int(evaluated[0][0])
    raw_values = np.array([item[2] for item in evaluated], dtype=np.float64)
    raw_values -= float(raw_values.max()) if len(raw_values) else 0.0
    temp = max(float(config.softmax_temperature), 1e-6)
    weights = np.exp(raw_values / temp)
    weights = weights / max(float(weights.sum()), 1e-12)
    action_set_weights = [0.0] * len(action_sets)
    for (action_idx, _key, _value), weight in zip(evaluated, weights.tolist()):
        action_set_weights[int(action_idx)] = float(weight)

    attach_positive_candidate_label(
        record,
        [int(i) for i in action_sets[best_idx].get("candidate_indices", [])],
        source="mcts",
        horizon=PRIMARY_HORIZON,
        action_set_index=best_idx,
        action_set_weights=action_set_weights,
    )
    record.setdefault("teacher", {})
    record["teacher"]["mcts"] = {
        "root_action_sets": int(config.root_action_sets),
        "opponent_samples": int(config.opponent_samples),
        "depth": int(config.depth),
        "rollout_horizon": int(config.rollout_horizon),
        "evaluated_action_sets": len(evaluated),
        "best_action_set": best_idx,
        "mode": "policy_response" if use_policy_response else "root_rollout",
    }
    return record


def apply_mcts_teacher(
    records: list[dict[str, Any]],
    *,
    rate: float,
    seed: int,
    model: CandidateRanker | None = None,
    device: str = "cpu",
    config: MCTSTeacherConfig | None = None,
    scenario: InitialScenario | None = None,
    geometry_cache: Any | None = None,
    candidate_limit: int = 40,
    max_launches: int = 6,
    beam_width: int = 12,
    max_same_target: int = 3,
    include_support: bool = False,
    filters: ActionFilters | None = None,
) -> int:
    """Apply teacher labels to a sampled subset of records."""

    rng = random.Random(int(seed))
    applied = 0
    for record in records:
        if rng.random() > float(rate):
            continue
        teach_record_with_mcts(
            record,
            model=model,
            device=device,
            config=config,
            scenario=scenario,
            geometry_cache=geometry_cache,
            candidate_limit=candidate_limit,
            max_launches=max_launches,
            beam_width=beam_width,
            max_same_target=max_same_target,
            include_support=include_support,
            filters=filters,
        )
        applied += 1
    return applied
