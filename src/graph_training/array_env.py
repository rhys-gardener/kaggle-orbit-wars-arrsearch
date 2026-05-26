"""Array-only rollout helpers for cached graph_training records.

This module intentionally does not move fleets through geometry.  It consumes
pre-canonicalized candidate/action-set records where each launch already knows
which planet it hits and on which relative turn.  That makes it useful for
training-time scoring over cached scenarios.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RolloutScore:
    production: float
    production_lead: float
    ship_total: float
    ship_lead: float
    planet_count: int
    alive: bool


def _player_count(record: dict[str, Any]) -> int:
    planets = record["observation"].get("planets", [])
    fleets = record["observation"].get("fleets", [])
    owners = [int(p[1]) for p in planets if int(p[1]) >= 0]
    owners.extend(int(f[1]) for f in fleets if int(f[1]) >= 0)
    player = int(record["graph"]["player"])
    return max(owners + [player]) + 1 if owners or player >= 0 else 2


def _candidate_by_index(record: dict[str, Any], idx: int) -> dict[str, Any]:
    return record["candidates"][int(idx)]


def initial_arrays(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, list[tuple[int, int, int]]]]:
    """Return owner, ships, production, event_schedule.

    event_schedule maps relative ETA to `(target_index, owner, ships)`.
    Existing in-flight fleets are loaded from cached inferred events.
    """

    planets = record["observation"].get("planets", [])
    planet_ids = [int(pid) for pid in record["graph"]["planet_ids"]]
    id_to_index = {pid: i for i, pid in enumerate(planet_ids)}
    owners = np.array([int(p[1]) for p in planets], dtype=np.int16)
    ships = np.array([float(p[5]) for p in planets], dtype=np.float32)
    production = np.array([float(p[6]) for p in planets], dtype=np.float32)
    schedule: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    for event in record.get("inbound_events", []):
        target_idx = id_to_index.get(int(event["target_id"]))
        if target_idx is None:
            continue
        eta = max(1, int(event["eta"]))
        schedule[eta].append((target_idx, int(event["owner"]), int(event["ships"])))

    return owners, ships, production, schedule


def schedule_action_set(
    record: dict[str, Any],
    action_set: dict[str, Any],
    owners: np.ndarray,
    ships: np.ndarray,
    schedule: dict[int, list[tuple[int, int, int]]],
) -> None:
    """Apply launch costs and schedule arrivals for one cached action set."""

    player = int(record["graph"]["player"])
    planet_ids = [int(pid) for pid in record["graph"]["planet_ids"]]
    id_to_index = {pid: i for i, pid in enumerate(planet_ids)}

    for idx in action_set.get("candidate_indices", []):
        candidate = _candidate_by_index(record, int(idx))
        src_idx = id_to_index.get(int(candidate["src_id"]))
        hit_id = candidate.get("actual_hit_id")
        hit_idx = None if hit_id is None else id_to_index.get(int(hit_id))
        if src_idx is None or hit_idx is None:
            continue
        amount = min(float(candidate["ships"]), max(ships[src_idx], 0.0))
        if amount <= 0:
            continue
        ships[src_idx] -= amount
        eta = max(1, int(candidate["eta"]))
        schedule[eta].append((hit_idx, player, int(amount)))


def _resolve_combat(
    owners: np.ndarray,
    ships: np.ndarray,
    arrivals: list[tuple[int, int, int]],
) -> None:
    by_target: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for target_idx, owner, amount in arrivals:
        by_target[int(target_idx)][int(owner)] += int(amount)

    for target_idx, player_ships in by_target.items():
        if not player_ships:
            continue
        sorted_players = sorted(player_ships.items(), key=lambda item: item[1], reverse=True)
        top_player, top_ships = sorted_players[0]
        if len(sorted_players) > 1:
            second_ships = sorted_players[1][1]
            survivor_ships = top_ships - second_ships
            if sorted_players[0][1] == sorted_players[1][1]:
                survivor_ships = 0
            survivor_owner = top_player if survivor_ships > 0 else -1
        else:
            survivor_owner = top_player
            survivor_ships = top_ships

        if survivor_ships <= 0:
            continue
        if int(owners[target_idx]) == survivor_owner:
            ships[target_idx] += survivor_ships
        else:
            ships[target_idx] -= survivor_ships
            if ships[target_idx] < 0:
                owners[target_idx] = survivor_owner
                ships[target_idx] = abs(ships[target_idx])


def rollout_cached_action_set(
    record: dict[str, Any],
    action_set_index: int,
    *,
    horizon: int = 50,
) -> tuple[np.ndarray, np.ndarray, RolloutScore]:
    """Roll one cached first-turn action set forward with no future launches."""

    owners, ships, production, schedule = initial_arrays(record)
    action_sets = record.get("action_sets", [])
    if action_set_index < 0 or action_set_index >= len(action_sets):
        raise IndexError(action_set_index)
    schedule_action_set(record, action_sets[action_set_index], owners, ships, schedule)

    for rel_turn in range(1, max(1, int(horizon)) + 1):
        owned_mask = owners >= 0
        ships[owned_mask] += production[owned_mask]
        arrivals = schedule.get(rel_turn)
        if arrivals:
            _resolve_combat(owners, ships, arrivals)

    return owners, ships, score_arrays(record, owners, ships, production)


def score_arrays(
    record: dict[str, Any],
    owners: np.ndarray,
    ships: np.ndarray,
    production: np.ndarray,
) -> RolloutScore:
    player = int(record["graph"]["player"])
    num_players = _player_count(record)
    prod_by_player = np.zeros(num_players, dtype=np.float32)
    ships_by_player = np.zeros(num_players, dtype=np.float32)
    planets_by_player = np.zeros(num_players, dtype=np.int16)

    for idx, owner in enumerate(owners.tolist()):
        if owner < 0 or owner >= num_players:
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


def rank_action_sets_by_rollout(record: dict[str, Any], *, horizon: int = 50) -> list[tuple[int, RolloutScore]]:
    scores = []
    for idx in range(len(record.get("action_sets", []))):
        _owners, _ships, score = rollout_cached_action_set(record, idx, horizon=horizon)
        scores.append((idx, score))
    return sorted(
        scores,
        key=lambda item: (
            item[1].production,
            item[1].production_lead,
            item[1].ship_total,
            item[1].ship_lead,
        ),
        reverse=True,
    )
