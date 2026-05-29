"""Candidate-as-row feature construction for the array-search ranker."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.graph_training.state import MAX_TURNS, production_rank


FEATURE_NAMES = (
    "valid_mask",
    "eta_norm",
    "distance_norm",
    "ship_cost_min_norm",
    "ship_cost_125pct_norm",
    "ship_cost_full_norm",
    "target_ships_now_norm",
    "target_prod_norm",
    "target_owner_mine",
    "target_owner_neutral",
    "target_owner_enemy",
    "target_radius_norm",
    "target_garrison_at_eta_norm",
    "friendly_arriving_before_eta_norm",
    "enemy_arriving_before_eta_norm",
    "enemy_min_eta_to_target_norm",
    "target_prod_rank",
    "future_prod_value_norm",
    "path_clear",
    "target_orbit_radius_norm",
    "target_phase_sin_at_eta",
    "target_phase_cos_at_eta",
    "target_is_my_threatened_planet",
    "source_ships_norm",
    "source_prod_norm",
    "source_under_threat",
    "is_below_typical_min_ships",
    "is_high_eta_launch",
    "turn_remaining",
    "my_planet_share",
    "my_prod_share",
    "num_players_norm",
)

FEATURE_DIM = len(FEATURE_NAMES)


def _planet_maps(record: dict[str, Any]) -> tuple[dict[int, list], dict[int, int], list[list]]:
    planets = [list(p) for p in record.get("observation", {}).get("planets", [])]
    by_id = {int(p[0]): p for p in planets}
    index = {int(p[0]): i for i, p in enumerate(planets)}
    return by_id, index, planets


def _inbound_before(record: dict[str, Any], target_id: int, player: int, eta: int) -> tuple[int, int, int | None]:
    friendly = 0
    enemy = 0
    enemy_min_eta = None
    for event in record.get("inbound_events", []):
        if int(event["target_id"]) != int(target_id):
            continue
        event_eta = int(event["eta"])
        if event_eta > int(eta):
            continue
        if int(event["owner"]) == int(player):
            friendly += int(event["ships"])
        else:
            enemy += int(event["ships"])
            enemy_min_eta = event_eta if enemy_min_eta is None else min(enemy_min_eta, event_eta)
    return friendly, enemy, enemy_min_eta


def _under_threat(record: dict[str, Any], planet_id: int, player: int) -> bool:
    by_id, _index, _planets = _planet_maps(record)
    planet = by_id.get(int(planet_id))
    if planet is None:
        return False
    friendly, enemy, enemy_min_eta = _inbound_before(record, planet_id, player, 10_000)
    eta = enemy_min_eta or 0
    garrison = float(planet[5]) + (float(planet[6]) * eta if int(planet[1]) >= 0 else 0.0) + friendly
    return enemy > garrison


def candidate_feature_row(record: dict[str, Any], candidate: dict[str, Any]) -> np.ndarray:
    player = int(record["graph"]["player"])
    step = int(record["graph"]["step"])
    by_id, _index, planets = _planet_maps(record)
    source = by_id.get(int(candidate["src_id"]))
    target = by_id.get(int(candidate["intended_tgt_id"]))
    row = np.zeros(FEATURE_DIM, dtype=np.float32)
    if source is None or target is None:
        return row

    eta = max(1, int(candidate.get("eta", 1)))
    ships = max(1, int(candidate.get("ships", 1)))
    required = max(1, int(candidate.get("required_ships", ships)))
    friendly, enemy, enemy_min_eta = _inbound_before(record, int(target[0]), player, eta)

    dx = float(target[2]) - float(source[2])
    dy = float(target[3]) - float(source[3])
    distance = math.hypot(dx, dy)
    target_owner = int(target[1])
    target_garrison = float(target[5]) + (float(target[6]) * eta if target_owner >= 0 else 0.0) + friendly - enemy
    target_dx = float(target[2]) - 50.0
    target_dy = float(target[3]) - 50.0
    orbit_radius = math.hypot(target_dx, target_dy)
    phase = math.atan2(target_dy, target_dx) + float(record["graph"].get("angular_velocity", 0.035)) * eta
    owned_planets = [p for p in planets if int(p[1]) >= 0]
    my_planets = [p for p in planets if int(p[1]) == player]
    my_prod = sum(float(p[6]) for p in my_planets)
    total_prod = sum(float(p[6]) for p in owned_planets)
    owners = {int(p[1]) for p in owned_planets}
    flags = candidate.get("flags", {})

    row[:] = np.array(
        [
            1.0 if candidate.get("actual_hit_id") == candidate.get("intended_tgt_id") else 0.0,
            min(eta / 120.0, 1.0),
            min(distance / 150.0, 1.0),
            min(required / 250.0, 1.0),
            min(math.ceil(required * 1.25) / 250.0, 1.0),
            min(ships / 250.0, 1.0),
            min(float(target[5]) / 250.0, 1.0),
            min(float(target[6]) / 5.0, 1.0),
            1.0 if target_owner == player else 0.0,
            1.0 if target_owner == -1 else 0.0,
            1.0 if target_owner >= 0 and target_owner != player else 0.0,
            min(float(target[4]) / 8.0, 1.0),
            min(max(target_garrison, 0.0) / 250.0, 1.0),
            min(friendly / 250.0, 1.0),
            min(enemy / 250.0, 1.0),
            1.0 if enemy_min_eta is None else min(enemy_min_eta / 120.0, 1.0),
            production_rank(planets, int(target[0])),
            min(float(target[6]) * max(0, MAX_TURNS - step - eta) / 2500.0, 1.0),
            1.0 if candidate.get("actual_hit_id") == candidate.get("intended_tgt_id") else 0.0,
            min(orbit_radius / 50.0, 1.0),
            math.sin(phase),
            math.cos(phase),
            1.0 if target_owner == player and _under_threat(record, int(target[0]), player) else 0.0,
            min(float(source[5]) / 250.0, 1.0),
            min(float(source[6]) / 5.0, 1.0),
            1.0 if _under_threat(record, int(source[0]), player) else 0.0,
            1.0 if flags.get("is_below_typical_min_ships", False) else 0.0,
            1.0 if flags.get("is_high_eta_launch", False) else 0.0,
            max(0.0, (MAX_TURNS - step) / MAX_TURNS),
            len(my_planets) / max(len(planets), 1),
            my_prod / max(total_prod, 1.0),
            min(max(len(owners), 1) / 4.0, 1.0),
        ],
        dtype=np.float32,
    )
    return row


def candidate_feature_matrix(record: dict[str, Any]) -> np.ndarray:
    candidates = record.get("candidates", [])
    if not candidates:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.vstack([candidate_feature_row(record, c) for c in candidates]).astype(np.float32, copy=False)
