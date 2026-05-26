"""Graph state extraction for production-first Orbit Wars search.

The representation has three layers:

* planet_features[P, Fp]
* edge_features[P, P, Fe]
* valid_edge_mask[P, P]

The mask is deliberately separate from feature values.  Zero is meaningful in
many feature columns, so unreachable or irrelevant edges should never be encoded
as a magic numeric feature row.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.physics import fleet_speed, intercept, is_orbiting, sun_blocked


MAX_TURNS = 500
RESERVE = 0.05

PLANET_FEATURE_NAMES = (
    "owner_mine",
    "owner_neutral",
    "owner_enemy",
    "ships_norm",
    "production_norm",
    "radius_norm",
    "x_norm",
    "y_norm",
    "orbit_radius_norm",
    "orbit_sin",
    "orbit_cos",
    "is_orbiting",
    "is_comet",
    "friendly_inbound_norm",
    "enemy_inbound_norm",
    "enemy_min_eta_norm",
    "available_norm",
    "under_threat",
    "turn_remaining",
    "production_rank",
)

EDGE_FEATURE_NAMES = (
    "src_owner_mine",
    "tgt_owner_mine",
    "tgt_owner_neutral",
    "tgt_owner_enemy",
    "src_ships_norm",
    "src_available_norm",
    "src_production_norm",
    "tgt_ships_norm",
    "tgt_production_norm",
    "tgt_radius_norm",
    "distance_norm",
    "eta_norm",
    "required_ships_norm",
    "affordability",
    "friendly_before_eta_norm",
    "enemy_before_eta_norm",
    "enemy_min_eta_norm",
    "future_prod_value_norm",
    "target_prod_rank",
    "source_surplus",
    "target_is_orbiting",
    "target_is_comet",
    "rough_path_clear",
    "turn_remaining",
    "source_under_threat_norm",
    "target_under_threat_norm",
)

PLANET_FEATURE_DIM = len(PLANET_FEATURE_NAMES)
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)


@dataclass(frozen=True)
class FleetEvent:
    """Inferred destination and timing for an in-flight fleet."""

    fleet_id: int
    owner: int
    source_id: int
    target_id: int
    ships: int
    eta: int
    angle_diff: float


@dataclass(frozen=True)
class EdgeEstimate:
    """Cheap source-target estimate used by both features and candidates."""

    source_id: int
    target_id: int
    tx: float
    ty: float
    eta: int
    distance: float
    required_ships: int
    arrival_garrison: float
    friendly_before_eta: int
    enemy_before_eta: int
    enemy_min_eta: int | None
    rough_path_clear: bool


@dataclass
class GraphState:
    player: int
    step: int
    angular_velocity: float
    planets: list[list]
    fleets: list[list]
    comet_ids: set[int]
    raw_comets: list
    planet_ids: list[int]
    id_to_index: dict[int, int]
    my_planet_ids: set[int]
    enemy_player_ids: set[int]
    inbound_events: dict[int, list[FleetEvent]]
    planet_features: np.ndarray
    edge_features: np.ndarray
    valid_edge_mask: np.ndarray
    edge_estimates: dict[tuple[int, int], EdgeEstimate] = field(default_factory=dict)

    @property
    def planet_by_id(self) -> dict[int, list]:
        return {int(p[0]): p for p in self.planets}

    @property
    def my_planets(self) -> list[list]:
        return [p for p in self.planets if int(p[1]) == self.player]

    @property
    def enemy_or_neutral_planets(self) -> list[list]:
        return [p for p in self.planets if int(p[1]) != self.player]


def get_obs_value(obs: Any, key: str, default=None):
    return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)


def parse_observation(obs: Any) -> dict[str, Any]:
    return {
        "player": int(get_obs_value(obs, "player", 0) or 0),
        "planets": [list(p) for p in (get_obs_value(obs, "planets", []) or [])],
        "fleets": [list(f) for f in (get_obs_value(obs, "fleets", []) or [])],
        "angular_velocity": float(get_obs_value(obs, "angular_velocity", 0.035) or 0.035),
        "comet_ids": {int(x) for x in (get_obs_value(obs, "comet_planet_ids", []) or [])},
        "raw_comets": list(get_obs_value(obs, "comets", []) or []),
        "step": int(get_obs_value(obs, "step", 0) or 0),
    }


def available_ships(planet: list, comet_ids: set[int], reserve: float = RESERVE) -> int:
    if int(planet[0]) in comet_ids:
        return max(0, int(planet[5]))
    return max(0, int(float(planet[5]) * (1.0 - reserve)))


def infer_fleet_events(
    fleets: list[list],
    planets: list[list],
    angular_velocity: float,
    comet_ids: set[int],
    raw_comets: list,
    *,
    angle_tolerance: float = 0.35,
) -> dict[int, list[FleetEvent]]:
    """Infer destination events from fleet rays using moving-target angles."""

    events: dict[int, list[FleetEvent]] = defaultdict(list)
    for fleet in fleets:
        fleet_id = int(fleet[0])
        owner = int(fleet[1])
        if owner < 0:
            continue
        fx, fy, angle, ships = float(fleet[2]), float(fleet[3]), float(fleet[4]), int(fleet[6])
        best_pid = None
        best_eta = None
        best_diff = angle_tolerance
        for planet in planets:
            tx, ty, eta = intercept(fx, fy, planet, angular_velocity, ships, comet_ids, raw_comets)
            if tx is None or eta is None:
                continue
            predicted = math.atan2(float(ty) - fy, float(tx) - fx)
            diff = abs((predicted - angle + math.pi) % (2.0 * math.pi) - math.pi)
            if diff < best_diff:
                best_pid = int(planet[0])
                best_eta = int(eta)
                best_diff = diff
        if best_pid is not None and best_eta is not None:
            events[best_pid].append(
                FleetEvent(
                    fleet_id=fleet_id,
                    owner=owner,
                    source_id=int(fleet[5]),
                    target_id=best_pid,
                    ships=ships,
                    eta=best_eta,
                    angle_diff=best_diff,
                )
            )
    for target_events in events.values():
        target_events.sort(key=lambda item: (item.eta, item.owner, item.fleet_id))
    return dict(events)


def inbound_summary(
    ctx: GraphState | None,
    events: dict[int, list[FleetEvent]],
    target_id: int,
    player: int,
    eta: int | None = None,
) -> tuple[int, int, int | None]:
    """Return friendly ships, enemy ships, and earliest enemy ETA."""

    friendly = 0
    enemy = 0
    enemy_min_eta = None
    for event in events.get(int(target_id), []):
        if eta is not None and event.eta > eta:
            continue
        if event.owner == player:
            friendly += event.ships
        else:
            enemy += event.ships
            if enemy_min_eta is None or event.eta < enemy_min_eta:
                enemy_min_eta = event.eta
    return friendly, enemy, enemy_min_eta


def production_rank(planets: list[list], planet_id: int) -> float:
    if not planets:
        return 0.0
    prods = sorted(float(p[6]) for p in planets)
    prod = next((float(p[6]) for p in planets if int(p[0]) == int(planet_id)), 0.0)
    return sum(1 for value in prods if value <= prod) / max(len(prods), 1)


def estimate_edge(
    source: list,
    target: list,
    player: int,
    planets: list[list],
    events: dict[int, list[FleetEvent]],
    angular_velocity: float,
    comet_ids: set[int],
    raw_comets: list,
    step: int,
    ships: int,
) -> EdgeEstimate | None:
    """Estimate an edge with a specific probe ship count."""

    if ships <= 0 or int(source[0]) == int(target[0]):
        return None
    tx, ty, eta = intercept(
        float(source[2]),
        float(source[3]),
        target,
        angular_velocity,
        int(ships),
        comet_ids,
        raw_comets,
    )
    if tx is None or ty is None or eta is None:
        return None
    eta = int(eta)
    distance = math.hypot(float(tx) - float(source[2]), float(ty) - float(source[3]))
    friendly_before, enemy_before, enemy_min_eta = inbound_summary(
        None, events, int(target[0]), player, eta
    )
    arrival_garrison = float(target[5])
    if int(target[1]) >= 0:
        arrival_garrison += float(target[6]) * eta
    required = max(1, int(math.ceil(arrival_garrison + 1.0 - friendly_before)))
    rough_path_clear = not sun_blocked(float(source[2]), float(source[3]), float(tx), float(ty))
    return EdgeEstimate(
        source_id=int(source[0]),
        target_id=int(target[0]),
        tx=float(tx),
        ty=float(ty),
        eta=eta,
        distance=distance,
        required_ships=required,
        arrival_garrison=arrival_garrison,
        friendly_before_eta=friendly_before,
        enemy_before_eta=enemy_before,
        enemy_min_eta=enemy_min_eta,
        rough_path_clear=rough_path_clear,
    )


def refine_capture_need(ctx: GraphState, source: list, target: list, available: int) -> EdgeEstimate | None:
    """Iterate because fleet speed changes with the number of ships launched."""

    estimate = estimate_edge(
        source,
        target,
        ctx.player,
        ctx.planets,
        ctx.inbound_events,
        ctx.angular_velocity,
        ctx.comet_ids,
        ctx.raw_comets,
        ctx.step,
        available,
    )
    if estimate is None:
        return None
    ships = min(max(1, estimate.required_ships), max(available, 1))
    last_required = None
    for _ in range(5):
        estimate = estimate_edge(
            source,
            target,
            ctx.player,
            ctx.planets,
            ctx.inbound_events,
            ctx.angular_velocity,
            ctx.comet_ids,
            ctx.raw_comets,
            ctx.step,
            ships,
        )
        if estimate is None:
            return None
        if estimate.required_ships == last_required:
            return estimate
        last_required = estimate.required_ships
        ships = min(max(1, estimate.required_ships), max(available, 1))
    return estimate


def _planet_features(
    planets: list[list],
    events: dict[int, list[FleetEvent]],
    player: int,
    comet_ids: set[int],
    step: int,
) -> np.ndarray:
    features = np.zeros((len(planets), PLANET_FEATURE_DIM), dtype=np.float32)
    turn_remaining = max(0.0, (MAX_TURNS - float(step)) / MAX_TURNS)
    prod_rank_by_id = {int(p[0]): production_rank(planets, int(p[0])) for p in planets}
    for i, planet in enumerate(planets):
        pid = int(planet[0])
        owner = int(planet[1])
        x, y = float(planet[2]), float(planet[3])
        dx, dy = x - 50.0, y - 50.0
        orbit_radius = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        friendly, enemy, enemy_min_eta = inbound_summary(None, events, pid, player, None)

        features[i, 0] = 1.0 if owner == player else 0.0
        features[i, 1] = 1.0 if owner == -1 else 0.0
        features[i, 2] = 1.0 if owner >= 0 and owner != player else 0.0
        features[i, 3] = min(float(planet[5]) / 200.0, 1.0)
        features[i, 4] = min(float(planet[6]) / 5.0, 1.0)
        features[i, 5] = min(float(planet[4]) / 8.0, 1.0)
        features[i, 6] = x / 100.0
        features[i, 7] = y / 100.0
        features[i, 8] = min(orbit_radius / 50.0, 1.0)
        features[i, 9] = math.sin(angle)
        features[i, 10] = math.cos(angle)
        features[i, 11] = 1.0 if is_orbiting(planet) else 0.0
        features[i, 12] = 1.0 if pid in comet_ids else 0.0
        features[i, 13] = min(friendly / 200.0, 1.0)
        features[i, 14] = min(enemy / 200.0, 1.0)
        features[i, 15] = 1.0 if enemy_min_eta is None else min(enemy_min_eta / 120.0, 1.0)
        features[i, 16] = min(available_ships(planet, comet_ids) / 200.0, 1.0)
        features[i, 17] = 1.0 if enemy > float(planet[5]) + float(planet[6]) * (enemy_min_eta or 0) else 0.0
        features[i, 18] = turn_remaining
        features[i, 19] = prod_rank_by_id[pid]
    return features


def _edge_features(
    planets: list[list],
    events: dict[int, list[FleetEvent]],
    player: int,
    comet_ids: set[int],
    raw_comets: list,
    angular_velocity: float,
    step: int,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], EdgeEstimate]]:
    n = len(planets)
    features = np.zeros((n, n, EDGE_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((n, n), dtype=bool)
    estimates: dict[tuple[int, int], EdgeEstimate] = {}
    my_ships = [float(p[5]) for p in planets if int(p[1]) == player]
    mean_my = sum(my_ships) / max(len(my_ships), 1)
    max_my = max(my_ships) if my_ships else 1.0
    turn_remaining = max(0.0, (MAX_TURNS - float(step)) / MAX_TURNS)
    prod_rank_by_id = {int(p[0]): production_rank(planets, int(p[0])) for p in planets}

    for i, source in enumerate(planets):
        src_owner = int(source[1])
        if src_owner != player:
            continue
        available = available_ships(source, comet_ids)
        source_friendly, source_enemy, _ = inbound_summary(None, events, int(source[0]), player, None)
        for j, target in enumerate(planets):
            if i == j:
                continue
            target_id = int(target[0])
            target_owner = int(target[1])
            estimate = estimate_edge(
                source,
                target,
                player,
                planets,
                events,
                angular_velocity,
                comet_ids,
                raw_comets,
                step,
                max(available, 1),
            )
            if estimate is None:
                continue
            estimates[(int(source[0]), target_id)] = estimate
            target_friendly, target_enemy, _ = inbound_summary(None, events, target_id, player, None)
            future_prod_value = float(target[6]) * max(0.0, MAX_TURNS - step - estimate.eta)

            features[i, j, 0] = 1.0 if src_owner == player else 0.0
            features[i, j, 1] = 1.0 if target_owner == player else 0.0
            features[i, j, 2] = 1.0 if target_owner == -1 else 0.0
            features[i, j, 3] = 1.0 if target_owner >= 0 and target_owner != player else 0.0
            features[i, j, 4] = min(float(source[5]) / 200.0, 1.0)
            features[i, j, 5] = min(available / 200.0, 1.0)
            features[i, j, 6] = min(float(source[6]) / 5.0, 1.0)
            features[i, j, 7] = min(float(target[5]) / 200.0, 1.0)
            features[i, j, 8] = min(float(target[6]) / 5.0, 1.0)
            features[i, j, 9] = min(float(target[4]) / 8.0, 1.0)
            features[i, j, 10] = min(estimate.distance / 150.0, 1.0)
            features[i, j, 11] = min(estimate.eta / 120.0, 1.0)
            features[i, j, 12] = min(estimate.required_ships / 200.0, 1.0)
            features[i, j, 13] = min(available / max(estimate.required_ships, 1), 2.0) / 2.0
            features[i, j, 14] = min(estimate.friendly_before_eta / 200.0, 1.0)
            features[i, j, 15] = min(estimate.enemy_before_eta / 200.0, 1.0)
            features[i, j, 16] = 1.0 if estimate.enemy_min_eta is None else min(estimate.enemy_min_eta / 120.0, 1.0)
            features[i, j, 17] = min(future_prod_value / 2500.0, 1.0)
            features[i, j, 18] = prod_rank_by_id[target_id]
            features[i, j, 19] = (float(source[5]) - mean_my) / max(max_my, 1.0)
            features[i, j, 20] = 1.0 if is_orbiting(target) else 0.0
            features[i, j, 21] = 1.0 if target_id in comet_ids else 0.0
            features[i, j, 22] = 1.0 if estimate.rough_path_clear else 0.0
            features[i, j, 23] = turn_remaining
            features[i, j, 24] = min(source_enemy / max(float(source[5]) + source_friendly + 1.0, 1.0), 1.0)
            features[i, j, 25] = min(target_enemy / max(float(target[5]) + target_friendly + 1.0, 1.0), 1.0)

            mask[i, j] = src_owner == player and available > 0 and estimate.rough_path_clear
    return features, mask, estimates


def build_graph_state(obs: Any) -> GraphState:
    parsed = parse_observation(obs)
    planets = parsed["planets"]
    planet_ids = [int(p[0]) for p in planets]
    events = infer_fleet_events(
        parsed["fleets"],
        planets,
        parsed["angular_velocity"],
        parsed["comet_ids"],
        parsed["raw_comets"],
    )
    player = parsed["player"]
    enemy_players = {int(p[1]) for p in planets if int(p[1]) >= 0 and int(p[1]) != player}
    planet_features = _planet_features(
        planets,
        events,
        player,
        parsed["comet_ids"],
        parsed["step"],
    )
    edge_features, edge_mask, estimates = _edge_features(
        planets,
        events,
        player,
        parsed["comet_ids"],
        parsed["raw_comets"],
        parsed["angular_velocity"],
        parsed["step"],
    )
    return GraphState(
        player=player,
        step=parsed["step"],
        angular_velocity=parsed["angular_velocity"],
        planets=planets,
        fleets=parsed["fleets"],
        comet_ids=parsed["comet_ids"],
        raw_comets=parsed["raw_comets"],
        planet_ids=planet_ids,
        id_to_index={pid: i for i, pid in enumerate(planet_ids)},
        my_planet_ids={int(p[0]) for p in planets if int(p[1]) == player},
        enemy_player_ids=enemy_players,
        inbound_events=events,
        planet_features=planet_features,
        edge_features=edge_features,
        valid_edge_mask=edge_mask,
        edge_estimates=estimates,
    )
