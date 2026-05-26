"""Arrays → kaggle-shape obs dict adapter.

``arrays_to_obs`` reconstructs the obs dict that ``build_graph_state`` expects.
This lets the array-only rollout loop drive the existing graph-state /
candidate-generation primitives without duplicating their feature-extraction
code.

The hard part is fleets. In-flight fleets are synthesised from the schedule
of ``ScheduledFleet`` entries; each is placed along its launch ray so that
``state.infer_fleet_events`` (which back-solves destination via intercept
angle) re-identifies the same target the scheduler intended.

For PR-1, this adapter only needs to round-trip:
  * step-0 obs (no in-flight fleets) → same planet/edge features.
  * a one-turn-after-launch obs (synthetic in-flight fleet) → infer_fleet_events
    returns the scheduled target.

Snapshot fleets that pre-existed in the seed's step-0 obs are passed through
verbatim; they will be incorrect after rel_step > 0 (we'd need to advance
them along their own trajectories), but no PR-1 seed has step-0 fleets so
this is deferred to PR-2.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from src.array_search.scenarios import InitialScenario, planet_positions_at
from src.graph_training.array_env import ScheduledFleet
from src.physics import LAUNCH_RADIUS_OFFSET, fleet_speed, intercept


_SYNTH_FLEET_ID_BASE = 1_000_000


def _advance_raw_comets(raw_comets: list[dict], rel_step: int) -> list[dict]:
    """Return a deep-copied comets list with ``path_index`` bumped by rel_step.

    Comets whose advanced path_index runs off the end of any path are dropped.
    """
    advanced: list[dict] = []
    for group in raw_comets:
        if not group.get("paths"):
            continue
        new_index = int(group.get("path_index", 0)) + int(rel_step)
        if any(new_index >= len(path) for path in group["paths"]):
            continue
        group_copy = copy.deepcopy(group)
        group_copy["path_index"] = new_index
        advanced.append(group_copy)
    return advanced


def _comet_position_at(scenario: InitialScenario, planet_id: int, rel_step: int) -> tuple[float, float] | None:
    for group in scenario.raw_comets:
        if planet_id in group.get("planet_ids", []):
            i = group["planet_ids"].index(planet_id)
            idx = int(group["path_index"]) + int(rel_step)
            path = group["paths"][i]
            if 0 <= idx < len(path):
                return float(path[idx][0]), float(path[idx][1])
            return None
    return None


def _build_planet(
    scenario: InitialScenario,
    i: int,
    owner: int,
    ships_at: float,
    positions: np.ndarray,
    rel_step: int,
) -> list | None:
    pid = int(scenario.planet_ids[i])
    if pid in scenario.comet_planet_ids:
        pos = _comet_position_at(scenario, pid, rel_step)
        if pos is None:
            return None
        x, y = pos
    else:
        x, y = float(positions[i, 0]), float(positions[i, 1])
    return [
        pid,
        int(owner),
        float(x),
        float(y),
        float(scenario.radii[i]),
        float(ships_at),
        float(scenario.production[i]),
    ]


def _synthesise_fleet(
    scenario: InitialScenario,
    entry: ScheduledFleet,
    rel_step: int,
    arrival_rel_turn: int,
    fid: int,
) -> list | None:
    """Place an in-flight fleet so that infer_fleet_events resolves it back
    to ``entry.target_idx``.

    Uses intercept() at launch time to derive the launch angle — the same
    function infer_fleet_events runs to re-identify the target. As long as
    the source/target/ships triple uniquely identifies the launch among
    nearby planets, the round-trip is exact.
    """
    src_idx = int(entry.source_idx)
    tgt_idx = int(entry.target_idx)
    if src_idx < 0 or src_idx >= len(scenario.planet_ids):
        return None
    if tgt_idx < 0 or tgt_idx >= len(scenario.planet_ids):
        return None

    launch_positions = planet_positions_at(scenario, entry.launch_rel_turn)
    sx = float(launch_positions[src_idx, 0])
    sy = float(launch_positions[src_idx, 1])
    src_radius = float(scenario.radii[src_idx])

    # Build a target-planet list reflecting state at launch time so intercept
    # can propagate it forward by the eta it computes internally.
    tgt_planet = [
        int(scenario.planet_ids[tgt_idx]),
        -1,  # owner irrelevant to intercept
        float(launch_positions[tgt_idx, 0]),
        float(launch_positions[tgt_idx, 1]),
        float(scenario.radii[tgt_idx]),
        0.0,
        0.0,
    ]
    raw_comets_at_launch = _advance_raw_comets(scenario.raw_comets, entry.launch_rel_turn)
    comet_ids = set(scenario.comet_planet_ids)

    tx, ty, _eta = intercept(
        sx, sy, tgt_planet, scenario.angular_velocity, int(entry.ships),
        comet_ids, raw_comets_at_launch,
    )
    if tx is None or ty is None:
        return None

    angle = math.atan2(ty - sy, tx - sx)
    ux, uy = math.cos(angle), math.sin(angle)

    launch_x = sx + ux * (src_radius + LAUNCH_RADIUS_OFFSET)
    launch_y = sy + uy * (src_radius + LAUNCH_RADIUS_OFFSET)

    age = int(rel_step) - int(entry.launch_rel_turn)
    speed = fleet_speed(int(entry.ships))
    fx = launch_x + ux * speed * age
    fy = launch_y + uy * speed * age

    src_id = int(scenario.planet_ids[src_idx])
    return [fid, int(entry.owner), float(fx), float(fy), float(angle), int(src_id), int(entry.ships)]


def arrays_to_obs(
    scenario: InitialScenario,
    *,
    seat: int,
    rel_step: int,
    owners: np.ndarray,
    ships: np.ndarray,
    schedule: dict[int, list[ScheduledFleet]] | None = None,
    snapshot_fleets: list[list] | None = None,
    next_fleet_id: int = _SYNTH_FLEET_ID_BASE,
) -> dict[str, Any]:
    """Reconstruct a kaggle-shape obs dict from rollout arrays.

    Parameters
    ----------
    scenario
        Frozen step-0 bundle from ``generate_initial_arrays``.
    seat
        Which player's perspective the obs is for.
    rel_step
        Turns elapsed since ``scenario.step``. The returned obs has
        ``step = scenario.step + rel_step``.
    owners, ships
        Current per-planet owner/ship arrays (mutated by the rollout).
    schedule
        Pending arrivals keyed by relative arrival turn. May be None or empty.
    snapshot_fleets
        Fleets present in the original step-0 obs (typically [] for fresh
        seeds). Passed through verbatim; only valid at rel_step == 0 in PR-1.
    next_fleet_id
        Base for synthetic fleet ids. Synthesised fleets are numbered
        ``next_fleet_id, next_fleet_id + 1, …`` so they never collide with
        engine-issued ids in tests.
    """
    schedule = schedule or {}
    positions = planet_positions_at(scenario, rel_step)

    planets: list[list] = []
    for i in range(len(scenario.planet_ids)):
        planet = _build_planet(scenario, i, int(owners[i]), float(ships[i]), positions, rel_step)
        if planet is not None:
            planets.append(planet)

    fleets: list[list] = []
    if snapshot_fleets:
        for f in snapshot_fleets:
            fleets.append(list(f))

    fid = int(next_fleet_id)
    for arrival_rel_turn, entries in schedule.items():
        if int(arrival_rel_turn) <= int(rel_step):
            continue
        for entry in entries:
            if int(entry.launch_rel_turn) > int(rel_step):
                continue
            fleet = _synthesise_fleet(scenario, entry, int(rel_step), int(arrival_rel_turn), fid)
            if fleet is not None:
                fleets.append(fleet)
                fid += 1

    raw_comets = _advance_raw_comets(scenario.raw_comets, rel_step)

    return {
        "step": int(scenario.step + rel_step),
        "player": int(seat),
        "angular_velocity": float(scenario.angular_velocity),
        "planets": planets,
        "fleets": fleets,
        "comets": raw_comets,
        "comet_planet_ids": list(scenario.comet_planet_ids),
    }
