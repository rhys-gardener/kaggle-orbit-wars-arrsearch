"""Random initial-scenario generator.

Calls ``sim.make_env(seed).reset(num_players)`` once and converts the step-0
obs into a frozen array bundle. The env is discarded immediately — the
training pipeline never invokes ``.step()`` on it.

The bundle preserves everything needed to:
  * compute every planet's position at any future ``step`` (closed-form orbit),
  * carry comet paths forward,
  * reconstruct a kaggle-shaped obs dict via ``state_adapter.arrays_to_obs``.

Stored fields are deliberately verbose. ``phase0`` and ``orbit_radius`` are
derived from the step-0 ``(x, y)`` so future-position math has no hidden
dependency on the original obs object.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim import make_env
from sim.interpreter_fast import COMET_PRODUCTION, COMET_RADIUS, COMET_SPAWN_STEPS, generate_comet_paths
from src.physics import CENTER, is_orbiting


DEFAULT_COMET_SPEED = 4.0


@dataclass(frozen=True)
class InitialScenario:
    seed: int
    num_players: int
    step: int  # always 0 for an initial scenario

    # Per-planet arrays (length P, indexed by planet position in `planet_ids`)
    planet_ids: np.ndarray         # int64[P]
    owners: np.ndarray             # int16[P]   (-1 for neutral)
    ships: np.ndarray              # float32[P]
    production: np.ndarray         # float32[P]
    radii: np.ndarray              # float32[P]
    planet_xy0: np.ndarray         # float32[P, 2] — position at step 0
    phase0: np.ndarray             # float32[P]   — atan2(y0-CENTER, x0-CENTER); 0 if non-orbiting
    orbit_radius: np.ndarray       # float32[P]   — hypot(x0-CENTER, y0-CENTER); 0 if non-orbiting
    orbiting: np.ndarray           # bool[P]

    # Global
    angular_velocity: float
    raw_comets: list[dict]         # includes future spawns; advance via path_index
    comet_planet_ids: list[int]

    # Snapshot of the original step-0 obs for parity / debugging. Plain dict,
    # no _AttrDict, safe to pickle.
    initial_obs: dict[str, Any]


def _planet_to_list(p: Any) -> list:
    return [
        int(p[0]),
        int(p[1]),
        float(p[2]),
        float(p[3]),
        float(p[4]),
        float(p[5]),
        float(p[6]),
    ]


def _snapshot_obs(obs: Any) -> dict[str, Any]:
    """Convert a kaggle/_AttrDict obs into a plain-dict snapshot."""

    def _get(key, default=None):
        return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)

    return {
        "step": int(_get("step", 0) or 0),
        "player": int(_get("player", 0) or 0),
        "angular_velocity": float(_get("angular_velocity", 0.035) or 0.035),
        "planets": [_planet_to_list(p) for p in (_get("planets", []) or [])],
        "fleets": [list(f) for f in (_get("fleets", []) or [])],
        "comets": copy.deepcopy(list(_get("comets", []) or [])),
        "comet_planet_ids": [int(x) for x in (_get("comet_planet_ids", []) or [])],
    }


def _future_comet_groups(
    *,
    seed: int,
    planets: list[list],
    angular_velocity: float,
    existing_comet_ids: list[int],
    comet_speed: float,
) -> tuple[list[list], list[dict], list[int]]:
    """Precompute deterministic future comet spawns without stepping the env.

    The engine seeds each comet group with
    ``random.Random(f"orbit_wars-comet-{episode_seed}-{spawn_step}")`` and
    stores newly spawned comets with ``path_index == -1`` before the turn is
    advanced.  In an observation at absolute step ``spawn_step`` that group has
    ``path_index == 0``.  We store future groups up front with
    ``path_index == -spawn_step`` so ``path_index + rel_step`` naturally
    reproduces that observation-time index.
    """

    future_planets: list[list] = []
    comet_ids = [int(x) for x in existing_comet_ids]
    groups: list[dict] = []
    next_id = max((int(p[0]) for p in planets), default=-1) + 1

    for spawn_step in COMET_SPAWN_STEPS:
        comet_rng = random.Random(f"orbit_wars-comet-{int(seed)}-{int(spawn_step)}")
        paths = generate_comet_paths(
            planets,
            float(angular_velocity),
            int(spawn_step),
            set(existing_comet_ids),
            float(comet_speed),
            rng=comet_rng,
        )
        if not paths:
            continue
        comet_ships = min(
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
        )
        planet_ids = [next_id + i for i in range(len(paths))]
        group = {
            "planet_ids": list(planet_ids),
            "paths": paths,
            "path_index": -int(spawn_step),
            "spawn_step": int(spawn_step),
            "initial_ships": int(comet_ships),
        }
        groups.append(group)
        for pid in planet_ids:
            planet = [int(pid), -1, -99.0, -99.0, COMET_RADIUS, float(comet_ships), COMET_PRODUCTION]
            future_planets.append(planet)
            comet_ids.append(int(pid))
        next_id += len(paths)

    return future_planets, groups, comet_ids


def generate_initial_arrays(
    seed: int,
    num_players: int = 2,
    *,
    include_future_comets: bool = True,
    comet_speed: float = DEFAULT_COMET_SPEED,
) -> InitialScenario:
    """Build an InitialScenario from sim.make_env(seed) at step 0.

    Never calls .step(). Discards the env after reading.
    """

    env = make_env({"seed": int(seed)})
    env.reset(int(num_players))
    obs = env.state[0].observation
    snapshot = _snapshot_obs(obs)

    planets = [list(p) for p in snapshot["planets"]]
    raw_comets = copy.deepcopy(snapshot["comets"])
    comet_planet_ids = [int(x) for x in snapshot["comet_planet_ids"]]
    if include_future_comets:
        future_planets, future_groups, comet_planet_ids = _future_comet_groups(
            seed=int(seed),
            planets=planets,
            angular_velocity=float(snapshot["angular_velocity"]),
            existing_comet_ids=comet_planet_ids,
            comet_speed=float(comet_speed),
        )
        planets.extend(future_planets)
        raw_comets.extend(future_groups)

    n = len(planets)

    planet_ids = np.array([int(p[0]) for p in planets], dtype=np.int64)
    owners = np.array([int(p[1]) for p in planets], dtype=np.int16)
    radii = np.array([float(p[4]) for p in planets], dtype=np.float32)
    ships = np.array([float(p[5]) for p in planets], dtype=np.float32)
    production = np.array([float(p[6]) for p in planets], dtype=np.float32)
    planet_xy0 = np.array([[float(p[2]), float(p[3])] for p in planets], dtype=np.float32)

    phase0 = np.zeros(n, dtype=np.float32)
    orbit_radius = np.zeros(n, dtype=np.float32)
    orbiting = np.zeros(n, dtype=bool)
    for i, p in enumerate(planets):
        if is_orbiting(p):
            dx = float(p[2]) - CENTER
            dy = float(p[3]) - CENTER
            phase0[i] = math.atan2(dy, dx)
            orbit_radius[i] = math.hypot(dx, dy)
            orbiting[i] = True

    return InitialScenario(
        seed=int(seed),
        num_players=int(num_players),
        step=0,
        planet_ids=planet_ids,
        owners=owners,
        ships=ships,
        production=production,
        radii=radii,
        planet_xy0=planet_xy0,
        phase0=phase0,
        orbit_radius=orbit_radius,
        orbiting=orbiting,
        angular_velocity=float(snapshot["angular_velocity"]),
        raw_comets=raw_comets,
        comet_planet_ids=comet_planet_ids,
        initial_obs=snapshot,
    )


def planet_positions_at(scenario: InitialScenario, step: int) -> np.ndarray:
    """Return ndarray[P, 2] of planet positions at the given step.

    For orbiting planets: ``CENTER + r * (cos, sin)(phase0 + av*step)``.
    For non-orbiting planets: the step-0 position (they don't move).
    Comets are *not* handled here — their positions come from
    ``raw_comets[*]["paths"]`` indexed by ``path_index + step``.
    """

    av = scenario.angular_velocity
    angles = scenario.phase0 + av * float(step)
    moving = np.column_stack(
        [
            CENTER + scenario.orbit_radius * np.cos(angles),
            CENTER + scenario.orbit_radius * np.sin(angles),
        ]
    ).astype(np.float32)
    out = scenario.planet_xy0.copy()
    out[scenario.orbiting] = moving[scenario.orbiting]
    return out


def active_planet_mask(scenario: InitialScenario, step: int) -> np.ndarray:
    """Return a bool mask of planets physically present at ``step``."""

    active = np.ones(len(scenario.planet_ids), dtype=bool)
    comet_ids = set(int(x) for x in scenario.comet_planet_ids)
    for i, pid in enumerate(scenario.planet_ids.tolist()):
        if int(pid) not in comet_ids:
            continue
        active[i] = False
        for group in scenario.raw_comets:
            if int(pid) not in group.get("planet_ids", []):
                continue
            comet_idx = group["planet_ids"].index(int(pid))
            path_index = int(group.get("path_index", 0)) + int(step)
            path = group["paths"][comet_idx]
            active[i] = 0 <= path_index < len(path)
            break
    return active
