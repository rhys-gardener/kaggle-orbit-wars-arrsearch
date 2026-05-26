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
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim import make_env
from src.physics import CENTER, is_orbiting


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
    raw_comets: list[dict]         # untouched from obs; advance via path_index
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


def generate_initial_arrays(seed: int, num_players: int = 2) -> InitialScenario:
    """Build an InitialScenario from sim.make_env(seed) at step 0.

    Never calls .step(). Discards the env after reading.
    """

    env = make_env({"seed": int(seed)})
    env.reset(int(num_players))
    obs = env.state[0].observation
    snapshot = _snapshot_obs(obs)

    planets = snapshot["planets"]
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
        raw_comets=snapshot["comets"],
        comet_planet_ids=snapshot["comet_planet_ids"],
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
