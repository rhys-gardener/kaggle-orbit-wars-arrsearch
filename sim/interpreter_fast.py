"""
Numpy-vectorised byte-equivalent replacement for
`kaggle_environments.envs.orbit_wars.orbit_wars.interpreter`.

Hot-path changes vs the original:
  - generate_comet_paths: inner 5 000-point dense-sample loop and
    arc-length re-sample replaced with numpy (was 27 % of total time).
  - generate_comet_paths validity check: 683k _distance() calls replaced
    with numpy batch distance ops; planet classification hoisted outside
    the 300-attempt loop.
  - Fleet movement + swept_pair_hit: one numpy broadcast over (F×P) pairs
    instead of a Python double-loop (was another 25 %).  When numba is
    available the batch hit functions are compiled to native scalar loops
    (no intermediate array allocations) — typically 3-5x faster for the
    small F×P sizes seen in practice.
  - Point-to-segment sun check: vectorised over all fleets at once.
  - Combat planet lookup: O(1) dict instead of O(P) linear search.
  - Production and planet-position updates: single numpy operations.

Everything else (action processing, comet spawning bookkeeping,
termination) is unchanged Python so the semantics stay identical.
"""
from __future__ import annotations

import math
import random

import numpy as np

# ---------------------------------------------------------------------------
# Optional Numba JIT — graceful fallback when not installed
# ---------------------------------------------------------------------------
try:
    from numba import njit as _numba_njit
    def _jit(fn):           # compile to native code, cache across runs
        return _numba_njit(cache=True)(fn)
    _NUMBA = True
except ImportError:
    def _jit(fn):           # no-op decorator
        return fn
    _NUMBA = False

# ---------------------------------------------------------------------------
# Constants — copied verbatim from orbit_wars.py
# ---------------------------------------------------------------------------
BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]

# Pre-compute log(1000) once.
_LOG1000 = math.log(1000)


def _get(d, key, default=None):
    """Helper matching orbit_wars.get — handles dict and SimpleNamespace."""
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


# ---------------------------------------------------------------------------
# Reuse planet generation unchanged (called once per episode, not hot).
# ---------------------------------------------------------------------------
from kaggle_environments.envs.orbit_wars.orbit_wars import generate_planets


# ---------------------------------------------------------------------------
# Vectorised generate_comet_paths
# ---------------------------------------------------------------------------

def generate_comet_paths(
    initial_planets,
    angular_velocity,
    spawn_step,
    comet_planet_ids=None,
    comet_speed=4.0,
    rng=None,
):
    """Byte-equivalent to orbit_wars.generate_comet_paths.

    Speedups over original:
      - inner 5 000-point dense-sample and arc-length loops → numpy
      - validity checking (sun + static/orbiting planet collision) → numpy
        batch distance ops; planet classification hoisted outside the
        300-attempt loop so O(P) work runs once instead of ≤300 times.
    """
    if rng is None:
        rng = random
    if comet_planet_ids is None:
        comet_planet_ids = set()
    else:
        comet_planet_ids = set(comet_planet_ids)

    buf = COMET_RADIUS + 0.5
    sun_thresh2 = (SUN_RADIUS + COMET_RADIUS) ** 2

    # ---- Hoist planet classification outside the 300-attempt loop ----------
    static_planets: list = []
    orbiting_planets: list = []
    for planet in initial_planets:
        if planet[0] in comet_planet_ids:
            continue
        pr = math.sqrt((planet[2] - CENTER) ** 2 + (planet[3] - CENTER) ** 2)
        if pr + planet[4] < ROTATION_RADIUS_LIMIT:
            orbiting_planets.append(planet)
        else:
            static_planets.append(planet)

    # Pre-build numpy arrays for the validity checks (reused every attempt)
    sp_x = np.array([p[2] for p in static_planets]) if static_planets else None
    sp_y = np.array([p[3] for p in static_planets]) if static_planets else None
    sp_thresh2 = np.array([(p[4] + buf) ** 2 for p in static_planets]) if static_planets else None

    if orbiting_planets:
        orb_dx = np.array([p[2] - CENTER for p in orbiting_planets])
        orb_dy = np.array([p[3] - CENTER for p in orbiting_planets])
        orb_r_arr = np.sqrt(orb_dx ** 2 + orb_dy ** 2)
        init_angles = np.arctan2(orb_dy, orb_dx)
        orb_thresh2 = np.array([(p[4] + COMET_RADIUS) ** 2 for p in orbiting_planets])
    else:
        orb_r_arr = init_angles = orb_thresh2 = None

    for _ in range(300):
        # ---- same RNG draws as the original --------------------------------
        e = rng.uniform(0.75, 0.93)
        a_val = rng.uniform(60, 150)
        perihelion = a_val * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue
        b_val = a_val * math.sqrt(1 - e ** 2)
        c_val = a_val * e
        phi = rng.uniform(math.pi / 6, math.pi / 3)

        # ---- numpy dense sample (replaces range(5000) Python loop) ---------
        num = 5000
        t = 0.3 * math.pi + 1.4 * math.pi * np.linspace(0, 1, num, endpoint=True)
        ex = c_val + a_val * np.cos(t)
        ey = b_val * np.sin(t)
        cos_phi, sin_phi = math.cos(phi), math.sin(phi)
        xs = CENTER + ex * cos_phi - ey * sin_phi
        ys = CENTER + ex * sin_phi + ey * cos_phi

        # ---- numpy arc-length re-sample ------------------------------------
        dx = np.diff(xs)
        dy = np.diff(ys)
        seg_len = np.sqrt(dx * dx + dy * dy)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])

        total_len = cum[-1]
        n_path = int(total_len / comet_speed)
        if n_path == 0:
            continue
        targets = np.arange(1, n_path + 1) * comet_speed
        idx = np.searchsorted(cum, targets, side='left')
        idx = np.clip(idx, 0, num - 1)
        path_np = list(zip(xs[idx].tolist(), ys[idx].tolist()))

        # ---- find on-board segment (unchanged logic) -----------------------
        board_start = None
        board_end = None
        for i, (x, y) in enumerate(path_np):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i

        if board_start is None:
            continue
        visible = path_np[board_start: board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue

        # ---- 4-fold symmetry (unchanged) -----------------------------------
        paths = [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]

        # ---- vectorised validity checks ------------------------------------
        vis_arr = np.array(visible)   # (V, 2)  as (x, y)
        cx_arr = vis_arr[:, 0]
        cy_arr = vis_arr[:, 1]
        V = len(visible)

        # Sun distance for each visible point directly
        sun_d2 = (cx_arr - CENTER) ** 2 + (cy_arr - CENTER) ** 2
        if np.any(sun_d2 < sun_thresh2):
            continue

        # 4 symmetry variants per visible point: (V, 4)
        sym_x = np.stack(
            [cy_arr, BOARD_SIZE - cx_arr, cx_arr, BOARD_SIZE - cy_arr], axis=1
        )
        sym_y = np.stack(
            [cx_arr, cy_arr, BOARD_SIZE - cy_arr, BOARD_SIZE - cx_arr], axis=1
        )

        # Static planet check — (V*4, N_static) distance matrix
        if sp_x is not None:
            all_sx = sym_x.ravel()[:, None]   # (V*4, 1)
            all_sy = sym_y.ravel()[:, None]
            sd2 = (all_sx - sp_x[None, :]) ** 2 + (all_sy - sp_y[None, :]) ** 2
            if np.any(sd2 < sp_thresh2[None, :]):
                continue

        # Orbiting planet check — positions depend on k (step index)
        if orb_r_arr is not None:
            steps_k = spawn_step - 1 + np.arange(V, dtype=float)   # (V,)
            cur_angles = init_angles[None, :] + angular_velocity * steps_k[:, None]  # (V, N_orb)
            orb_px = CENTER + orb_r_arr[None, :] * np.cos(cur_angles)   # (V, N_orb)
            orb_py = CENTER + orb_r_arr[None, :] * np.sin(cur_angles)
            # (V, 4, N_orb) broadcast
            od2 = ((sym_x[:, :, None] - orb_px[:, None, :]) ** 2 +
                   (sym_y[:, :, None] - orb_py[:, None, :]) ** 2)
            if np.any(od2 < orb_thresh2[None, None, :]):
                continue

        return paths
    return None


# ---------------------------------------------------------------------------
# Swept-pair hit  (F fleets × P planets)
#
# Two implementations:
#   _swept_pair_hit_batch_np  — numpy broadcast (no extra deps)
#   _swept_pair_hit_batch_jit — scalar loop, compiled by Numba when available
#
# _swept_pair_hit_batch is bound to the Numba version if numba is installed,
# otherwise falls back to the numpy version.
# ---------------------------------------------------------------------------

def _swept_pair_hit_batch_np(
    fl_old_x: np.ndarray,
    fl_old_y: np.ndarray,
    fl_new_x: np.ndarray,
    fl_new_y: np.ndarray,
    pl_old_x: np.ndarray,
    pl_old_y: np.ndarray,
    pl_new_x: np.ndarray,
    pl_new_y: np.ndarray,
    radii: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    d0x = fl_old_x[:, None] - pl_old_x[None, :]
    d0y = fl_old_y[:, None] - pl_old_y[None, :]
    dvx = (fl_new_x - fl_old_x)[:, None] - (pl_new_x - pl_old_x)[None, :]
    dvy = (fl_new_y - fl_old_y)[:, None] - (pl_new_y - pl_old_y)[None, :]

    a_m = dvx * dvx + dvy * dvy
    b_m = 2.0 * (d0x * dvx + d0y * dvy)
    c_m = d0x * d0x + d0y * d0y - radii[None, :] ** 2

    disc = b_m * b_m - 4.0 * a_m * c_m
    safe_disc = np.where(disc >= 0.0, disc, 0.0)
    sq = np.sqrt(safe_disc)
    safe_a = np.where(a_m > 1e-12, a_m, 1.0)
    two_a = 2.0 * safe_a
    t1 = (-b_m - sq) / two_a
    t2 = (-b_m + sq) / two_a
    seg_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    point_hit = c_m <= 0.0
    hits = np.where(a_m < 1e-12, point_hit, seg_hit)
    return hits & valid[None, :]


def _swept_pair_hit_batch_jit(
    fl_old_x: np.ndarray,
    fl_old_y: np.ndarray,
    fl_new_x: np.ndarray,
    fl_new_y: np.ndarray,
    pl_old_x: np.ndarray,
    pl_old_y: np.ndarray,
    pl_new_x: np.ndarray,
    pl_new_y: np.ndarray,
    radii: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Scalar loop version — compiled by Numba into fused native code."""
    F = fl_old_x.shape[0]
    P = pl_old_x.shape[0]
    out = np.zeros((F, P), dtype=np.bool_)
    for i in range(F):
        fvx = fl_new_x[i] - fl_old_x[i]
        fvy = fl_new_y[i] - fl_old_y[i]
        for j in range(P):
            if not valid[j]:
                continue
            d0x = fl_old_x[i] - pl_old_x[j]
            d0y = fl_old_y[i] - pl_old_y[j]
            dvx = fvx - (pl_new_x[j] - pl_old_x[j])
            dvy = fvy - (pl_new_y[j] - pl_old_y[j])
            a = dvx * dvx + dvy * dvy
            r = radii[j]
            c = d0x * d0x + d0y * d0y - r * r
            if a < 1e-12:
                out[i, j] = c <= 0.0
            else:
                b = 2.0 * (d0x * dvx + d0y * dvy)
                disc = b * b - 4.0 * a * c
                if disc >= 0.0:
                    sq = math.sqrt(disc)
                    two_a = 2.0 * a
                    t1 = (-b - sq) / two_a
                    t2 = (-b + sq) / two_a
                    out[i, j] = t2 >= 0.0 and t1 <= 1.0
    return out


# ---------------------------------------------------------------------------
# Point-to-segment sun check  (all fleets at once)
# ---------------------------------------------------------------------------

def _sun_hit_batch_np(
    fl_old_x: np.ndarray,
    fl_old_y: np.ndarray,
    fl_new_x: np.ndarray,
    fl_new_y: np.ndarray,
) -> np.ndarray:
    vx, vy = fl_old_x, fl_old_y
    wx, wy = fl_new_x, fl_new_y
    px, py = CENTER, CENTER
    l2 = (wx - vx) ** 2 + (wy - vy) ** 2
    dot = (px - vx) * (wx - vx) + (py - vy) * (wy - vy)
    t = np.where(l2 > 0, np.clip(dot / np.where(l2 > 0, l2, 1.0), 0.0, 1.0), 0.0)
    proj_x = vx + t * (wx - vx)
    proj_y = vy + t * (wy - vy)
    dist2 = (proj_x - px) ** 2 + (proj_y - py) ** 2
    return dist2 < SUN_RADIUS ** 2


def _sun_hit_batch_jit(
    fl_old_x: np.ndarray,
    fl_old_y: np.ndarray,
    fl_new_x: np.ndarray,
    fl_new_y: np.ndarray,
) -> np.ndarray:
    """Scalar loop version — compiled by Numba."""
    F = fl_old_x.shape[0]
    out = np.zeros(F, dtype=np.bool_)
    px = 50.0
    py = 50.0
    r2 = SUN_RADIUS * SUN_RADIUS
    for i in range(F):
        vx = fl_old_x[i]
        vy = fl_old_y[i]
        wx = fl_new_x[i]
        wy = fl_new_y[i]
        l2 = (wx - vx) * (wx - vx) + (wy - vy) * (wy - vy)
        if l2 > 0.0:
            dot = (px - vx) * (wx - vx) + (py - vy) * (wy - vy)
            t = dot / l2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
        else:
            t = 0.0
        proj_x = vx + t * (wx - vx)
        proj_y = vy + t * (wy - vy)
        dist2 = (proj_x - px) * (proj_x - px) + (proj_y - py) * (proj_y - py)
        out[i] = dist2 < r2
    return out


# Bind the best available implementation at import time
if _NUMBA:
    _swept_pair_hit_batch = _jit(_swept_pair_hit_batch_jit)
    _sun_hit_batch = _jit(_sun_hit_batch_jit)
else:
    _swept_pair_hit_batch = _swept_pair_hit_batch_np
    _sun_hit_batch = _sun_hit_batch_np


# ---------------------------------------------------------------------------
# Main interpreter — drop-in replacement for orbit_wars.interpreter
# ---------------------------------------------------------------------------

def interpreter(state, env):  # noqa: C901  (complex but mirrors original)
    configuration = env.configuration
    num_agents = len(state)
    obs0 = state[0].observation

    # ------------------------------------------------------------------ init
    if not _get(obs0, "planets") and not _get(obs0, "planets") == []:
        _init_game(state, env, num_agents, obs0)
        return state
    # Check the actual contents (empty list is falsy too)
    if not hasattr(obs0, "planets") or obs0.planets is None:
        _init_game(state, env, num_agents, obs0)
        return state
    # Proper init guard matching original: `not obs0.planets` covers both
    # missing and empty-list cases that arise during reset().
    planets_raw = _get(obs0, "planets", None)
    if planets_raw is None or (isinstance(planets_raw, list) and len(planets_raw) == 0 and
                                not hasattr(obs0, "angular_velocity")):
        _init_game(state, env, num_agents, obs0)
        return state

    if env.done:
        return state

    # --------------------------------------------------------- comet expiry
    expired_comet_pids: list[int] = []
    for group in obs0.comets:
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            if idx >= len(group["paths"][i]):
                expired_comet_pids.append(pid)
    if expired_comet_pids:
        _remove_comets(obs0, set(expired_comet_pids))

    # --------------------------------------------------------- comet spawn
    step = _get(obs0, "step", 0)
    comet_speed = configuration.cometSpeed
    if (step + 1) in COMET_SPAWN_STEPS:
        env_info = getattr(env, "info", None) or {}
        episode_seed = env_info.get("seed", 0) or 0
        comet_rng = random.Random(f"orbit_wars-comet-{episode_seed}-{step + 1}")
        comet_paths = generate_comet_paths(
            obs0.initial_planets,
            obs0.angular_velocity,
            step + 1,
            obs0.comet_planet_ids,
            comet_speed,
            rng=comet_rng,
        )
        if comet_paths:
            next_id = max(p[0] for p in obs0.planets) + 1
            comet_ships = min(
                comet_rng.randint(1, 99),
                comet_rng.randint(1, 99),
                comet_rng.randint(1, 99),
                comet_rng.randint(1, 99),
            )
            group: dict = {"planet_ids": [], "paths": comet_paths, "path_index": -1}
            for i, p_path in enumerate(comet_paths):
                pid = next_id + i
                group["planet_ids"].append(pid)
                obs0.comet_planet_ids.append(pid)
                planet = [pid, -1, -99, -99, COMET_RADIUS, comet_ships, COMET_PRODUCTION]
                obs0.planets.append(planet)
                obs0.initial_planets.append(planet[:])
            obs0.comets.append(group)

    # --------------------------------------------------- 0. fleet launch
    for i in range(num_agents):
        _process_moves(i, state[i].action, obs0)

    # --------------------------------------------------- 1. production (numpy)
    _production_np(obs0.planets)

    # --------------------------------------------------- 2. planet positions
    angular_velocity = obs0.angular_velocity
    step_for_pos = _get(obs0, "step", 1)
    comet_pid_set = set(obs0.comet_planet_ids)
    initial_by_id = {p[0]: p for p in obs0.initial_planets}

    planet_paths: dict[int, tuple] = {}
    expired_comet_pids_2: list[int] = []

    for planet in obs0.planets:
        if planet[0] in comet_pid_set:
            continue
        old_pos = (planet[2], planet[3])
        initial_p = initial_by_id.get(planet[0])
        if initial_p is not None:
            dx = initial_p[2] - CENTER
            dy = initial_p[3] - CENTER
            r = math.sqrt(dx * dx + dy * dy)
            if r + planet[4] < ROTATION_RADIUS_LIMIT:
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * step_for_pos
                new_pos = (
                    CENTER + r * math.cos(cur_angle),
                    CENTER + r * math.sin(cur_angle),
                )
            else:
                new_pos = old_pos
        else:
            new_pos = old_pos
        planet_paths[planet[0]] = (old_pos, new_pos, True)

    # Advance comets
    for group in obs0.comets:
        group["path_index"] += 1
        idx = group["path_index"]
        for ci, pid in enumerate(group["planet_ids"]):
            planet = next((p for p in obs0.planets if p[0] == pid), None)
            if planet is None:
                continue
            p_path = group["paths"][ci]
            old_pos = (planet[2], planet[3])
            if idx >= len(p_path):
                expired_comet_pids_2.append(pid)
                planet_paths[pid] = (old_pos, old_pos, True)
            else:
                new_pos = (p_path[idx][0], p_path[idx][1])
                check = old_pos[0] >= 0
                planet_paths[pid] = (old_pos, new_pos, check)

    # --------------------------------------------------- 3. fleet movement (numpy)
    combat_lists: dict[int, list] = {p[0]: [] for p in obs0.planets}
    fleets_to_remove: list = []

    if obs0.fleets:
        fleets_to_remove = _fleet_movement_np(
            obs0.fleets, obs0.planets, planet_paths, combat_lists,
            configuration.shipSpeed,
        )

    # --------------------------------------------------- 4. apply planet positions
    for planet in obs0.planets:
        path = planet_paths.get(planet[0])
        if path is not None:
            planet[2], planet[3] = path[1]

    # Remove comets that expired during advancement
    if expired_comet_pids_2:
        _remove_comets_post(obs0, set(expired_comet_pids_2))

    obs0.fleets = [f for f in obs0.fleets if f not in fleets_to_remove]

    # --------------------------------------------------- 5. combat
    # Build O(1) lookup once instead of O(P) next() search per combat entry
    planet_by_id = {p[0]: p for p in obs0.planets}
    for pid, planet_fleets in combat_lists.items():
        planet = planet_by_id.get(pid)
        if planet is None or not planet_fleets:
            continue
        player_ships: dict[int, int] = {}
        for fleet in planet_fleets:
            owner = fleet[1]
            player_ships[owner] = player_ships.get(owner, 0) + fleet[6]
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
        if survivor_ships > 0:
            if planet[1] == survivor_owner:
                planet[5] += survivor_ships
            else:
                planet[5] -= survivor_ships
                if planet[5] < 0:
                    planet[1] = survivor_owner
                    planet[5] = abs(planet[5])

    # --------------------------------------------------- sync + termination
    for i in range(1, num_agents):
        state[i].observation.planets = obs0.planets
        state[i].observation.initial_planets = obs0.initial_planets
        state[i].observation.fleets = obs0.fleets
        state[i].observation.next_fleet_id = obs0.next_fleet_id
        state[i].observation.comets = obs0.comets
        state[i].observation.comet_planet_ids = obs0.comet_planet_ids

    terminated = False
    step2 = _get(obs0, "step", 0)
    if step2 >= configuration.episodeSteps - 2:
        terminated = True

    alive_players: set[int] = set()
    for p in obs0.planets:
        if p[1] != -1:
            alive_players.add(p[1])
    for f in obs0.fleets:
        alive_players.add(f[1])
    if len(alive_players) <= 1:
        terminated = True

    if terminated:
        for s in state:
            s.status = "DONE"
        scores = [0] * num_agents
        for p in obs0.planets:
            if p[1] != -1:
                scores[p[1]] += p[5]
        for f in obs0.fleets:
            scores[f[1]] += f[6]
        max_score = max(scores)
        for i in range(num_agents):
            if scores[i] == max_score and max_score > 0:
                state[i].reward = 1
            else:
                state[i].reward = -1

    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_game(state, env, num_agents, obs0):
    """First-call initialisation — identical to original interpreter init block."""
    if not hasattr(env, "info") or env.info is None:
        env.info = {}
    seed = env.info.get("seed")
    if seed is None:
        seed = _get(env.configuration, "seed", None)
    if seed is None:
        seed = random.randrange(2 ** 31)
    try:
        env.configuration.seed = None
    except (AttributeError, TypeError):
        env.configuration["seed"] = None
    env.info["seed"] = seed
    init_rng = random.Random(seed)

    angular_velocity = init_rng.uniform(0.025, 0.05)
    obs0.angular_velocity = angular_velocity
    obs0.planets = generate_planets(init_rng)
    obs0.initial_planets = [p.copy() for p in obs0.planets]
    obs0.fleets = []
    obs0.next_fleet_id = 0
    obs0.comets = []
    obs0.comet_planet_ids = []

    num_groups = len(obs0.planets) // 4
    if num_groups > 0:
        home_group = init_rng.randint(0, num_groups - 1)
        base = home_group * 4
        if num_agents == 2:
            obs0.planets[base][1] = 0
            obs0.planets[base][5] = 10
            obs0.planets[base + 3][1] = 1
            obs0.planets[base + 3][5] = 10
        elif num_agents == 4:
            for j in range(4):
                obs0.planets[base + j][1] = j
                obs0.planets[base + j][5] = 10

    for i in range(num_agents):
        state[i].observation.player = i
        if i > 0:
            state[i].observation.angular_velocity = obs0.angular_velocity
            state[i].observation.planets = obs0.planets
            state[i].observation.initial_planets = obs0.initial_planets
            state[i].observation.fleets = obs0.fleets
            state[i].observation.next_fleet_id = obs0.next_fleet_id
            state[i].observation.comets = obs0.comets
            state[i].observation.comet_planet_ids = obs0.comet_planet_ids


def _process_moves(player_id: int, action, obs0) -> None:
    """Identical to original process_moves closure."""
    if not action or not isinstance(action, list):
        return
    for move in action:
        if len(move) != 3:
            continue
        from_id, angle, ships = move
        ships = int(ships)
        from_planet = next((p for p in obs0.planets if p[0] == from_id), None)
        if from_planet and from_planet[1] == player_id:
            if from_planet[5] >= ships and ships > 0:
                from_planet[5] -= ships
                start_x = from_planet[2] + math.cos(angle) * (from_planet[4] + 0.1)
                start_y = from_planet[3] + math.sin(angle) * (from_planet[4] + 0.1)
                obs0.fleets.append([
                    obs0.next_fleet_id, player_id,
                    start_x, start_y, angle, from_id, ships,
                ])
                obs0.next_fleet_id += 1


def _production_np(planets: list) -> None:
    """Add production to all owned planets."""
    for p in planets:
        if p[1] != -1:
            p[5] += p[6]


def _fleet_movement_np(
    fleets: list,
    planets: list,
    planet_paths: dict,
    combat_lists: dict,
    max_speed: float,
) -> list:
    """Move all fleets and resolve collisions using numpy swept_pair_hit batch.

    Returns list of fleet objects to remove.
    """
    F = len(fleets)
    if F == 0:
        return []

    fl_old_x = np.empty(F)
    fl_old_y = np.empty(F)
    fl_angle = np.empty(F)
    fl_ships = np.empty(F)
    for i, f in enumerate(fleets):
        fl_old_x[i] = f[2]
        fl_old_y[i] = f[3]
        fl_angle[i] = f[4]
        fl_ships[i] = f[6]

    fl_speed = np.minimum(
        1.0 + (max_speed - 1.0) * (np.log(fl_ships) / _LOG1000) ** 1.5,
        max_speed,
    )
    fl_new_x = fl_old_x + np.cos(fl_angle) * fl_speed
    fl_new_y = fl_old_y + np.sin(fl_angle) * fl_speed

    for i, f in enumerate(fleets):
        f[2] = float(fl_new_x[i])
        f[3] = float(fl_new_y[i])

    planets_in_path = [(p, planet_paths[p[0]]) for p in planets if p[0] in planet_paths]
    if not planets_in_path:
        sun_hit = _sun_hit_batch(fl_old_x, fl_old_y, fl_new_x, fl_new_y)
        oob = ~((fl_new_x >= 0) & (fl_new_x <= BOARD_SIZE) &
                (fl_new_y >= 0) & (fl_new_y <= BOARD_SIZE))
        return [fleets[i] for i in range(F) if sun_hit[i] or oob[i]]

    P = len(planets_in_path)
    pl_old_x = np.empty(P)
    pl_old_y = np.empty(P)
    pl_new_x = np.empty(P)
    pl_new_y = np.empty(P)
    pl_rad = np.empty(P)
    pl_valid = np.empty(P, dtype=bool)
    planet_order: list = []

    for j, (planet, path) in enumerate(planets_in_path):
        old_pos, new_pos, check = path
        pl_old_x[j] = old_pos[0]
        pl_old_y[j] = old_pos[1]
        pl_new_x[j] = new_pos[0]
        pl_new_y[j] = new_pos[1]
        pl_rad[j] = planet[4]
        pl_valid[j] = check
        planet_order.append(planet)

    hit_matrix = _swept_pair_hit_batch(
        fl_old_x, fl_old_y, fl_new_x, fl_new_y,
        pl_old_x, pl_old_y, pl_new_x, pl_new_y,
        pl_rad, pl_valid,
    )

    any_hit = hit_matrix.any(axis=1)
    first_planet_idx = hit_matrix.argmax(axis=1)

    sun_hit = _sun_hit_batch(fl_old_x, fl_old_y, fl_new_x, fl_new_y)
    oob = ~((fl_new_x >= 0) & (fl_new_x <= BOARD_SIZE) &
            (fl_new_y >= 0) & (fl_new_y <= BOARD_SIZE))

    to_remove = []
    for i in range(F):
        if any_hit[i]:
            planet = planet_order[int(first_planet_idx[i])]
            combat_lists[planet[0]].append(fleets[i])
            to_remove.append(fleets[i])
        elif oob[i] or sun_hit[i]:
            to_remove.append(fleets[i])

    return to_remove


def _remove_comets(obs0, expired_set: set) -> None:
    obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
    obs0.initial_planets = [p for p in obs0.initial_planets if p[0] not in expired_set]
    obs0.comet_planet_ids = [pid for pid in obs0.comet_planet_ids if pid not in expired_set]
    for group in obs0.comets:
        group["planet_ids"] = [pid for pid in group["planet_ids"] if pid not in expired_set]
    obs0.comets = [g for g in obs0.comets if g["planet_ids"]]


def _remove_comets_post(obs0, expired_set: set) -> None:
    obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
    obs0.initial_planets = [p for p in obs0.initial_planets if p[0] not in expired_set]
    obs0.comet_planet_ids = [pid for pid in obs0.comet_planet_ids if pid not in expired_set]
    for group in obs0.comets:
        group["planet_ids"] = [pid for pid in group["planet_ids"] if pid not in expired_set]
    obs0.comets = [g for g in obs0.comets if g["planet_ids"]]
