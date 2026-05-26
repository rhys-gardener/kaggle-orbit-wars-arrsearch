"""Physics helpers shared between submission code and training pipeline.

These are copies of the helpers defined in ``main.py``. They live here so the
training pipeline can import them without dragging in the 1500-line submission
module (and its embedded v17 weights). ``main.py`` must remain self-contained
for Kaggle submission, so it keeps its own copies; ``tests/test_physics_parity``
asserts the two implementations agree.

Symbols:
    fleet_speed
    sun_blocked
    is_orbiting
    intercept
    _trajectory_first_hit
    _solve_engine_angle

Internal helpers (also used by the geometry cache):
    p2seg
    predict_pos
    _comet_path_for
    comet_predict
    _planet_pos_at
"""

from __future__ import annotations

import math


CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
BOARD_SIZE = 100.0
LAUNCH_RADIUS_OFFSET = 0.1


def fleet_speed(ships):
    return min(MAX_SPEED, 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000)) ** 1.5)


def p2seg(p, v, w):
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0:
        return math.hypot(p[0] - v[0], p[1] - v[1])
    t = max(0.0, min(1.0, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    proj = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return math.hypot(p[0] - proj[0], p[1] - proj[1])


def sun_blocked(sx, sy, tx, ty):
    return p2seg((CENTER, CENTER), (sx, sy), (tx, ty)) < SUN_RADIUS


def is_orbiting(planet):
    r = math.hypot(planet[2] - CENTER, planet[3] - CENTER)
    return r + planet[4] < ROTATION_RADIUS_LIMIT


def predict_pos(planet, angular_velocity, extra_turns):
    if not is_orbiting(planet):
        return planet[2], planet[3]
    dx, dy = planet[2] - CENTER, planet[3] - CENTER
    r = math.sqrt(dx ** 2 + dy ** 2)
    future_angle = math.atan2(dy, dx) + angular_velocity * extra_turns
    return CENTER + r * math.cos(future_angle), CENTER + r * math.sin(future_angle)


def _comet_path_for(planet_id, raw_comets):
    for group in raw_comets:
        if planet_id in group["planet_ids"]:
            i = group["planet_ids"].index(planet_id)
            return group["paths"][i], group["path_index"]
    return None, None


def comet_predict(planet_id, raw_comets, eta):
    path, path_idx = _comet_path_for(planet_id, raw_comets)
    if path is None:
        return None, None
    target_idx = path_idx + int(round(eta))
    if 0 <= target_idx < len(path):
        return float(path[target_idx][0]), float(path[target_idx][1])
    return None, None


def _planet_pos_at(planet, av, extra_turns, comet_ids, raw_comets):
    if planet[0] in comet_ids:
        return comet_predict(planet[0], raw_comets, extra_turns)
    return predict_pos(planet, av, extra_turns)


def intercept(sx, sy, planet, angular_velocity, ships, comet_ids=None, raw_comets=None):
    is_comet = comet_ids and planet[0] in comet_ids
    tx, ty = planet[2], planet[3]
    eta = 0.0
    for _ in range(15):
        dist = math.hypot(tx - sx, ty - sy)
        eta = dist / fleet_speed(ships)
        if is_comet:
            tx, ty = comet_predict(planet[0], raw_comets, eta)
            if tx is None:
                return None, None, None
        else:
            tx, ty = predict_pos(planet, angular_velocity, eta)
    final_eta = int(eta)
    if is_comet:
        tx, ty = comet_predict(planet[0], raw_comets, final_eta)
        if tx is None:
            return None, None, None
    else:
        tx, ty = predict_pos(planet, angular_velocity, final_eta)
    return tx, ty, final_eta


def _trajectory_first_hit(source, angle, ships, planets, av, comet_ids, raw_comets, max_steps=180):
    speed = fleet_speed(ships)
    dx, dy = math.cos(angle), math.sin(angle)
    x = source[2] + dx * (source[4] + LAUNCH_RADIUS_OFFSET)
    y = source[3] + dy * (source[4] + LAUNCH_RADIUS_OFFSET)

    for step in range(max_steps):
        nx = x + speed * dx
        ny = y + speed * dy

        if not (0.0 <= nx <= BOARD_SIZE and 0.0 <= ny <= BOARD_SIZE):
            return None, "bounds", step + 1

        if p2seg((CENTER, CENTER), (x, y), (nx, ny)) < SUN_RADIUS:
            return None, "sun", step + 1

        for planet in planets:
            px, py = _planet_pos_at(planet, av, step, comet_ids, raw_comets)
            if px is None:
                continue
            if p2seg((px, py), (x, y), (nx, ny)) < planet[4]:
                return planet[0], "fleet", step + 1

        for planet in planets:
            ox, oy = _planet_pos_at(planet, av, step, comet_ids, raw_comets)
            px, py = _planet_pos_at(planet, av, step + 1, comet_ids, raw_comets)
            if ox is None or px is None:
                continue
            if ox == px and oy == py:
                continue
            if p2seg((nx, ny), (ox, oy), (px, py)) < planet[4]:
                return planet[0], "sweep", step + 1

        x, y = nx, ny

    return None, "max_steps", max_steps


def _solve_engine_angle(source, target, ships, planets, av, comet_ids, raw_comets,
                        max_steps=180, eta_hint=None, window=6):
    speed = fleet_speed(ships)
    checked_angles = set()
    checked_steps = set()
    steps = []

    if eta_hint is not None:
        center = max(0, int(eta_hint))
        for delta in range(window + 1):
            for s in (center - delta, center + delta):
                if 0 <= s < max_steps and s not in checked_steps:
                    checked_steps.add(s)
                    steps.append(s)

    for step in range(max_steps):
        if step in checked_steps:
            continue
        plausible = False
        for extra in (step, step + 1):
            px, py = _planet_pos_at(target, av, extra, comet_ids, raw_comets)
            if px is None:
                continue
            angle = math.atan2(py - source[3], px - source[2])
            sx = source[2] + math.cos(angle) * (source[4] + LAUNCH_RADIUS_OFFSET)
            sy = source[3] + math.sin(angle) * (source[4] + LAUNCH_RADIUS_OFFSET)
            travel_eta = math.hypot(px - sx, py - sy) / speed
            if abs(travel_eta - (step + 1)) <= 2.5:
                plausible = True
                break
        if plausible:
            checked_steps.add(step)
            steps.append(step)

    for step in steps:
        for extra in (step, step + 1):
            px, py = _planet_pos_at(target, av, extra, comet_ids, raw_comets)
            if px is None:
                continue
            angle = math.atan2(py - source[3], px - source[2])
            key = round(angle, 10)
            if key in checked_angles:
                continue
            checked_angles.add(key)
            hit_id, _reason, hit_steps = _trajectory_first_hit(
                source, angle, ships, planets, av, comet_ids, raw_comets, max_steps
            )
            if hit_id == target[0]:
                return angle, px, py, hit_steps
    return None
