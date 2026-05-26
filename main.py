"""
Orbit Wars agent — v12a

Builds on v12 (engine-confirmed launch geometry) with:
  - Attack intercepts solved against the fleet size that will actually launch (no misses)
  - Owned comets can launch attacks; comet evacuation before expiry
  - Defensive reinforcements use engine simulation for accurate angles
  - Enemy attacks include one extra turn of production as a capture cushion

WEIGHT_DIM=22: 20 feature weights + alpha + beta.
"""

import math

CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
RESERVE = 0.05
BOARD_SIZE = 100.0
LAUNCH_RADIUS_OFFSET = 0.1
COMET_EVAC_BASE_SCORE = 4.0
ENEMY_PROD_BUFFER_TURNS = 1

WEIGHT_DIM = 22

DEFAULT_WEIGHTS = [
    2.0,   # [0]  prod_dist_ratio   — primary targeting signal
    0.0,   # [1]  dist_decay
    0.3,   # [2]  eta_decay         — mild preference for nearby targets
    1.2,   # [3]  production_norm   — strongly reward high-production targets
    0.0,   # [4]  neg_ships_cost    — top agents drain planets fully
    1.5,   # [5]  is_neutral        — strongly prefer uncaptured planets early
    0.5,   # [6]  is_enemy          — mildly positive; willing to attack enemies
    0.0,   # [7]  is_comet          — never targeted
    0.0,   # [8]  enemy_threat_src
    -1.0,  # [9]  enemy_race_tgt    — avoid targets where enemy arrives first
    0.0,   # [10] turns_remaining
    0.0,   # [11] my_ship_fraction
    0.0,   # [12] my_planets_frac
    1.2,   # [13] enemy_prod_norm   — attack high-production enemy planets
    1.0,   # [14] winning_x_enemy   — press attack when ahead
    0.8,   # [15] prod_uncontested  — prefer uncontested high-prod targets
    0.5,   # [16] src_ship_surplus  — prefer draining surplus-heavy planets
    1.5,   # [17] relative_prod_rank
    0.4,   # [18] future_prod_value
    0.0,   # [19] enemy_near_tgt
    1.0,   # [20] alpha             — garrison multiplier
    0.0,   # [21] beta
]


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def fleet_speed(ships):
    return min(MAX_SPEED, 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000)) ** 1.5)


def p2seg(p, v, w):
    """Minimum distance from point p to line segment v-w."""
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


# ---------------------------------------------------------------------------
# Fleet intelligence helpers
# ---------------------------------------------------------------------------

def _fleet_ships_targeting(raw_fleets, planet, player):
    px, py, r = planet[2], planet[3], planet[4]
    total = 0
    for f in raw_fleets:
        if f[1] == player or f[1] < 0:
            continue
        fx, fy, fa = f[2], f[3], f[4]
        dx, dy = math.cos(fa), math.sin(fa)
        cx, cy = fx - px, fy - py
        b = 2.0 * (cx * dx + cy * dy)
        c = cx * cx + cy * cy - r * r
        disc = b * b - 4.0 * c
        if disc >= 0 and (-b - math.sqrt(disc)) / 2.0 > 0:
            total += f[6]
    return total


def _enemy_arrives_first(raw_fleets, target, our_eta, player):
    px, py, r = target[2], target[3], target[4]
    for f in raw_fleets:
        if f[1] == player or f[1] < 0:
            continue
        fx, fy, fa = f[2], f[3], f[4]
        dx, dy = math.cos(fa), math.sin(fa)
        cx, cy = fx - px, fy - py
        b = 2.0 * (cx * dx + cy * dy)
        c = cx * cx + cy * cy - r * r
        disc = b * b - 4.0 * c
        if disc >= 0 and (-b - math.sqrt(disc)) / 2.0 > 0:
            enemy_eta = math.hypot(fx - px, fy - py) / fleet_speed(f[6])
            if enemy_eta < our_eta:
                return 1.0
    return 0.0


def _friendly_ships_en_route(raw_fleets, player, planets, av, comet_ids, raw_comets):
    en_route = {}
    for f in raw_fleets:
        if f[1] != player:
            continue
        fx, fy, fa, fships = f[2], f[3], f[4], f[6]
        best_pid, best_diff = None, 0.30
        for p in planets:
            tx, ty, _ = intercept(fx, fy, p, av, fships, comet_ids, raw_comets)
            if tx is None:
                continue
            predicted_angle = math.atan2(ty - fy, tx - fx)
            diff = abs((predicted_angle - fa + math.pi) % (2 * math.pi) - math.pi)
            if diff < best_diff:
                best_diff = diff
                best_pid = p[0]
        if best_pid is not None:
            en_route[best_pid] = en_route.get(best_pid, 0) + fships
    return en_route


def _enemy_threats(raw_fleets, player, planets, av, comet_ids, raw_comets):
    threat = {}
    for f in raw_fleets:
        if f[1] == player or f[1] < 0:
            continue
        fx, fy, fa, fships = f[2], f[3], f[4], f[6]
        best_pid, best_diff, best_eta = None, 0.25, None
        for p in planets:
            tx, ty, eta = intercept(fx, fy, p, av, fships, comet_ids, raw_comets)
            if tx is None:
                continue
            predicted_angle = math.atan2(ty - fy, tx - fx)
            diff = abs((predicted_angle - fa + math.pi) % (2 * math.pi) - math.pi)
            if diff < best_diff:
                best_diff = diff
                best_pid = p[0]
                best_eta = eta
        if best_pid is not None:
            if best_pid not in threat:
                threat[best_pid] = [0, float('inf')]
            threat[best_pid][0] += fships
            threat[best_pid][1] = min(threat[best_pid][1], best_eta)
    return threat


# ---------------------------------------------------------------------------
# Feature extraction (20 features, indices 0-19)
# ---------------------------------------------------------------------------

def _extract_features(mine, t, eta, arrival_garrison, dist,
                      available, ships_score, comet_ids, player,
                      my_planets, all_planets, raw_fleets, step,
                      my_total_ships, all_total_ships, enemy_targets):
    f = [0.0] * 20

    f[0] = t[6] / (dist + 1.0)
    f[1] = math.exp(-dist / 30.0)
    f[2] = math.exp(-eta / 20.0) if eta > 0 else 1.0
    f[3] = t[6] / 8.0
    f[4] = 1.0 - min(ships_score / max(available, 1), 1.0)
    f[5] = 1.0 if t[1] == -1 else 0.0
    f[6] = 1.0 if (t[1] >= 0 and t[1] != player) else 0.0
    f[7] = 1.0 if t[0] in comet_ids else 0.0

    enemy_ships = _fleet_ships_targeting(raw_fleets, mine, player)
    f[8] = min(enemy_ships / max(available, 1), 1.0)
    f[9] = _enemy_arrives_first(raw_fleets, t, eta, player)

    f[10] = (500 - step) / 500.0
    f[11] = my_total_ships / max(all_total_ships, 1)
    f[12] = len(my_planets) / max(len(all_planets), 1)

    f[13] = f[6] * f[3]
    f[14] = (f[11] - 0.5) * f[6]
    f[15] = f[3] * (1.0 - f[9])

    my_ships = [p[5] for p in my_planets]
    mean_my = sum(my_ships) / max(len(my_ships), 1)
    max_my = max(my_ships) if my_ships else 1
    f[16] = (mine[5] - mean_my) / max(max_my, 1)

    prods = sorted(p[6] for p in enemy_targets)
    if prods:
        f[17] = sum(1 for p in prods if p <= t[6]) / len(prods)
    else:
        f[17] = 0.5

    f[18] = t[6] * max(0.0, 500.0 - step - eta) / 500.0

    enemy_owned = [p for p in all_planets if p[1] >= 0 and p[1] != player]
    if enemy_owned:
        near = sum(1 for ep in enemy_owned if math.hypot(ep[2] - t[2], ep[3] - t[3]) < 30)
        f[19] = near / len(enemy_owned)

    return f


# ---------------------------------------------------------------------------
# v12: engine-accurate trajectory solving
# ---------------------------------------------------------------------------

def _arrival_garrison(target, eta):
    if target[1] >= 0:
        return target[5] + target[6] * (eta + ENEMY_PROD_BUFFER_TURNS)
    return target[5]


def _available_ships(planet, comet_ids):
    if planet[0] in comet_ids:
        return int(planet[5])
    return int(planet[5] * (1 - RESERVE))


def _comet_turns_remaining(planet_id, raw_comets):
    path, path_idx = _comet_path_for(planet_id, raw_comets)
    if path is None:
        return 0
    return max(0, len(path) - path_idx - 1)


def _planet_pos_at(planet, av, extra_turns, comet_ids, raw_comets):
    if planet[0] in comet_ids:
        return comet_predict(planet[0], raw_comets, extra_turns)
    return predict_pos(planet, av, extra_turns)


def _trajectory_first_hit(source, angle, ships, planets, av, comet_ids, raw_comets, max_steps=180):
    """Replay the engine's movement checks and return the first body hit."""
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

        # Engine moves planets then checks if planet sweep crosses fleet position
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
    """Find an angle that the engine simulation confirms hits the target."""
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


def _rough_attack(mine, target, available, av, comet_ids, raw_comets, friendly_en_route):
    """Cheap fixed-point estimate used to choose which attacks deserve exact solve."""
    if available <= 0:
        return None

    already_covered = friendly_en_route.get(target[0], 0)
    tx, ty, eta = intercept(mine[2], mine[3], target, av, available, comet_ids, raw_comets)
    if tx is None:
        return None

    arrival_garrison = _arrival_garrison(target, eta)
    ships_score = arrival_garrison + 1 - already_covered
    if ships_score <= 0 or ships_score > available:
        return None

    for _ in range(8):
        tx, ty, eta = intercept(mine[2], mine[3], target, av, ships_score, comet_ids, raw_comets)
        if tx is None:
            return None
        arrival_garrison = _arrival_garrison(target, eta)
        required = arrival_garrison + 1 - already_covered
        if required <= 0 or required > available:
            return None
        if required <= ships_score:
            break
        ships_score = required
    else:
        return None

    dist = math.hypot(tx - mine[2], ty - mine[3])
    return tx, ty, eta, arrival_garrison, ships_score, dist


def _solve_attack(mine, target, ships_score, available, planets, av, comet_ids, raw_comets,
                  friendly_en_route, eta_hint):
    already_covered = friendly_en_route.get(target[0], 0)
    eta = eta_hint
    for _ in range(4):
        solved = _solve_engine_angle(mine, target, ships_score, planets, av, comet_ids, raw_comets, eta_hint=eta)
        if solved is None:
            return None
        angle, tx, ty, eta = solved
        arrival_garrison = _arrival_garrison(target, eta)
        required = arrival_garrison + 1 - already_covered
        if required <= 0 or required > available:
            return None
        if required <= ships_score:
            break
        ships_score = required
    else:
        return None

    dist = math.hypot(tx - mine[2], ty - mine[3])
    return angle, tx, ty, eta, arrival_garrison, ships_score, dist


def _find_evac_target(comet, target_planets, all_planets, av, ships_to_send, comet_ids, raw_comets):
    best = None
    for target in target_planets:
        if target[0] == comet[0] or target[0] in comet_ids:
            continue
        tx, ty, eta = intercept(comet[2], comet[3], target, av, ships_to_send, comet_ids, raw_comets)
        if tx is None:
            continue
        solved = _solve_engine_angle(
            comet, target, ships_to_send, all_planets, av, comet_ids, raw_comets, eta_hint=eta
        )
        if solved is None:
            continue
        angle, tx, ty, eta = solved
        dist = math.hypot(tx - comet[2], ty - comet[3])
        item = (dist, target[0], target, angle, tx, ty, eta)
        if best is None or item < best:
            best = item
    return best


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def _make_agent(weights):
    w = list(map(float, weights))
    alpha = w[20]
    beta = w[21]

    def _agent(obs):
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
        av = obs.get("angular_velocity", 0.035) if isinstance(obs, dict) else obs.angular_velocity
        comet_ids = set(obs.get("comet_planet_ids", []) if isinstance(obs, dict) else obs.comet_planet_ids)
        raw_comets = obs.get("comets", []) if isinstance(obs, dict) else obs.comets
        step = int(obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0))

        planets = [list(p) for p in raw_planets]
        my_planets = [p for p in planets if p[1] == player]
        enemy_targets = [p for p in planets if p[1] != player]

        if not my_planets:
            return []

        my_total_ships = sum(p[5] for p in my_planets)
        all_total_ships = sum(p[5] for p in planets)

        friendly_en_route = _friendly_ships_en_route(raw_fleets, player, planets, av, comet_ids, raw_comets)
        enemy_threat = _enemy_threats(raw_fleets, player, planets, av, comet_ids, raw_comets)

        candidates = []

        # Attack candidates (comets excluded as targets; owned comets may be sources)
        for mine in my_planets:
            available = _available_ships(mine, comet_ids)
            if available <= (0 if mine[0] in comet_ids else 1):
                continue
            for target in enemy_targets:
                if target[0] in comet_ids:
                    continue

                rough = _rough_attack(mine, target, available, av, comet_ids, raw_comets, friendly_en_route)
                if rough is None:
                    continue
                tx, ty, eta, arrival_garrison, ships_score, dist = rough

                feats = _extract_features(
                    mine, target, eta, arrival_garrison, dist,
                    available, ships_score, comet_ids, player,
                    my_planets, planets, raw_fleets, step,
                    my_total_ships, all_total_ships,
                    enemy_targets,
                )
                score = sum(w[i] * feats[i] for i in range(20))
                if mine[0] in comet_ids:
                    score += 1.0
                ships_to_send = min(max(ships_score, int(ships_score * alpha + beta * target[6] * eta)), available)
                candidates.append((score, "atk", mine, target, None, ships_to_send, ships_score, eta, available, None))

        # Comet evacuation candidates
        for comet in my_planets:
            if comet[0] not in comet_ids:
                continue
            ships_to_send = int(comet[5])
            if ships_to_send <= 0:
                continue
            evac = _find_evac_target(comet, my_planets, planets, av, ships_to_send, comet_ids, raw_comets)
            evac_score = COMET_EVAC_BASE_SCORE
            if evac is None:
                evac = _find_evac_target(comet, planets, planets, av, ships_to_send, comet_ids, raw_comets)
                evac_score = COMET_EVAC_BASE_SCORE - 0.75
            if evac is None:
                continue
            dist, _target_id, target, angle, tx, ty, eta = evac
            remaining = _comet_turns_remaining(comet[0], raw_comets)
            urgency = max(0.0, (40.0 - remaining) / 10.0)
            score = evac_score + urgency + ships_to_send / 20.0
            candidates.append((score, "evac", comet, target, angle, ships_to_send, ships_to_send, eta, ships_to_send, None))

        # Defensive reinforce candidates
        for target in my_planets:
            threat = enemy_threat.get(target[0])
            if not threat:
                continue
            threat_ships, enemy_eta = threat[0], threat[1]
            garrison_at_arrival = target[5] + target[6] * int(enemy_eta)
            already_defending = friendly_en_route.get(target[0], 0)
            if garrison_at_arrival + already_defending >= threat_ships:
                continue
            ships_needed = max(1, threat_ships - garrison_at_arrival - already_defending + 1)
            for mine in my_planets:
                if mine[0] == target[0]:
                    continue
                available = _available_ships(mine, comet_ids)
                if available <= (0 if mine[0] in comet_ids else 1):
                    continue
                ships_to_send = min(ships_needed, available)
                tx_r, ty_r, eta_r = intercept(mine[2], mine[3], target, av, ships_to_send, comet_ids, raw_comets)
                if tx_r is None:
                    continue
                if eta_r > enemy_eta + 5:
                    continue
                dist = math.hypot(tx_r - mine[2], ty_r - mine[3])
                feats = _extract_features(
                    mine, target, eta_r, 0, dist,
                    available, ships_to_send, comet_ids, player,
                    my_planets, planets, raw_fleets, step,
                    my_total_ships, all_total_ships,
                    enemy_targets,
                )
                score = sum(w[i] * feats[i] for i in range(20))
                candidates.append((score, "def", mine, target, None, ships_to_send, ships_needed, eta_r, available, enemy_eta + 5))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        used_src = set()
        used_atk_tgt = set()
        defense_covered = {}
        moves = []
        for score, kind, source, target, angle, ships_to_send, ships_needed, eta_hint, available, max_eta in candidates:
            src_id = source[0]
            tgt_id = target[0]
            if src_id in used_src:
                continue

            if kind == "atk":
                if tgt_id in used_atk_tgt:
                    continue
                solved = _solve_attack(
                    source, target, ships_needed, available, planets, av, comet_ids, raw_comets,
                    friendly_en_route, eta_hint,
                )
                if solved is None:
                    continue
                angle, tx, ty, eta, arrival_garrison, exact_need, dist = solved
                send = min(max(exact_need, int(exact_need * alpha + beta * target[6] * eta)), available)
                if send != exact_need:
                    exact = _solve_engine_angle(
                        source, target, send, planets, av, comet_ids, raw_comets, eta_hint=eta
                    )
                    if exact is None:
                        continue
                    angle, tx, ty, eta = exact
                    required = _arrival_garrison(target, eta) + 1
                    required -= friendly_en_route.get(tgt_id, 0)
                    if required > send:
                        continue
                moves.append([src_id, angle, send])
                used_src.add(src_id)
                used_atk_tgt.add(tgt_id)

            elif kind == "def":
                covered = defense_covered.get(tgt_id, 0)
                if covered >= ships_needed:
                    continue
                send = min(ships_to_send, ships_needed - covered)
                solved = _solve_engine_angle(source, target, send, planets, av, comet_ids, raw_comets, eta_hint=eta_hint)
                if solved is None:
                    continue
                angle, tx, ty, eta = solved
                if max_eta is not None and eta > max_eta:
                    continue
                moves.append([src_id, angle, send])
                used_src.add(src_id)
                defense_covered[tgt_id] = covered + send

            elif kind == "evac":
                if angle is None:
                    continue
                hit_id, _reason, _steps = _trajectory_first_hit(
                    source, angle, ships_to_send, planets, av, comet_ids, raw_comets
                )
                if hit_id != tgt_id:
                    continue
                moves.append([src_id, angle, ships_to_send])
                used_src.add(src_id)

        return moves

    return _agent


agent = _make_agent(DEFAULT_WEIGHTS)

# v12a fallback binding intentionally replaced by v17 when the packaged
# candidate can register its embedded modules.



# ============================================================================
# v17 packaged candidate: MLP ranker + diagnostic pressure override gate
# Built by scripts/build_v17_main.py.
# ============================================================================

import base64 as _v17_b64
import io as _v17_io
import numpy as _v17_np
import sys as _v17_sys
import types as _v17_types

_v17_current_mod = _v17_sys.modules.get(__name__)
if _v17_current_mod is not None and __name__ != "builtins":
    _v17_sys.modules["main"] = _v17_current_mod
else:
    _v17_main_mod = _v17_types.ModuleType("main")
    _v17_main_mod.__dict__.update(globals())
    _v17_sys.modules["main"] = _v17_main_mod
try:
    import agents as _agents_pkg
except Exception:
    _agents_pkg = _v17_types.ModuleType("agents")
    _agents_pkg.__path__ = []
    _v17_sys.modules["agents"] = _agents_pkg

_V17_MODULE_SOURCES = [
    ('agents.kaggle_ender_v49', '"""Port of example_notebooks/orbit-wars-heuristic-agent-scored-1000.py."""\n"""\n\nOrbit Wars -Enders FleetScocred >1000 on leaderboard 5/2026"\n"""\n\nimport os\n\nos.environ[\'KAGGLE_ENVELOPES\'] = \'0\'\n\n\n\nimport math\n\n\n\nSUN_X, SUN_Y = 50.0, 50.0\n\nSUN_RADIUS = 10.0\n\nMAX_SPEED = 6.0\n\nDECOY_THRESHOLD = 8\n\n\n\n\n\ndef fleet_speed(ships: int) -> float:\n\n    if ships <= 0:\n\n        return 1.0\n\n    return 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000)) ** 1.5\n\n\n\n\n\ndef travel_time(x1: float, y1: float, x2: float, y2: float, ships: int) -> float:\n\n    dist = math.hypot(x2 - x1, y2 - y1)\n\n    return dist / fleet_speed(ships) if ships > 0 else 999.0\n\n\n\n\n\ndef line_seg_min_dist(x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> float:\n\n    dx, dy = x2 - x1, y2 - y1\n\n    len_sq = dx * dx + dy * dy\n\n    if len_sq == 0:\n\n        return math.hypot(x1 - px, y1 - py)\n\n    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))\n\n    return math.hypot(x1 + t * dx - px, y1 + t * dy - py)\n\n\n\n\n\ndef path_crosses_sun(x1: float, y1: float, x2: float, y2: float, margin: float = 1.5) -> bool:\n\n    return line_seg_min_dist(x1, y1, x2, y2, SUN_X, SUN_Y) < SUN_RADIUS + margin\n\n\n\n\n\ndef predict_orbit(x: float, y: float, omega: float, dt: float):\n\n    theta = math.atan2(y - SUN_Y, x - SUN_X)\n\n    r = math.hypot(x - SUN_X, y - SUN_Y)\n\n    return SUN_X + r * math.cos(theta + omega * dt), SUN_Y + r * math.sin(theta + omega * dt)\n\n\n\n\n\ndef solve_intercept(fx: float, fy: float, tx: float, ty: float, orbiting: bool, omega: float, ships: int, iterations: int = 25):\n\n    if not orbiting:\n\n        t = travel_time(fx, fy, tx, ty, ships)\n\n        return tx, ty, t\n\n    theta = math.atan2(ty - SUN_Y, tx - SUN_X)\n\n    r = math.hypot(tx - SUN_X, ty - SUN_Y)\n\n    t = travel_time(fx, fy, tx, ty, ships)\n\n    ix, iy = tx, ty\n\n    for _ in range(iterations):\n\n        ix, iy = predict_orbit(tx, ty, omega, t)\n\n        t2 = travel_time(fx, fy, ix, iy, ships)\n\n        if abs(t2 - t) < 0.05:\n\n            break\n\n        t = t2\n\n    return ix, iy, t\n\n\n\n\n\ndef safe_angle(x1: float, y1: float, x2: float, y2: float) -> float:\n\n    direct = math.atan2(y2 - y1, x2 - x1)\n\n    if not path_crosses_sun(x1, y1, x2, y2, margin=1.5):\n\n        return direct\n\n    d = math.hypot(x1 - SUN_X, y1 - SUN_Y)\n\n    if d <= SUN_RADIUS + 1.0:\n\n        return direct\n\n    half = math.asin(min(1.0, (SUN_RADIUS + 1.0) / d))\n\n    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)\n\n    cw = to_sun + half\n\n    ccw = to_sun - half\n\n    def adiff(a):\n\n        dd = (a - direct) % (2 * math.pi)\n\n        return min(dd, 2 * math.pi - dd)\n\n    return cw if adiff(cw) < adiff(ccw) else ccw\n\n\n\n\n\ndef is_decoy_fleet(fleet, planets, omega):\n\n    if fleet[\'ships\'] < DECOY_THRESHOLD:\n\n        return True\n\n    tgt_id = None\n\n    best_dist = float(\'inf\')\n\n    for p in planets.values():\n\n        d = math.hypot(fleet[\'x\'] - p[\'x\'], fleet[\'y\'] - p[\'y\'])\n\n        if d < best_dist:\n\n            best_dist = d\n\n            tgt_id = p[\'id\']\n\n    if tgt_id is None:\n\n        return True\n\n    tgt = planets.get(tgt_id)\n\n    if tgt is None:\n\n        return True\n\n    r = math.hypot(tgt[\'x\'] - SUN_X, tgt[\'y\'] - SUN_Y)\n\n    is_orb = (r + tgt[\'radius\']) < 48.0\n\n    ships_needed = tgt[\'ships\'] + 1\n\n    if fleet[\'ships\'] < ships_needed * 0.4:\n\n        return True\n\n    return False\n\n\n\n\n\ndef ships_needed_for_takeover(tgt_ships, tgt_prod, tt, owner, margin=1.05):\n\n    if owner == -1:\n\n        return int(tgt_ships * margin) + 1\n\n    growth = tgt_prod * tt\n\n    return int((tgt_ships + growth) * margin) + 1\n\n\n\n\n\ndef planet_under_threat(p_id, fleets, planets, player, omega):\n\n    incoming = 0\n\n    for f in fleets.values():\n\n        if f[\'owner\'] == player:\n\n            continue\n\n        best_tgt, best_d = None, float(\'inf\')\n\n        for p in planets.values():\n\n            if p[\'id\'] == f[\'from\']:\n\n                continue\n\n            d = math.hypot(f[\'x\'] - p[\'x\'], f[\'y\'] - p[\'y\'])\n\n            if d < best_d:\n\n                best_d = d\n\n                best_tgt = p[\'id\']\n\n        if best_tgt == p_id:\n\n            r = math.hypot(planets[p_id][\'x\'] - SUN_X, planets[p_id][\'y\'] - SUN_Y)\n\n            is_orbiting = (r + planets[p_id][\'radius\']) < 48.0\n\n            if is_orbiting:\n\n                ix, iy = predict_orbit(planets[p_id][\'x\'], planets[p_id][\'y\'], omega, travel_time(f[\'x\'], f[\'y\'], planets[p_id][\'x\'], planets[p_id][\'y\'], int(f[\'ships\'])))\n\n                d = math.hypot(ix - planets[p_id][\'x\'], iy - planets[p_id][\'y\'])\n\n            else:\n\n                d = math.hypot(f[\'x\'] - planets[p_id][\'x\'], f[\'y\'] - planets[p_id][\'y\'])\n\n            if d < 50:\n\n                incoming += f[\'ships\']\n\n    return incoming\n\n\n\n\n\n# =============================================================================\n\n# MULTI-LEG PATH PLANNER (minimal - just for hard targets)\n\n# =============================================================================\n\n\n\ndef compute_tangent_points(x1: float, y1: float, margin: float = 2.0):\n\n    d = math.hypot(x1 - SUN_X, y1 - SUN_Y)\n\n    if d <= SUN_RADIUS + margin:\n\n        return None, None\n\n    half_angle = math.asin(min(1.0, (SUN_RADIUS + margin) / d))\n\n    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)\n\n    return to_sun + half_angle, to_sun - half_angle\n\n\n\n\n\ndef multi_leg_path(x1: float, y1: float, x2: float, y2: float, margin: float = 2.0):\n\n    """Only use multi-leg for targets whose direct path crosses sun."""\n\n    if not path_crosses_sun(x1, y1, x2, y2, margin):\n\n        return [(x2, y2)], math.hypot(x2 - x1, y2 - y1)\n\n    \n\n    # Try beacon points\n\n    beacon_ring = SUN_RADIUS + 15.0\n\n    waypoints = []\n\n    for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:\n\n        bx = SUN_X + beacon_ring * math.cos(angle)\n\n        by = SUN_Y + beacon_ring * math.sin(angle)\n\n        if not path_crosses_sun(x1, y1, bx, by, margin) and not path_crosses_sun(bx, by, x2, y2, margin):\n\n            waypoints.append((bx, by))\n\n    \n\n    if not waypoints:\n\n        return None, float(\'inf\')\n\n    \n\n    # Find shortest\n\n    best_wp = None\n\n    best_dist = float(\'inf\')\n\n    for wx, wy in waypoints:\n\n        d = math.hypot(wx - x1, wy - y1) + math.hypot(x2 - wx, y2 - wy)\n\n        if d < best_dist:\n\n            best_dist = d\n\n            best_wp = (wx, wy)\n\n    \n\n    if best_wp:\n\n        return [best_wp, (x2, y2)], best_dist\n\n    \n\n    return None, float(\'inf\')\n\n\n\n\n\n# =============================================================================\n\n# CAPTURE WINDOW ESTIMATION (simplified - just scoring bonus)\n\n# =============================================================================\n\n\n\ndef estimate_capture_bonus(src_x: float, src_y: float, planet, omega: float, ships: int) -> float:\n\n    """Return a bonus score for targets with wide capture windows."""\n\n    r = math.hypot(planet[\'x\'] - SUN_X, planet[\'y\'] - SUN_Y)\n\n    if (r + planet[\'radius\']) >= 48.0:\n\n        return 0.0  # Not orbiting, no penalty\n\n    \n\n    # Simple check: see if direct path doesn\'t cross sun\n\n    if not path_crosses_sun(src_x, src_y, planet[\'x\'], planet[\'y\'], margin=2.0):\n\n        return 3.0  # Easy intercept\n\n    \n\n    # Check a few future positions\n\n    safe_count = 0\n\n    for offset in range(-6, 7):\n\n        fx, fy = predict_orbit(planet[\'x\'], planet[\'y\'], omega, offset)\n\n        if not path_crosses_sun(src_x, src_y, fx, fy, margin=2.0):\n\n            safe_count += 1\n\n    \n\n    # More safe positions = wider window = bonus\n\n    return (safe_count / 13.0) * 5.0  # 0 to 5 bonus\n\n\n\n\n\n# =============================================================================\n\n# MAIN AGENT - v48 core with minimal enhancements\n\n# =============================================================================\n\n\n\ndef agent(obs):\n\n    if isinstance(obs, dict):\n\n        player = obs.get(\'player\', 0)\n\n        planets_data = obs.get(\'planets\', [])\n\n        fleets_data = obs.get(\'fleets\', [])\n\n        step = obs.get(\'step\', 0)\n\n        omega = obs.get(\'angular_velocity\', 0.03)\n\n    else:\n\n        player = getattr(obs, \'player\', 0)\n\n        planets_data = getattr(obs, \'planets\', [])\n\n        fleets_data = getattr(obs, \'fleets\', [])\n\n        step = getattr(obs, \'step\', 0)\n\n        omega = getattr(obs, \'angular_velocity\', 0.03)\n\n\n\n    planets = {}\n\n    for p in planets_data:\n\n        pid, owner, x, y, radius, ships, prod = p[:7]\n\n        r = math.hypot(x - SUN_X, y - SUN_Y)\n\n        planets[pid] = {\n\n            \'id\': pid, \'owner\': owner, \'x\': x, \'y\': y,\n\n            \'radius\': radius, \'ships\': float(ships), \'prod\': float(prod),\n\n            \'is_orb\': (r + radius) < 48.0\n\n        }\n\n\n\n    fleets = {}\n\n    for f in fleets_data:\n\n        fleets[f[0]] = {\n\n            \'id\': f[0], \'owner\': f[1], \'x\': f[2], \'y\': f[3],\n\n            \'angle\': f[4], \'from\': f[5], \'ships\': float(f[6])\n\n        }\n\n\n\n    my = [p for p in planets.values() if p[\'owner\'] == player]\n\n    if not my:\n\n        return []\n\n\n\n    enemy = [p for p in planets.values() if p[\'owner\'] != player and p[\'owner\'] != -1]\n\n    neutrals = [p for p in planets.values() if p[\'owner\'] == -1]\n\n\n\n    my_prod = sum(p[\'prod\'] for p in my)\n\n    my_ships = sum(p[\'ships\'] for p in my)\n\n    enemy_prod = sum(p[\'prod\'] for p in enemy) if enemy else 0\n\n    enemy_ships = sum(p[\'ships\'] for p in enemy) if enemy else 0\n\n\n\n    prod_ratio = my_prod / enemy_prod if enemy_prod > 0 else 999\n\n    ship_ratio = my_ships / enemy_ships if enemy_ships > 0 else 999\n\n\n\n    my_planet_count = len(my)\n\n    neighbor_count = sum(1 for t in neutrals if any(math.hypot(t[\'x\'] - p[\'x\'], t[\'y\'] - p[\'y\']) < 35 for p in my))\n\n\n\n    nearby_larger_planets = []\n\n    for src in my:\n\n        for t in (neutrals + enemy):\n\n            d = math.hypot(t[\'x\'] - src[\'x\'], t[\'y\'] - src[\'y\'])\n\n            if d < 40 and t[\'prod\'] >= src[\'prod\'] * 0.8 and t[\'radius\'] >= src[\'radius\'] * 0.8:\n\n                nearby_larger_planets.append((src[\'id\'], t[\'id\'], d))\n\n\n\n    real_enemy_fleets = {f_id: f for f_id, f in fleets.items() if f[\'owner\'] != player and not is_decoy_fleet(f, planets, omega)}\n\n\n\n    in_flight_from = set()\n\n    in_flight_to = set()\n\n    for f in fleets.values():\n\n        if f[\'owner\'] == player and f[\'from\'] is not None:\n\n            in_flight_from.add(f[\'from\'])\n\n            best_tgt, best_d = None, float(\'inf\')\n\n            for p in planets.values():\n\n                if p[\'id\'] == f[\'from\']:\n\n                    continue\n\n                d = math.hypot(f[\'x\'] - p[\'x\'], f[\'y\'] - p[\'y\'])\n\n                if d < best_d:\n\n                    best_d = d\n\n                    best_tgt = p[\'id\']\n\n            if best_tgt:\n\n                in_flight_to.add(best_tgt)\n\n\n\n    threats = {}\n\n    for p in planets.values():\n\n        if p[\'owner\'] == player:\n\n            threats[p[\'id\']] = planet_under_threat(p[\'id\'], fleets, planets, player, omega)\n\n\n\n    smash_targets = set()\n\n    for e in enemy:\n\n        nearby_my_ships = sum(p[\'ships\'] for p in my if math.hypot(p[\'x\'] - e[\'x\'], p[\'y\'] - e[\'y\']) < 50)\n\n        if nearby_my_ships > e[\'ships\'] * 0.95:\n\n            smash_targets.add(e[\'id\'])\n\n\n\n    if smash_targets:\n\n        phase = \'smash\'\n\n    elif my_ships > 120 and my_planet_count < 4 and enemy:\n\n        phase = \'rush\'\n\n    elif my_planet_count < 3 or (neighbor_count > 0 and my_planet_count < 5):\n\n        phase = \'expand\'\n\n    elif threats and any(t > my_ships * 0.25 for t in threats.values()):\n\n        phase = \'counter_attack\'\n\n    elif prod_ratio > 4 and my_ships > 80 and my_planet_count >= 3:\n\n        phase = \'crush\'\n\n    elif prod_ratio > 2.0 or ship_ratio > 2.5:\n\n        phase = \'aggressive\'\n\n    elif my_prod < enemy_prod * 0.7:\n\n        phase = \'defend\'\n\n    elif len(enemy) > 0 and len(my) >= 3 and my_prod > enemy_prod * 1.0:\n\n        phase = \'dominate\'\n\n    else:\n\n        phase = \'grow\'\n\n\n\n    moves = []\n\n\n\n    targeted_this_turn = set()\n\n\n\n    for src in my:\n\n        if src[\'id\'] in in_flight_from:\n\n            continue\n\n\n\n        if src[\'ships\'] < 10:\n\n            continue\n\n\n\n        if phase == \'expand\':\n\n            nearby_larger = {nl[1] for nl in nearby_larger_planets if nl[0] == src[\'id\']}\n\n            best_target = None\n\n            best_score = -1e9\n\n            for t in neutrals:\n\n                if t[\'id\'] == src[\'id\']:\n\n                    continue\n\n                if t[\'id\'] in in_flight_to or t[\'id\'] in targeted_this_turn:\n\n                    continue\n\n                d = math.hypot(t[\'x\'] - src[\'x\'], t[\'y\'] - src[\'y\'])\n\n                score = -d * 3 + t[\'prod\'] * 3\n\n                if nearby_larger and t[\'radius\'] < src[\'radius\'] * 0.7 and d > 25:\n\n                    score -= 50\n\n                if score > best_score:\n\n                    best_score = score\n\n                    best_target = t\n\n            if best_target:\n\n                r = math.hypot(best_target[\'x\'] - SUN_X, best_target[\'y\'] - SUN_Y)\n\n                is_orbiting = (r + best_target[\'radius\']) < 48.0\n\n                ix, iy, tt = solve_intercept(src[\'x\'], src[\'y\'], best_target[\'x\'], best_target[\'y\'], is_orbiting, omega, int(src[\'ships\']))\n\n                if not path_crosses_sun(src[\'x\'], src[\'y\'], ix, iy, margin=1.5):\n\n                    send = ships_needed_for_takeover(best_target[\'ships\'], best_target[\'prod\'], tt, best_target[\'owner\'])\n\n                    if src[\'ships\'] >= send:\n\n                        angle = safe_angle(src[\'x\'], src[\'y\'], ix, iy)\n\n                        moves.append([src[\'id\'], angle, send])\n\n                        targeted_this_turn.add(best_target[\'id\'])\n\n                        src[\'ships\'] -= send\n\n                        if src[\'ships\'] < 5:\n\n                            break\n\n            elif src[\'ships\'] > 40:\n\n                decoy_tgt = None\n\n                decoy_score = -1e9\n\n                for t in (enemy + neutrals):\n\n                    if t[\'id\'] == src[\'id\']:\n\n                        continue\n\n                    if t[\'id\'] in targeted_this_turn:\n\n                        continue\n\n                    d = math.hypot(t[\'x\'] - src[\'x\'], t[\'y\'] - src[\'y\'])\n\n                    score = -d + (t[\'prod\'] if t[\'owner\'] != -1 else 0) * 5\n\n                    if nearby_larger and t[\'radius\'] < src[\'radius\'] * 0.7 and d > 25:\n\n                        score -= 50\n\n                    if score > decoy_score:\n\n                        decoy_score = score\n\n                        decoy_tgt = t\n\n                if decoy_tgt and src[\'ships\'] > 25:\n\n                    send = min(8, int(src[\'ships\'] * 0.15))\n\n                    if send >= 5:\n\n                        r = math.hypot(decoy_tgt[\'x\'] - SUN_X, decoy_tgt[\'y\'] - SUN_Y)\n\n                        is_orbiting = (r + decoy_tgt[\'radius\']) < 48.0\n\n                        ix, iy, tt = solve_intercept(src[\'x\'], src[\'y\'], decoy_tgt[\'x\'], decoy_tgt[\'y\'], is_orbiting, omega, int(src[\'ships\']))\n\n                        if not path_crosses_sun(src[\'x\'], src[\'y\'], ix, iy, margin=1.5):\n\n                            angle = safe_angle(src[\'x\'], src[\'y\'], ix, iy)\n\n                            moves.append([src[\'id\'], angle, send])\n\n                            targeted_this_turn.add(decoy_tgt[\'id\'])\n\n                            src[\'ships\'] -= send\n\n                            if src[\'ships\'] < 10:\n\n                                break\n\n\n\n        need_defense = threats.get(src[\'id\'], 0) > src[\'ships\'] * 0.3\n\n\n\n        if need_defense and phase != \'counter_attack\':\n\n            continue\n\n\n\n        if need_defense and phase == \'counter_attack\' and threats.get(src[\'id\'], 0) >= src[\'ships\'] * 0.5:\n\n            continue\n\n\n\n        if phase == \'counter_attack\':\n\n            best_enemy = None\n\n            best_score = -1e9\n\n            for t in enemy:\n\n                if t[\'id\'] in targeted_this_turn:\n\n                    continue\n\n                d = math.hypot(t[\'x\'] - src[\'x\'], t[\'y\'] - src[\'y\'])\n\n                score = t[\'ships\'] * 0.8 + t[\'prod\'] * 8 - d\n\n                if t[\'id\'] in smash_targets:\n\n                    score += 50\n\n                if score > best_score:\n\n                    best_score = score\n\n                    best_enemy = t\n\n            if best_enemy:\n\n                r = math.hypot(best_enemy[\'x\'] - SUN_X, best_enemy[\'y\'] - SUN_Y)\n\n                is_orbiting = (r + best_enemy[\'radius\']) < 48.0\n\n                ix, iy, tt = solve_intercept(src[\'x\'], src[\'y\'], best_enemy[\'x\'], best_enemy[\'y\'], is_orbiting, omega, int(src[\'ships\']))\n\n                if not path_crosses_sun(src[\'x\'], src[\'y\'], ix, iy, margin=1.5):\n\n                    send = int(src[\'ships\'] * 0.8)\n\n                    send = max(send, int(best_enemy[\'ships\'] * 1.1))\n\n                    send = min(send, int(src[\'ships\'] * 0.95))\n\n                    if src[\'ships\'] > send + 3:\n\n                        angle = safe_angle(src[\'x\'], src[\'y\'], ix, iy)\n\n                        moves.append([src[\'id\'], angle, send])\n\n                        targeted_this_turn.add(best_enemy[\'id\'])\n\n                        src[\'ships\'] -= send\n\n\n\n        best_tgt = None\n\n        best_score = -1e9\n\n\n\n        if phase == \'smash\':\n\n            candidates = [t for t in enemy if t[\'id\'] in smash_targets]\n\n        elif phase == \'rush\':\n\n            candidates = enemy\n\n        elif phase == \'expand\' or phase == \'opportunistic\' or phase == \'aggressive\' or phase == \'dominate\':\n\n            candidates = neutrals if phase not in (\'aggressive\', \'dominate\') else (enemy + neutrals)\n\n        elif phase == \'grow\':\n\n            candidates = [t for t in neutrals if threats.get(t[\'id\'], 0) == 0]\n\n        else:\n\n            candidates = []\n\n\n\n        for t in candidates:\n\n            if t[\'id\'] == src[\'id\']:\n\n                continue\n\n            if t[\'id\'] in in_flight_to:\n\n                continue\n\n            if t[\'id\'] in targeted_this_turn:\n\n                continue\n\n\n\n            incoming = threats.get(t[\'id\'], 0)\n\n            if incoming > 0:\n\n                continue\n\n\n\n            r = math.hypot(t[\'x\'] - SUN_X, t[\'y\'] - SUN_Y)\n\n            is_orbiting = t[\'is_orb\']\n\n\n\n            ix, iy, tt = solve_intercept(src[\'x\'], src[\'y\'], t[\'x\'], t[\'y\'], is_orbiting, omega, int(src[\'ships\']))\n\n\n\n            if path_crosses_sun(src[\'x\'], src[\'y\'], ix, iy, margin=1.5):\n\n                # Try multi-leg path for this target\n\n                waypoints, _ = multi_leg_path(src[\'x\'], src[\'y\'], ix, iy)\n\n                if waypoints is None:\n\n                    continue\n\n                # Use multi-leg\n\n                final_x, final_y = waypoints[-1]\n\n                if path_crosses_sun(src[\'x\'], src[\'y\'], final_x, final_y, margin=1.5):\n\n                    continue\n\n\n\n            if is_orbiting:\n\n                planet_future = predict_orbit(t[\'x\'], t[\'y\'], omega, tt)\n\n                to_planet = math.atan2(planet_future[1] - src[\'y\'], planet_future[0] - src[\'x\'])\n\n                to_target = math.atan2(t[\'y\'] - src[\'y\'], t[\'x\'] - src[\'x\'])\n\n                diff = abs((to_planet - to_target) % (2 * math.pi))\n\n                if diff > 0.5 and diff < (2 * math.pi - 0.5):\n\n                    continue\n\n\n\n            score = t[\'prod\'] * 18 - tt * 2.5\n\n\n\n            if t[\'owner\'] == -1:\n\n                score += 25\n\n\n\n            if phase == \'aggressive\' and t[\'owner\'] != -1:\n\n                score += 35 - t[\'ships\'] * 0.12\n\n\n\n            if phase == \'dominate\' and t[\'owner\'] != -1:\n\n                score += 45 - t[\'ships\'] * 0.08\n\n\n\n            if phase == \'dominate\' and t[\'owner\'] == -1:\n\n                score += 20\n\n\n\n            if is_orbiting:\n\n                score -= 6\n\n\n\n            if src[\'ships\'] > 50 and t[\'owner\'] == -1:\n\n                score += 12\n\n\n\n            if src[\'prod\'] > t[\'prod\'] * 0.7:\n\n                score += 8\n\n\n\n            # ADD CAPTURE WINDOW BONUS\n\n            score += estimate_capture_bonus(src[\'x\'], src[\'y\'], t, omega, int(src[\'ships\']))\n\n\n\n            if score > best_score:\n\n                best_score = score\n\n                best_tgt = (t, ix, iy, tt)\n\n\n\n        if best_tgt is None:\n\n            continue\n\n\n\n        tgt, ix, iy, tt = best_tgt\n\n\n\n        if phase == \'smash\':\n\n            send = int(src[\'ships\'] * 0.9)\n\n            send = max(send, ships_needed_for_takeover(tgt[\'ships\'], tgt[\'prod\'], tt, tgt[\'owner\']))\n\n        elif phase == \'rush\':\n\n            send = int(src[\'ships\'] * 0.8)\n\n        elif phase == \'aggressive\':\n\n            send = int(src[\'ships\'] * 0.4)\n\n            send = max(send, ships_needed_for_takeover(tgt[\'ships\'], tgt[\'prod\'], tt, tgt[\'owner\']))\n\n            send = min(send, int(src[\'ships\'] * 0.7))\n\n        elif phase == \'dominate\':\n\n            send = int(src[\'ships\'] * 0.5)\n\n            send = max(send, ships_needed_for_takeover(tgt[\'ships\'], tgt[\'prod\'], tt, tgt[\'owner\']))\n\n            send = min(send, int(src[\'ships\'] * 0.8))\n\n        elif phase == \'opportunistic\':\n\n            send = ships_needed_for_takeover(tgt[\'ships\'], tgt[\'prod\'], tt, tgt[\'owner\'])\n\n            send = min(send, int(src[\'ships\'] * 0.5))\n\n        else:\n\n            send = ships_needed_for_takeover(tgt[\'ships\'], tgt[\'prod\'], tt, tgt[\'owner\'])\n\n\n\n        if src[\'ships\'] < send:\n\n            continue\n\n\n\n        angle = safe_angle(src[\'x\'], src[\'y\'], ix, iy)\n\n        moves.append([src[\'id\'], angle, send])\n\n        targeted_this_turn.add(tgt[\'id\'])\n\n\n\n    if phase == \'expand\':\n\n        for src in my:\n\n            if src[\'id\'] in in_flight_from:\n\n                continue\n\n            if src[\'ships\'] < 10:\n\n                continue\n\n            nearby_larger = [nl for nl in nearby_larger_planets if nl[0] == src[\'id\']]\n\n            if not nearby_larger:\n\n                continue\n\n            candidates = [t for t in (neutrals + enemy)\n\n                          if t[\'id\'] not in targeted_this_turn\n\n                          and t[\'id\'] not in in_flight_to\n\n                          and t[\'owner\'] != player]\n\n            if not candidates:\n\n                continue\n\n            best_tgt = None\n\n            best_score = -1e9\n\n            for t in candidates:\n\n                d = math.hypot(t[\'x\'] - src[\'x\'], t[\'y\'] - src[\'y\'])\n\n                if d > 40:\n\n                    continue\n\n                score = t[\'prod\'] * 5 - d\n\n                if t[\'radius\'] >= src[\'radius\'] * 0.8 and t[\'prod\'] >= src[\'prod\'] * 0.8:\n\n                    score += 40\n\n                if score > best_score:\n\n                    best_score = score\n\n                    best_tgt = t\n\n            if best_tgt:\n\n                r = math.hypot(best_tgt[\'x\'] - SUN_X, best_tgt[\'y\'] - SUN_Y)\n\n                is_orbiting = (r + best_tgt[\'radius\']) < 48.0\n\n                ix, iy, tt = solve_intercept(src[\'x\'], src[\'y\'], best_tgt[\'x\'], best_tgt[\'y\'], is_orbiting, omega, int(src[\'ships\']))\n\n                if not path_crosses_sun(src[\'x\'], src[\'y\'], ix, iy, margin=1.5):\n\n                    send = ships_needed_for_takeover(best_tgt[\'ships\'], best_tgt[\'prod\'], tt, best_tgt[\'owner\'])\n\n                    if src[\'ships\'] >= send:\n\n                        angle = safe_angle(src[\'x\'], src[\'y\'], ix, iy)\n\n                        moves.append([src[\'id\'], angle, send])\n\n                        targeted_this_turn.add(best_tgt[\'id\'])\n\n                        src[\'ships\'] -= send\n\n\n\n    return moves\n\n\n\n\n\nif __name__ == \'__main__\':\n\n    print("v49c Minimal Strategic Enhancement loaded!")\n\n# %% [code]\n\n'),
    ('agents.scored_agent', '"""\nPhase 3: Parameterised scoring agent.\n\nInterface for Phase 4 CMA-ES optimizer:\n    from agents.scored_agent import make_agent, DEFAULT_WEIGHTS, WEIGHT_DIM, WEIGHT_NAMES\n    agent_fn = make_agent(weights)   # weights: list or numpy array of length WEIGHT_DIM\n\nWeight vector layout (WEIGHT_DIM = 22):\n    [0]  prod_dist_ratio      — production / (dist+1); primary v4 signal\n    [1]  dist_decay           — exp(-dist/30); smooth distance penalty\n    [2]  eta_decay            — exp(-eta/20); smooth ETA penalty\n    [3]  production_norm      — production / 8\n    [4]  neg_ships_cost       — 1 - ships_score/available; prefer cheap captures\n    [5]  is_neutral           — 1 if target is neutral\n    [6]  is_enemy             — 1 if target is enemy-owned\n    [7]  is_comet             — 1 if target is a comet\n    [8]  enemy_threat_src     — enemy fleet ships inbound to source / available\n    [9]  enemy_race_tgt       — 1 if any enemy fleet reaches target before us\n    [10] turns_remaining      — (500 - step) / 500\n    [11] my_ship_fraction     — our planet ships / all planet ships\n    [12] my_planets_frac      — our planet count / total planet count\n    [13] enemy_prod_norm      — is_enemy * production_norm (interaction: enemy × prod)\n    [14] winning_x_enemy      — (my_ship_fraction - 0.5) * is_enemy (attack enemies when winning)\n    [15] prod_uncontested     — production_norm * (1 - enemy_race_tgt) (high-prod + uncontested)\n    [16] src_ship_surplus     — (src_ships - mean_my_ships) / max_my_ships; drain surplus planets\n    [17] relative_prod_rank   — normalised rank of target production among non-owned planets\n    [18] future_prod_value    — production * max(0, turns_remaining - eta) / 500; lifetime value of capture\n    [19] enemy_near_tgt       — fraction of enemy planets within dist 30 of target\n    [20] alpha                — garrison multiplier for ship allocation\n    [21] beta                 — production*eta multiplier for ship allocation\n\nDEFAULT_WEIGHTS encode the strategy observed in top-agent replays:\n  - Aggressively expand to high-production planets early (prod_dist_ratio, production_norm, is_neutral)\n  - Switch to attacking enemies once ahead (enemy_prod_norm, winning_x_enemy)\n  - Drain source planets fully rather than holding reserve (RESERVE=0.05)\n  - Send minimal fleet to capture (alpha=0.85) — avoid wasting ships on overkill\n  - Avoid races where enemy arrives first (enemy_race_tgt negative)\n"""\n\nimport math\nimport sys\nimport os\n\nsys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))\n\nfrom main import fleet_speed, sun_blocked, intercept\n\nWEIGHT_DIM = 22\n\nWEIGHT_NAMES = [\n    "prod_dist_ratio",\n    "dist_decay",\n    "eta_decay",\n    "production_norm",\n    "neg_ships_cost",\n    "is_neutral",\n    "is_enemy",\n    "is_comet",\n    "enemy_threat_src",\n    "enemy_race_tgt",\n    "turns_remaining",\n    "my_ship_fraction",\n    "my_planets_frac",\n    "enemy_prod_norm",\n    "winning_x_enemy",\n    "prod_uncontested",\n    "src_ship_surplus",\n    "relative_prod_rank",\n    "future_prod_value",\n    "enemy_near_tgt",\n    "alpha",\n    "beta",\n]\n\nDEFAULT_WEIGHTS = [\n    2.0,   # [0]  prod_dist_ratio   — doubled from v4; primary targeting signal\n    0.0,   # [1]  dist_decay\n    0.3,   # [2]  eta_decay         — mild preference for nearby targets\n    1.2,   # [3]  production_norm   — strongly reward high-production targets\n    0.0,   # [4]  neg_ships_cost    — top agents drain planets fully; don\'t penalise cost\n    1.5,   # [5]  is_neutral        — strongly prefer uncaptured planets early\n    0.5,   # [6]  is_enemy          — mildly positive; willing to attack enemies\n    0.0,   # [7]  is_comet          — zero; comets never targeted in top-agent replays\n    0.0,   # [8]  enemy_threat_src\n    -1.0,  # [9]  enemy_race_tgt    — avoid targets where enemy arrives first\n    0.0,   # [10] turns_remaining\n    0.0,   # [11] my_ship_fraction  — removed "coast when winning" bias\n    0.0,   # [12] my_planets_frac\n    1.2,   # [13] enemy_prod_norm   — attack high-production enemy planets\n    1.0,   # [14] winning_x_enemy   — press attack on enemies when ahead\n    0.8,   # [15] prod_uncontested  — prefer uncontested high-prod targets\n    0.5,   # [16] src_ship_surplus  — prefer draining surplus-heavy planets\n    1.5,   # [17] relative_prod_rank — highest-production target among options\n    0.4,   # [18] future_prod_value — prod * turns_remaining_after_capture / 500\n    0.0,   # [19] enemy_near_tgt    — neutral; proximity to enemy does not penalise attack\n    1.0,   # [20] alpha             — send minimum required ships (1.0 = exact minimum)\n    0.0,   # [21] beta\n]\n\n# Top agents drain planets to 0-1 ships; 0.05 allows evolution to go lower if needed\nRESERVE = 0.05\n\n\n# ---------------------------------------------------------------------------\n# Private helpers (same ray-circle pattern as fleet_target_planets in main.py)\n# ---------------------------------------------------------------------------\n\ndef _fleet_ships_targeting(raw_fleets, planet, player):\n    """Sum of ships in enemy fleets whose ray intersects planet."""\n    px, py, r = planet[2], planet[3], planet[4]\n    total = 0\n    for f in raw_fleets:\n        if f[1] == player or f[1] < 0:\n            continue\n        fx, fy, fa = f[2], f[3], f[4]\n        dx, dy = math.cos(fa), math.sin(fa)\n        cx, cy = fx - px, fy - py\n        b = 2.0 * (cx * dx + cy * dy)\n        c = cx * cx + cy * cy - r * r\n        disc = b * b - 4.0 * c\n        if disc >= 0 and (-b - math.sqrt(disc)) / 2.0 > 0:\n            total += f[6]\n    return total\n\n\ndef _enemy_arrives_first(raw_fleets, target, our_eta, player):\n    """1.0 if any enemy fleet is heading toward target with estimated ETA < ours."""\n    px, py, r = target[2], target[3], target[4]\n    for f in raw_fleets:\n        if f[1] == player or f[1] < 0:\n            continue\n        fx, fy, fa = f[2], f[3], f[4]\n        dx, dy = math.cos(fa), math.sin(fa)\n        cx, cy = fx - px, fy - py\n        b = 2.0 * (cx * dx + cy * dy)\n        c = cx * cx + cy * cy - r * r\n        disc = b * b - 4.0 * c\n        if disc >= 0 and (-b - math.sqrt(disc)) / 2.0 > 0:\n            enemy_eta = math.hypot(fx - px, fy - py) / fleet_speed(f[6])\n            if enemy_eta < our_eta:\n                return 1.0\n    return 0.0\n\n\ndef _friendly_ships_en_route(raw_fleets, player, planets, av, comet_ids, raw_comets):\n    """Find which planet each friendly in-flight fleet is heading to via intercept matching.\n    Ray-circle intersection against current positions is wrong for orbiting planets."""\n    en_route = {}\n    for f in raw_fleets:\n        if f[1] != player:\n            continue\n        fx, fy, fa, fships = f[2], f[3], f[4], f[6]\n        best_pid, best_diff = None, 0.30  # ~17 deg; mid-flight drift: 5 turns * 0.035 av * r40 ≈ 0.27 rad\n        for p in planets:\n            tx, ty, _ = intercept(fx, fy, p, av, fships, comet_ids, raw_comets)\n            if tx is None:\n                continue\n            predicted_angle = math.atan2(ty - fy, tx - fx)\n            diff = abs((predicted_angle - fa + math.pi) % (2 * math.pi) - math.pi)\n            if diff < best_diff:\n                best_diff = diff\n                best_pid = p[0]\n        if best_pid is not None:\n            en_route[best_pid] = en_route.get(best_pid, 0) + fships\n    return en_route\n\n\ndef _enemy_threats(raw_fleets, player, planets, av, comet_ids, raw_comets):\n    """Find which planet each enemy fleet is heading to via intercept matching."""\n    threat = {}\n    for f in raw_fleets:\n        if f[1] == player or f[1] < 0:\n            continue\n        fx, fy, fa, fships = f[2], f[3], f[4], f[6]\n        best_pid, best_diff, best_eta = None, 0.25, None\n        for p in planets:\n            tx, ty, eta = intercept(fx, fy, p, av, fships, comet_ids, raw_comets)\n            if tx is None:\n                continue\n            predicted_angle = math.atan2(ty - fy, tx - fx)\n            diff = abs((predicted_angle - fa + math.pi) % (2 * math.pi) - math.pi)\n            if diff < best_diff:\n                best_diff = diff\n                best_pid = p[0]\n                best_eta = eta\n        if best_pid is not None:\n            if best_pid not in threat:\n                threat[best_pid] = [0, float(\'inf\')]\n            threat[best_pid][0] += fships\n            threat[best_pid][1] = min(threat[best_pid][1], best_eta)\n    return threat\n\n\n# ---------------------------------------------------------------------------\n# Feature extraction\n# ---------------------------------------------------------------------------\n\ndef extract_features(\n    mine, t, eta, arrival_garrison, dist,\n    available, ships_score, comet_ids, player,\n    my_planets, all_planets, raw_fleets, step,\n    my_total_ships, all_total_ships,\n    enemy_targets,\n):\n    """Return a list of 20 floats representing the (source, target) action."""\n    f = [0.0] * 20\n\n    f[0] = t[6] / (dist + 1.0)                               # prod_dist_ratio\n    f[1] = math.exp(-dist / 30.0)                            # dist_decay\n    f[2] = math.exp(-eta / 20.0) if eta > 0 else 1.0        # eta_decay\n    f[3] = t[6] / 8.0                                        # production_norm\n    f[4] = 1.0 - min(ships_score / max(available, 1), 1.0)  # neg_ships_cost\n    f[5] = 1.0 if t[1] == -1 else 0.0                       # is_neutral\n    f[6] = 1.0 if (t[1] >= 0 and t[1] != player) else 0.0  # is_enemy\n    f[7] = 1.0 if t[0] in comet_ids else 0.0                # is_comet\n\n    enemy_ships = _fleet_ships_targeting(raw_fleets, mine, player)\n    f[8] = min(enemy_ships / max(available, 1), 1.0)         # enemy_threat_src\n\n    f[9] = _enemy_arrives_first(raw_fleets, t, eta, player)  # enemy_race_tgt\n\n    f[10] = (500 - step) / 500.0                             # turns_remaining\n    f[11] = my_total_ships / max(all_total_ships, 1)         # my_ship_fraction\n    f[12] = len(my_planets) / max(len(all_planets), 1)       # my_planets_frac\n\n    # Interaction features\n    f[13] = f[6] * f[3]                   # enemy_prod_norm\n    f[14] = (f[11] - 0.5) * f[6]         # winning_x_enemy\n    f[15] = f[3] * (1.0 - f[9])          # prod_uncontested\n\n    # New features informed by top-agent replay analysis\n    my_ships = [p[5] for p in my_planets]\n    mean_my = sum(my_ships) / max(len(my_ships), 1)\n    max_my = max(my_ships) if my_ships else 1\n    f[16] = (mine[5] - mean_my) / max(max_my, 1)            # src_ship_surplus\n\n    # Production rank among all non-owned targets (0=lowest, 1=highest)\n    prods = sorted(p[6] for p in enemy_targets)\n    if prods:\n        rank = sum(1 for p in prods if p <= t[6])\n        f[17] = rank / len(prods)\n    else:\n        f[17] = 0.5                                          # relative_prod_rank\n\n    f[18] = t[6] * max(0.0, 500.0 - step - eta) / 500.0    # future_prod_value\n\n    # Fraction of enemy-owned planets within distance 30 of target\n    enemy_owned = [p for p in all_planets if p[1] >= 0 and p[1] != player]\n    if enemy_owned:\n        near = sum(1 for ep in enemy_owned if math.hypot(ep[2] - t[2], ep[3] - t[3]) < 30)\n        f[19] = near / len(enemy_owned)\n    else:\n        f[19] = 0.0                                          # enemy_near_tgt\n\n    return f\n\n\n# ---------------------------------------------------------------------------\n# Agent factory\n# ---------------------------------------------------------------------------\n\ndef make_agent(weights):\n    """\n    Return an agent(obs) callable parameterised by weights.\n    weights: list or numpy array of length WEIGHT_DIM.\n    """\n    w = list(map(float, weights))\n    if len(w) != WEIGHT_DIM:\n        raise ValueError(f"Expected {WEIGHT_DIM} weights, got {len(w)}")\n    alpha = w[20]\n    beta = w[21]\n\n    def agent(obs):\n        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player\n        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets\n        raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets\n        av = obs.get("angular_velocity", 0.035) if isinstance(obs, dict) else obs.angular_velocity\n        comet_ids = set(\n            obs.get("comet_planet_ids", []) if isinstance(obs, dict) else obs.comet_planet_ids\n        )\n        raw_comets = obs.get("comets", []) if isinstance(obs, dict) else obs.comets\n        step = int(obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0))\n\n        planets = [list(p) for p in raw_planets]\n        my_planets = [p for p in planets if p[1] == player]\n        enemy_targets = [p for p in planets if p[1] != player]\n\n        if not enemy_targets or not my_planets:\n            return []\n\n        my_total_ships = sum(p[5] for p in my_planets)\n        all_total_ships = sum(p[5] for p in planets)\n\n        # B: in-flight context for the whole turn\n        friendly_en_route = _friendly_ships_en_route(raw_fleets, player, planets, av, comet_ids, raw_comets)\n        # C: friendly planets under enemy threat\n        enemy_threat = _enemy_threats(raw_fleets, player, planets, av, comet_ids, raw_comets)\n\n        # Candidate tuple: (score, src_id, tgt_id, sx, sy, tx, ty, ships_to_send, deficit_contribution)\n        candidates = []\n\n        # --- A+B: Attack candidates ---\n        for mine in my_planets:\n            if mine[0] in comet_ids:\n                continue\n            available = int(mine[5] * (1 - RESERVE))\n            if available <= 1:\n                continue\n            for t in enemy_targets:\n                if t[0] in comet_ids:\n                    continue\n                tx, ty, eta = intercept(mine[2], mine[3], t, av, available, comet_ids, raw_comets)\n                if tx is None or sun_blocked(mine[2], mine[3], tx, ty):\n                    continue\n                arrival_garrison = t[5] + (t[6] * eta if t[1] >= 0 else 0)\n                full_cost = arrival_garrison + 1\n                # B: only commit what in-flight friendly fleets haven\'t already covered\n                already_covered = friendly_en_route.get(t[0], 0)\n                remaining = full_cost - already_covered\n                if remaining <= 0:\n                    continue\n                ships_score = remaining\n                if ships_score > available:\n                    continue\n                # Refine intercept with ships_score for accurate ETA\n                tx, ty, eta = intercept(mine[2], mine[3], t, av, ships_score, comet_ids, raw_comets)\n                if tx is None or sun_blocked(mine[2], mine[3], tx, ty):\n                    continue\n                arrival_garrison = t[5] + (t[6] * eta if t[1] >= 0 else 0)\n                full_cost = arrival_garrison + 1\n                already_covered = friendly_en_route.get(t[0], 0)\n                remaining = full_cost - already_covered\n                if remaining <= 0:\n                    continue\n                ships_score = remaining\n                if ships_score > available:\n                    continue\n                dist = math.hypot(tx - mine[2], ty - mine[3])\n                if eta > 0 and abs(dist / fleet_speed(ships_score) - eta) > 2.0:\n                    continue\n                feats = extract_features(\n                    mine, t, eta, arrival_garrison, dist,\n                    available, ships_score, comet_ids, player,\n                    my_planets, planets, raw_fleets, step,\n                    my_total_ships, all_total_ships,\n                    enemy_targets,\n                )\n                score = sum(w[i] * feats[i] for i in range(20))\n                ships_to_send = min(max(ships_score, int(ships_score * alpha)), available)\n                candidates.append((score, mine[0], t[0], mine[2], mine[3], tx, ty, ships_to_send, ships_score))\n\n        # --- C: Defensive reinforce candidates ---\n        for t in my_planets:\n            threat = enemy_threat.get(t[0])\n            if not threat:\n                continue\n            threat_ships, enemy_eta = threat[0], threat[1]\n            garrison_at_arrival = t[5] + t[6] * int(enemy_eta)\n            already_defending = friendly_en_route.get(t[0], 0)\n            if garrison_at_arrival + already_defending >= threat_ships:\n                continue\n            ships_needed = max(1, threat_ships - garrison_at_arrival - already_defending + 1)\n            for mine in my_planets:\n                if mine[0] == t[0]:\n                    continue\n                available = int(mine[5] * (1 - RESERVE))\n                if available <= 1:\n                    continue\n                tx_r, ty_r, eta_r = intercept(mine[2], mine[3], t, av, available, comet_ids, raw_comets)\n                if tx_r is None or sun_blocked(mine[2], mine[3], tx_r, ty_r):\n                    continue\n                if eta_r > enemy_eta + 5:\n                    continue\n                dist = math.hypot(tx_r - mine[2], ty_r - mine[3])\n                ships_to_send = min(ships_needed, available)\n                feats = extract_features(\n                    mine, t, eta_r, 0, dist,\n                    available, ships_to_send, comet_ids, player,\n                    my_planets, planets, raw_fleets, step,\n                    my_total_ships, all_total_ships,\n                    enemy_targets,\n                )\n                score = sum(w[i] * feats[i] for i in range(20))\n                candidates.append((score, mine[0], t[0], mine[2], mine[3], tx_r, ty_r, ships_to_send, ships_needed))\n\n        if not candidates:\n            return []\n\n        # Greedy assignment: one fleet per attack target; multiple allowed for defense.\n        my_planet_ids = {p[0] for p in my_planets}\n        candidates.sort(reverse=True)\n        used_src = set()\n        used_atk_tgt = set()\n        defense_covered = {}\n        moves = []\n        for score, src_id, tgt_id, sx, sy, tx, ty, ships_to_send, ships_needed in candidates:\n            if src_id in used_src:\n                continue\n            if tgt_id in my_planet_ids:\n                covered = defense_covered.get(tgt_id, 0)\n                if covered >= ships_needed:\n                    continue\n                send = min(ships_to_send, ships_needed - covered)\n                moves.append([src_id, math.atan2(ty - sy, tx - sx), send])\n                used_src.add(src_id)\n                defense_covered[tgt_id] = covered + send\n            else:\n                if tgt_id in used_atk_tgt:\n                    continue\n                moves.append([src_id, math.atan2(ty - sy, tx - sx), ships_to_send])\n                used_src.add(src_id)\n                used_atk_tgt.add(tgt_id)\n\n        return moves\n\n    return agent\n'),
    ('agents.policy_features', '"""Feature extraction shared by imitation collection, training, and PPO.\n\nThe policy is a per-candidate scorer. For each turn we enumerate the same\ncandidate set v12 enumerates (attack + defense), compute a 35-dim static\nfeature vector per candidate, and produce 3 dispatch-slot dynamic features\nthat are appended at score time (committed-from-source, committed-to-target,\ndispatch-slot index).\n\nA virtual STOP candidate is always appended at the end of the candidate list,\nso the policy can explicitly choose "no more launches this turn".\n\nLayout (FEATURE_DIM = 38):\n    [0:20]   v12 features from scored_agent.extract_features()\n    [20:26]  6 broadcast global features (my_prod_share, my_ship_share,\n             enemy_count_norm, turn_norm, my_planets_norm, neutrals_norm)\n    [26:30]  target owner one-hot (is_mine, is_neutral, is_strongest_enemy,\n             is_weak_enemy)\n    [30]     src_garrison_norm\n    [31]     src_prod_norm\n    [32]     tgt_dist_to_nearest_my_planet_norm\n    [33]     tgt_dist_to_nearest_enemy_planet_norm\n    [34]     tgt_inbound_friendly_ships_norm\n    [35]     committed_from_src_norm   (dispatch-slot dynamic)\n    [36]     committed_to_tgt_norm     (dispatch-slot dynamic)\n    [37]     dispatch_slot_norm        (dispatch-slot dynamic)\n\nGlobal value-head vector (GLOBAL_DIM = 12):\n    my_prod_share, my_ship_share, enemy_count_norm, turn_norm,\n    my_planets_norm, neutrals_norm, turn_remaining, my_lead_margin,\n    my_planets_under_threat_norm, total_enemy_ships_inflight_norm,\n    comet_active_count_norm, next_comet_spawn_in_norm\n"""\nfrom __future__ import annotations\n\nimport math\nfrom dataclasses import dataclass, field\nfrom typing import Iterable\n\nimport numpy as np\n\nfrom agents.scored_agent import (\n    extract_features,\n    RESERVE,\n    _friendly_ships_en_route,\n    _enemy_threats,\n)\nfrom main import fleet_speed, sun_blocked, intercept\n\n\nFEATURE_DIM = 38\nSTATIC_DIM = 35\nGLOBAL_DIM = 12\nCOMET_SPAWN_STEPS = (50, 150, 250, 350, 450)\nMAX_EPISODE_STEPS = 500\n\n\n@dataclass\nclass Candidate:\n    src_id: int\n    tgt_id: int\n    sx: float\n    sy: float\n    tx: float\n    ty: float\n    angle: float\n    ships_to_send: int\n    ships_needed: int\n    is_defense: bool\n    static_features: np.ndarray = field(default_factory=lambda: np.zeros(STATIC_DIM, dtype=np.float32))\n\n\ndef _parse_obs(obs):\n    """Return a dict of unpacked observation fields, accepting dict or SimpleNamespace."""\n    g = (lambda k, d=None: obs.get(k, d)) if isinstance(obs, dict) else (lambda k, d=None: getattr(obs, k, d))\n    return {\n        "player": g("player", 0),\n        "planets": list(g("planets", []) or []),\n        "fleets": list(g("fleets", []) or []),\n        "av": g("angular_velocity", 0.035),\n        "comet_ids": set(g("comet_planet_ids", []) or []),\n        "raw_comets": list(g("comets", []) or []),\n        "step": int(g("step", 0) or 0),\n    }\n\n\ndef compute_globals(obs):\n    """Return (broadcast_globals[6], value_globals[12]).\n\n    broadcast_globals are reused per candidate and stored in feature columns\n    [20:26]. value_globals are used only by the value head during PPO.\n    """\n    o = _parse_obs(obs)\n    player = o["player"]\n    planets = [list(p) for p in o["planets"]]\n    fleets = o["fleets"]\n    step = o["step"]\n\n    my_planets = [p for p in planets if p[1] == player]\n    enemy_owned = [p for p in planets if p[1] >= 0 and p[1] != player]\n    neutrals = [p for p in planets if p[1] == -1]\n    all_planets = planets\n\n    my_prod = sum(p[6] for p in my_planets)\n    enemy_prod = sum(p[6] for p in enemy_owned)\n    neutral_prod = sum(p[6] for p in neutrals)\n    total_prod = max(my_prod + enemy_prod + neutral_prod, 1)\n\n    my_ships = sum(p[5] for p in my_planets)\n    my_fleet_ships = sum(f[6] for f in fleets if f[1] == player)\n    my_total = my_ships + my_fleet_ships\n    enemy_ships = sum(p[5] for p in enemy_owned)\n    enemy_fleet_ships = sum(f[6] for f in fleets if f[1] >= 0 and f[1] != player)\n    enemy_inflight = enemy_fleet_ships\n    enemy_total = enemy_ships + enemy_fleet_ships\n    total_ships = max(my_total + enemy_total + sum(p[5] for p in neutrals), 1)\n\n    enemy_players = {p[1] for p in enemy_owned}\n    enemy_count = len(enemy_players)\n\n    turn_norm = min(step / MAX_EPISODE_STEPS, 1.0)\n    turn_remaining = 1.0 - turn_norm\n\n    # enemy strength by player for tgt_owner_onehot later; precompute here\n    enemy_strength = {}\n    for pid in enemy_players:\n        s = sum(p[5] for p in planets if p[1] == pid) + sum(f[6] for f in fleets if f[1] == pid)\n        enemy_strength[pid] = s\n    strongest_enemy = max(enemy_strength, key=enemy_strength.get) if enemy_strength else None\n\n    # Planets under threat (any enemy fleet inbound)\n    threat_map = _enemy_threats(fleets, player, planets, o["av"], o["comet_ids"], o["raw_comets"])\n    my_under_threat = sum(1 for pid in threat_map if any(p[0] == pid and p[1] == player for p in planets))\n    my_planets_under_threat_norm = my_under_threat / max(len(my_planets), 1)\n\n    # Comet timing\n    active_comets = len(o["raw_comets"])\n    upcoming = [s for s in COMET_SPAWN_STEPS if s > step]\n    next_spawn_in = (upcoming[0] - step) / MAX_EPISODE_STEPS if upcoming else 1.0\n\n    broadcast = np.array([\n        my_prod / total_prod,                                # my_prod_share\n        my_total / total_ships,                              # my_ship_share\n        enemy_count / 3.0,                                   # enemy_count_norm (max 3 in 4p)\n        turn_norm,                                           # turn_norm\n        len(my_planets) / max(len(all_planets), 1),          # my_planets_norm\n        len(neutrals) / max(len(all_planets), 1),            # neutrals_norm\n    ], dtype=np.float32)\n\n    my_lead = (my_total - max((enemy_strength[p] for p in enemy_strength), default=0)) / max(total_ships, 1)\n\n    value_vec = np.array([\n        *broadcast,\n        turn_remaining,\n        max(min(my_lead, 1.0), -1.0),\n        my_planets_under_threat_norm,\n        enemy_inflight / max(total_ships, 1),\n        active_comets / 4.0,\n        next_spawn_in,\n    ], dtype=np.float32)\n\n    extras = {\n        "strongest_enemy": strongest_enemy,\n        "enemy_strength": enemy_strength,\n        "my_planets": my_planets,\n        "enemy_owned": enemy_owned,\n        "neutrals": neutrals,\n        "all_planets": all_planets,\n        "fleets": fleets,\n        "av": o["av"],\n        "comet_ids": o["comet_ids"],\n        "raw_comets": o["raw_comets"],\n        "step": step,\n        "player": player,\n        "threat_map": threat_map,\n    }\n    return broadcast, value_vec, extras\n\n\ndef _tactical_features(t, mine, extras, friendly_en_route):\n    """Return the 9 candidate-side features beyond v12 + globals:\n       4 owner-onehot, src_garrison_norm, src_prod_norm,\n       tgt_dist_to_nearest_my, tgt_dist_to_nearest_enemy, tgt_inbound_friendly_norm.\n    """\n    player = extras["player"]\n    strongest = extras["strongest_enemy"]\n    out = np.zeros(9, dtype=np.float32)\n\n    # owner one-hot\n    if t[1] == player:\n        out[0] = 1.0\n    elif t[1] == -1:\n        out[1] = 1.0\n    elif strongest is not None and t[1] == strongest:\n        out[2] = 1.0\n    else:\n        out[3] = 1.0\n\n    out[4] = min(mine[5] / 100.0, 1.0)            # src_garrison_norm\n    out[5] = mine[6] / 5.0                        # src_prod_norm\n\n    my_planets = extras["my_planets"]\n    enemy_owned = extras["enemy_owned"]\n    if my_planets:\n        d_my = min(math.hypot(p[2] - t[2], p[3] - t[3]) for p in my_planets)\n        out[6] = min(d_my / 100.0, 1.0)           # tgt_dist_my\n    else:\n        out[6] = 1.0\n    if enemy_owned:\n        d_en = min(math.hypot(p[2] - t[2], p[3] - t[3]) for p in enemy_owned)\n        out[7] = min(d_en / 100.0, 1.0)           # tgt_dist_enemy\n    else:\n        out[7] = 1.0\n\n    inbound = friendly_en_route.get(t[0], 0)\n    out[8] = min(inbound / max(t[5] + 1, 1), 1.0)  # tgt_inbound_friendly_norm\n    return out\n\n\ndef enumerate_candidates(obs) -> tuple[list[Candidate], np.ndarray, np.ndarray]:\n    """Return (candidates, broadcast_globals, value_globals).\n\n    Mirrors v12\'s candidate enumeration: attack (every my_planet × every\n    non-owned non-comet target) plus defense (every my_planet × every\n    my_planet_under_threat). Each candidate has its 35-dim static feature\n    vector populated.\n    """\n    broadcast, value_vec, extras = compute_globals(obs)\n    player = extras["player"]\n    planets = extras["all_planets"]\n    fleets = extras["fleets"]\n    av = extras["av"]\n    comet_ids = extras["comet_ids"]\n    raw_comets = extras["raw_comets"]\n    step = extras["step"]\n    my_planets = extras["my_planets"]\n    enemy_targets = [p for p in planets if p[1] != player]\n    my_total_ships = sum(p[5] for p in my_planets)\n    all_total_ships = sum(p[5] for p in planets)\n\n    friendly_en_route = _friendly_ships_en_route(fleets, player, planets, av, comet_ids, raw_comets)\n    threat_map = extras["threat_map"]\n\n    candidates: list[Candidate] = []\n\n    # Attack candidates (mirror scored_agent.py)\n    for mine in my_planets:\n        if mine[0] in comet_ids:\n            continue\n        available = int(mine[5] * (1 - RESERVE))\n        if available <= 1:\n            continue\n        for t in enemy_targets:\n            if t[0] in comet_ids:\n                continue\n            tx, ty, eta = intercept(mine[2], mine[3], t, av, available, comet_ids, raw_comets)\n            if tx is None or sun_blocked(mine[2], mine[3], tx, ty):\n                continue\n            arrival_garrison = t[5] + (t[6] * eta if t[1] >= 0 else 0)\n            full_cost = arrival_garrison + 1\n            already_covered = friendly_en_route.get(t[0], 0)\n            remaining = full_cost - already_covered\n            if remaining <= 0:\n                continue\n            ships_score = remaining\n            if ships_score > available:\n                continue\n            tx, ty, eta = intercept(mine[2], mine[3], t, av, ships_score, comet_ids, raw_comets)\n            if tx is None or sun_blocked(mine[2], mine[3], tx, ty):\n                continue\n            arrival_garrison = t[5] + (t[6] * eta if t[1] >= 0 else 0)\n            full_cost = arrival_garrison + 1\n            already_covered = friendly_en_route.get(t[0], 0)\n            remaining = full_cost - already_covered\n            if remaining <= 0:\n                continue\n            ships_score = remaining\n            if ships_score > available:\n                continue\n            dist = math.hypot(tx - mine[2], ty - mine[3])\n            if eta > 0 and abs(dist / fleet_speed(ships_score) - eta) > 2.0:\n                continue\n            v12_feats = extract_features(\n                mine, t, eta, arrival_garrison, dist,\n                available, ships_score, comet_ids, player,\n                my_planets, planets, fleets, step,\n                my_total_ships, all_total_ships,\n                enemy_targets,\n            )\n            tactical = _tactical_features(t, mine, extras, friendly_en_route)\n            static = np.zeros(STATIC_DIM, dtype=np.float32)\n            static[:20] = v12_feats\n            static[20:26] = broadcast\n            static[26:35] = tactical\n            angle = math.atan2(ty - mine[3], tx - mine[2])\n            candidates.append(Candidate(\n                src_id=mine[0], tgt_id=t[0],\n                sx=mine[2], sy=mine[3], tx=tx, ty=ty, angle=angle,\n                ships_to_send=ships_score, ships_needed=ships_score,\n                is_defense=False, static_features=static,\n            ))\n\n    # Defense candidates (mirror scored_agent.py)\n    for t in my_planets:\n        threat = threat_map.get(t[0])\n        if not threat:\n            continue\n        threat_ships, enemy_eta = threat[0], threat[1]\n        garrison_at_arrival = t[5] + t[6] * int(enemy_eta)\n        already_defending = friendly_en_route.get(t[0], 0)\n        if garrison_at_arrival + already_defending >= threat_ships:\n            continue\n        ships_needed = max(1, threat_ships - garrison_at_arrival - already_defending + 1)\n        for mine in my_planets:\n            if mine[0] == t[0]:\n                continue\n            available = int(mine[5] * (1 - RESERVE))\n            if available <= 1:\n                continue\n            tx_r, ty_r, eta_r = intercept(mine[2], mine[3], t, av, available, comet_ids, raw_comets)\n            if tx_r is None or sun_blocked(mine[2], mine[3], tx_r, ty_r):\n                continue\n            if eta_r > enemy_eta + 5:\n                continue\n            dist = math.hypot(tx_r - mine[2], ty_r - mine[3])\n            ships_to_send = min(ships_needed, available)\n            v12_feats = extract_features(\n                mine, t, eta_r, 0, dist,\n                available, ships_to_send, comet_ids, player,\n                my_planets, planets, fleets, step,\n                my_total_ships, all_total_ships,\n                enemy_targets,\n            )\n            tactical = _tactical_features(t, mine, extras, friendly_en_route)\n            static = np.zeros(STATIC_DIM, dtype=np.float32)\n            static[:20] = v12_feats\n            static[20:26] = broadcast\n            static[26:35] = tactical\n            angle = math.atan2(ty_r - mine[3], tx_r - mine[2])\n            candidates.append(Candidate(\n                src_id=mine[0], tgt_id=t[0],\n                sx=mine[2], sy=mine[3], tx=tx_r, ty=ty_r, angle=angle,\n                ships_to_send=ships_to_send, ships_needed=ships_needed,\n                is_defense=True, static_features=static,\n            ))\n\n    return candidates, broadcast, value_vec\n\n\ndef build_dispatch_features(candidates: list[Candidate], used_src: set, defense_covered: dict,\n                            slot_idx: int, k_max: int) -> tuple[np.ndarray, np.ndarray]:\n    """Return (features [N+1, 38], valid_mask [N+1] of {0,1}).\n\n    Index N is the virtual STOP candidate (always valid).\n    """\n    n = len(candidates)\n    feats = np.zeros((n + 1, FEATURE_DIM), dtype=np.float32)\n    mask = np.zeros(n + 1, dtype=np.float32)\n\n    slot_norm = slot_idx / max(k_max, 1)\n\n    for i, c in enumerate(candidates):\n        feats[i, :STATIC_DIM] = c.static_features\n        # dynamic features\n        committed_from_src = 1.0 if c.src_id in used_src else 0.0\n        if c.is_defense:\n            covered = defense_covered.get(c.tgt_id, 0)\n            committed_to_tgt = min(covered / max(c.ships_needed, 1), 1.0)\n        else:\n            committed_to_tgt = 0.0\n        feats[i, 35] = committed_from_src\n        feats[i, 36] = committed_to_tgt\n        feats[i, 37] = slot_norm\n        mask[i] = _is_valid(c, used_src, defense_covered)\n\n    # STOP slot — features are slot_norm only; always valid.\n    feats[n, 37] = slot_norm\n    mask[n] = 1.0\n    return feats, mask\n\n\ndef apply_attack_target_mask(candidates: list[Candidate], mask: np.ndarray,\n                             used_atk_tgt: set) -> np.ndarray:\n    """Apply the v12/v15 one-attack-per-target rule to an existing mask.\n\n    `build_dispatch_features` only knows source and defense coverage state.\n    The attack target rule is kept separate so callers that intentionally allow\n    same-target pressure (v16) can skip it, while v15 imitation, PPO rollout,\n    and inference can stay exactly aligned.\n    """\n    for i, c in enumerate(candidates):\n        if not c.is_defense and c.tgt_id in used_atk_tgt:\n            mask[i] = 0.0\n    return mask\n\n\ndef _is_valid(c: Candidate, used_src: set, defense_covered: dict) -> float:\n    if c.src_id in used_src:\n        return 0.0\n    if c.is_defense:\n        covered = defense_covered.get(c.tgt_id, 0)\n        if covered >= c.ships_needed:\n            return 0.0\n    return 1.0\n\n\ndef find_chosen_candidate(candidates: list[Candidate], move, used_src: set, used_atk_tgt: set,\n                          defense_covered: dict) -> int:\n    """Match a v12 action [src_id, angle, ships] to a candidate index.\n\n    Returns -1 if no match (rare; usually a v12 internal allocation we don\'t\n    reproduce exactly — the caller can drop the sample).\n    """\n    src_id, angle, ships = move[0], move[1], move[2]\n    best_i = -1\n    best_diff = 0.20  # ~11 degrees tolerance\n    for i, c in enumerate(candidates):\n        if c.src_id != src_id:\n            continue\n        if not _is_valid(c, used_src, defense_covered):\n            continue\n        if c.tgt_id in used_atk_tgt and not c.is_defense:\n            continue\n        diff = abs((c.angle - angle + math.pi) % (2 * math.pi) - math.pi)\n        if diff < best_diff:\n            best_diff = diff\n            best_i = i\n    return best_i\n\n\ndef commit(candidates: list[Candidate], idx: int, used_src: set, used_atk_tgt: set,\n           defense_covered: dict) -> None:\n    """Update dispatch state after the policy picks candidate idx (not STOP)."""\n    c = candidates[idx]\n    used_src.add(c.src_id)\n    if c.is_defense:\n        defense_covered[c.tgt_id] = defense_covered.get(c.tgt_id, 0) + c.ships_to_send\n    else:\n        used_atk_tgt.add(c.tgt_id)\n'),
    ('agents.v16_open_features', '"""Open-action candidate generation for v16 PPO.\n\nv15 learned inside the v12 action surface: one attack per target, exact\ncaptures only, and defense only for owned planets. This module keeps the\ngeometry solved by candidate generation, but widens the tactical surface so\nself-play can choose bundles, pressure, overkill, drain-style launches, and\nvoluntary support.\n\nThe policy still scores a variable-length list of legal candidates plus a STOP\nslot. Source planets remain single-use within a turn; attack targets do not.\n"""\nfrom __future__ import annotations\n\nimport math\nfrom dataclasses import dataclass, field\n\nimport numpy as np\n\nfrom agents.policy_features import (\n    compute_globals,\n    _tactical_features,\n)\nfrom agents.scored_agent import (\n    extract_features,\n    RESERVE,\n    _friendly_ships_en_route,\n)\nfrom main import fleet_speed, sun_blocked, intercept\n\n\nTYPE_NAMES = (\n    "capture_exact",\n    "capture_overkill",\n    "pressure",\n    "defense",\n    "support",\n    "consolidate",\n)\nBUCKET_NAMES = (\n    "exact",\n    "x125",\n    "x150",\n    "half_available",\n    "full_available",\n)\n\nTYPE_TO_IDX = {name: i for i, name in enumerate(TYPE_NAMES)}\nBUCKET_TO_IDX = {name: i for i, name in enumerate(BUCKET_NAMES)}\n\nOPEN_STATIC_DIM = 50\nOPEN_FEATURE_DIM = 55\nOPEN_GLOBAL_DIM = 12\n\n\n@dataclass\nclass OpenCandidate:\n    src_id: int\n    tgt_id: int\n    sx: float\n    sy: float\n    tx: float\n    ty: float\n    angle: float\n    ships_to_send: int\n    ships_needed: int\n    available: int\n    cand_type: str\n    bucket: str\n    is_attack: bool\n    is_defense: bool\n    static_features: np.ndarray = field(\n        default_factory=lambda: np.zeros(OPEN_STATIC_DIM, dtype=np.float32)\n    )\n\n\ndef _ceil_ships(x: float) -> int:\n    return max(1, int(math.ceil(float(x))))\n\n\ndef _arrival_need(mine, target, av, ships: int, comet_ids, raw_comets,\n                  friendly_en_route: dict[int, int]):\n    tx, ty, eta = intercept(mine[2], mine[3], target, av, ships, comet_ids, raw_comets)\n    if tx is None or sun_blocked(mine[2], mine[3], tx, ty):\n        return None\n    arrival_garrison = target[5] + (target[6] * eta if target[1] >= 0 else 0)\n    remaining = arrival_garrison + 1 - friendly_en_route.get(target[0], 0)\n    dist = math.hypot(tx - mine[2], ty - mine[3])\n    if eta > 0 and abs(dist / fleet_speed(ships) - eta) > 2.0:\n        return None\n    return tx, ty, eta, arrival_garrison, remaining, dist\n\n\ndef _refined_capture_need(mine, target, av, available: int, comet_ids, raw_comets,\n                          friendly_en_route: dict[int, int]):\n    probe = _arrival_need(mine, target, av, available, comet_ids, raw_comets, friendly_en_route)\n    if probe is None:\n        return None\n    needed = _ceil_ships(probe[4])\n    if needed <= 0:\n        return None\n    if needed > available:\n        return (*probe, needed)\n\n    for _ in range(3):\n        refined = _arrival_need(mine, target, av, needed, comet_ids, raw_comets, friendly_en_route)\n        if refined is None:\n            return None\n        next_needed = _ceil_ships(refined[4])\n        if next_needed == needed:\n            return (*refined, needed)\n        needed = next_needed\n        if needed > available:\n            return (*refined, needed)\n    return (*refined, needed)\n\n\ndef _static_features(mine, target, eta: float, arrival_garrison: float, dist: float,\n                     available: int, ships_to_send: int, ships_needed: int,\n                     cand_type: str, bucket: str, broadcast: np.ndarray,\n                     extras: dict, friendly_en_route: dict[int, int],\n                     enemy_targets: list) -> np.ndarray:\n    player = extras["player"]\n    planets = extras["all_planets"]\n    fleets = extras["fleets"]\n    step = extras["step"]\n    comet_ids = extras["comet_ids"]\n    my_planets = extras["my_planets"]\n    my_total_ships = sum(p[5] for p in my_planets)\n    all_total_ships = sum(p[5] for p in planets)\n\n    v12_feats = extract_features(\n        mine, target, eta, arrival_garrison, dist,\n        available, ships_to_send, comet_ids, player,\n        my_planets, planets, fleets, step,\n        my_total_ships, all_total_ships,\n        enemy_targets,\n    )\n    tactical = _tactical_features(target, mine, extras, friendly_en_route)\n\n    static = np.zeros(OPEN_STATIC_DIM, dtype=np.float32)\n    static[:20] = v12_feats\n    static[20:26] = broadcast\n    static[26:35] = tactical\n    static[35 + TYPE_TO_IDX[cand_type]] = 1.0\n    static[41 + BUCKET_TO_IDX[bucket]] = 1.0\n    static[46] = min(ships_to_send / max(available, 1), 1.0)\n    static[47] = min(available / 100.0, 1.0)\n    static[48] = min(ships_needed / max(available, 1), 2.0) / 2.0\n    static[49] = min(max(ships_needed - available, 0) / 100.0, 1.0)\n    return static\n\n\ndef _add_candidate(candidates: list[OpenCandidate], seen: set,\n                   mine, target, av, comet_ids, raw_comets,\n                   broadcast: np.ndarray, extras: dict,\n                   friendly_en_route: dict[int, int], enemy_targets: list,\n                   available: int, ships_to_send: int, ships_needed: int,\n                   cand_type: str, bucket: str, is_attack: bool,\n                   is_defense: bool, arrival_garrison: float | None = None) -> None:\n    ships_to_send = max(1, min(int(ships_to_send), available))\n    if available <= 1 or ships_to_send <= 0:\n        return\n    key = (mine[0], target[0], ships_to_send)\n    if key in seen:\n        return\n\n    tx, ty, eta = intercept(mine[2], mine[3], target, av, ships_to_send, comet_ids, raw_comets)\n    if tx is None or sun_blocked(mine[2], mine[3], tx, ty):\n        return\n    dist = math.hypot(tx - mine[2], ty - mine[3])\n    if eta > 0 and abs(dist / fleet_speed(ships_to_send) - eta) > 2.0:\n        return\n    if arrival_garrison is None:\n        arrival_garrison = target[5] + (target[6] * eta if target[1] >= 0 else 0)\n\n    static = _static_features(\n        mine, target, eta, arrival_garrison, dist,\n        available, ships_to_send, max(ships_needed, 1),\n        cand_type, bucket, broadcast, extras, friendly_en_route, enemy_targets,\n    )\n    candidates.append(OpenCandidate(\n        src_id=mine[0],\n        tgt_id=target[0],\n        sx=mine[2],\n        sy=mine[3],\n        tx=tx,\n        ty=ty,\n        angle=math.atan2(ty - mine[3], tx - mine[2]),\n        ships_to_send=ships_to_send,\n        ships_needed=max(ships_needed, 1),\n        available=available,\n        cand_type=cand_type,\n        bucket=bucket,\n        is_attack=is_attack,\n        is_defense=is_defense,\n        static_features=static,\n    ))\n    seen.add(key)\n\n\ndef enumerate_open_candidates(obs) -> tuple[list[OpenCandidate], np.ndarray, np.ndarray]:\n    """Return open-action candidates plus broadcast and value globals."""\n    broadcast, value_vec, extras = compute_globals(obs)\n    player = extras["player"]\n    planets = extras["all_planets"]\n    fleets = extras["fleets"]\n    av = extras["av"]\n    comet_ids = extras["comet_ids"]\n    raw_comets = extras["raw_comets"]\n    my_planets = extras["my_planets"]\n    enemy_targets = [p for p in planets if p[1] != player]\n\n    friendly_en_route = _friendly_ships_en_route(\n        fleets, player, planets, av, comet_ids, raw_comets\n    )\n    threat_map = extras["threat_map"]\n\n    candidates: list[OpenCandidate] = []\n    seen: set = set()\n\n    for mine in my_planets:\n        if mine[0] in comet_ids:\n            continue\n        available = int(mine[5] * (1 - RESERVE))\n        if available <= 1:\n            continue\n\n        for target in enemy_targets:\n            if target[0] in comet_ids:\n                continue\n            refined = _refined_capture_need(\n                mine, target, av, available, comet_ids, raw_comets, friendly_en_route\n            )\n            if refined is None:\n                continue\n            tx, ty, eta, arrival_garrison, remaining, dist, needed = refined\n            if remaining <= 0:\n                continue\n\n            if needed <= available:\n                _add_candidate(\n                    candidates, seen, mine, target, av, comet_ids, raw_comets,\n                    broadcast, extras, friendly_en_route, enemy_targets,\n                    available, needed, needed, "capture_exact", "exact",\n                    is_attack=True, is_defense=False, arrival_garrison=arrival_garrison,\n                )\n                for mult, bucket in ((1.25, "x125"), (1.50, "x150")):\n                    amount = min(available, _ceil_ships(needed * mult))\n                    if amount > needed:\n                        _add_candidate(\n                            candidates, seen, mine, target, av, comet_ids, raw_comets,\n                            broadcast, extras, friendly_en_route, enemy_targets,\n                            available, amount, needed, "capture_overkill", bucket,\n                            is_attack=True, is_defense=False,\n                        )\n\n            pressure_target = target[1] >= 0 or target[6] >= 3 or needed > available\n            if pressure_target:\n                pressure_amounts = [\n                    (max(1, available // 2), "half_available"),\n                    (available, "full_available"),\n                ]\n                for amount, bucket in pressure_amounts:\n                    _add_candidate(\n                        candidates, seen, mine, target, av, comet_ids, raw_comets,\n                        broadcast, extras, friendly_en_route, enemy_targets,\n                        available, amount, needed, "pressure", bucket,\n                        is_attack=True, is_defense=False,\n                    )\n\n        for target in my_planets:\n            if target[0] == mine[0] or target[0] in comet_ids:\n                continue\n            threat = threat_map.get(target[0])\n            if threat:\n                threat_ships, enemy_eta = threat[0], threat[1]\n                garrison_at_arrival = target[5] + target[6] * int(enemy_eta)\n                already_defending = friendly_en_route.get(target[0], 0)\n                ships_needed = max(1, threat_ships - garrison_at_arrival - already_defending + 1)\n                if ships_needed > 0:\n                    for amount, bucket in (\n                        (min(ships_needed, available), "exact"),\n                        (min(_ceil_ships(ships_needed * 1.25), available), "x125"),\n                        (available, "full_available"),\n                    ):\n                        _add_candidate(\n                            candidates, seen, mine, target, av, comet_ids, raw_comets,\n                            broadcast, extras, friendly_en_route, enemy_targets,\n                            available, amount, ships_needed, "defense", bucket,\n                            is_attack=False, is_defense=True,\n                            arrival_garrison=0.0,\n                        )\n\n            if available >= 5:\n                ctype = "consolidate" if mine[5] > target[5] + 10 else "support"\n                for amount, bucket in (\n                    (max(1, available // 2), "half_available"),\n                    (available, "full_available"),\n                ):\n                    _add_candidate(\n                        candidates, seen, mine, target, av, comet_ids, raw_comets,\n                        broadcast, extras, friendly_en_route, enemy_targets,\n                        available, amount, amount, ctype, bucket,\n                        is_attack=False, is_defense=False,\n                        arrival_garrison=0.0,\n                    )\n\n    return candidates, broadcast, value_vec\n\n\ndef build_open_dispatch_features(candidates: list[OpenCandidate], used_src: set,\n                                 target_committed: dict, target_counts: dict,\n                                 slot_idx: int, k_max: int) -> tuple[np.ndarray, np.ndarray]:\n    """Return `(features, valid_mask)` for v16 open-action dispatch."""\n    n = len(candidates)\n    feats = np.zeros((n + 1, OPEN_FEATURE_DIM), dtype=np.float32)\n    mask = np.zeros(n + 1, dtype=np.float32)\n    slot_norm = slot_idx / max(k_max, 1)\n\n    for i, c in enumerate(candidates):\n        feats[i, :OPEN_STATIC_DIM] = c.static_features\n        committed = target_committed.get(c.tgt_id, 0)\n        count = target_counts.get(c.tgt_id, 0)\n        feats[i, 50] = 1.0 if c.src_id in used_src else 0.0\n        feats[i, 51] = min(committed / max(c.ships_needed, 1), 2.0) / 2.0\n        feats[i, 52] = min(committed / 100.0, 1.0)\n        feats[i, 53] = min(count / max(k_max, 1), 1.0)\n        feats[i, 54] = slot_norm\n        valid = c.src_id not in used_src\n        if c.is_defense and committed >= c.ships_needed:\n            valid = False\n        mask[i] = 1.0 if valid else 0.0\n\n    feats[n, 54] = slot_norm\n    mask[n] = 1.0\n    return feats, mask\n\n\ndef commit_open(candidates: list[OpenCandidate], idx: int, used_src: set,\n                target_committed: dict, target_counts: dict) -> None:\n    c = candidates[idx]\n    used_src.add(c.src_id)\n    target_committed[c.tgt_id] = target_committed.get(c.tgt_id, 0) + c.ships_to_send\n    target_counts[c.tgt_id] = target_counts.get(c.tgt_id, 0) + 1\n\n\ndef find_open_chosen_candidate(candidates: list[OpenCandidate], move,\n                               mask: np.ndarray,\n                               angle_tol: float = 0.35) -> int:\n    """Match an expert action to the nearest valid open-action candidate.\n\n    v12/v12a/v14 teachers may send a ship amount that is not exactly one of the\n    v16 buckets, and different ship counts shift the predicted intercept angle\n    (target is orbiting, fleet speed depends on ships). Matching ranks valid\n    same-source candidates by combined angle + ship-count distance and falls\n    back to the best candidate within angle_tol.\n    """\n    src_id, angle, ships = move[0], move[1], move[2]\n    best_i = -1\n    best_score = float("inf")\n    for i, c in enumerate(candidates):\n        if i >= len(mask) or mask[i] < 0.5 or c.src_id != src_id:\n            continue\n        angle_diff = abs((c.angle - angle + math.pi) % (2 * math.pi) - math.pi)\n        if angle_diff > angle_tol:\n            continue\n        ship_diff = abs(c.ships_to_send - ships) / max(c.ships_to_send, ships, 1)\n        score = angle_diff + 0.10 * ship_diff\n        if score < best_score:\n            best_score = score\n            best_i = i\n    return best_i\n\n\n__all__ = [\n    "OpenCandidate",\n    "OPEN_FEATURE_DIM",\n    "OPEN_GLOBAL_DIM",\n    "OPEN_STATIC_DIM",\n    "TYPE_NAMES",\n    "BUCKET_NAMES",\n    "enumerate_open_candidates",\n    "build_open_dispatch_features",\n    "commit_open",\n    "find_open_chosen_candidate",\n]\n'),
    ('agents.v17_action_sets', '"""Action-set proposals for v17 value-guided policy improvement.\n\nThis module lifts the v16 open-action primitives into whole-turn action sets.\nEvery launch is canonicalized by the planet the engine-style trajectory replay\nactually hits, so branch mutations reason about real fleet destinations rather\nthan intended target labels.\n"""\nfrom __future__ import annotations\n\nimport math\nimport random\nfrom dataclasses import dataclass, field\nfrom typing import Iterable\n\nimport numpy as np\n\nfrom agents.policy_features import compute_globals\nfrom agents.scored_agent import _friendly_ships_en_route\nfrom agents.v16_open_features import (\n    BUCKET_NAMES,\n    OPEN_STATIC_DIM,\n    OpenCandidate,\n    TYPE_NAMES,\n    enumerate_open_candidates,\n    _static_features,\n)\nfrom main import (\n    _solve_engine_angle,\n    _trajectory_first_hit,\n    fleet_speed,\n    intercept,\n)\n\n\nACTION_SET_FEATURE_DIM = 128\nACTION_SET_BASE_FEATURE_DIM = 112\nMAX_ACTIONS_PER_SET = 16\nANGLE_KEY_SCALE = 1_000_000\n\n\n@dataclass(frozen=True)\nclass CanonicalLaunch:\n    src_id: int\n    angle: float\n    ships: int\n    hit_id: int\n    hit_reason: str\n    eta: int\n    cand_type: str = "teacher"\n    bucket: str = "teacher"\n    intended_tgt_id: int | None = None\n    static_features: tuple[float, ...] = field(default_factory=tuple)\n\n    def action(self) -> list:\n        return [self.src_id, self.angle, self.ships]\n\n\n@dataclass\nclass ActionSetProposal:\n    actions: list[list]\n    canonical: tuple[CanonicalLaunch, ...]\n    features: np.ndarray\n    provenance: tuple[str, ...]\n    key: tuple\n    is_teacher: bool = False\n\n\n@dataclass\nclass _ObsContext:\n    player: int\n    planets: list[list]\n    fleets: list[list]\n    av: float\n    comet_ids: set[int]\n    raw_comets: list\n    step: int\n    source_by_id: dict[int, list]\n    planet_by_id: dict[int, list]\n    my_planets: list[list]\n    enemy_targets: list[list]\n    broadcast: np.ndarray\n    value_globals: np.ndarray\n    extras: dict\n    friendly_en_route: dict[int, int]\n\n\n_OPEN_TYPE_SET = set(TYPE_NAMES)\n_OPEN_BUCKET_SET = set(BUCKET_NAMES)\n\n\ndef _get(obj, key: str, default=None):\n    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)\n\n\ndef _obs_context(obs) -> _ObsContext:\n    broadcast, value_globals, extras = compute_globals(obs)\n    planets = [list(p) for p in extras["all_planets"]]\n    player = int(extras["player"])\n    my_planets = [p for p in planets if int(p[1]) == player]\n    enemy_targets = [p for p in planets if int(p[1]) != player]\n    friendly_en_route = _friendly_ships_en_route(\n        extras["fleets"],\n        player,\n        planets,\n        float(extras["av"]),\n        extras["comet_ids"],\n        extras["raw_comets"],\n    )\n    return _ObsContext(\n        player=player,\n        planets=planets,\n        fleets=[list(f) for f in extras["fleets"]],\n        av=float(extras["av"]),\n        comet_ids={int(pid) for pid in extras["comet_ids"]},\n        raw_comets=list(extras["raw_comets"]),\n        step=int(extras["step"]),\n        source_by_id={int(p[0]): p for p in my_planets},\n        planet_by_id={int(p[0]): p for p in planets},\n        my_planets=my_planets,\n        enemy_targets=enemy_targets,\n        broadcast=broadcast,\n        value_globals=value_globals,\n        extras=extras,\n        friendly_en_route=friendly_en_route,\n    )\n\n\ndef _angle_key(angle: float) -> int:\n    return int(round(float(angle) * ANGLE_KEY_SCALE))\n\n\ndef _launch_key(launch: CanonicalLaunch) -> tuple:\n    return (\n        int(launch.src_id),\n        int(launch.hit_id),\n        int(launch.ships),\n        _angle_key(launch.angle),\n    )\n\n\ndef action_set_key(launches: Iterable[CanonicalLaunch]) -> tuple:\n    return tuple(sorted(_launch_key(launch) for launch in launches))\n\n\ndef _move_static_tuple(candidate: OpenCandidate | None) -> tuple[float, ...]:\n    if candidate is None:\n        return ()\n    return tuple(float(x) for x in candidate.static_features.tolist())\n\n\ndef _infer_open_type(\n    ctx: _ObsContext,\n    source,\n    target,\n    ships: int,\n    ships_needed: int,\n    preferred: str,\n) -> str:\n    if preferred in _OPEN_TYPE_SET:\n        return preferred\n    if int(target[1]) == ctx.player:\n        if ctx.extras["threat_map"].get(int(target[0])):\n            return "defense"\n        return "consolidate" if float(source[5]) > float(target[5]) + 10.0 else "support"\n    if ships >= max(1, ships_needed):\n        if ships >= int(math.ceil(max(1, ships_needed) * 1.25)):\n            return "capture_overkill"\n        return "capture_exact"\n    return "pressure"\n\n\ndef _infer_open_bucket(ships: int, available: int, ships_needed: int, preferred: str) -> str:\n    if preferred in _OPEN_BUCKET_SET:\n        return preferred\n    ships_needed = max(1, int(math.ceil(float(ships_needed))))\n    if ships >= available:\n        return "full_available"\n    if ships <= ships_needed:\n        return "exact"\n    if ships <= int(math.ceil(ships_needed * 1.25)):\n        return "x125"\n    if ships <= int(math.ceil(ships_needed * 1.50)):\n        return "x150"\n    if ships <= max(1, available // 2):\n        return "half_available"\n    return "x150"\n\n\ndef _build_static_tuple(\n    ctx: _ObsContext,\n    source,\n    target,\n    ships: int,\n    eta: float,\n    cand_type: str,\n    bucket: str,\n) -> tuple[float, ...]:\n    if source is None or target is None:\n        return ()\n    available = max(int(source[5]), 1)\n    ships = max(1, min(int(ships), available))\n\n    tx, ty, eta_est = intercept(\n        source[2], source[3], target, ctx.av, ships, ctx.comet_ids, ctx.raw_comets\n    )\n    if tx is None or ty is None:\n        tx, ty = float(target[2]), float(target[3])\n    eta_value = float(eta if eta and eta > 0 else eta_est or 0.0)\n    dist = math.hypot(float(tx) - float(source[2]), float(ty) - float(source[3]))\n\n    target_id = int(target[0])\n    if int(target[1]) == ctx.player:\n        threat = ctx.extras["threat_map"].get(target_id)\n        if threat:\n            threat_ships, enemy_eta = threat[0], threat[1]\n            garrison_at_arrival = float(target[5]) + float(target[6]) * int(enemy_eta)\n            already_defending = ctx.friendly_en_route.get(target_id, 0)\n            ships_needed = max(1, int(math.ceil(threat_ships - garrison_at_arrival - already_defending + 1)))\n            arrival_garrison = 0.0\n        else:\n            ships_needed = ships\n            arrival_garrison = 0.0\n    else:\n        arrival_garrison = float(target[5])\n        if int(target[1]) >= 0:\n            arrival_garrison += float(target[6]) * eta_value\n        ships_needed = max(\n            1,\n            int(math.ceil(arrival_garrison + 1.0 - ctx.friendly_en_route.get(target_id, 0))),\n        )\n\n    inferred_type = _infer_open_type(ctx, source, target, ships, ships_needed, cand_type)\n    inferred_bucket = _infer_open_bucket(ships, available, ships_needed, bucket)\n    static = _static_features(\n        source,\n        target,\n        eta_value,\n        arrival_garrison,\n        dist,\n        available,\n        ships,\n        ships_needed,\n        inferred_type,\n        inferred_bucket,\n        ctx.broadcast,\n        ctx.extras,\n        ctx.friendly_en_route,\n        ctx.enemy_targets,\n    )\n    return tuple(float(x) for x in static.tolist())\n\n\ndef _static_tuple_for_launch(ctx: _ObsContext, launch: CanonicalLaunch) -> tuple[float, ...]:\n    return _build_static_tuple(\n        ctx,\n        ctx.source_by_id.get(int(launch.src_id)),\n        ctx.planet_by_id.get(int(launch.hit_id)),\n        int(launch.ships),\n        float(launch.eta),\n        str(launch.cand_type),\n        str(launch.bucket),\n    )\n\n\ndef _canonicalize_move(\n    ctx: _ObsContext,\n    move,\n    committed_by_source: dict[int, int],\n    candidate: OpenCandidate | None = None,\n    max_steps: int = 180,\n) -> CanonicalLaunch | None:\n    if not isinstance(move, (list, tuple)) or len(move) != 3:\n        return None\n    try:\n        src_id = int(move[0])\n        angle = float(move[1])\n        ships = int(move[2])\n    except (TypeError, ValueError):\n        return None\n    if ships <= 0 or not math.isfinite(angle):\n        return None\n    source = ctx.source_by_id.get(src_id)\n    if source is None:\n        return None\n    already = committed_by_source.get(src_id, 0)\n    if already + ships > int(source[5]):\n        return None\n\n    hit_id, reason, eta = _trajectory_first_hit(\n        source,\n        angle,\n        ships,\n        ctx.planets,\n        ctx.av,\n        ctx.comet_ids,\n        ctx.raw_comets,\n        max_steps=max_steps,\n    )\n    if hit_id is None and reason == "sun":\n        return None\n\n    committed_by_source[src_id] = already + ships\n    actual_hit_id = int(hit_id) if hit_id is not None else -1\n    target = ctx.planet_by_id.get(actual_hit_id)\n    if candidate is not None and int(candidate.tgt_id) == actual_hit_id:\n        static_features = _move_static_tuple(candidate)\n    else:\n        static_features = _build_static_tuple(\n            ctx,\n            source,\n            target,\n            ships,\n            int(eta),\n            candidate.cand_type if candidate is not None else "teacher",\n            candidate.bucket if candidate is not None else "teacher",\n        )\n\n    return CanonicalLaunch(\n        src_id=src_id,\n        angle=angle,\n        ships=ships,\n        hit_id=actual_hit_id,\n        hit_reason=str(reason),\n        eta=int(eta),\n        cand_type=candidate.cand_type if candidate is not None else "teacher",\n        bucket=candidate.bucket if candidate is not None else "teacher",\n        intended_tgt_id=int(candidate.tgt_id) if candidate is not None else None,\n        static_features=static_features,\n    )\n\n\ndef canonicalize_action_set(obs, actions) -> tuple[tuple[CanonicalLaunch, ...], bool]:\n    """Return engine-hit canonical launches and whether all raw moves were valid."""\n    ctx = _obs_context(obs)\n    launches: list[CanonicalLaunch] = []\n    committed: dict[int, int] = {}\n    ok = True\n    for move in actions or []:\n        launch = _canonicalize_move(ctx, move, committed)\n        if launch is None:\n            ok = False\n            continue\n        launches.append(launch)\n    return tuple(launches), ok\n\n\ndef validate_action_set(obs, actions) -> tuple[bool, str]:\n    ctx = _obs_context(obs)\n    committed: dict[int, int] = {}\n    for idx, move in enumerate(actions or []):\n        before = dict(committed)\n        launch = _canonicalize_move(ctx, move, committed)\n        if launch is None:\n            return False, f"invalid launch {idx}: {move!r}"\n        if committed == before:\n            return False, f"launch {idx} did not reserve source ships"\n    return True, ""\n\n\ndef _candidate_action(candidate: OpenCandidate) -> list:\n    return [int(candidate.src_id), float(candidate.angle), int(candidate.ships_to_send)]\n\n\ndef _canonical_candidate(ctx: _ObsContext, candidate: OpenCandidate) -> CanonicalLaunch | None:\n    return _canonicalize_move(ctx, _candidate_action(candidate), {}, candidate=candidate)\n\n\ndef _trusted_candidate(ctx: _ObsContext, candidate: OpenCandidate) -> CanonicalLaunch | None:\n    """Cheap candidate wrapper for inference-time scoring.\n\n    `enumerate_open_candidates` has already solved an intercept and filtered\n    sun-blocked paths. Offline search uses `_canonical_candidate`; online\n    inference can trade exact first-hit replay for speed because the submitted\n    action remains engine-legal even if this feature-side target is imperfect.\n    """\n    try:\n        src_id = int(candidate.src_id)\n        tgt_id = int(candidate.tgt_id)\n        ships = int(candidate.ships_to_send)\n        angle = float(candidate.angle)\n    except (TypeError, ValueError):\n        return None\n    source = ctx.source_by_id.get(src_id)\n    if source is None or ships <= 0 or ships > int(source[5]) or not math.isfinite(angle):\n        return None\n    dist = math.hypot(float(candidate.tx) - float(candidate.sx), float(candidate.ty) - float(candidate.sy))\n    eta = max(1, int(round(dist / max(fleet_speed(ships), 1e-6))))\n    return CanonicalLaunch(\n        src_id=src_id,\n        angle=angle,\n        ships=ships,\n        hit_id=tgt_id,\n        hit_reason="candidate",\n        eta=eta,\n        cand_type=candidate.cand_type,\n        bucket=candidate.bucket,\n        intended_tgt_id=tgt_id,\n        static_features=_move_static_tuple(candidate),\n    )\n\n\ndef _add_proposal(\n    proposals: list[ActionSetProposal],\n    seen: set,\n    ctx: _ObsContext,\n    launches: Iterable[CanonicalLaunch],\n    provenance: Iterable[str],\n    is_teacher: bool = False,\n    validate_trajectory: bool = True,\n) -> None:\n    canonical = tuple(launches)\n    if len(canonical) > MAX_ACTIONS_PER_SET:\n        return\n    key = action_set_key(canonical)\n    if key in seen:\n        return\n    actions = [launch.action() for launch in canonical]\n    ok, _msg = _validate_canonical_constraints(\n        ctx,\n        canonical,\n        validate_trajectory=validate_trajectory,\n    )\n    if not ok:\n        return\n    features = featurize_action_set_from_canonical(ctx, canonical)\n    _add_proposal_context_features(features, provenance, len(proposals), canonical)\n    proposals.append(ActionSetProposal(\n        actions=actions,\n        canonical=canonical,\n        features=features,\n        provenance=tuple(provenance),\n        key=key,\n        is_teacher=is_teacher,\n    ))\n    seen.add(key)\n\n\ndef _add_proposal_context_features(\n    features: np.ndarray,\n    provenance: Iterable[str],\n    proposal_idx: int,\n    launches: tuple[CanonicalLaunch, ...],\n) -> None:\n    tags = tuple(provenance)\n    family = tags[0].split(":", 1)[0] if tags else "unknown"\n    features[112] = 1.0 if family == "teacher" else 0.0\n    features[113] = 1.0 if family == "delay_stop" else 0.0\n    features[114] = 1.0 if family == "remove" else 0.0\n    features[115] = 1.0 if family == "single" else 0.0\n    features[116] = 1.0 if family == "add" else 0.0\n    features[117] = 1.0 if family == "resize" else 0.0\n    features[118] = 1.0 if family == "swap" else 0.0\n    features[119] = 1.0 if family == "bundle" else 0.0\n    features[120] = 1.0 if family == "diversity_combo" else 0.0\n    features[121] = min(float(proposal_idx) / 50.0, 1.0)\n\n    types = {launch.cand_type for launch in launches}\n    features[122] = 1.0 if "capture_exact" in types else 0.0\n    features[123] = 1.0 if "capture_overkill" in types else 0.0\n    features[124] = 1.0 if "pressure" in types else 0.0\n    features[125] = 1.0 if "defense" in types else 0.0\n    features[126] = 1.0 if ("support" in types or "consolidate" in types) else 0.0\n    features[127] = 1.0\n\n\ndef _validate_canonical_constraints(\n    ctx: _ObsContext,\n    launches: Iterable[CanonicalLaunch],\n    validate_trajectory: bool = True,\n) -> tuple[bool, str]:\n    committed: dict[int, int] = {}\n    for launch in launches:\n        source = ctx.source_by_id.get(int(launch.src_id))\n        if source is None:\n            return False, f"source {launch.src_id} is not owned"\n        committed[launch.src_id] = committed.get(launch.src_id, 0) + int(launch.ships)\n        if committed[launch.src_id] > int(source[5]):\n            return False, f"source {launch.src_id} overcommitted"\n        if validate_trajectory:\n            hit_id, reason, _eta = _trajectory_first_hit(\n                source,\n                launch.angle,\n                launch.ships,\n                ctx.planets,\n                ctx.av,\n                ctx.comet_ids,\n                ctx.raw_comets,\n            )\n            if hit_id is None and reason == "sun":\n                return False, f"launch from {launch.src_id} crosses sun"\n            actual_hit_id = int(hit_id) if hit_id is not None else -1\n            if actual_hit_id != int(launch.hit_id):\n                return False, f"launch from {launch.src_id} hit changed"\n    return True, ""\n\n\ndef _dedupe_launches(launches: Iterable[CanonicalLaunch]) -> tuple[CanonicalLaunch, ...]:\n    out: list[CanonicalLaunch] = []\n    seen_sources: set[int] = set()\n    for launch in launches:\n        if launch.src_id in seen_sources:\n            continue\n        out.append(launch)\n        seen_sources.add(launch.src_id)\n    return tuple(out)\n\n\ndef _solve_launch_to_target(\n    ctx: _ObsContext,\n    source,\n    target,\n    ships: int,\n    cand_type: str,\n    bucket: str,\n) -> CanonicalLaunch | None:\n    ships = max(1, min(int(ships), int(source[5])))\n    tx, ty, eta_hint = intercept(\n        source[2], source[3], target, ctx.av, ships, ctx.comet_ids, ctx.raw_comets\n    )\n    if tx is None or ty is None:\n        eta_hint = None\n    solved = _solve_engine_angle(\n        source,\n        target,\n        ships,\n        ctx.planets,\n        ctx.av,\n        ctx.comet_ids,\n        ctx.raw_comets,\n        eta_hint=eta_hint,\n    )\n    if solved is None:\n        return None\n    angle, _x, _y, hit_steps = solved\n    hit_id, reason, eta = _trajectory_first_hit(\n        source, angle, ships, ctx.planets, ctx.av, ctx.comet_ids, ctx.raw_comets\n    )\n    if hit_id != target[0]:\n        return None\n    return CanonicalLaunch(\n        src_id=int(source[0]),\n        angle=float(angle),\n        ships=int(ships),\n        hit_id=int(hit_id),\n        hit_reason=str(reason),\n        eta=int(eta or hit_steps),\n        cand_type=cand_type,\n        bucket=bucket,\n        intended_tgt_id=int(target[0]),\n        static_features=_build_static_tuple(\n            ctx,\n            source,\n            target,\n            ships,\n            int(eta or hit_steps),\n            cand_type,\n            bucket,\n        ),\n    )\n\n\ndef _extra_engine_primitives(ctx: _ObsContext, limit: int = 300) -> list[CanonicalLaunch]:\n    """Broaden v16 with engine-confirmed amount buckets for legal targets."""\n    primitives: list[CanonicalLaunch] = []\n    seen: set[tuple] = set()\n    targets = [p for p in ctx.planets if int(p[0]) not in ctx.comet_ids]\n    targets.sort(key=lambda p: (\n        0 if p[1] != ctx.player else 1,\n        -int(p[6]),\n        math.hypot(p[2] - 50.0, p[3] - 50.0),\n        int(p[0]),\n    ))\n    for source in ctx.my_planets:\n        if int(source[0]) in ctx.comet_ids:\n            continue\n        available = int(source[5])\n        if available <= 1:\n            continue\n        for target in targets:\n            if int(target[0]) == int(source[0]):\n                continue\n            if target[1] == ctx.player and available < 5:\n                continue\n            amounts = {\n                max(1, available // 4),\n                max(1, available // 2),\n                max(1, int(available * 0.75)),\n                available,\n            }\n            if target[1] != ctx.player:\n                amounts.add(max(1, min(available, int(target[5]) + 1)))\n            for ships in sorted(amounts):\n                bucket = (\n                    "full_available" if ships == available\n                    else "half_available" if ships <= available // 2\n                    else "x150"\n                )\n                cand_type = "support" if target[1] == ctx.player else "pressure"\n                launch = _solve_launch_to_target(ctx, source, target, ships, cand_type, bucket)\n                if launch is None:\n                    continue\n                key = _launch_key(launch)\n                if key in seen:\n                    continue\n                primitives.append(launch)\n                seen.add(key)\n                if len(primitives) >= limit:\n                    return primitives\n    return primitives\n\n\ndef _primitive_priority(ctx: _ObsContext, launch: CanonicalLaunch) -> tuple:\n    target = ctx.planet_by_id.get(launch.hit_id)\n    target_prod = int(target[6]) if target is not None else 0\n    owner = int(target[1]) if target is not None else -99\n    owner_priority = 0 if owner != ctx.player else 1\n    type_priority = {\n        "capture_exact": 0,\n        "capture_overkill": 1,\n        "pressure": 2,\n        "defense": 3,\n        "support": 4,\n        "consolidate": 5,\n    }.get(launch.cand_type, 6)\n    return (owner_priority, type_priority, -target_prod, launch.eta, launch.src_id, launch.ships)\n\n\ndef _candidate_priority_hint(candidate: OpenCandidate) -> tuple:\n    type_priority = {\n        "capture_exact": 0,\n        "capture_overkill": 1,\n        "pressure": 2,\n        "defense": 3,\n        "support": 4,\n        "consolidate": 5,\n    }.get(candidate.cand_type, 6)\n    static = candidate.static_features\n    eta_hint = float(static[3]) if len(static) > 3 else 999.0\n    prod_hint = float(static[15]) if len(static) > 15 else 0.0\n    ship_frac = float(candidate.ships_to_send) / max(float(candidate.available), 1.0)\n    return (\n        type_priority,\n        -prod_hint,\n        eta_hint,\n        ship_frac,\n        int(candidate.src_id),\n        int(candidate.tgt_id),\n    )\n\n\ndef _replace_launch(\n    base: tuple[CanonicalLaunch, ...],\n    old_idx: int,\n    new_launch: CanonicalLaunch,\n) -> tuple[CanonicalLaunch, ...]:\n    out = list(base)\n    out[old_idx] = new_launch\n    return tuple(out)\n\n\ndef _committed_by_hit(launches: Iterable[CanonicalLaunch]) -> dict[int, int]:\n    out: dict[int, int] = {}\n    for launch in launches:\n        out[launch.hit_id] = out.get(launch.hit_id, 0) + launch.ships\n    return out\n\n\ndef generate_action_set_proposals(\n    obs,\n    base_actions,\n    max_proposals: int = 300,\n    rng: random.Random | None = None,\n    candidate_limit: int | None = None,\n    validate_trajectory: bool = True,\n    allow_engine_primitives: bool = True,\n    trust_candidate_targets: bool = False,\n) -> list[ActionSetProposal]:\n    """Generate valid whole-turn proposals around a teacher action set."""\n    ctx = _obs_context(obs)\n    rng = rng or random.Random(0)\n    proposals: list[ActionSetProposal] = []\n    seen: set = set()\n\n    teacher_launches, _teacher_ok = canonicalize_action_set(obs, base_actions or [])\n    _add_proposal(\n        proposals,\n        seen,\n        ctx,\n        teacher_launches,\n        ("teacher",),\n        is_teacher=True,\n        validate_trajectory=validate_trajectory,\n    )\n\n    open_primitives: list[CanonicalLaunch] = []\n    open_candidates = enumerate_open_candidates(obs)[0]\n    if candidate_limit is not None and candidate_limit > 0 and len(open_candidates) > candidate_limit:\n        open_candidates = sorted(open_candidates, key=_candidate_priority_hint)[:candidate_limit]\n    for candidate in open_candidates:\n        launch = (\n            _trusted_candidate(ctx, candidate)\n            if trust_candidate_targets\n            else _canonical_candidate(ctx, candidate)\n        )\n        if launch is not None and launch.hit_id >= 0:\n            open_primitives.append(launch)\n    engine_primitives: list[CanonicalLaunch] = []\n    if allow_engine_primitives and len(open_primitives) < 50:\n        engine_primitives = _extra_engine_primitives(\n            ctx, limit=max(80, min(max_proposals, 160))\n        )\n\n    primitive_by_key: dict[tuple, CanonicalLaunch] = {}\n    for launch in open_primitives + engine_primitives:\n        primitive_by_key.setdefault(_launch_key(launch), launch)\n    primitives = sorted(primitive_by_key.values(), key=lambda launch: _primitive_priority(ctx, launch))\n\n    # Delay/stop and remove-one-launch branches.\n    _add_proposal(proposals, seen, ctx, (), ("delay_stop",), validate_trajectory=validate_trajectory)\n    for idx, _launch in enumerate(teacher_launches):\n        branch = tuple(launch for j, launch in enumerate(teacher_launches) if j != idx)\n        _add_proposal(\n            proposals,\n            seen,\n            ctx,\n            branch,\n            (f"remove:{idx}",),\n            validate_trajectory=validate_trajectory,\n        )\n\n    # Single primitive branches and teacher-plus-one branches.\n    teacher_sources = {launch.src_id for launch in teacher_launches}\n    for launch in primitives:\n        _add_proposal(\n            proposals,\n            seen,\n            ctx,\n            (launch,),\n            (f"single:{launch.cand_type}",),\n            validate_trajectory=validate_trajectory,\n        )\n        if launch.src_id not in teacher_sources:\n            branch = tuple(list(teacher_launches) + [launch])\n            _add_proposal(\n                proposals,\n                seen,\n                ctx,\n                branch,\n                (f"add:{launch.cand_type}",),\n                validate_trajectory=validate_trajectory,\n            )\n        if len(proposals) >= max_proposals:\n            return proposals[:max_proposals]\n\n    # Resize teacher launches while preserving their actual hit target.\n    for idx, launch in enumerate(teacher_launches):\n        source = ctx.source_by_id.get(launch.src_id)\n        target = ctx.planet_by_id.get(launch.hit_id)\n        if source is None or target is None:\n            continue\n        available = int(source[5])\n        amounts = [\n            launch.ships,\n            max(1, int(math.ceil(launch.ships * 1.25))),\n            max(1, int(math.ceil(launch.ships * 1.50))),\n            max(1, available // 2),\n            available,\n        ]\n        for ships in sorted({min(max(1, amount), available) for amount in amounts}):\n            resized = _solve_launch_to_target(ctx, source, target, ships, "resize", "amount_bucket")\n            if resized is None:\n                continue\n            _add_proposal(\n                proposals,\n                seen,\n                ctx,\n                _replace_launch(teacher_launches, idx, resized),\n                (f"resize:{idx}",),\n                validate_trajectory=validate_trajectory,\n            )\n            if len(proposals) >= max_proposals:\n                return proposals[:max_proposals]\n\n    # Swap teacher launch to another actual-hit target reachable from same source.\n    for idx, launch in enumerate(teacher_launches):\n        swaps = [p for p in primitives if p.src_id == launch.src_id and p.hit_id != launch.hit_id]\n        for swap in swaps[:20]:\n            _add_proposal(\n                proposals,\n                seen,\n                ctx,\n                _replace_launch(teacher_launches, idx, swap),\n                (f"swap:{idx}:{swap.cand_type}",),\n                validate_trajectory=validate_trajectory,\n            )\n            if len(proposals) >= max_proposals:\n                return proposals[:max_proposals]\n\n    # Same-target bundles by actual hit_id.\n    by_hit: dict[int, list[CanonicalLaunch]] = {}\n    for launch in primitives:\n        by_hit.setdefault(launch.hit_id, []).append(launch)\n    for hit_id, group in sorted(by_hit.items(), key=lambda item: (-len(item[1]), item[0])):\n        if hit_id < 0:\n            continue\n        distinct = _dedupe_launches(sorted(group, key=lambda launch: _primitive_priority(ctx, launch)))\n        if len(distinct) < 2:\n            continue\n        for width in (2, 3, 4):\n            if len(distinct) < width:\n                continue\n            branch = distinct[:width]\n            _add_proposal(\n                proposals,\n                seen,\n                ctx,\n                branch,\n                (f"bundle:{hit_id}:{width}",),\n                validate_trajectory=validate_trajectory,\n            )\n            if len(proposals) >= max_proposals:\n                return proposals[:max_proposals]\n\n    # Deterministic diversity combinations.\n    top = primitives[: min(80, len(primitives))]\n    for _ in range(max_proposals * 2):\n        if len(proposals) >= max_proposals or not top:\n            break\n        width = rng.randint(2, min(5, max(2, len(ctx.my_planets))))\n        shuffled = top[:]\n        rng.shuffle(shuffled)\n        branch = _dedupe_launches(shuffled[: width * 2])[:width]\n        _add_proposal(\n            proposals,\n            seen,\n            ctx,\n            branch,\n            ("diversity_combo",),\n            validate_trajectory=validate_trajectory,\n        )\n\n    return proposals[:max_proposals]\n\n\ndef featurize_action_set(obs, actions) -> np.ndarray:\n    ctx = _obs_context(obs)\n    launches, _ok = canonicalize_action_set(obs, actions or [])\n    return featurize_action_set_from_canonical(ctx, launches)\n\n\ndef featurize_action_set_from_canonical(\n    ctx: _ObsContext,\n    launches: tuple[CanonicalLaunch, ...],\n) -> np.ndarray:\n    out = np.zeros(ACTION_SET_FEATURE_DIM, dtype=np.float32)\n    out[: len(ctx.value_globals)] = ctx.value_globals\n\n    static_rows = []\n    for launch in launches:\n        static = launch.static_features or _static_tuple_for_launch(ctx, launch)\n        if static:\n            static_rows.append(np.array(static, dtype=np.float32))\n    if static_rows:\n        mat = np.stack(static_rows)\n        out[12:12 + OPEN_STATIC_DIM] = mat.mean(axis=0)\n\n    agg = np.zeros(50, dtype=np.float32)\n    if not launches:\n        out[62:ACTION_SET_BASE_FEATURE_DIM] = agg\n        return out\n\n    n = len(launches)\n    owned_count = max(len(ctx.my_planets), 1)\n    planet_count = max(len(ctx.planets), 1)\n    available_total = max(sum(int(p[5]) for p in ctx.my_planets), 1)\n    ships = np.array([launch.ships for launch in launches], dtype=np.float32)\n    etas = np.array([launch.eta for launch in launches], dtype=np.float32)\n    unique_sources = {launch.src_id for launch in launches}\n    unique_targets = {launch.hit_id for launch in launches}\n    hit_committed = _committed_by_hit(launches)\n    hit_counts: dict[int, int] = {}\n    for launch in launches:\n        hit_counts[launch.hit_id] = hit_counts.get(launch.hit_id, 0) + 1\n\n    target_rows = [ctx.planet_by_id.get(launch.hit_id) for launch in launches]\n    target_prod = np.array([\n        float(t[6]) if t is not None else 0.0 for t in target_rows\n    ], dtype=np.float32)\n    target_owner = [int(t[1]) if t is not None else -99 for t in target_rows]\n    source_after = []\n    send_frac = []\n    for launch in launches:\n        source = ctx.source_by_id.get(launch.src_id)\n        if source is None:\n            continue\n        committed = sum(item.ships for item in launches if item.src_id == launch.src_id)\n        source_after.append(max(0, int(source[5]) - committed))\n        send_frac.append(launch.ships / max(int(source[5]), 1))\n\n    type_counts: dict[str, int] = {}\n    for launch in launches:\n        type_counts[launch.cand_type] = type_counts.get(launch.cand_type, 0) + 1\n\n    need_ratios = []\n    for hit_id, committed in hit_committed.items():\n        target = ctx.planet_by_id.get(hit_id)\n        if target is None:\n            continue\n        need = max(1.0, float(target[5]) + 1.0)\n        need_ratios.append(min(committed / need, 3.0))\n\n    prod_swing_20 = 0.0\n    prod_swing_40 = 0.0\n    for launch, target in zip(launches, target_rows):\n        if target is None or int(target[1]) == ctx.player:\n            continue\n        prod_swing_20 += float(target[6]) * max(0.0, 20.0 - float(launch.eta))\n        prod_swing_40 += float(target[6]) * max(0.0, 40.0 - float(launch.eta))\n\n    agg[0] = min(n / MAX_ACTIONS_PER_SET, 1.0)\n    agg[1] = min(float(ships.sum()) / 500.0, 1.0)\n    agg[2] = min(float(ships.sum()) / available_total, 2.0) / 2.0\n    agg[3] = len(unique_sources) / owned_count\n    agg[4] = len(unique_targets) / planet_count\n    agg[5] = max(0, n - len(unique_targets)) / max(n, 1)\n    agg[6] = sum(1 for owner in target_owner if owner != ctx.player) / n\n    agg[7] = sum(1 for owner in target_owner if owner == ctx.player) / n\n    agg[8] = type_counts.get("defense", 0) / n\n    agg[9] = type_counts.get("pressure", 0) / n\n    agg[10] = type_counts.get("capture_exact", 0) / n\n    agg[11] = type_counts.get("capture_overkill", 0) / n\n    agg[12] = min(float(etas.max()) / 120.0, 1.0)\n    agg[13] = min(float(etas.mean()) / 120.0, 1.0)\n    agg[14] = min(float(target_prod.max()) / 5.0, 1.0)\n    agg[15] = min(float(target_prod.mean()) / 5.0, 1.0)\n    agg[16] = min(float(ships.mean()) / 100.0, 1.0)\n    agg[17] = min(float(ships.max()) / 100.0, 1.0)\n    agg[18] = float(np.mean(send_frac)) if send_frac else 0.0\n    agg[19] = float(np.max(send_frac)) if send_frac else 0.0\n    agg[20] = sum(1 for owner in target_owner if owner >= 0 and owner != ctx.player) / n\n    agg[21] = sum(1 for owner in target_owner if owner == -1) / n\n    agg[22] = sum(1 for owner in target_owner if owner == ctx.player) / n\n    agg[23] = sum(1 for launch in launches if launch.hit_id in ctx.comet_ids) / n\n    agg[24] = min(float(min(source_after)) / 100.0, 1.0) if source_after else 0.0\n    agg[25] = min(float(np.mean(source_after)) / 100.0, 1.0) if source_after else 0.0\n    agg[26] = sum(1 for launch in launches if ctx.extras["threat_map"].get(launch.hit_id)) / n\n    agg[27] = len([count for count in hit_committed.values() if count > 0]) / planet_count\n    agg[28] = max(need_ratios) / 3.0 if need_ratios else 0.0\n    agg[29] = float(np.mean(need_ratios)) / 3.0 if need_ratios else 0.0\n    agg[30] = min(prod_swing_20 / 500.0, 1.0)\n    agg[31] = min(prod_swing_40 / 1000.0, 1.0)\n    agg[32] = min(float(target_prod.sum()) / 50.0, 1.0)\n    agg[33] = min(float(etas.std()) / 120.0, 1.0)\n    agg[34] = type_counts.get("support", 0) / n\n    agg[35] = type_counts.get("consolidate", 0) / n\n    agg[36] = type_counts.get("resize", 0) / n\n    agg[37] = len([count for count in hit_counts.values() if count > 1]) / max(len(unique_targets), 1)\n    agg[38] = ctx.step / 500.0\n    agg[39] = 1.0 if ctx.step < 75 else 0.0\n    agg[40] = 1.0 if 75 <= ctx.step < 250 else 0.0\n    agg[41] = 1.0 if ctx.step >= 250 else 0.0\n    agg[42] = len(ctx.raw_comets) / 4.0\n    agg[43] = len(ctx.my_planets) / planet_count\n    agg[44] = sum(1 for p in ctx.planets if p[1] == -1) / planet_count\n    agg[45] = min(len(ctx.fleets) / 100.0, 1.0)\n    agg[46] = min(sum(f[6] for f in ctx.fleets if f[1] == ctx.player) / 500.0, 1.0)\n    agg[47] = min(sum(f[6] for f in ctx.fleets if f[1] >= 0 and f[1] != ctx.player) / 500.0, 1.0)\n    agg[48] = 1.0 if any(launch.cand_type == "teacher" for launch in launches) else 0.0\n    agg[49] = 1.0\n\n    out[62:ACTION_SET_BASE_FEATURE_DIM] = agg\n    return out\n\n\n__all__ = [\n    "ACTION_SET_FEATURE_DIM",\n    "ActionSetProposal",\n    "CanonicalLaunch",\n    "action_set_key",\n    "canonicalize_action_set",\n    "featurize_action_set",\n    "generate_action_set_proposals",\n    "validate_action_set",\n]\n'),
    ('agents.v17_value_agent', '"""v17 value-guided proposal ranker.\n\nThis is a development agent for the value-guided policy-improvement loop. It\nkeeps kaggle_ender as the safe fallback, generates a small set of whole-turn\naction-set proposals, and uses the Stage 4 compact ranker to pick among them.\n\nIt is intentionally file-backed for now. Packaging into self-contained\n`main.py` belongs to the later v17 packaging stage.\n"""\nfrom __future__ import annotations\n\nimport base64\nimport io\nimport json\nimport random\nfrom pathlib import Path\n\nimport numpy as np\n\nfrom agents import kaggle_ender_v49\nfrom agents.v17_action_sets import ACTION_SET_FEATURE_DIM, generate_action_set_proposals\n\n\nDEFAULT_WEIGHTS = None\nDEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "v17_linear_ranker.npz"\nDEFAULT_OVERRIDE_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "v17_override.npz"\nDEFAULT_PROPOSAL_CAP = 20\nDEFAULT_CANDIDATE_LIMIT = 40\nDEFAULT_OVERRIDE_MARGIN = 2.00\n_EMBEDDED_RANKER_NPZ_B64 = (\n    "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"\n)\n\n\ndef _get(obj, key: str, default=None):\n    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)\n\n\ndef _obs_seed(obs) -> int:\n    planets = _get(obs, "planets", []) or []\n    fleets = _get(obs, "fleets", []) or []\n    player = int(_get(obs, "player", 0) or 0)\n    step = int(_get(obs, "step", 0) or 0)\n    return (\n        17017\n        + step * 1009\n        + player * 9176\n        + len(planets) * 131\n        + len(fleets) * 37\n    )\n\n\ndef _load_ranker(model_path: str | Path | None):\n    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH\n    try:\n        data = np.load(path)\n    except Exception:\n        return None\n    try:\n        mean = data["mean"].astype(np.float32)\n        scale = data["scale"].astype(np.float32)\n    except Exception:\n        return None\n    if len(mean) != ACTION_SET_FEATURE_DIM or len(scale) != ACTION_SET_FEATURE_DIM:\n        return None\n    scale = scale.copy()\n    scale[scale < 1e-6] = 1.0\n    if "W1" in data:\n        try:\n            return {\n                "type": "mlp",\n                "mean": mean,\n                "scale": scale,\n                "W1": data["W1"].astype(np.float32),\n                "b1": data["b1"].astype(np.float32),\n                "W2": data["W2"].astype(np.float32),\n                "b2": data["b2"].astype(np.float32),\n                "W3": data["W3"].astype(np.float32),\n                "b3": data["b3"].astype(np.float32),\n            }\n        except Exception:\n            return None\n    try:\n        weights = data["w"].astype(np.float32)\n        bias = float(data["b"])\n    except Exception:\n        return None\n    if len(weights) != ACTION_SET_FEATURE_DIM:\n        return None\n    return {"type": "linear", "mean": mean, "scale": scale, "w": weights, "b": bias}\n\n\ndef _load_override_model(model_path: str | Path | None):\n    if model_path is None:\n        return None\n    path = Path(model_path)\n    try:\n        data = np.load(path)\n        mean = data["mean"].astype(np.float32)\n        scale = data["scale"].astype(np.float32)\n        weights = data["w"].astype(np.float32)\n        bias = float(data["b"])\n        threshold = float(data["threshold"])\n    except Exception:\n        return None\n    if len(mean) != ACTION_SET_FEATURE_DIM or len(scale) != ACTION_SET_FEATURE_DIM:\n        return None\n    if len(weights) != ACTION_SET_FEATURE_DIM:\n        return None\n    scale = scale.copy()\n    scale[scale < 1e-6] = 1.0\n    return {\n        "mean": mean,\n        "scale": scale,\n        "w": weights,\n        "b": bias,\n        "threshold": threshold,\n    }\n\n\ndef _score(features: np.ndarray, ranker) -> np.ndarray:\n    x = (features.astype(np.float32) - ranker["mean"]) / ranker["scale"]\n    if ranker.get("type") == "mlp":\n        h1 = np.maximum(x @ ranker["W1"] + ranker["b1"], 0.0)\n        h2 = np.maximum(h1 @ ranker["W2"] + ranker["b2"], 0.0)\n        return (h2 @ ranker["W3"] + ranker["b3"]).reshape(-1)\n    return x @ ranker["w"] + ranker["b"]\n\n\ndef _override_prob(features: np.ndarray, override_model) -> np.ndarray:\n    x = (features.astype(np.float32) - override_model["mean"]) / override_model["scale"]\n    z = np.clip(x @ override_model["w"] + override_model["b"], -60.0, 60.0)\n    return 1.0 / (1.0 + np.exp(-z))\n\n\ndef _canonical_for_log(launch) -> dict:\n    return {\n        "src_id": int(launch.src_id),\n        "angle": float(launch.angle),\n        "ships": int(launch.ships),\n        "hit_id": int(launch.hit_id),\n        "hit_reason": str(launch.hit_reason),\n        "eta": int(launch.eta),\n        "cand_type": str(launch.cand_type),\n        "bucket": str(launch.bucket),\n        "intended_tgt_id": (\n            None if launch.intended_tgt_id is None else int(launch.intended_tgt_id)\n        ),\n    }\n\n\ndef _proposal_for_log(proposal) -> dict:\n    return {\n        "actions": [list(move) for move in proposal.actions],\n        "provenance": list(proposal.provenance),\n        "canonical": [_canonical_for_log(launch) for launch in proposal.canonical],\n        "is_teacher": bool(proposal.is_teacher),\n    }\n\n\ndef _emit_decision_log(\n    decision_logger,\n    decision_log_path: Path | None,\n    payload: dict,\n    *,\n    strict_logging: bool = False,\n) -> None:\n    try:\n        if decision_logger is not None:\n            decision_logger(payload)\n        if decision_log_path is not None:\n            decision_log_path.parent.mkdir(parents=True, exist_ok=True)\n            with open(decision_log_path, "a", encoding="utf-8") as f:\n                f.write(json.dumps(payload, separators=(",", ":"), sort_keys=True))\n                f.write("\\n")\n    except Exception:\n        if strict_logging:\n            raise\n        pass\n\n\ndef make_agent(\n    weights=None,\n    *,\n    model_path: str | Path | None = None,\n    override_model_path: str | Path | None = DEFAULT_OVERRIDE_MODEL_PATH,\n    proposal_cap: int = DEFAULT_PROPOSAL_CAP,\n    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,\n    override_margin: float = DEFAULT_OVERRIDE_MARGIN,\n    decision_log_path: str | Path | None = None,\n    decision_logger=None,\n    strict_logging: bool = False,\n):\n    """Create a guarded value-ranker agent.\n\n    The `weights` positional parameter is accepted for compatibility with the\n    repo\'s tournament loader; it is ignored unless it is a string/path model.\n    """\n    if model_path is None and isinstance(weights, (str, Path)):\n        model_path = weights\n    ranker = _load_ranker(model_path)\n    override_model = _load_override_model(override_model_path)\n    fallback = kaggle_ender_v49.agent\n    cap = max(1, int(proposal_cap))\n    cand_limit = max(1, int(candidate_limit))\n    margin = float(override_margin)\n    log_path = Path(decision_log_path) if decision_log_path is not None else None\n\n    def agent(obs):\n        fallback_actions = fallback(obs)\n        if ranker is None:\n            return fallback_actions\n\n        try:\n            proposals = generate_action_set_proposals(\n                obs,\n                fallback_actions,\n                max_proposals=cap,\n                candidate_limit=cand_limit,\n                validate_trajectory=False,\n                allow_engine_primitives=False,\n                trust_candidate_targets=True,\n                rng=random.Random(_obs_seed(obs)),\n            )\n        except Exception:\n            return fallback_actions\n        if not proposals:\n            return fallback_actions\n\n        baseline_idx = next(\n            (idx for idx, proposal in enumerate(proposals) if proposal.is_teacher),\n            0,\n        )\n        try:\n            features = np.stack([proposal.features for proposal in proposals]).astype(np.float32)\n            scores = _score(features, ranker)\n        except Exception:\n            return fallback_actions\n\n        best_idx = int(np.argmax(scores))\n        if best_idx != baseline_idx and scores[best_idx] > scores[baseline_idx] + margin:\n            score_gap = float(scores[best_idx] - scores[baseline_idx])\n            override_prob = None\n            override_threshold = None\n            if override_model is not None:\n                try:\n                    prob = float(_override_prob(features[best_idx:best_idx + 1], override_model)[0])\n                except Exception:\n                    return fallback_actions\n                override_prob = prob\n                override_threshold = float(override_model["threshold"])\n                if prob < override_threshold:\n                    return fallback_actions\n            chosen_actions = [list(move) for move in proposals[best_idx].actions]\n            if decision_logger is not None or log_path is not None:\n                top_indices = sorted(\n                    range(len(proposals)),\n                    key=lambda idx: float(scores[idx]),\n                    reverse=True,\n                )[:5]\n                _emit_decision_log(\n                    decision_logger,\n                    log_path,\n                    {\n                        "event": "v17_override",\n                        "step": int(_get(obs, "step", 0) or 0),\n                        "player": int(_get(obs, "player", 0) or 0),\n                        "override_margin": margin,\n                        "proposal_count": int(len(proposals)),\n                        "baseline_index": int(baseline_idx),\n                        "best_index": int(best_idx),\n                        "baseline_score": float(scores[baseline_idx]),\n                        "best_score": float(scores[best_idx]),\n                        "score_gap": score_gap,\n                        "override_prob": override_prob,\n                        "override_threshold": override_threshold,\n                        "baseline_features": features[baseline_idx].astype(float).tolist(),\n                        "selected_features": features[best_idx].astype(float).tolist(),\n                        "fallback_actions": [list(move) for move in fallback_actions],\n                        "selected_actions": chosen_actions,\n                        "baseline_proposal": _proposal_for_log(proposals[baseline_idx]),\n                        "selected_proposal": _proposal_for_log(proposals[best_idx]),\n                        "top_candidates": [\n                            {\n                                "index": int(idx),\n                                "score": float(scores[idx]),\n                                "score_gap": float(scores[idx] - scores[baseline_idx]),\n                                "provenance": list(proposals[idx].provenance),\n                                "actions": [list(move) for move in proposals[idx].actions],\n                            }\n                            for idx in top_indices\n                        ],\n                    },\n                    strict_logging=strict_logging,\n                )\n            return chosen_actions\n        return fallback_actions\n\n    return agent\n\n\nagent = make_agent()\n')
]


def _v17_register_module(_name, _source):
    _existing = _v17_sys.modules.get(_name)
    if _existing is not None and getattr(_existing, "__file__", None) not in (None, "main.py"):
        return _existing
    _mod = _v17_types.ModuleType(_name)
    _mod.__file__ = "main.py"
    _mod.__package__ = _name.rpartition(".")[0]
    _v17_sys.modules[_name] = _mod
    if "." in _name:
        _parent_name, _child_name = _name.rsplit(".", 1)
        setattr(_v17_sys.modules[_parent_name], _child_name, _mod)
    exec(compile(_source, _name, "exec"), _mod.__dict__)
    return _mod


_V17_LOCAL_IMPORT_GUARD_MODULES = {
    "agents.policy_features",
    "agents.scored_agent",
    "agents.v16_open_features",
    "agents.v17_action_sets",
    "agents.v17_value_agent",
}
_v17_local_agents_active = any(
    _name in _V17_LOCAL_IMPORT_GUARD_MODULES
    and getattr(_mod, "__file__", None) not in (None, "main.py")
    for _name, _mod in list(_v17_sys.modules.items())
)

if not _v17_local_agents_active:
    for _v17_module_name, _v17_module_source in _V17_MODULE_SOURCES:
        _v17_register_module(_v17_module_name, _v17_module_source)

_V17_RANKER_NPZ_B64 = """\
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
7p8AAAAA
"""

_V17_GATE_NPZ_B64 = """\
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
ZS5ucHlQSwUGAAAAAAYABgBKAQAA+QgAAAAA
"""


def _v17_npz_from_b64(_blob):
    return _v17_np.load(_v17_io.BytesIO(_v17_b64.b64decode(_blob.replace("\n", ""))))


def _v17_embedded_ranker(_model_path=None):
    _data = _v17_npz_from_b64(_V17_RANKER_NPZ_B64)
    _mean = _data["mean"].astype(_v17_np.float32)
    _scale = _data["scale"].astype(_v17_np.float32)
    _scale = _scale.copy()
    _scale[_scale < 1e-6] = 1.0
    return {
        "type": "mlp",
        "mean": _mean,
        "scale": _scale,
        "W1": _data["W1"].astype(_v17_np.float32),
        "b1": _data["b1"].astype(_v17_np.float32),
        "W2": _data["W2"].astype(_v17_np.float32),
        "b2": _data["b2"].astype(_v17_np.float32),
        "W3": _data["W3"].astype(_v17_np.float32),
        "b3": _data["b3"].astype(_v17_np.float32),
    }


def _v17_embedded_gate(_model_path=None):
    _data = _v17_npz_from_b64(_V17_GATE_NPZ_B64)
    _mean = _data["mean"].astype(_v17_np.float32)
    _scale = _data["scale"].astype(_v17_np.float32)
    _scale = _scale.copy()
    _scale[_scale < 1e-6] = 1.0
    return {
        "mean": _mean,
        "scale": _scale,
        "w": _data["w"].astype(_v17_np.float32),
        "b": float(_data["b"]),
        "threshold": float(_data["threshold"]),
    }


_v17_agent_mod = _v17_sys.modules.get("agents.v17_value_agent")
if _v17_agent_mod is not None and hasattr(_v17_agent_mod, "make_agent"):
    _v17_agent_mod._load_ranker = _v17_embedded_ranker
    _v17_agent_mod._load_override_model = _v17_embedded_gate

    agent = _v17_agent_mod.make_agent(
        model_path=None,
        override_model_path="embedded",
        override_margin=15.0,
    )
    _v17_sys.modules["main"].__dict__.update(globals())
