"""Candidate launch generation for the production-first graph track."""

from __future__ import annotations

import math
from dataclasses import dataclass

from main import _solve_engine_angle, _trajectory_first_hit, intercept

from .state import (
    MAX_TURNS,
    GraphState,
    available_ships,
    refine_capture_need,
)
from .geometry_cache import GeometryCache


@dataclass(frozen=True)
class LaunchCandidate:
    src_id: int
    intended_tgt_id: int
    actual_hit_id: int | None
    angle: float
    ships: int
    eta: int
    hit_reason: str
    kind: str
    bucket: str
    score: float
    required_ships: int
    target_production: float
    target_owner: int

    def action(self) -> list:
        return [self.src_id, self.angle, self.ships]

    @property
    def hits_intended_target(self) -> bool:
        return self.actual_hit_id == self.intended_tgt_id


def canonicalize_launch(
    ctx: GraphState,
    source: list,
    target: list,
    ships: int,
    *,
    kind: str,
    bucket: str,
    score: float,
    required_ships: int,
    allow_off_target: bool = False,
    solve_cache: dict[tuple[int, int, int], tuple[float, int | None, str, int]] | None = None,
    max_steps: int = 140,
) -> LaunchCandidate | None:
    """Solve an engine-confirmed launch and record what it actually hits."""

    ships = int(max(1, min(ships, int(source[5]))))
    key = (int(source[0]), int(target[0]), ships)
    cached = solve_cache.get(key) if solve_cache is not None else None
    if cached is not None:
        angle, hit_id, reason, hit_steps = cached
        if hit_id != int(target[0]) and not allow_off_target:
            return None
        return LaunchCandidate(
            src_id=int(source[0]),
            intended_tgt_id=int(target[0]),
            actual_hit_id=None if hit_id is None else int(hit_id),
            angle=float(angle),
            ships=ships,
            eta=int(hit_steps),
            hit_reason=str(reason),
            kind=kind,
            bucket=bucket,
            score=float(score),
            required_ships=max(1, int(required_ships)),
            target_production=float(target[6]),
            target_owner=int(target[1]),
        )

    tx, ty, eta_hint = intercept(
        float(source[2]),
        float(source[3]),
        target,
        ctx.angular_velocity,
        ships,
        ctx.comet_ids,
        ctx.raw_comets,
    )
    if tx is None or eta_hint is None:
        return None
    solved = _solve_engine_angle(
        source,
        target,
        ships,
        ctx.planets,
        ctx.angular_velocity,
        ctx.comet_ids,
        ctx.raw_comets,
        eta_hint=eta_hint,
        max_steps=max_steps,
    )
    if solved is None:
        return None
    angle, _tx, _ty, eta = solved
    hit_id, reason, hit_steps = _trajectory_first_hit(
        source,
        angle,
        ships,
        ctx.planets,
        ctx.angular_velocity,
        ctx.comet_ids,
        ctx.raw_comets,
        max_steps=max_steps,
    )
    if solve_cache is not None:
        solve_cache[key] = (
            float(angle),
            None if hit_id is None else int(hit_id),
            str(reason),
            int(hit_steps),
        )
    if hit_id != int(target[0]) and not allow_off_target:
        return None
    return LaunchCandidate(
        src_id=int(source[0]),
        intended_tgt_id=int(target[0]),
        actual_hit_id=None if hit_id is None else int(hit_id),
        angle=float(angle),
        ships=ships,
        eta=int(hit_steps),
        hit_reason=str(reason),
        kind=kind,
        bucket=bucket,
        score=float(score),
        required_ships=max(1, int(required_ships)),
        target_production=float(target[6]),
        target_owner=int(target[1]),
    )


def _attack_score(
    ctx: GraphState,
    source: list,
    target: list,
    eta: int,
    ships: int,
    required: int,
    available: int,
    kind: str,
) -> float:
    remaining = max(0.0, MAX_TURNS - ctx.step - eta)
    prod = float(target[6])
    prod_value = prod * remaining / 45.0
    eta_penalty = eta * 0.08
    cost_penalty = 1.3 * ships / max(available, 1)
    owner_bonus = 1.5 if int(target[1]) == -1 else 0.7
    affordability = min(available / max(required, 1), 2.0)
    source_surplus = (float(source[5]) - 12.0) / 100.0
    pressure_scale = 0.45 if kind == "pressure" else 1.0
    overkill_penalty = max(0.0, ships - required * 1.5) / max(available, 1)
    return pressure_scale * (
        prod_value
        + prod * 2.0
        + owner_bonus
        + affordability
        + source_surplus
        - eta_penalty
        - cost_penalty
        - overkill_penalty
    )


def _defense_score(ctx: GraphState, source: list, target: list, eta: int, ships: int, required: int) -> float:
    prod = float(target[6])
    urgency = max(0.0, 1.0 - eta / 40.0)
    return 12.0 + prod * 3.0 + urgency * 6.0 - ships / max(float(source[5]), 1.0)


def _support_score(ctx: GraphState, source: list, target: list, eta: int, ships: int) -> float:
    prod_delta = float(target[6]) - float(source[6])
    source_surplus = max(0.0, float(source[5]) - 30.0) / 60.0
    return 0.8 + prod_delta * 0.8 + source_surplus - eta * 0.04 - ships / 120.0


def _bucket_amounts(required: int, available: int) -> list[tuple[str, int]]:
    amounts = [("exact", required)]
    x125 = int(math.ceil(required * 1.25))
    x150 = int(math.ceil(required * 1.50))
    if x125 <= available and x125 > required:
        amounts.append(("x125", x125))
    if x150 <= available and x150 > x125:
        amounts.append(("x150", x150))
    if available > max(required * 2, 12):
        amounts.append(("full_available", available))
    return amounts


def _pressure_amounts(available: int) -> list[tuple[str, int]]:
    out = []
    if available >= 4:
        out.append(("half_available", max(1, available // 2)))
    if available >= 2:
        out.append(("full_available", available))
    return out


def generate_launch_candidates(
    ctx: GraphState,
    *,
    top_targets_per_source: int = 3,
    max_candidates: int = 40,
    include_support: bool = True,
) -> list[LaunchCandidate]:
    """Generate engine-confirmed launch candidates from graph state."""

    candidates: list[LaunchCandidate] = []
    solve_cache: dict[tuple[int, int, int], tuple[float, int | None, str, int]] = {}
    planet_by_id = ctx.planet_by_id
    my_planets = ctx.my_planets
    non_owned = [p for p in ctx.planets if int(p[1]) != ctx.player and int(p[0]) not in ctx.comet_ids]

    for source in my_planets:
        if int(source[0]) in ctx.comet_ids:
            available = available_ships(source, ctx.comet_ids)
        else:
            available = available_ships(source, ctx.comet_ids)
        if available <= 1:
            continue

        rough_attacks = []
        for target in non_owned:
            estimate = refine_capture_need(ctx, source, target, available)
            if estimate is None or not estimate.rough_path_clear:
                continue
            rough_score = _attack_score(
                ctx,
                source,
                target,
                estimate.eta,
                min(estimate.required_ships, available),
                estimate.required_ships,
                available,
                "capture",
            )
            rough_attacks.append((rough_score, target, estimate))
        rough_attacks.sort(key=lambda item: item[0], reverse=True)

        for rough_score, target, estimate in rough_attacks[:top_targets_per_source]:
            required = max(1, int(estimate.required_ships))
            if required <= available:
                for bucket, amount in _bucket_amounts(required, available):
                    score = _attack_score(
                        ctx,
                        source,
                        target,
                        estimate.eta,
                        amount,
                        required,
                        available,
                        "capture",
                    )
                    kind = "capture_exact" if bucket == "exact" else "capture_overkill"
                    candidate = canonicalize_launch(
                        ctx,
                        source,
                        target,
                        amount,
                        kind=kind,
                        bucket=bucket,
                        score=score,
                        required_ships=required,
                        solve_cache=solve_cache,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
            elif float(target[6]) >= 3.0 or int(target[1]) >= 0:
                for bucket, amount in _pressure_amounts(available):
                    score = _attack_score(
                        ctx,
                        source,
                        target,
                        estimate.eta,
                        amount,
                        required,
                        available,
                        "pressure",
                    )
                    candidate = canonicalize_launch(
                        ctx,
                        source,
                        target,
                        amount,
                        kind="pressure",
                        bucket=bucket,
                        score=score,
                        required_ships=required,
                        solve_cache=solve_cache,
                    )
                    if candidate is not None:
                        candidates.append(candidate)

        for target in my_planets:
            if int(target[0]) == int(source[0]):
                continue
            enemy_events = [e for e in ctx.inbound_events.get(int(target[0]), []) if e.owner != ctx.player]
            if enemy_events:
                first_enemy_eta = min(e.eta for e in enemy_events)
                threat_ships = sum(e.ships for e in enemy_events if e.eta <= first_enemy_eta + 2)
                friendly_ships = sum(e.ships for e in ctx.inbound_events.get(int(target[0]), []) if e.owner == ctx.player and e.eta <= first_enemy_eta)
                garrison = float(target[5]) + float(target[6]) * first_enemy_eta + friendly_ships
                need = max(1, int(math.ceil(threat_ships - garrison + 1.0)))
                if need > 0:
                    for bucket, amount in (("exact", min(need, available)), ("full_available", available)):
                        if amount <= 0:
                            continue
                        estimate = refine_capture_need(ctx, source, target, available)
                        eta = first_enemy_eta if estimate is None else estimate.eta
                        score = _defense_score(ctx, source, target, eta, amount, need)
                        candidate = canonicalize_launch(
                            ctx,
                            source,
                            target,
                            amount,
                            kind="defense",
                            bucket=bucket,
                            score=score,
                            required_ships=need,
                            solve_cache=solve_cache,
                        )
                        if candidate is not None:
                            candidates.append(candidate)

            if include_support and available >= 8 and float(source[5]) > float(target[5]) + 15.0:
                edge = ctx.edge_estimates.get((int(source[0]), int(target[0])))
                eta = 60 if edge is None else edge.eta
                for bucket, amount in (("half_available", max(1, available // 2)), ("full_available", available)):
                    score = _support_score(ctx, source, target, eta, amount)
                    if score <= 0.0:
                        continue
                    candidate = canonicalize_launch(
                        ctx,
                        source,
                        target,
                        amount,
                        kind="support" if float(target[6]) >= float(source[6]) else "consolidate",
                        bucket=bucket,
                        score=score,
                        required_ships=amount,
                        solve_cache=solve_cache,
                    )
                    if candidate is not None:
                        candidates.append(candidate)

    dedup: dict[tuple[int, int | None, int, str, str], LaunchCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.src_id,
            candidate.actual_hit_id,
            candidate.ships,
            candidate.kind,
            candidate.bucket,
        )
        old = dedup.get(key)
        if old is None or candidate.score > old.score:
            dedup[key] = candidate
    return sorted(dedup.values(), key=lambda item: item.score, reverse=True)[:max_candidates]


def _friendly_before_eta(ctx: GraphState, target_id: int, eta: int) -> int:
    return sum(
        event.ships
        for event in ctx.inbound_events.get(int(target_id), [])
        if event.owner == ctx.player and event.eta <= eta
    )


def _enemy_before_eta(ctx: GraphState, target_id: int, eta: int) -> int:
    return sum(
        event.ships
        for event in ctx.inbound_events.get(int(target_id), [])
        if event.owner != ctx.player and event.eta <= eta
    )


def _required_at_eta(ctx: GraphState, target: list, eta: int) -> int:
    friendly = _friendly_before_eta(ctx, int(target[0]), eta)
    arrival_garrison = float(target[5])
    if int(target[1]) >= 0:
        arrival_garrison += float(target[6]) * int(eta)
    return max(1, int(math.ceil(arrival_garrison + 1.0 - friendly)))


def _bucket_label(ships: int, required: int, available: int) -> str:
    if ships >= available:
        return "full_available"
    if ships <= required:
        return "exact"
    if ships <= int(math.ceil(required * 1.25)):
        return "x125"
    if ships <= int(math.ceil(required * 1.50)):
        return "x150"
    return f"bucket_{ships}"


def generate_cached_launch_candidates(
    ctx: GraphState,
    geometry_cache: GeometryCache,
    *,
    max_candidates: int = 80,
    include_support: bool = True,
    max_chase_ratio: float = 0.0,
    chase_eta_allow: int = 12,
    nearest_step: bool = True,
) -> list[LaunchCandidate]:
    """Generate candidates from a precomputed geometry cache.

    This is the training-time path: no engine angle solving happens here.  The
    action space is restricted to precomputed ship buckets whose cached geometry
    reaches the intended target.
    """

    candidates: list[LaunchCandidate] = []
    my_planets = ctx.my_planets
    non_owned = [p for p in ctx.planets if int(p[1]) != ctx.player and int(p[0]) not in ctx.comet_ids]
    bucket_values = [int(x) for x in geometry_cache.ship_buckets.tolist()]

    for source in my_planets:
        source_id = int(source[0])
        available = available_ships(source, ctx.comet_ids)
        if available <= 1:
            continue

        for target in non_owned:
            target_id = int(target[0])
            for bucket in bucket_values:
                if bucket > available:
                    continue
                geom = geometry_cache.lookup(
                    ctx.step,
                    source_id,
                    target_id,
                    bucket,
                    nearest_step=nearest_step,
                    nearest_bucket=False,
                )
                if geom is None or not geom.useful or not geom.reachable:
                    continue
                if geom.actual_hit_id != target_id or geom.angle is None:
                    continue
                if (
                    max_chase_ratio > 0.0
                    and geom.chase_ratio > max_chase_ratio
                    and geom.eta > chase_eta_allow
                ):
                    continue

                required = _required_at_eta(ctx, target, geom.eta)
                if bucket >= required:
                    kind = "capture_exact" if bucket <= int(math.ceil(required * 1.15)) else "capture_overkill"
                elif float(target[6]) >= 3.0 or int(target[1]) >= 0:
                    kind = "pressure"
                else:
                    continue
                score = _attack_score(
                    ctx,
                    source,
                    target,
                    geom.eta,
                    bucket,
                    required,
                    available,
                    "pressure" if kind == "pressure" else "capture",
                )
                candidates.append(
                    LaunchCandidate(
                        src_id=source_id,
                        intended_tgt_id=target_id,
                        actual_hit_id=geom.actual_hit_id,
                        angle=float(geom.angle),
                        ships=int(bucket),
                        eta=int(geom.eta),
                        hit_reason=str(geom.hit_reason),
                        kind=kind,
                        bucket=_bucket_label(bucket, required, available),
                        score=score,
                        required_ships=required,
                        target_production=float(target[6]),
                        target_owner=int(target[1]),
                    )
                )

        for target in my_planets:
            target_id = int(target[0])
            if target_id == source_id:
                continue
            enemy_events = [
                event
                for event in ctx.inbound_events.get(target_id, [])
                if event.owner != ctx.player
            ]
            first_enemy_eta = min((event.eta for event in enemy_events), default=None)
            threat_need = 0
            if first_enemy_eta is not None:
                threat_ships = sum(event.ships for event in enemy_events if event.eta <= first_enemy_eta + 2)
                friendly = _friendly_before_eta(ctx, target_id, first_enemy_eta)
                garrison = float(target[5]) + float(target[6]) * first_enemy_eta + friendly
                threat_need = max(1, int(math.ceil(threat_ships - garrison + 1.0)))

            for bucket in bucket_values:
                if bucket > available:
                    continue
                geom = geometry_cache.lookup(
                    ctx.step,
                    source_id,
                    target_id,
                    bucket,
                    nearest_step=nearest_step,
                    nearest_bucket=False,
                )
                if geom is None or not geom.useful or not geom.reachable:
                    continue
                if geom.actual_hit_id != target_id or geom.angle is None:
                    continue
                if first_enemy_eta is not None and threat_need > 0:
                    if geom.eta > first_enemy_eta + 5:
                        continue
                    score = _defense_score(ctx, source, target, geom.eta, bucket, threat_need)
                    kind = "defense"
                    required = threat_need
                elif include_support and float(source[5]) > float(target[5]) + 15.0:
                    score = _support_score(ctx, source, target, geom.eta, bucket)
                    if score <= 0.0:
                        continue
                    kind = "support" if float(target[6]) >= float(source[6]) else "consolidate"
                    required = bucket
                else:
                    continue
                candidates.append(
                    LaunchCandidate(
                        src_id=source_id,
                        intended_tgt_id=target_id,
                        actual_hit_id=geom.actual_hit_id,
                        angle=float(geom.angle),
                        ships=int(bucket),
                        eta=int(geom.eta),
                        hit_reason=str(geom.hit_reason),
                        kind=kind,
                        bucket=_bucket_label(bucket, required, available),
                        score=score,
                        required_ships=max(1, int(required)),
                        target_production=float(target[6]),
                        target_owner=int(target[1]),
                    )
                )

    dedup: dict[tuple[int, int | None, int, str], LaunchCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.src_id,
            candidate.actual_hit_id,
            candidate.ships,
            candidate.kind,
        )
        old = dedup.get(key)
        if old is None or candidate.score > old.score:
            dedup[key] = candidate
    return sorted(dedup.values(), key=lambda item: item.score, reverse=True)[:max_candidates]
