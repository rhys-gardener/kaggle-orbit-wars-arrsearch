"""Precomputed geometry lookup for array-only graph training.

Geometry in Orbit Wars depends on:

* current turn / planet positions
* source planet id
* intended target planet id
* fleet ship count, through ``fleet_speed(ships)``

It does not depend on ownership or current garrison.  This makes it a good
candidate for heavy preprocessing.  The lookup is intentionally bucketed by
ship count: training can operate on bucketed launch amounts, then a later exact
candidate-labeling pass can refine any promising action.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.physics import _solve_engine_angle, _trajectory_first_hit, intercept


REASON_CODES = {
    "none": 0,
    "fleet": 1,
    "sweep": 2,
    "sun": 3,
    "bounds": 4,
    "max_steps": 5,
    "no_intercept": 6,
    "no_solution": 7,
    "dominated_speed_bucket": 8,
    "chase_pruned": 9,
}
CODE_REASONS = {value: key for key, value in REASON_CODES.items()}


@dataclass(frozen=True)
class GeometryResult:
    step: int
    source_id: int
    target_id: int
    ship_bucket: int
    reachable: bool
    angle: float | None
    eta: int
    actual_hit_id: int | None
    hit_reason: str
    useful: bool
    chase_ratio: float


@dataclass
class GeometryCache:
    steps: np.ndarray
    planet_ids: np.ndarray
    ship_buckets: np.ndarray
    angle: np.ndarray
    eta: np.ndarray
    hit_index: np.ndarray
    reason: np.ndarray
    reachable: np.ndarray
    useful: np.ndarray
    chase_ratio: np.ndarray
    metadata: dict[str, Any]

    def lookup(
        self,
        step: int,
        source_id: int,
        target_id: int,
        ships: int,
        *,
        nearest_step: bool = True,
        nearest_bucket: bool = True,
    ) -> GeometryResult | None:
        s_idx = _lookup_index(self.steps, int(step), nearest=nearest_step)
        b_idx = _lookup_index(self.ship_buckets, int(ships), nearest=nearest_bucket)
        src_idx = _exact_index(self.planet_ids, int(source_id))
        tgt_idx = _exact_index(self.planet_ids, int(target_id))
        if s_idx is None or b_idx is None or src_idx is None or tgt_idx is None:
            return None
        hit_idx = int(self.hit_index[s_idx, b_idx, src_idx, tgt_idx])
        hit_id = None if hit_idx < 0 else int(self.planet_ids[hit_idx])
        eta = int(self.eta[s_idx, b_idx, src_idx, tgt_idx])
        angle = float(self.angle[s_idx, b_idx, src_idx, tgt_idx])
        if not math.isfinite(angle):
            angle_value = None
        else:
            angle_value = angle
        reason_code = int(self.reason[s_idx, b_idx, src_idx, tgt_idx])
        return GeometryResult(
            step=int(self.steps[s_idx]),
            source_id=int(source_id),
            target_id=int(target_id),
            ship_bucket=int(self.ship_buckets[b_idx]),
            reachable=bool(self.reachable[s_idx, b_idx, src_idx, tgt_idx]),
            angle=angle_value,
            eta=eta,
            actual_hit_id=hit_id,
            hit_reason=CODE_REASONS.get(reason_code, "unknown"),
            useful=bool(self.useful[s_idx, b_idx, src_idx, tgt_idx]),
            chase_ratio=float(self.chase_ratio[s_idx, b_idx, src_idx, tgt_idx]),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            steps=self.steps.astype(np.int16, copy=False),
            planet_ids=self.planet_ids.astype(np.int16, copy=False),
            ship_buckets=self.ship_buckets.astype(np.int16, copy=False),
            angle=self.angle.astype(np.float32, copy=False),
            eta=self.eta.astype(np.int16, copy=False),
            hit_index=self.hit_index.astype(np.int16, copy=False),
            reason=self.reason.astype(np.uint8, copy=False),
            reachable=self.reachable.astype(np.bool_, copy=False),
            useful=self.useful.astype(np.bool_, copy=False),
            chase_ratio=self.chase_ratio.astype(np.float32, copy=False),
            metadata=json.dumps(self.metadata, sort_keys=True),
        )


def load_geometry_cache(path: str | Path) -> GeometryCache:
    data = np.load(path, allow_pickle=False)
    metadata_raw = data["metadata"]
    metadata = json.loads(str(metadata_raw.tolist()))
    return GeometryCache(
        steps=data["steps"],
        planet_ids=data["planet_ids"],
        ship_buckets=data["ship_buckets"],
        angle=data["angle"],
        eta=data["eta"],
        hit_index=data["hit_index"],
        reason=data["reason"],
        reachable=data["reachable"],
        useful=data["useful"] if "useful" in data else data["reachable"],
        chase_ratio=data["chase_ratio"] if "chase_ratio" in data else np.zeros_like(data["angle"], dtype=np.float32),
        metadata=metadata,
    )


def _lookup_index(values: np.ndarray, value: int, *, nearest: bool) -> int | None:
    exact = np.where(values == value)[0]
    if len(exact):
        return int(exact[0])
    if not nearest or len(values) == 0:
        return None
    return int(np.argmin(np.abs(values.astype(np.int64) - int(value))))


def _exact_index(values: np.ndarray, value: int) -> int | None:
    exact = np.where(values == value)[0]
    if len(exact):
        return int(exact[0])
    return None


def _reason_code(reason: str) -> int:
    return REASON_CODES.get(str(reason), REASON_CODES["none"])


def solve_geometry_entry(
    source: list,
    target: list,
    planets: list[list],
    angular_velocity: float,
    comet_ids: set[int],
    raw_comets: list,
    ships: int,
    *,
    max_steps: int = 160,
) -> tuple[float, int, int | None, str, bool]:
    """Return angle, eta, hit_id, reason, reachable for one bucketed edge."""

    tx, ty, eta_hint = intercept(
        float(source[2]),
        float(source[3]),
        target,
        angular_velocity,
        int(ships),
        comet_ids,
        raw_comets,
    )
    if tx is None or eta_hint is None:
        return math.nan, -1, None, "no_intercept", False

    solved = _solve_engine_angle(
        source,
        target,
        int(ships),
        planets,
        angular_velocity,
        comet_ids,
        raw_comets,
        eta_hint=eta_hint,
        max_steps=max_steps,
    )
    if solved is not None:
        angle, _tx, _ty, _eta = solved
    else:
        angle = math.atan2(float(ty) - float(source[3]), float(tx) - float(source[2]))

    hit_id, reason, hit_steps = _trajectory_first_hit(
        source,
        float(angle),
        int(ships),
        planets,
        angular_velocity,
        comet_ids,
        raw_comets,
        max_steps=max_steps,
    )
    reachable = hit_id == int(target[0])
    if solved is None and not reachable:
        reason = "no_solution" if reason in ("max_steps", "bounds", "sun") else reason
    return float(angle), int(hit_steps), None if hit_id is None else int(hit_id), str(reason), bool(reachable)


def cheap_geometry_probe(
    source: list,
    target: list,
    angular_velocity: float,
    comet_ids: set[int],
    raw_comets: list,
    ships: int,
) -> tuple[int, float] | None:
    """Return cheap ETA and chase ratio without engine trajectory solving."""

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
    direct_dist = math.hypot(float(target[2]) - float(source[2]), float(target[3]) - float(source[3]))
    solved_dist = math.hypot(float(tx) - float(source[2]), float(ty) - float(source[3]))
    chase_ratio = solved_dist / max(direct_dist, 1e-6)
    return int(eta), float(chase_ratio)


def build_geometry_cache_from_observations(
    observations: list[dict[str, Any]],
    ship_buckets: list[int],
    *,
    max_steps: int = 160,
    max_pairs_per_step: int = 0,
    keep_only_eta_improvements: bool = False,
    min_eta_improvement: float = 1.0,
    max_chase_ratio: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> GeometryCache:
    """Build a geometry cache from sampled observations.

    Each observation supplies one absolute game step and the planet positions at
    that step.  The same shared planet state is sufficient for every player.
    """

    if not observations:
        raise ValueError("observations must not be empty")

    by_step: dict[int, dict[str, Any]] = {}
    for obs in observations:
        by_step[int(obs.get("step", 0) or 0)] = obs
    steps = np.array(sorted(by_step), dtype=np.int16)
    all_planet_ids = sorted(
        {
            int(p[0])
            for obs in by_step.values()
            for p in (obs.get("planets", []) or [])
        }
    )
    planet_ids = np.array(all_planet_ids, dtype=np.int16)
    bucket_arr = np.array(sorted({max(1, int(x)) for x in ship_buckets}), dtype=np.int16)

    s_count = len(steps)
    b_count = len(bucket_arr)
    p_count = len(planet_ids)
    shape = (s_count, b_count, p_count, p_count)
    angle = np.full(shape, np.nan, dtype=np.float32)
    eta = np.full(shape, -1, dtype=np.int16)
    hit_index = np.full(shape, -1, dtype=np.int16)
    reason = np.full(shape, REASON_CODES["none"], dtype=np.uint8)
    reachable = np.zeros(shape, dtype=np.bool_)
    useful = np.zeros(shape, dtype=np.bool_)
    chase_ratio = np.zeros(shape, dtype=np.float32)

    id_to_index = {int(pid): idx for idx, pid in enumerate(planet_ids)}

    for s_idx, step in enumerate(steps.tolist()):
        obs = by_step[int(step)]
        planets = [list(p) for p in obs.get("planets", [])]
        av = float(obs.get("angular_velocity", 0.035) or 0.035)
        comet_ids = {int(x) for x in (obs.get("comet_planet_ids", []) or [])}
        raw_comets = list(obs.get("comets", []) or [])
        pairs_done = 0
        for source in planets:
            src_idx = id_to_index.get(int(source[0]))
            if src_idx is None:
                continue
            for target in planets:
                tgt_idx = id_to_index.get(int(target[0]))
                if tgt_idx is None:
                    continue
                if int(source[0]) == int(target[0]):
                    continue
                if max_pairs_per_step and pairs_done >= max_pairs_per_step:
                    break
                best_kept_eta: int | None = None
                for b_idx, ships in enumerate(bucket_arr.tolist()):
                    probe = cheap_geometry_probe(
                        source,
                        target,
                        av,
                        comet_ids,
                        raw_comets,
                        int(ships),
                    )
                    if probe is None:
                        eta[s_idx, b_idx, src_idx, tgt_idx] = -1
                        reason[s_idx, b_idx, src_idx, tgt_idx] = REASON_CODES["no_intercept"]
                        continue
                    cheap_eta, cheap_chase_ratio = probe
                    eta[s_idx, b_idx, src_idx, tgt_idx] = int(cheap_eta)
                    chase_ratio[s_idx, b_idx, src_idx, tgt_idx] = float(cheap_chase_ratio)

                    if max_chase_ratio > 0.0 and cheap_chase_ratio > max_chase_ratio:
                        reason[s_idx, b_idx, src_idx, tgt_idx] = REASON_CODES["chase_pruned"]
                        continue
                    if (
                        keep_only_eta_improvements
                        and best_kept_eta is not None
                        and cheap_eta > best_kept_eta - float(min_eta_improvement)
                    ):
                        reason[s_idx, b_idx, src_idx, tgt_idx] = REASON_CODES["dominated_speed_bucket"]
                        continue

                    item = solve_geometry_entry(
                        source,
                        target,
                        planets,
                        av,
                        comet_ids,
                        raw_comets,
                        int(ships),
                        max_steps=max_steps,
                    )
                    angle_value, eta_value, hit_id, reason_text, is_reachable = item
                    angle[s_idx, b_idx, src_idx, tgt_idx] = angle_value
                    eta[s_idx, b_idx, src_idx, tgt_idx] = eta_value
                    hit_index[s_idx, b_idx, src_idx, tgt_idx] = -1 if hit_id is None else id_to_index.get(hit_id, -1)
                    reason[s_idx, b_idx, src_idx, tgt_idx] = _reason_code(reason_text)
                    reachable[s_idx, b_idx, src_idx, tgt_idx] = is_reachable
                    useful[s_idx, b_idx, src_idx, tgt_idx] = True
                    if is_reachable and (best_kept_eta is None or eta_value < best_kept_eta):
                        best_kept_eta = eta_value
                pairs_done += 1
            if max_pairs_per_step and pairs_done >= max_pairs_per_step:
                break

    return GeometryCache(
        steps=steps,
        planet_ids=planet_ids,
        ship_buckets=bucket_arr,
        angle=angle,
        eta=eta,
        hit_index=hit_index,
        reason=reason,
        reachable=reachable,
        useful=useful,
        chase_ratio=chase_ratio,
        metadata=dict(metadata or {}),
    )
