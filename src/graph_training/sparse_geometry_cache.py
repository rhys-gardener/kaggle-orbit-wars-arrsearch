"""Persistent sparse exact geometry lookup.

Dense geometry precomputes every ``(step, bucket, source, target)`` entry.
This cache keeps the same ``lookup`` surface but solves exact entries on demand
and can be persisted between repeated training games on the same scenario.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.array_search.scenarios import InitialScenario
from src.array_search.state_adapter import arrays_to_obs
from src.graph_training.geometry_cache import GeometryResult, cheap_geometry_probe, solve_geometry_entry


DEFAULT_SHIP_BUCKETS = (
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    80,
    96,
    112,
    128,
    160,
    192,
    224,
    256,
)


class SparseGeometryCache:
    """GeometryCache-compatible sparse lookup for one ``InitialScenario``."""

    def __init__(
        self,
        scenario: InitialScenario,
        ship_buckets: list[int] | tuple[int, ...] = DEFAULT_SHIP_BUCKETS,
        *,
        max_steps: int = 160,
        max_chase_ratio: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.scenario = scenario
        self.ship_buckets = np.array(sorted({max(1, int(x)) for x in ship_buckets}), dtype=np.int16)
        self.planet_ids = scenario.planet_ids.astype(np.int16, copy=False)
        self.steps = np.array([], dtype=np.int16)
        self.max_steps = int(max_steps)
        self.max_chase_ratio = float(max_chase_ratio)
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("geometry_mode", "sparse")
        self.metadata.setdefault("seed", int(scenario.seed))
        self._obs_by_step: dict[int, dict[str, Any]] = {}
        self._entries: dict[tuple[int, int, int, int], GeometryResult] = {}
        self.lookup_count = 0
        self.solve_count = 0
        self.cache_hits = 0

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def _obs(self, step: int) -> dict[str, Any]:
        step = int(step)
        obs = self._obs_by_step.get(step)
        if obs is None:
            obs = arrays_to_obs(
                self.scenario,
                seat=0,
                rel_step=step,
                owners=self.scenario.owners,
                ships=self.scenario.ships,
                schedule={},
            )
            self._obs_by_step[step] = obs
        return obs

    def _bucket(self, ships: int, *, nearest: bool) -> int | None:
        value = int(ships)
        if value in set(int(x) for x in self.ship_buckets.tolist()):
            return value
        if not nearest or len(self.ship_buckets) == 0:
            return None
        idx = int(np.argmin(np.abs(self.ship_buckets.astype(np.int64) - value)))
        return int(self.ship_buckets[idx])

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
        del nearest_step
        bucket = self._bucket(int(ships), nearest=nearest_bucket)
        if bucket is None:
            return None
        key = (int(step), int(source_id), int(target_id), int(bucket))
        self.lookup_count += 1
        existing = self._entries.get(key)
        if existing is not None:
            self.cache_hits += 1
            return existing

        obs = self._obs(int(step))
        planet_by_id = {int(p[0]): list(p) for p in obs.get("planets", [])}
        source = planet_by_id.get(int(source_id))
        target = planet_by_id.get(int(target_id))
        if source is None or target is None:
            return None
        comet_ids = {int(x) for x in obs.get("comet_planet_ids", [])}
        raw_comets = list(obs.get("comets", []) or [])
        probe = cheap_geometry_probe(
            source,
            target,
            float(obs.get("angular_velocity", 0.035) or 0.035),
            comet_ids,
            raw_comets,
            int(bucket),
        )
        chase_ratio = 0.0 if probe is None else float(probe[1])
        if probe is None or (self.max_chase_ratio > 0.0 and chase_ratio > self.max_chase_ratio):
            result = GeometryResult(
                step=int(step),
                source_id=int(source_id),
                target_id=int(target_id),
                ship_bucket=int(bucket),
                reachable=False,
                angle=None,
                eta=-1,
                actual_hit_id=None,
                hit_reason="no_intercept" if probe is None else "chase_pruned",
                useful=False,
                chase_ratio=chase_ratio,
            )
            self._entries[key] = result
            return result

        self.solve_count += 1
        angle, eta, hit_id, reason, reachable = solve_geometry_entry(
            source,
            target,
            [list(p) for p in obs.get("planets", [])],
            float(obs.get("angular_velocity", 0.035) or 0.035),
            comet_ids,
            raw_comets,
            int(bucket),
            max_steps=self.max_steps,
        )
        result = GeometryResult(
            step=int(step),
            source_id=int(source_id),
            target_id=int(target_id),
            ship_bucket=int(bucket),
            reachable=bool(reachable),
            angle=None if not math.isfinite(angle) else float(angle),
            eta=int(eta),
            actual_hit_id=None if hit_id is None else int(hit_id),
            hit_reason=str(reason),
            useful=True,
            chase_ratio=chase_ratio,
        )
        self._entries[key] = result
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = list(self._entries.values())
        np.savez_compressed(
            path,
            keys=np.array(
                [[e.step, e.source_id, e.target_id, e.ship_bucket] for e in entries],
                dtype=np.int32,
            ),
            angle=np.array(
                [np.nan if e.angle is None else float(e.angle) for e in entries],
                dtype=np.float32,
            ),
            eta=np.array([e.eta for e in entries], dtype=np.int16),
            actual_hit_id=np.array([-1 if e.actual_hit_id is None else e.actual_hit_id for e in entries], dtype=np.int16),
            reachable=np.array([e.reachable for e in entries], dtype=np.bool_),
            useful=np.array([e.useful for e in entries], dtype=np.bool_),
            chase_ratio=np.array([e.chase_ratio for e in entries], dtype=np.float32),
            hit_reason=np.array([e.hit_reason for e in entries], dtype="U32"),
            ship_buckets=self.ship_buckets.astype(np.int16, copy=False),
            planet_ids=self.planet_ids.astype(np.int16, copy=False),
            max_steps=np.array([self.max_steps], dtype=np.int16),
            max_chase_ratio=np.array([self.max_chase_ratio], dtype=np.float32),
            metadata=json.dumps(self.metadata, sort_keys=True),
        )

    @classmethod
    def load(cls, path: str | Path, scenario: InitialScenario) -> "SparseGeometryCache":
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"].tolist()))
        cache = cls(
            scenario,
            [int(x) for x in data["ship_buckets"].tolist()],
            max_steps=int(data["max_steps"][0]),
            max_chase_ratio=float(data["max_chase_ratio"][0]),
            metadata=metadata,
        )
        keys = data["keys"]
        angles = data["angle"]
        etas = data["eta"]
        hit_ids = data["actual_hit_id"]
        reachable = data["reachable"]
        useful = data["useful"]
        chase_ratio = data["chase_ratio"]
        reasons = data["hit_reason"]
        for i, key in enumerate(keys.tolist()):
            angle = float(angles[i])
            hit_id = int(hit_ids[i])
            result = GeometryResult(
                step=int(key[0]),
                source_id=int(key[1]),
                target_id=int(key[2]),
                ship_bucket=int(key[3]),
                reachable=bool(reachable[i]),
                angle=None if not math.isfinite(angle) else angle,
                eta=int(etas[i]),
                actual_hit_id=None if hit_id < 0 else hit_id,
                hit_reason=str(reasons[i]),
                useful=bool(useful[i]),
                chase_ratio=float(chase_ratio[i]),
            )
            cache._entries[(result.step, result.source_id, result.target_id, result.ship_bucket)] = result
        return cache


def load_sparse_geometry_cache(path: str | Path, scenario: InitialScenario) -> SparseGeometryCache:
    return SparseGeometryCache.load(path, scenario)
