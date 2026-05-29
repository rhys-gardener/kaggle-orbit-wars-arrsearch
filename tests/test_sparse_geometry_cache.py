from __future__ import annotations

import math

import pytest

from src.array_search.scenarios import generate_initial_arrays
from src.graph_training.sparse_geometry_cache import SparseGeometryCache


def test_sparse_geometry_lookup_persists(tmp_path):
    scenario = generate_initial_arrays(0, num_players=4)
    source_id = int(scenario.planet_ids[16])
    target_id = int(scenario.planet_ids[17])
    cache = SparseGeometryCache(scenario, [1, 2, 4, 8], max_steps=30)

    first = cache.lookup(0, source_id, target_id, 4, nearest_bucket=False)
    assert first is not None
    assert cache.entry_count == 1

    path = tmp_path / "sparse_geometry.npz"
    cache.save(path)
    loaded = SparseGeometryCache.load(path, scenario)
    second = loaded.lookup(0, source_id, target_id, 4, nearest_bucket=False)

    assert second.step == first.step
    assert second.source_id == first.source_id
    assert second.target_id == first.target_id
    assert second.ship_bucket == first.ship_bucket
    assert second.reachable == first.reachable
    assert second.eta == first.eta
    assert second.actual_hit_id == first.actual_hit_id
    assert second.hit_reason == first.hit_reason
    assert second.useful == first.useful
    assert second.chase_ratio == pytest.approx(first.chase_ratio)
    assert second.angle == pytest.approx(first.angle)
    assert loaded.cache_hits == 1
    if first.angle is not None:
        assert math.isfinite(first.angle)
