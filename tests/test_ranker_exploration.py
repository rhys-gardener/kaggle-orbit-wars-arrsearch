from __future__ import annotations

import numpy as np

from src.array_search.ranker import greedy_pack_action_set


def _record(n: int = 6):
    # n candidates, each from a distinct source with ample garrison so the budget
    # never prunes; targets distinct so multi_source_bonus stays out of the way.
    planets = [[i, 0, 0.0, 0.0, 1.0, 1000.0, 1.0] for i in range(n)]
    candidates = [
        {"src_id": i, "actual_hit_id": n + i, "ships": 5, "angle": 0.1 * i}
        for i in range(n)
    ]
    return {
        "graph": {"planet_ids": [i for i in range(n)]},
        "observation": {"planets": planets},
        "candidates": candidates,
    }


def test_temperature_zero_matches_argsort():
    record = _record()
    scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.2], dtype=np.float32)
    out = greedy_pack_action_set(record, scores, max_launches=6, temperature=0.0)
    assert out["candidate_indices"] == list(np.argsort(-scores))


def test_temperature_sampling_is_rng_deterministic():
    record = _record()
    scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.2], dtype=np.float32)
    a = greedy_pack_action_set(
        record, scores, max_launches=6, temperature=1.0, rng=np.random.default_rng(123)
    )
    b = greedy_pack_action_set(
        record, scores, max_launches=6, temperature=1.0, rng=np.random.default_rng(123)
    )
    assert a["candidate_indices"] == b["candidate_indices"]


def test_temperature_sampling_can_reorder():
    record = _record()
    scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.2], dtype=np.float32)
    greedy = list(np.argsort(-scores))
    # Across many seeds at least one sampled ordering should differ from greedy.
    reordered = False
    for seed in range(50):
        out = greedy_pack_action_set(
            record, scores, max_launches=6, temperature=2.0, rng=np.random.default_rng(seed)
        )
        if out["candidate_indices"] != greedy:
            reordered = True
            break
    assert reordered
