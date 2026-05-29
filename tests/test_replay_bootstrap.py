from __future__ import annotations

from scripts.build_replay_bootstrap import find_or_inject_candidate
from src.graph_training.actions import LaunchCandidate


def _candidate(src: int, tgt: int, ships: int, angle: float, kind: str = "capture_exact") -> LaunchCandidate:
    return LaunchCandidate(
        src_id=src,
        intended_tgt_id=tgt,
        actual_hit_id=tgt,
        angle=angle,
        ships=ships,
        eta=10,
        hit_reason="fleet",
        kind=kind,
        bucket=str(ships),
        score=1.0,
        required_ships=ships,
        target_production=2.0,
        target_owner=-1,
    )


def test_replay_candidate_injected_when_bucket_candidate_misses_ship_count():
    candidates = [_candidate(1, 2, 8, 0.5)]
    replay = _candidate(1, 2, 20, 0.5, kind="replay")

    idx = find_or_inject_candidate(candidates, replay)

    assert idx == 1
    assert candidates[idx].kind == "replay"


def test_replay_candidate_matches_close_existing_candidate():
    candidates = [_candidate(1, 2, 18, 0.51)]
    replay = _candidate(1, 2, 20, 0.5, kind="replay")

    idx = find_or_inject_candidate(candidates, replay)

    assert idx == 0
    assert len(candidates) == 1
