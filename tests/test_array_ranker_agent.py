from __future__ import annotations

import math

from agents.array_ranker_agent import make_agent
from src.array_search.ranker import CandidateRanker, save_ranker
from src.array_search.scenarios import generate_initial_arrays
from src.array_search.state_adapter import arrays_to_obs


def test_online_ranker_agent_emits_wellformed_moves(tmp_path):
    """Guard the yardstick's core assumption: the online agent must actually emit
    launches from a live obs (a silent [] would make any benchmark meaningless)."""
    ckpt = tmp_path / "ranker.pt"
    save_ranker(ckpt, CandidateRanker(), extra={"test": True})
    agent = make_agent(ckpt, candidate_limit=24, max_launches=6)

    scenario = generate_initial_arrays(200_000, num_players=2)
    emitted = 0
    for step in (0, 5, 20):
        obs = arrays_to_obs(
            scenario, seat=0, rel_step=step, owners=scenario.owners, ships=scenario.ships, schedule={}
        )
        moves = agent(obs)
        assert isinstance(moves, list)
        for move in moves:
            assert len(move) == 3  # [src_id, angle, ships]
            assert isinstance(int(move[0]), int)
            assert math.isfinite(float(move[1]))
            assert int(move[2]) >= 1
        emitted += len(moves)
    # Across early turns a fresh net should still produce at least one launch.
    assert emitted > 0
