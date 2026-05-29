from __future__ import annotations

from src.array_search.mcts_teacher import MCTSTeacherConfig, teach_record_with_mcts
from src.array_search.records import build_record_dict
from src.array_search.scenarios import generate_initial_arrays
from src.array_search.state_adapter import arrays_to_obs
from src.graph_training.actions import generate_cached_launch_candidates
from src.graph_training.search import generate_action_sets
from src.graph_training.sparse_geometry_cache import SparseGeometryCache
from src.graph_training.state import build_graph_state


def test_mcts_teacher_attaches_reachable_positive_label():
    scenario = generate_initial_arrays(0, num_players=4)
    obs = arrays_to_obs(
        scenario,
        seat=0,
        rel_step=0,
        owners=scenario.owners,
        ships=scenario.ships,
        schedule={},
    )
    ctx = build_graph_state(obs)
    geometry = SparseGeometryCache(scenario, [1, 2, 4, 8], max_steps=30)
    candidates = generate_cached_launch_candidates(ctx, geometry, max_candidates=12)
    action_sets = generate_action_sets(ctx, candidates, max_launches=3, beam_width=6, candidate_limit=12)
    record = build_record_dict(
        scenario_id="test",
        seed=0,
        rel_turn=0,
        seat=0,
        policy_id="A",
        obs=obs,
        ctx=ctx,
        candidates=candidates,
        candidate_flags=[],
        action_sets=action_sets,
    )

    teach_record_with_mcts(record, config=MCTSTeacherConfig(root_action_sets=4, rollout_horizon=5))

    positives = record["labels"]["best_candidate_indices_by_horizon"]["60"]
    assert record["labels"]["source"] == "mcts"
    assert all(record["candidates"][idx]["actual_hit_id"] == record["candidates"][idx]["intended_tgt_id"] for idx in positives)


def test_mcts_teacher_policy_response_depth_uses_scenario_geometry():
    scenario = generate_initial_arrays(0, num_players=4)
    obs = arrays_to_obs(
        scenario,
        seat=0,
        rel_step=0,
        owners=scenario.owners,
        ships=scenario.ships,
        schedule={},
    )
    ctx = build_graph_state(obs)
    geometry = SparseGeometryCache(scenario, [1, 2, 4, 8], max_steps=30)
    candidates = generate_cached_launch_candidates(ctx, geometry, max_candidates=12)
    action_sets = generate_action_sets(ctx, candidates, max_launches=3, beam_width=6, candidate_limit=12)
    record = build_record_dict(
        scenario_id="test",
        seed=0,
        rel_turn=0,
        seat=0,
        policy_id="A",
        obs=obs,
        ctx=ctx,
        candidates=candidates,
        candidate_flags=[],
        action_sets=action_sets,
    )

    teach_record_with_mcts(
        record,
        config=MCTSTeacherConfig(root_action_sets=2, depth=2, rollout_horizon=5),
        scenario=scenario,
        geometry_cache=geometry,
        candidate_limit=8,
        max_launches=2,
        beam_width=4,
    )

    assert record["labels"]["source"] == "mcts"
    assert record["teacher"]["mcts"]["mode"] == "policy_response"
    assert record["teacher"]["mcts"]["depth"] == 2
