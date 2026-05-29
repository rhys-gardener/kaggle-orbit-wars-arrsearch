"""Online agent wrapping a trained array-search CandidateRanker.

This is a *local evaluation* agent for the real-engine yardstick — NOT a Kaggle
submission. It imports ``src/`` and loads a torch ``.pt`` checkpoint directly.
The self-contained, weight-embedded, numpy-only ``main.py`` submission path is
separate future work.

Inference chain (no precomputed geometry cache needed — engine candidate path):
    obs -> build_graph_state -> generate_launch_candidates -> filter_candidates
        -> build_record_dict -> score_candidates -> greedy_pack_action_set
        -> [[src_id, angle, ships], ...]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.array_search.action_filters import ActionFilters, filter_candidates
from src.array_search.ranker import greedy_pack_action_set, load_ranker, score_candidates
from src.array_search.records import build_record_dict
from src.graph_training.actions import generate_launch_candidates
from src.graph_training.state import build_graph_state


def _normalized_obs(ctx) -> dict[str, Any]:
    """Plain-dict obs for the record so downstream subscripting is safe even when
    the engine hands the agent a Structify/attribute-style observation."""
    return {
        "player": int(ctx.player),
        "planets": ctx.planets,
        "fleets": ctx.fleets,
        "angular_velocity": float(ctx.angular_velocity),
        "comets": ctx.raw_comets,
        "comet_planet_ids": list(ctx.comet_ids),
        "step": int(ctx.step),
    }


def make_agent(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    candidate_limit: int = 80,
    max_launches: int = 10,
    include_support: bool = False,
    multi_source_bonus: float = 0.15,
) -> Callable[[Any], list[list[float]]]:
    """Build a greedy ``agent(obs) -> moves`` closure from a ranker checkpoint."""

    model = load_ranker(str(checkpoint_path), map_location=device)
    filters = ActionFilters()

    def agent(obs: Any) -> list[list[float]]:
        ctx = build_graph_state(obs)
        raw = generate_launch_candidates(
            ctx, max_candidates=candidate_limit, include_support=include_support
        )
        candidates, flags, reject_counts = filter_candidates(raw, filters)
        record = build_record_dict(
            scenario_id="yardstick",
            seed=0,
            rel_turn=int(ctx.step),
            seat=int(ctx.player),
            policy_id="ranker",
            obs=_normalized_obs(ctx),
            ctx=ctx,
            candidates=candidates,
            candidate_flags=flags,
            action_sets=[],
            reject_counts=reject_counts,
        )
        if not record["candidates"]:
            return []
        scores = score_candidates(model, record, device=device)
        # Greedy (temperature 0): strength must be measured deterministically.
        chosen = greedy_pack_action_set(
            record,
            scores,
            max_launches=max_launches,
            multi_source_bonus=multi_source_bonus,
        )
        return [list(action) for action in chosen.get("actions", [])]

    return agent
