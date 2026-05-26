"""Playable adapter for the production-first graph search."""

from __future__ import annotations

from dataclasses import dataclass

from .search import choose_action_set
from .state import build_graph_state


@dataclass(frozen=True)
class GraphProdConfig:
    max_launches: int = 4
    beam_width: int = 12
    max_same_target: int = 3
    candidate_limit: int = 24
    top_targets_per_source: int = 2
    include_support: bool = False


def make_agent(config: GraphProdConfig | None = None):
    cfg = config or GraphProdConfig()

    def _agent(obs):
        try:
            ctx = build_graph_state(obs)
            action_set = choose_action_set(
                ctx,
                max_launches=cfg.max_launches,
                beam_width=cfg.beam_width,
                max_same_target=cfg.max_same_target,
                candidate_limit=cfg.candidate_limit,
                top_targets_per_source=cfg.top_targets_per_source,
                include_support=cfg.include_support,
            )
            return action_set.actions()
        except Exception:
            return []

    return _agent


agent = make_agent()
