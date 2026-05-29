"""Production-first graph search experiments for Orbit Wars.

This package is intentionally separate from the Kaggle submission path.  The
goal is to build a richer state/action representation and prove it locally
before any of it is copied into ``main.py``.
"""

from .agent import GraphProdConfig, agent, make_agent
from .state import GraphState, build_graph_state
from .actions import LaunchCandidate, generate_cached_launch_candidates, generate_launch_candidates
from .search import ActionSet, generate_action_sets
from .geometry_cache import GeometryCache, GeometryResult, load_geometry_cache
from .sparse_geometry_cache import DEFAULT_SHIP_BUCKETS, SparseGeometryCache, load_sparse_geometry_cache

__all__ = [
    "ActionSet",
    "GraphProdConfig",
    "GraphState",
    "GeometryCache",
    "GeometryResult",
    "DEFAULT_SHIP_BUCKETS",
    "SparseGeometryCache",
    "LaunchCandidate",
    "agent",
    "build_graph_state",
    "generate_action_sets",
    "generate_cached_launch_candidates",
    "generate_launch_candidates",
    "load_geometry_cache",
    "load_sparse_geometry_cache",
    "make_agent",
]
