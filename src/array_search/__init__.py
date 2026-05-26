"""Array-only search/training pipeline for Orbit Wars.

Modules in this package operate on ndarrays and never call kaggle_environments
or sim.make_env(...).step() inside the training hot loop. The only env touch
is reading a step-0 obs via ``scenarios.generate_initial_arrays``.
"""

from .action_filters import ActionFilters, CandidateFlags, compute_flags, filter_candidates, is_strict_rejected
from .scenarios import InitialScenario, generate_initial_arrays, planet_positions_at
from .state_adapter import arrays_to_obs
from .training_log import TrainingLogger, read_jsonl

__all__ = [
    "ActionFilters",
    "CandidateFlags",
    "InitialScenario",
    "TrainingLogger",
    "arrays_to_obs",
    "compute_flags",
    "filter_candidates",
    "generate_initial_arrays",
    "is_strict_rejected",
    "planet_positions_at",
    "read_jsonl",
]
