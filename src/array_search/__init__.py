"""Array-only search/training pipeline for Orbit Wars.

Modules in this package operate on ndarrays and never call kaggle_environments
or sim.make_env(...).step() inside the training hot loop. The only env touch
is reading a step-0 obs via ``scenarios.generate_initial_arrays``.
"""

from .action_filters import ActionFilters, CandidateFlags, compute_flags, filter_candidates, is_strict_rejected
from .features import FEATURE_DIM, FEATURE_NAMES, candidate_feature_matrix, candidate_feature_row
from .labels import DEFAULT_HORIZONS, PRIMARY_HORIZON, attach_positive_candidate_label, label_record, label_records
from .ranker import CandidateRanker, load_ranker, save_ranker
from .mcts_teacher import MCTSTeacherConfig, apply_mcts_teacher, teach_record_with_mcts
from .scenarios import InitialScenario, active_planet_mask, generate_initial_arrays, planet_positions_at
from .state_adapter import arrays_to_obs
from .training_log import TrainingLogger, read_jsonl

__all__ = [
    "ActionFilters",
    "CandidateFlags",
    "CandidateRanker",
    "DEFAULT_HORIZONS",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "InitialScenario",
    "MCTSTeacherConfig",
    "PRIMARY_HORIZON",
    "TrainingLogger",
    "active_planet_mask",
    "arrays_to_obs",
    "attach_positive_candidate_label",
    "apply_mcts_teacher",
    "candidate_feature_matrix",
    "candidate_feature_row",
    "compute_flags",
    "filter_candidates",
    "generate_initial_arrays",
    "is_strict_rejected",
    "label_record",
    "label_records",
    "load_ranker",
    "planet_positions_at",
    "read_jsonl",
    "save_ranker",
    "teach_record_with_mcts",
]
