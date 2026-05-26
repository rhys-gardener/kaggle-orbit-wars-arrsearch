"""Array-only search/training pipeline for Orbit Wars.

Modules in this package operate on ndarrays and never call kaggle_environments
or sim.make_env(...).step() inside the training hot loop. The only env touch
is reading a step-0 obs via ``scenarios.generate_initial_arrays``.
"""

from .scenarios import InitialScenario, generate_initial_arrays, planet_positions_at
from .state_adapter import arrays_to_obs

__all__ = [
    "InitialScenario",
    "arrays_to_obs",
    "generate_initial_arrays",
    "planet_positions_at",
]
