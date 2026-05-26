"""Curriculum env for Orbit Wars training.

Wraps the upstream `kaggle_environments` orbit_wars env with a `make_env(level)`
factory that patches a few module-level constants (planet group bounds, comet
spawn schedule) to produce smaller, faster training environments.

Levels:
- "tiny"   : 1 planet group (4 planets), no comets, 50 steps.
- "medium" : 2 planet groups (8 planets), no comets, 150 steps.
- "full"   : upstream defaults, 500 steps.

Patches are process-global. A given process should stick to one level at a
time. Multiprocessing workers should call `make_env(level)` themselves at
startup so each worker applies the patch in its own address space.
"""
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars import orbit_wars as _ow


_ORIG_MIN_GROUPS = _ow.MIN_PLANET_GROUPS
_ORIG_MAX_GROUPS = _ow.MAX_PLANET_GROUPS
_ORIG_MIN_STATIC = _ow.MIN_STATIC_GROUPS
_ORIG_COMET_STEPS = list(_ow.COMET_SPAWN_STEPS)


LEVELS = {
    "tiny": {
        "min_groups": 1,
        "max_groups": 1,
        "min_static": 1,
        "comet_steps": [],
        "episodeSteps": 50,
    },
    "medium": {
        "min_groups": 2,
        "max_groups": 2,
        "min_static": 1,
        "comet_steps": [],
        "episodeSteps": 150,
    },
    "full": {
        "min_groups": _ORIG_MIN_GROUPS,
        "max_groups": _ORIG_MAX_GROUPS,
        "min_static": _ORIG_MIN_STATIC,
        "comet_steps": list(_ORIG_COMET_STEPS),
        "episodeSteps": 500,
    },
}


_current_level = None


def _apply_level(level: str) -> None:
    """Patch the upstream orbit_wars module constants for this level."""
    global _current_level
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {list(LEVELS)}")
    cfg = LEVELS[level]
    _ow.MIN_PLANET_GROUPS = cfg["min_groups"]
    _ow.MAX_PLANET_GROUPS = cfg["max_groups"]
    _ow.MIN_STATIC_GROUPS = cfg["min_static"]
    _ow.COMET_SPAWN_STEPS = list(cfg["comet_steps"])
    _current_level = level


def make_env(level: str = "full", configuration: dict | None = None, debug: bool = False):
    """Create a kaggle_environments env at the given curriculum level.

    Parameters
    ----------
    level
        One of "tiny", "medium", "full".
    configuration
        Extra configuration passed to `kaggle_environments.make`. Overrides
        the level default for any keys present (e.g. episodeSteps).
    """
    _apply_level(level)
    cfg = {"episodeSteps": LEVELS[level]["episodeSteps"]}
    if configuration:
        cfg.update(configuration)
    return make("orbit_wars", configuration=cfg, debug=debug)


def current_level() -> str | None:
    return _current_level


__all__ = ["make_env", "current_level", "LEVELS"]
