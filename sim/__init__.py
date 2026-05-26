"""Fast Orbit Wars simulator.

Drop-in replacement for `kaggle_environments.make("orbit_wars", ...)` that
bypasses the kaggle_environments framework wrapper (schema validation,
structify deepcopy, stdout redirection, per-step state cloning) and calls
the official `interpreter()` directly. Same game logic, same byte-for-byte
output, much faster.

Use `from sim import make_env`. Pass `kaggle_sim=True` to fall back to the
real kaggle env (for parity checks).
"""
from sim.fast_env import FastEnv, make_env

__all__ = ["FastEnv", "make_env"]
