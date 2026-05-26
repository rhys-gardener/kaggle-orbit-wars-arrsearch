"""Parity test: src/physics.py vs main.py.

main.py is the Kaggle submission file and must stay self-contained, so it
keeps its own copies of the physics helpers. This test asserts that the
src/physics.py copies produce identical outputs across a representative
input grid, so drift between the two copies is caught immediately.
"""

from __future__ import annotations

import math

import pytest

import main
from src import physics
from sim import make_env


SEEDS = (1, 3, 7, 11, 17, 23, 29, 42, 99)


@pytest.mark.parametrize("ships", [1, 10, 50, 100, 250, 500, 1000])
def test_fleet_speed_parity(ships):
    assert physics.fleet_speed(ships) == main.fleet_speed(ships)


@pytest.mark.parametrize(
    "sx,sy,tx,ty",
    [
        (10.0, 10.0, 90.0, 90.0),     # crosses sun
        (10.0, 10.0, 90.0, 10.0),     # horizontal, far below sun
        (50.0, 5.0, 50.0, 95.0),      # vertical through sun
        (15.0, 50.0, 85.0, 50.0),     # horizontal through sun
        (0.0, 0.0, 100.0, 100.0),     # diagonal through sun
        (60.0, 60.0, 80.0, 80.0),     # outside sun region
    ],
)
def test_sun_blocked_parity(sx, sy, tx, ty):
    assert physics.sun_blocked(sx, sy, tx, ty) == main.sun_blocked(sx, sy, tx, ty)


@pytest.mark.parametrize(
    "planet",
    [
        [0, 0, 50.0, 50.0, 5.0, 10, 4],   # at center
        [1, 0, 91.0, 75.0, 2.4, 10, 4],   # near edge
        [2, 0, 60.0, 50.0, 3.0, 10, 4],   # mid-orbit
        [3, 0, 95.0, 95.0, 6.0, 10, 4],   # corner-ish
    ],
)
def test_is_orbiting_parity(planet):
    assert physics.is_orbiting(planet) == main.is_orbiting(planet)


def test_intercept_parity_across_seeds():
    """For real step-0 obs maps, intercept should match exactly for every
    (source, target, ship_count) triple."""
    for seed in SEEDS:
        env = make_env({"seed": seed})
        env.reset(2)
        obs = env.state[0].observation
        planets = [list(p) for p in obs["planets"]]
        av = float(obs["angular_velocity"])
        for src in planets[:6]:
            for tgt in planets[:8]:
                if src[0] == tgt[0]:
                    continue
                for ships in (1, 25, 100, 500):
                    a = physics.intercept(src[2], src[3], tgt, av, ships)
                    b = main.intercept(src[2], src[3], tgt, av, ships)
                    assert a == b, f"seed={seed} src={src[0]} tgt={tgt[0]} ships={ships}: {a} vs {b}"


def test_intercept_parity_with_synthetic_comet():
    """Comet branch: build a tiny comet path and verify both copies agree."""
    raw_comets = [
        {
            "planet_ids": [999],
            "paths": [[[float(t), 30.0 + 0.1 * t] for t in range(120)]],
            "path_index": 0,
        }
    ]
    comet_ids = {999}
    planet = [999, -1, 0.0, 30.0, 1.5, 10, 0]
    for ships in (1, 50, 500):
        a = physics.intercept(20.0, 30.0, planet, 0.035, ships, comet_ids, raw_comets)
        b = main.intercept(20.0, 30.0, planet, 0.035, ships, comet_ids, raw_comets)
        assert a == b


def test_trajectory_first_hit_parity():
    """Walk a few launch angles and confirm hit identification matches."""
    env = make_env({"seed": 7})
    env.reset(2)
    obs = env.state[0].observation
    planets = [list(p) for p in obs["planets"]]
    av = float(obs["angular_velocity"])
    source = planets[0]
    for ships in (10, 100, 500):
        for angle_deg in range(0, 360, 30):
            angle = math.radians(angle_deg)
            a = physics._trajectory_first_hit(source, angle, ships, planets, av, set(), [])
            b = main._trajectory_first_hit(source, angle, ships, planets, av, set(), [])
            assert a == b, f"ships={ships} angle={angle_deg}: {a} vs {b}"


def test_solve_engine_angle_parity():
    """Sample a handful of (source, target, ships) combos and compare solver output."""
    env = make_env({"seed": 7})
    env.reset(2)
    obs = env.state[0].observation
    planets = [list(p) for p in obs["planets"]]
    av = float(obs["angular_velocity"])
    pairs = [(0, 5), (1, 7), (2, 9), (0, 12), (3, 4)]
    for src_idx, tgt_idx in pairs:
        if src_idx >= len(planets) or tgt_idx >= len(planets):
            continue
        src = planets[src_idx]
        tgt = planets[tgt_idx]
        for ships in (50, 200):
            a = physics._solve_engine_angle(src, tgt, ships, planets, av, set(), [])
            b = main._solve_engine_angle(src, tgt, ships, planets, av, set(), [])
            assert a == b, f"src={src_idx} tgt={tgt_idx} ships={ships}: {a} vs {b}"
