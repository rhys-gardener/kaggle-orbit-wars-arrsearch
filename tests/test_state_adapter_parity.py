"""Round-trip parity tests for src/array_search/state_adapter.py.

Two assertions gate this PR:

  1. **Step-0 parity.** Given a real seed's step-0 obs, going
     real_obs → initial_arrays → arrays_to_obs(rel_step=0) → build_graph_state
     produces the *same* planet_features / edge_features / valid_edge_mask as
     real_obs → build_graph_state directly.

  2. **Scheduled-launch round-trip.** After scheduling one launch from planet
     A to planet B at rel_step=0, rendering obs at rel_step=1 must produce
     a fleet that ``infer_fleet_events`` resolves to target B with the
     expected eta.

These two tests are the contract the rollout loop depends on. Anything else
that breaks first is downstream of them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.array_search import generate_initial_arrays, arrays_to_obs
from src.array_search.scenarios import planet_positions_at
from src.graph_training.array_env import ScheduledFleet
from src.graph_training.state import build_graph_state, infer_fleet_events
from src.physics import intercept, sun_blocked


SEEDS = (1, 7, 17, 42, 99)


@pytest.mark.parametrize("seed", SEEDS)
def test_step0_obs_round_trip(seed):
    """rel_step=0 reconstruction must give identical graph features."""
    scenario = generate_initial_arrays(seed, num_players=2)
    original = scenario.initial_obs
    original_with_player = {**original, "player": 0}

    owners = scenario.owners.copy()
    ships = scenario.ships.copy()
    obs = arrays_to_obs(
        scenario,
        seat=0,
        rel_step=0,
        owners=owners,
        ships=ships,
        schedule={},
        snapshot_fleets=original["fleets"],
    )

    # Planet payloads should match position-wise (tiny float32 round-trip drift OK).
    assert len(obs["planets"]) == len(original["planets"])
    for got, expected in zip(obs["planets"], original["planets"]):
        assert int(got[0]) == int(expected[0])
        assert int(got[1]) == int(expected[1])
        assert math.isclose(got[2], expected[2], abs_tol=1e-4)
        assert math.isclose(got[3], expected[3], abs_tol=1e-4)
        assert math.isclose(got[4], expected[4], abs_tol=1e-4)
        assert math.isclose(got[5], expected[5], abs_tol=1e-4)
        assert math.isclose(got[6], expected[6], abs_tol=1e-4)

    g_original = build_graph_state(original_with_player)
    g_round = build_graph_state(obs)

    np.testing.assert_allclose(g_round.planet_features, g_original.planet_features, atol=1e-4)
    np.testing.assert_allclose(g_round.edge_features, g_original.edge_features, atol=1e-4)
    np.testing.assert_array_equal(g_round.valid_edge_mask, g_original.valid_edge_mask)


@pytest.mark.parametrize("seed", SEEDS)
def test_scheduled_launch_resolves_to_target(seed):
    """A scheduled launch must be re-identified by infer_fleet_events.

    We pick a (source, target) pair the agent could realistically launch on
    (path clear, distinct planets), schedule it via the schedule dict the
    same way schedule_action_set does, render obs at rel_step=1, and assert
    the inferred event's target matches the scheduled target.
    """
    scenario = generate_initial_arrays(seed, num_players=2)
    n = len(scenario.planet_ids)

    # Pick a player-owned source and a neutral target with a clear path.
    owners = scenario.owners
    candidates = []
    positions = scenario.planet_xy0
    for src in range(n):
        if int(owners[src]) != 0:
            continue
        for tgt in range(n):
            if src == tgt:
                continue
            sx, sy = float(positions[src, 0]), float(positions[src, 1])
            tx, ty = float(positions[tgt, 0]), float(positions[tgt, 1])
            if sun_blocked(sx, sy, tx, ty):
                continue
            candidates.append((src, tgt))
        if candidates:
            break
    assert candidates, f"seed={seed}: no valid (src, tgt) pair found"

    src_idx, tgt_idx = candidates[0]
    ships_launched = 30

    # Compute expected eta the same way schedule_action_set would.
    src_planet_list = [
        int(scenario.planet_ids[src_idx]),
        int(scenario.owners[src_idx]),
        float(positions[src_idx, 0]),
        float(positions[src_idx, 1]),
        float(scenario.radii[src_idx]),
        float(scenario.ships[src_idx]),
        float(scenario.production[src_idx]),
    ]
    tgt_planet_list = [
        int(scenario.planet_ids[tgt_idx]),
        int(scenario.owners[tgt_idx]),
        float(positions[tgt_idx, 0]),
        float(positions[tgt_idx, 1]),
        float(scenario.radii[tgt_idx]),
        float(scenario.ships[tgt_idx]),
        float(scenario.production[tgt_idx]),
    ]
    _tx, _ty, eta = intercept(
        src_planet_list[2],
        src_planet_list[3],
        tgt_planet_list,
        scenario.angular_velocity,
        ships_launched,
    )
    assert eta is not None and eta >= 1

    schedule = {
        int(eta): [
            ScheduledFleet(
                source_idx=src_idx,
                target_idx=tgt_idx,
                owner=0,
                ships=ships_launched,
                launch_rel_turn=0,
            )
        ]
    }
    owners_arr = scenario.owners.copy()
    ships_arr = scenario.ships.copy()
    ships_arr[src_idx] -= ships_launched

    rel_step = 1
    obs = arrays_to_obs(
        scenario,
        seat=0,
        rel_step=rel_step,
        owners=owners_arr,
        ships=ships_arr,
        schedule=schedule,
    )

    assert len(obs["fleets"]) == 1, f"seed={seed}: expected 1 synth fleet, got {obs['fleets']}"

    events = infer_fleet_events(
        obs["fleets"],
        obs["planets"],
        obs["angular_velocity"],
        set(obs["comet_planet_ids"]),
        obs["comets"],
    )
    expected_tgt_id = int(scenario.planet_ids[tgt_idx])
    assert expected_tgt_id in events, (
        f"seed={seed}: inferred events {list(events.keys())} did not include "
        f"expected target id {expected_tgt_id}"
    )
    inferred = events[expected_tgt_id]
    assert len(inferred) == 1
    ev = inferred[0]
    assert ev.target_id == expected_tgt_id
    assert ev.owner == 0
    assert ev.ships == ships_launched
    # eta in the inferred event is what `intercept` says from the fleet's current
    # mid-flight position; that should match (arrival_eta - rel_step) within a turn.
    assert abs(int(ev.eta) - (int(eta) - rel_step)) <= 1


@pytest.mark.parametrize("seed", SEEDS)
def test_planet_positions_match_orbit_forward(seed):
    """planet_positions_at(scenario, k) must match the engine's orbit formula
    for every orbiting planet at multiple future steps."""
    scenario = generate_initial_arrays(seed)
    for rel_step in (1, 5, 25, 100):
        positions = planet_positions_at(scenario, rel_step)
        for i in range(len(scenario.planet_ids)):
            if not bool(scenario.orbiting[i]):
                continue
            x0, y0 = float(scenario.planet_xy0[i, 0]), float(scenario.planet_xy0[i, 1])
            dx, dy = x0 - 50.0, y0 - 50.0
            r = math.hypot(dx, dy)
            ang = math.atan2(dy, dx) + scenario.angular_velocity * rel_step
            ex = 50.0 + r * math.cos(ang)
            ey = 50.0 + r * math.sin(ang)
            assert math.isclose(float(positions[i, 0]), ex, abs_tol=1e-3)
            assert math.isclose(float(positions[i, 1]), ey, abs_tol=1e-3)
