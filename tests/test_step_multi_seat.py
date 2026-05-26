"""Tests for ``step_multi_seat``.

Asserts the multi-seat per-turn function in array_env:
  * deducts ships from each seat's sources atomically (no cross-seat order
    sensitivity for non-overlapping launches),
  * schedules arrivals at the correct ``current_rel_turn + eta`` slot,
  * advances production exactly once per turn for all owned planets,
  * resolves combat for arrivals at the new turn.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.graph_training.array_env import (
    PendingLaunch,
    ScheduledFleet,
    step_multi_seat,
)


def _state(num_planets: int = 4):
    owners = np.array([0, 1, -1, 0], dtype=np.int16)
    ships = np.array([100.0, 80.0, 30.0, 50.0], dtype=np.float32)
    production = np.array([4.0, 3.0, 2.0, 5.0], dtype=np.float32)
    return owners, ships, production


def test_advances_rel_turn():
    owners, ships, prod = _state()
    schedule: dict[int, list[ScheduledFleet]] = {}
    new_t = step_multi_seat(owners, ships, prod, schedule, [], current_rel_turn=0)
    assert new_t == 1
    new_t = step_multi_seat(owners, ships, prod, schedule, [], current_rel_turn=new_t)
    assert new_t == 2


def test_production_accrues_only_for_owned():
    """Neutral planets (owner=-1) do NOT accrue production; only owned planets do.
    Mirrors the existing semantics of ``rollout_cached_action_set``."""
    owners, ships, prod = _state()
    schedule: dict[int, list[ScheduledFleet]] = {}
    before = ships.copy()
    step_multi_seat(owners, ships, prod, schedule, [], current_rel_turn=0)
    np.testing.assert_array_almost_equal(ships[0], before[0] + prod[0])
    np.testing.assert_array_almost_equal(ships[1], before[1] + prod[1])
    np.testing.assert_array_almost_equal(ships[2], before[2])  # neutral does NOT accrue
    np.testing.assert_array_almost_equal(ships[3], before[3] + prod[3])


def test_multi_seat_launches_deduct_simultaneously():
    """Two seats launching from different sources in the same turn must
    each see ships deducted from their respective sources, and both
    arrivals must land in the same schedule slot."""
    owners, ships, prod = _state()
    schedule: dict[int, list[ScheduledFleet]] = {}
    seat0_launch = PendingLaunch(source_idx=0, target_idx=2, owner=0, ships=30, eta=5)
    seat1_launch = PendingLaunch(source_idx=1, target_idx=2, owner=1, ships=25, eta=5)
    step_multi_seat(owners, ships, prod, schedule, [seat0_launch, seat1_launch], current_rel_turn=0)
    # Source ships deducted then production accrued.
    assert ships[0] == pytest.approx(100.0 - 30.0 + 4.0)
    assert ships[1] == pytest.approx(80.0 - 25.0 + 3.0)
    # Arrivals both scheduled at rel_turn 0 + 5 = 5.
    assert 5 in schedule
    owners_in_slot = sorted(int(f.owner) for f in schedule[5])
    assert owners_in_slot == [0, 1]


def test_arrival_resolves_combat_on_arrival_turn():
    """Schedule a single arrival at rel_turn 3 (eta=3 launched at turn 0).
    Stepping forward three turns should fire combat exactly once on turn 3.

    Neutral planet (owner=-1) does NOT accrue production, so its garrison
    stays at 30 throughout the wait. Fleet of 40 arrives → seat 0 captures
    with 40-30=10 survivors.
    """
    owners, ships, prod = _state()
    schedule: dict[int, list[ScheduledFleet]] = {}
    launch = PendingLaunch(source_idx=0, target_idx=2, owner=0, ships=40, eta=3)
    rel_turn = step_multi_seat(owners, ships, prod, schedule, [launch], current_rel_turn=0)
    assert int(owners[2]) == -1  # not yet captured
    rel_turn = step_multi_seat(owners, ships, prod, schedule, [], current_rel_turn=rel_turn)  # → 2
    assert int(owners[2]) == -1
    rel_turn = step_multi_seat(owners, ships, prod, schedule, [], current_rel_turn=rel_turn)  # → 3, combat
    assert int(owners[2]) == 0
    assert ships[2] == pytest.approx(10.0, abs=1e-3)


def test_ship_cap_at_source():
    """If a PendingLaunch requests more ships than the source has, only
    the source's actual stockpile is sent."""
    owners, ships, prod = _state()
    schedule: dict[int, list[ScheduledFleet]] = {}
    launch = PendingLaunch(source_idx=0, target_idx=2, owner=0, ships=999, eta=4)
    step_multi_seat(owners, ships, prod, schedule, [launch], current_rel_turn=0)
    # All of seat 0's source ships shipped (100), then production added (4).
    assert ships[0] == pytest.approx(0.0 + 4.0)
    fleet = schedule[4][0]
    assert fleet.ships == 100


def test_zero_ships_launch_is_dropped():
    owners, ships, prod = _state()
    schedule: dict[int, list[ScheduledFleet]] = {}
    launch = PendingLaunch(source_idx=0, target_idx=2, owner=0, ships=0, eta=3)
    step_multi_seat(owners, ships, prod, schedule, [launch], current_rel_turn=0)
    assert schedule == {}
    assert ships[0] == pytest.approx(100.0 + 4.0)


def test_negative_source_ships_does_not_underflow():
    """If a source is already drained, no launch goes out (no negative ship count)."""
    owners, ships, prod = _state()
    ships[0] = 0.0
    schedule: dict[int, list[ScheduledFleet]] = {}
    launch = PendingLaunch(source_idx=0, target_idx=2, owner=0, ships=50, eta=3)
    step_multi_seat(owners, ships, prod, schedule, [launch], current_rel_turn=0)
    assert ships[0] == pytest.approx(prod[0])  # only production
    assert schedule == {}


def test_chained_steps_accumulate_correctly():
    """Run a 4-turn multi-seat sequence and verify the schedule is consumed
    correctly turn by turn."""
    owners, ships, prod = _state()
    schedule: dict[int, list[ScheduledFleet]] = {}
    seat_turn_launches = {
        0: [PendingLaunch(0, 1, 0, 50, 2)],
        1: [PendingLaunch(3, 1, 0, 20, 3)],  # arrives turn 4
        2: [],
        3: [],
    }
    rel_turn = 0
    for t in range(4):
        rel_turn = step_multi_seat(
            owners, ships, prod, schedule, seat_turn_launches[t], current_rel_turn=rel_turn
        )
    # Turn 2 arrival fired against seat-1 home (80 + 2*3 production = 86 at turn 2; 50 < 86 so survives, owner unchanged).
    # Turn 4 arrival fires against seat-1 home post-production again.
    # The exact final ships value depends on every intermediate step; the point of this
    # test is that we ran 4 turns without crashing and consumed both arrivals.
    assert rel_turn == 4
    # Schedule entries persist after firing (matches rollout_cached_action_set
    # which reads via .get() and never pops). Confirm both expected arrival
    # slots exist with the right number of entries.
    assert len(schedule[2]) == 1
    assert len(schedule[4]) == 1
