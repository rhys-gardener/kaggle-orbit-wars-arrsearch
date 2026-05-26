"""Tests for ``src/array_search/action_filters.py``.

Each strict rule is asserted to reject the right kind of candidate, and each
lax rule is asserted to set the right flag without rejecting.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.array_search.action_filters import (
    ActionFilters,
    compute_flags,
    filter_candidates,
    is_strict_rejected,
)


@dataclass
class _Cand:
    ships: int
    eta: int
    intended_tgt_id: int | None = 5
    actual_hit_id: int | None = 5


def test_default_filter_passes_typical_winner_move():
    """A 20-ship launch at eta=8 is dead-centre winning play; must not reject."""
    c = _Cand(ships=20, eta=8)
    assert is_strict_rejected(c, ActionFilters()) is None


def test_strict_eta_cap_rejects_long_journey():
    c = _Cand(ships=50, eta=90)
    assert is_strict_rejected(c, ActionFilters()) == "eta_gt_strict"


def test_strict_small_long_journey_rejects():
    # 2 ships at eta=50 → rejected (small fleet long journey)
    c = _Cand(ships=2, eta=50)
    assert is_strict_rejected(c, ActionFilters()) == "small_fleet_long_journey"


def test_small_short_journey_passes():
    """1-ship launches at moderate eta are legitimate winning play (~13% of all
    winning launches are <= 3 ships)."""
    c = _Cand(ships=1, eta=20)
    assert is_strict_rejected(c, ActionFilters()) is None


def test_zero_ships_rejected():
    c = _Cand(ships=0, eta=5)
    assert is_strict_rejected(c, ActionFilters()) == "ships_lt_1"


def test_off_target_rejected_by_default():
    """If the engine solver says the launch doesn't hit the intended target,
    reject by default — it'll confuse the ranker."""
    c = _Cand(ships=20, eta=8, intended_tgt_id=5, actual_hit_id=7)
    assert is_strict_rejected(c, ActionFilters()) == "off_target_hit"


def test_off_target_allowed_when_disabled():
    filters = ActionFilters(require_hits_intended=False)
    c = _Cand(ships=20, eta=8, intended_tgt_id=5, actual_hit_id=7)
    assert is_strict_rejected(c, filters) is None


def test_flag_below_typical_min_ships():
    c = _Cand(ships=2, eta=10)
    flags = compute_flags(c, ActionFilters())
    assert flags.is_below_typical_min_ships is True
    assert flags.is_high_eta_launch is False


def test_flag_high_eta_launch():
    c = _Cand(ships=15, eta=60)
    flags = compute_flags(c, ActionFilters())
    assert flags.is_below_typical_min_ships is False
    assert flags.is_high_eta_launch is True


def test_filter_candidates_returns_stats():
    cands = [
        _Cand(ships=20, eta=8),       # ok
        _Cand(ships=15, eta=60),      # ok, high_eta flag
        _Cand(ships=1, eta=70),       # strict reject (small_fleet_long_journey)
        _Cand(ships=10, eta=200),     # strict reject (eta_gt_strict)
        _Cand(ships=0, eta=5),        # strict reject (ships_lt_1)
        _Cand(ships=20, eta=8, intended_tgt_id=5, actual_hit_id=None),  # off_target
    ]
    kept, flags, stats = filter_candidates(cands, ActionFilters())
    assert len(kept) == 2
    assert len(flags) == 2
    assert stats["eta_gt_strict"] == 1
    assert stats["small_fleet_long_journey"] == 1
    assert stats["ships_lt_1"] == 1
    assert stats["off_target_hit"] == 1


def test_ablation_config_passes_everything():
    """The ablation hatch — relax all knobs, expect everything to pass."""
    ablation = ActionFilters(
        max_eta_strict=999,
        small_long_journey_ships=0,
        small_long_journey_eta=999,
        require_hits_intended=False,
    )
    cands = [
        _Cand(ships=1, eta=70),
        _Cand(ships=10, eta=200),
        _Cand(ships=20, eta=8, intended_tgt_id=5, actual_hit_id=99),
    ]
    kept, _, stats = filter_candidates(cands, ablation)
    assert len(kept) == 3
    assert sum(stats.values()) == 0
