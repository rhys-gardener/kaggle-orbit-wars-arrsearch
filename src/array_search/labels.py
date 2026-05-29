"""Rollout-based labels for cached array-search records."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.graph_training.array_env import RolloutScore, rollout_cached_action_set


DEFAULT_HORIZONS = (30, 60, 120)
PRIMARY_HORIZON = 60


def score_key(score: RolloutScore) -> tuple[float, float, float, float, int, int]:
    """Priority order from the initiative plan."""

    return (
        float(score.production),
        float(score.production_lead),
        float(score.ship_total),
        float(score.ship_lead),
        int(score.planet_count),
        1 if score.alive else 0,
    )


def label_record(
    record: dict[str, Any],
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    primary_horizon: int = PRIMARY_HORIZON,
) -> dict[str, Any]:
    """Evaluate every action set and attach multi-horizon rollout labels."""

    action_sets = record.get("action_sets", [])
    labels: dict[str, Any] = {
        "horizons": [int(h) for h in horizons],
        "primary_horizon": int(primary_horizon),
        "action_sets": [],
        "best_action_set_by_horizon": {},
        "best_candidate_indices_by_horizon": {},
    }
    if not action_sets:
        record["labels"] = labels
        return labels

    per_action: list[dict[str, Any]] = []
    best_by_horizon: dict[int, tuple[int, RolloutScore]] = {}
    for idx, action_set in enumerate(action_sets):
        item: dict[str, Any] = {
            "action_set_index": int(idx),
            "candidate_indices": [int(x) for x in action_set.get("candidate_indices", [])],
            "scores": {},
        }
        for horizon in horizons:
            _owners, _ships, score = rollout_cached_action_set(record, idx, horizon=int(horizon))
            item["scores"][str(int(horizon))] = asdict(score)
            old = best_by_horizon.get(int(horizon))
            if old is None or score_key(score) > score_key(old[1]):
                best_by_horizon[int(horizon)] = (idx, score)
        per_action.append(item)

    labels["action_sets"] = per_action
    for horizon, (idx, _score) in best_by_horizon.items():
        labels["best_action_set_by_horizon"][str(horizon)] = int(idx)
        labels["best_candidate_indices_by_horizon"][str(horizon)] = [
            int(x) for x in action_sets[idx].get("candidate_indices", [])
        ]
    record["labels"] = labels
    return labels


def label_records(
    records: list[dict[str, Any]],
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    primary_horizon: int = PRIMARY_HORIZON,
) -> list[dict[str, Any]]:
    for record in records:
        label_record(record, horizons=horizons, primary_horizon=primary_horizon)
    return records


def primary_positive_candidates(record: dict[str, Any], *, horizon: int = PRIMARY_HORIZON) -> set[int]:
    labels = record.get("labels") or label_record(record, primary_horizon=horizon)
    values = labels.get("best_candidate_indices_by_horizon", {}).get(str(int(horizon)), [])
    return {int(x) for x in values}


def attach_positive_candidate_label(
    record: dict[str, Any],
    candidate_indices: list[int],
    *,
    source: str,
    horizon: int = PRIMARY_HORIZON,
    action_set_index: int | None = None,
    action_set_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Attach non-rollout teacher labels in the same shape the ranker reads."""

    positives = sorted({int(i) for i in candidate_indices if int(i) >= 0})
    labels = {
        "horizons": [int(horizon)],
        "primary_horizon": int(horizon),
        "source": str(source),
        "action_sets": [],
        "best_action_set_by_horizon": {},
        "best_candidate_indices_by_horizon": {str(int(horizon)): positives},
    }
    if action_set_index is not None:
        labels["best_action_set_by_horizon"][str(int(horizon))] = int(action_set_index)
    if action_set_weights is not None:
        labels["action_set_weights"] = [float(x) for x in action_set_weights]
    record["labels"] = labels
    return labels
