"""Serialization helpers for array-search training records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.graph_training.actions import LaunchCandidate
from src.graph_training.search import ActionSet
from src.graph_training.state import GraphState


SCHEMA_VERSION = 2


def candidate_key(candidate: LaunchCandidate | dict[str, Any]) -> tuple:
    if isinstance(candidate, dict):
        return (
            int(candidate["src_id"]),
            int(candidate["intended_tgt_id"]),
            None if candidate.get("actual_hit_id") is None else int(candidate["actual_hit_id"]),
            int(candidate["ships"]),
            round(float(candidate["angle"]), 10),
            str(candidate["kind"]),
            str(candidate["bucket"]),
        )
    return (
        int(candidate.src_id),
        int(candidate.intended_tgt_id),
        None if candidate.actual_hit_id is None else int(candidate.actual_hit_id),
        int(candidate.ships),
        round(float(candidate.angle), 10),
        str(candidate.kind),
        str(candidate.bucket),
    )


def candidate_to_record(candidate: LaunchCandidate, flags: Any | None = None) -> dict[str, Any]:
    out = {
        "src_id": int(candidate.src_id),
        "intended_tgt_id": int(candidate.intended_tgt_id),
        "actual_hit_id": None if candidate.actual_hit_id is None else int(candidate.actual_hit_id),
        "angle": float(candidate.angle),
        "ships": int(candidate.ships),
        "eta": int(candidate.eta),
        "hit_reason": str(candidate.hit_reason),
        "kind": str(candidate.kind),
        "bucket": str(candidate.bucket),
        "score": float(candidate.score),
        "required_ships": int(candidate.required_ships),
        "target_production": float(candidate.target_production),
        "target_owner": int(candidate.target_owner),
    }
    if flags is not None:
        out["flags"] = {
            "is_below_typical_min_ships": bool(flags.is_below_typical_min_ships),
            "is_high_eta_launch": bool(flags.is_high_eta_launch),
        }
    return out


def action_set_to_record(action_set: ActionSet, candidate_index: dict[tuple, int]) -> dict[str, Any]:
    indices = []
    for launch in action_set.launches:
        idx = candidate_index.get(candidate_key(launch))
        if idx is not None:
            indices.append(int(idx))
    return {
        "score": float(action_set.score),
        "candidate_indices": indices,
        "actions": action_set.actions(),
    }


def graph_to_record(ctx: GraphState) -> dict[str, Any]:
    return {
        "player": int(ctx.player),
        "step": int(ctx.step),
        "angular_velocity": float(ctx.angular_velocity),
        "planet_ids": [int(pid) for pid in ctx.planet_ids],
        "my_planet_ids": sorted(int(pid) for pid in ctx.my_planet_ids),
        "enemy_player_ids": sorted(int(pid) for pid in ctx.enemy_player_ids),
        "planet_features": ctx.planet_features.astype(np.float32, copy=False),
        "edge_features": ctx.edge_features.astype(np.float32, copy=False),
        "valid_edge_mask": ctx.valid_edge_mask.astype(np.bool_, copy=False),
    }


def inbound_events_to_records(ctx: GraphState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for events in ctx.inbound_events.values():
        for event in events:
            rows.append(
                {
                    "fleet_id": int(event.fleet_id),
                    "owner": int(event.owner),
                    "source_id": int(event.source_id),
                    "target_id": int(event.target_id),
                    "ships": int(event.ships),
                    "eta": int(event.eta),
                    "angle_diff": float(event.angle_diff),
                }
            )
    rows.sort(key=lambda item: (item["eta"], item["target_id"], item["owner"], item["fleet_id"]))
    return rows


def build_record_dict(
    *,
    scenario_id: str,
    seed: int,
    rel_turn: int,
    seat: int,
    policy_id: str,
    obs: dict[str, Any],
    ctx: GraphState,
    candidates: list[LaunchCandidate],
    candidate_flags: list[Any],
    action_sets: list[ActionSet],
    reject_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    cidx = {candidate_key(candidate): i for i, candidate in enumerate(candidates)}
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "kind": "array_self_play",
            "scenario_id": str(scenario_id),
            "seed": int(seed),
            "rel_turn": int(rel_turn),
            "player": int(seat),
            "policy_id": str(policy_id),
        },
        "observation": obs,
        "graph": graph_to_record(ctx),
        "inbound_events": inbound_events_to_records(ctx),
        "candidates": [
            candidate_to_record(candidate, candidate_flags[i] if i < len(candidate_flags) else None)
            for i, candidate in enumerate(candidates)
        ],
        "action_sets": [action_set_to_record(action_set, cidx) for action_set in action_sets],
        "filter_stats": dict(reject_counts or {}),
    }


def flush_records(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {"file": path.name, "records": len(records), "bytes": path.stat().st_size}


def load_record_shards(paths: list[Path]) -> list[dict[str, Any]]:
    import pickle

    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, list):
            rows.extend(loaded)
    return rows
