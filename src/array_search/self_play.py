"""Array-only self-play rollout helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from src.array_search.action_filters import ActionFilters, filter_candidates
from src.array_search.ranker import CandidateRanker, greedy_pack_action_set, score_candidates
from src.array_search.records import build_record_dict
from src.array_search.scenarios import InitialScenario, active_planet_mask
from src.array_search.state_adapter import arrays_to_obs
from src.graph_training.actions import generate_cached_launch_candidates
from src.graph_training.array_env import PendingLaunch, ScheduledFleet, step_multi_seat
from src.graph_training.geometry_cache import GeometryCache
from src.graph_training.search import generate_action_sets
from src.graph_training.state import build_graph_state


class Policy(Protocol):
    policy_id: str

    def choose(self, record: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class RandomPolicy:
    policy_id: str = "random"
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(int(self.seed))

    def choose(self, record: dict[str, Any]) -> dict[str, Any]:
        action_sets = record.get("action_sets", [])
        if not action_sets:
            return {"score": 0.0, "candidate_indices": [], "actions": []}
        return dict(self._rng.choice(action_sets))


@dataclass
class HeuristicPolicy:
    policy_id: str = "heuristic"

    def choose(self, record: dict[str, Any]) -> dict[str, Any]:
        action_sets = record.get("action_sets", [])
        if not action_sets:
            return {"score": 0.0, "candidate_indices": [], "actions": []}
        return dict(max(action_sets, key=lambda item: float(item.get("score", 0.0))))


@dataclass
class RankerPolicy:
    model: CandidateRanker
    policy_id: str
    device: str = "cpu"
    max_launches: int = 10
    multi_source_bonus: float = 0.15
    temperature: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        # Decision-time exploration: temperature>0 perturbs candidate ordering so
        # self-play visits a wider action space. Eval/yardstick leave it at 0.
        self._rng = np.random.default_rng(int(self.seed)) if float(self.temperature) > 0.0 else None

    def choose(self, record: dict[str, Any]) -> dict[str, Any]:
        scores = score_candidates(self.model, record, device=self.device)
        return greedy_pack_action_set(
            record,
            scores,
            max_launches=self.max_launches,
            multi_source_bonus=self.multi_source_bonus,
            temperature=self.temperature,
            rng=self._rng,
        )


def action_set_to_pending_launches(
    record: dict[str, Any],
    action_set: dict[str, Any],
    *,
    owner: int,
    planet_id_to_index: dict[int, int] | None = None,
) -> list[PendingLaunch]:
    id_to_index = planet_id_to_index or {
        int(pid): i for i, pid in enumerate(record["graph"]["planet_ids"])
    }
    out: list[PendingLaunch] = []
    for idx in action_set.get("candidate_indices", []):
        candidate = record["candidates"][int(idx)]
        src_idx = id_to_index.get(int(candidate["src_id"]))
        hit_id = candidate.get("actual_hit_id")
        tgt_idx = None if hit_id is None else id_to_index.get(int(hit_id))
        if src_idx is None or tgt_idx is None:
            continue
        out.append(
            PendingLaunch(
                source_idx=int(src_idx),
                target_idx=int(tgt_idx),
                owner=int(owner),
                ships=int(candidate["ships"]),
                eta=max(1, int(candidate["eta"])),
            )
        )
    return out


def _policy_for_seat(
    seat: int,
    policies: dict[str, Policy],
    *,
    seed: int,
) -> tuple[str, Policy]:
    if "A" in policies and "B" in policies:
        # Rotate parity by seed so the two policies do not inherit permanent
        # first/second-player bias.
        use_a = ((int(seat) + int(seed)) % 2) == 0
        policy_id = "A" if use_a else "B"
        return policy_id, policies[policy_id]
    key = sorted(policies)[int(seat) % len(policies)]
    return key, policies[key]


def run_array_self_play(
    *,
    scenario: InitialScenario,
    geometry_cache: GeometryCache,
    policies: dict[str, Policy],
    horizon_turns: int,
    candidate_limit: int = 80,
    max_launches: int = 10,
    beam_width: int = 32,
    max_same_target: int = 3,
    include_support: bool = True,
    filters: ActionFilters | None = None,
    max_chase_ratio: float = 0.0,
    chase_eta_allow: int = 12,
    scenario_id: str | None = None,
    keep_records: bool = True,
) -> dict[str, Any]:
    """Roll one scenario forward using array-only state transitions."""

    filters = filters or ActionFilters()
    owners = scenario.owners.copy()
    ships = scenario.ships.copy()
    production = scenario.production.copy()
    schedule: dict[int, list[ScheduledFleet]] = {}
    planet_id_to_index = {int(pid): i for i, pid in enumerate(scenario.planet_ids.tolist())}
    rel_turn = 0
    records: list[dict[str, Any]] = []
    reject_totals: dict[str, int] = {}
    scenario_id = scenario_id or f"seed_{scenario.seed}"

    for _turn in range(int(horizon_turns)):
        pending: list[PendingLaunch] = []
        for seat in range(int(scenario.num_players)):
            obs = arrays_to_obs(
                scenario,
                seat=seat,
                rel_step=rel_turn,
                owners=owners,
                ships=ships,
                schedule=schedule,
            )
            ctx = build_graph_state(obs)
            raw_candidates = generate_cached_launch_candidates(
                ctx,
                geometry_cache,
                max_candidates=candidate_limit,
                include_support=include_support,
                max_chase_ratio=max_chase_ratio,
                chase_eta_allow=chase_eta_allow,
            )
            candidates, flags, reject_counts = filter_candidates(raw_candidates, filters)
            for key, value in reject_counts.items():
                reject_totals[key] = reject_totals.get(key, 0) + int(value)
            action_sets = generate_action_sets(
                ctx,
                candidates,
                max_launches=min(max_launches, filters.max_launches_per_turn),
                beam_width=beam_width,
                max_same_target=max_same_target,
                candidate_limit=candidate_limit,
                include_support=include_support,
            )
            policy_id, policy = _policy_for_seat(seat, policies, seed=scenario.seed)
            record = build_record_dict(
                scenario_id=scenario_id,
                seed=scenario.seed,
                rel_turn=rel_turn,
                seat=seat,
                policy_id=policy_id,
                obs=obs,
                ctx=ctx,
                candidates=candidates,
                candidate_flags=flags,
                action_sets=action_sets,
                reject_counts=reject_counts,
            )
            chosen = policy.choose(record)
            record["chosen_action_set"] = dict(chosen)
            if keep_records:
                records.append(record)
            pending.extend(
                action_set_to_pending_launches(
                    record,
                    chosen,
                    owner=seat,
                    planet_id_to_index=planet_id_to_index,
                )
            )

        rel_turn = step_multi_seat(
            owners,
            ships,
            production,
            schedule,
            pending,
            current_rel_turn=rel_turn,
            active_mask=active_planet_mask(scenario, rel_turn),
            arrival_active_mask=active_planet_mask(scenario, rel_turn + 1),
        )

    final_active_mask = active_planet_mask(scenario, rel_turn)
    return {
        "records": records,
        "owners": owners,
        "ships": ships,
        "production": production,
        "schedule": schedule,
        "rel_turn": rel_turn,
        "active_mask": final_active_mask,
        "reject_counts": reject_totals,
    }


def seat_scores(
    owners: np.ndarray,
    ships: np.ndarray,
    production: np.ndarray,
    num_players: int,
    active_mask: np.ndarray | None = None,
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    active = np.ones(len(owners), dtype=bool) if active_mask is None else active_mask.astype(bool, copy=False)
    for seat in range(int(num_players)):
        mask = (owners == seat) & active
        out[seat] = {
            "production": float(production[mask].sum()),
            "ships": float(ships[mask].sum()),
            "planets": float(mask.sum()),
        }
    return out


def clone_ranker(model: CandidateRanker) -> CandidateRanker:
    clone = CandidateRanker()
    clone.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
    return clone


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
