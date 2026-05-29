"""Small candidate-row ranker and action-set composition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.array_search.features import FEATURE_DIM, FEATURE_NAMES, candidate_feature_matrix
from src.array_search.labels import PRIMARY_HORIZON, label_record, primary_positive_candidates


class CandidateRanker(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class RankerTrainStats:
    loss: float
    loss_listwise: float
    loss_pairwise: float
    records_used: int
    mean_candidates_per_turn: float


def record_targets(record: dict[str, Any], *, horizon: int = PRIMARY_HORIZON) -> np.ndarray:
    n = len(record.get("candidates", []))
    target = np.zeros(n, dtype=np.float32)
    positives = primary_positive_candidates(record, horizon=horizon)
    positives = {idx for idx in positives if 0 <= idx < n}
    if positives:
        value = 1.0 / len(positives)
        for idx in positives:
            target[idx] = value
    return target


def train_ranker_epoch(
    model: CandidateRanker,
    records: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    horizon: int = PRIMARY_HORIZON,
    pairwise_weight: float = 0.1,
    device: str | torch.device = "cpu",
) -> RankerTrainStats:
    model.train()
    device = torch.device(device)
    model.to(device)
    total = 0.0
    total_list = 0.0
    total_pair = 0.0
    used = 0
    candidate_counts: list[int] = []
    for record in records:
        if "labels" not in record:
            label_record(record, primary_horizon=horizon)
        x_np = candidate_feature_matrix(record)
        y_np = record_targets(record, horizon=horizon)
        if len(x_np) < 2 or float(y_np.sum()) <= 0.0:
            continue
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        scores = model(x)
        log_probs = F.log_softmax(scores, dim=0)
        loss_list = -(y * log_probs).sum()

        pos_mask = y > 0
        bottom_k = max(1, len(y_np) // 4)
        bottom_idx = torch.argsort(scores.detach())[:bottom_k]
        if bool(pos_mask.any()):
            pos_scores = scores[pos_mask].mean()
            neg_scores = scores[bottom_idx]
            loss_pair = F.relu(1.0 - pos_scores + neg_scores).mean()
        else:
            loss_pair = torch.zeros((), device=device)
        source = str((record.get("labels") or {}).get("source", "rollout"))
        record_weight = 2.0 if source == "mcts" else (1.25 if source == "replay" else 1.0)
        loss = (loss_list + float(pairwise_weight) * loss_pair) * record_weight
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += float(loss.detach().cpu())
        total_list += float(loss_list.detach().cpu())
        total_pair += float(loss_pair.detach().cpu())
        used += 1
        candidate_counts.append(len(x_np))
    denom = max(used, 1)
    return RankerTrainStats(
        loss=total / denom,
        loss_listwise=total_list / denom,
        loss_pairwise=total_pair / denom,
        records_used=used,
        mean_candidates_per_turn=float(np.mean(candidate_counts)) if candidate_counts else 0.0,
    )


def score_candidates(model: CandidateRanker, record: dict[str, Any], *, device: str | torch.device = "cpu") -> np.ndarray:
    x_np = candidate_feature_matrix(record)
    if len(x_np) == 0:
        return np.zeros(0, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(x_np).to(device)).detach().cpu().numpy()
    return scores.astype(np.float32, copy=False)


def greedy_pack_action_set(
    record: dict[str, Any],
    scores: np.ndarray,
    *,
    max_launches: int = 10,
    multi_source_bonus: float = 0.15,
) -> dict[str, Any]:
    candidates = record.get("candidates", [])
    planet_ids = [int(pid) for pid in record["graph"]["planet_ids"]]
    id_to_index = {pid: i for i, pid in enumerate(planet_ids)}
    planets = record.get("observation", {}).get("planets", [])
    source_budget = {int(p[0]): float(p[5]) for p in planets}
    chosen: list[int] = []
    target_counts: dict[int, int] = {}
    order = list(np.argsort(-scores))
    for idx in order:
        if len(chosen) >= int(max_launches):
            break
        candidate = candidates[int(idx)]
        src_id = int(candidate["src_id"])
        tgt_id = candidate.get("actual_hit_id")
        if tgt_id is None or src_id not in id_to_index:
            continue
        ships = int(candidate["ships"])
        if source_budget.get(src_id, 0.0) < ships:
            continue
        source_budget[src_id] -= ships
        chosen.append(int(idx))
        target_counts[int(tgt_id)] = target_counts.get(int(tgt_id), 0) + 1

    packed_score = float(sum(float(scores[i]) for i in chosen))
    packed_score += float(multi_source_bonus) * sum(max(0, count - 1) for count in target_counts.values())
    return {
        "score": packed_score,
        "candidate_indices": chosen,
        "actions": [
            [candidates[i]["src_id"], candidates[i]["angle"], candidates[i]["ships"]]
            for i in chosen
        ],
    }


def save_ranker(path: str | Path, model: CandidateRanker, *, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
        "extra": dict(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_ranker(path: str | Path, *, map_location: str | torch.device = "cpu") -> CandidateRanker:
    payload = torch.load(path, map_location=map_location)
    model = CandidateRanker(input_dim=int(payload.get("input_dim", FEATURE_DIM)))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
