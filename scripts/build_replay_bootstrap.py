"""Build replay-bootstrap records and an optional initial ranker checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.array_search.action_filters import ActionFilters, filter_candidates
from src.array_search.labels import attach_positive_candidate_label
from src.array_search.ranker import CandidateRanker, save_ranker, train_ranker_epoch
from src.array_search.records import build_record_dict, flush_records
from src.graph_training.actions import LaunchCandidate, generate_launch_candidates
from src.graph_training.search import ActionSet, generate_action_sets
from src.graph_training.sparse_geometry_cache import DEFAULT_SHIP_BUCKETS
from src.graph_training.state import build_graph_state
from src.physics import _trajectory_first_hit


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _angle_diff(a: float, b: float) -> float:
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


def winners_for_episode(episode: dict[str, Any]) -> set[int]:
    rewards = episode.get("rewards") or []
    statuses = episode.get("statuses") or []
    if not rewards:
        return set()
    max_reward = max(r for r in rewards if r is not None)
    return {
        i
        for i, reward in enumerate(rewards)
        if reward == max_reward and (i >= len(statuses) or statuses[i] != "ERROR")
    }


def replay_candidate(ctx, action: list) -> LaunchCandidate | None:
    try:
        src_id = int(action[0])
        angle = float(action[1])
        ships = int(action[2])
    except (TypeError, ValueError, IndexError):
        return None
    if ships <= 0:
        return None
    source = ctx.planet_by_id.get(src_id)
    if source is None:
        return None
    hit_id, reason, hit_steps = _trajectory_first_hit(
        source,
        angle,
        ships,
        ctx.planets,
        ctx.angular_velocity,
        ctx.comet_ids,
        ctx.raw_comets,
        max_steps=180,
    )
    if hit_id is None:
        return None
    target = ctx.planet_by_id.get(int(hit_id))
    if target is None:
        return None
    return LaunchCandidate(
        src_id=src_id,
        intended_tgt_id=int(hit_id),
        actual_hit_id=int(hit_id),
        angle=angle,
        ships=ships,
        eta=int(hit_steps),
        hit_reason=str(reason),
        kind="replay",
        bucket=f"exact_{ships}",
        score=1000.0,
        required_ships=max(1, ships),
        target_production=float(target[6]),
        target_owner=int(target[1]),
    )


def find_or_inject_candidate(candidates: list[LaunchCandidate], replay: LaunchCandidate) -> int:
    best_idx = None
    best_key = (10**9, 10**9.0, 10**9)
    for idx, candidate in enumerate(candidates):
        if int(candidate.src_id) != int(replay.src_id):
            continue
        if candidate.actual_hit_id != replay.actual_hit_id:
            continue
        ship_delta = abs(int(candidate.ships) - int(replay.ships))
        angle_delta = _angle_diff(candidate.angle, replay.angle)
        key = (ship_delta, angle_delta, 0 if candidate.kind == "replay" else 1)
        if key < best_key:
            best_idx = idx
            best_key = key
    if best_idx is not None and best_key[0] <= max(2, int(round(replay.ships * 0.25))):
        return int(best_idx)
    candidates.append(replay)
    return len(candidates) - 1


def process_episode(
    episode: dict[str, Any],
    *,
    episode_path: Path,
    max_turns: int,
    winning_only: bool,
    candidate_limit: int,
    max_launches: int,
    beam_width: int,
    filters: ActionFilters,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    winners = winners_for_episode(episode)
    episode_id = str(episode.get("info", {}).get("EpisodeId") or episode.get("id") or episode_path.stem)
    seed = int(episode.get("info", {}).get("seed") or 0)
    for turn, step in enumerate(episode.get("steps") or []):
        if turn > int(max_turns):
            break
        for seat, seat_state in enumerate(step):
            if winning_only and seat not in winners:
                continue
            actions = seat_state.get("action") or []
            if not actions:
                continue
            obs = dict(seat_state.get("observation") or {})
            if not obs.get("planets"):
                continue
            obs["player"] = int(seat)
            ctx = build_graph_state(obs)
            replay_candidates = [c for action in actions if (c := replay_candidate(ctx, action)) is not None]
            if not replay_candidates:
                continue
            raw_candidates = generate_launch_candidates(
                ctx,
                max_candidates=max(candidate_limit, len(replay_candidates) + 16),
                include_support=True,
            )
            candidates, flags, reject_counts = filter_candidates(raw_candidates, filters)
            positive_indices = [find_or_inject_candidate(candidates, replay) for replay in replay_candidates]
            while len(flags) < len(candidates):
                flags.append(None)
            action_sets = generate_action_sets(
                ctx,
                candidates,
                max_launches=max(max_launches, len(positive_indices)),
                beam_width=beam_width,
                max_same_target=max_launches,
                candidate_limit=max(candidate_limit, len(candidates)),
                include_support=True,
            )
            replay_set = ActionSet(
                launches=tuple(candidates[i] for i in positive_indices),
                score=10_000.0,
            )
            action_sets = [replay_set] + [a for a in action_sets if set(positive_indices) != set(
                getattr(a, "candidate_indices", [])
            )]
            record = build_record_dict(
                scenario_id=f"replay_{episode_id}",
                seed=seed,
                rel_turn=turn,
                seat=seat,
                policy_id="replay",
                obs=obs,
                ctx=ctx,
                candidates=candidates,
                candidate_flags=flags,
                action_sets=action_sets,
                reject_counts=reject_counts,
            )
            record["source"]["kind"] = "replay_bootstrap"
            record["source"]["episode_id"] = episode_id
            record["source"]["episode_path"] = str(episode_path)
            attach_positive_candidate_label(
                record,
                positive_indices,
                source="replay",
                action_set_index=0,
            )
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", default="replays/sample")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-files", type=int, default=25)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--winning-only", action="store_true", default=True)
    parser.add_argument("--candidate-limit", type=int, default=80)
    parser.add_argument("--max-launches", type=int, default=10)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--ship-buckets", default=",".join(str(x) for x in DEFAULT_SHIP_BUCKETS))
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--train-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    records_dir = out_dir / "records"
    ckpt_dir = out_dir / "checkpoints"
    records_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    filters = ActionFilters(max_launches_per_turn=args.max_launches)
    paths = sorted(Path(args.replay_dir).glob("*.json"))[: max(0, int(args.max_files))]
    start = time.perf_counter()
    records: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            episode = json.load(f)
        records.extend(
            process_episode(
                episode,
                episode_path=path,
                max_turns=args.max_turns,
                winning_only=args.winning_only,
                candidate_limit=args.candidate_limit,
                max_launches=args.max_launches,
                beam_width=args.beam_width,
                filters=filters,
            )
        )

    shards = []
    for shard_idx, start_idx in enumerate(range(0, len(records), max(1, args.shard_size))):
        shards.append(flush_records(records_dir / f"records_{shard_idx:05d}.pkl", records[start_idx : start_idx + args.shard_size]))

    ckpt_path = None
    train_stats = []
    if records and args.train_epochs > 0:
        model = CandidateRanker()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for epoch in range(1, args.train_epochs + 1):
            random.shuffle(records)
            stats = train_ranker_epoch(model, records, optimizer, device=args.device)
            train_stats.append({**stats.__dict__, "epoch": epoch})
        ckpt_path = ckpt_dir / "replay_bootstrap_ranker.pt"
        save_ranker(ckpt_path, model, extra={"kind": "replay_bootstrap", "records": len(records)})

    manifest = {
        "wall_seconds": time.perf_counter() - start,
        "replay_dir": str(Path(args.replay_dir)),
        "files": [str(p) for p in paths],
        "record_count": len(records),
        "shards": shards,
        "checkpoint": None if ckpt_path is None else str(ckpt_path),
        "train_stats": train_stats,
        "ship_buckets": parse_ints(args.ship_buckets),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
