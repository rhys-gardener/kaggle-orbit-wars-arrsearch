"""Build array-only graph scenario caches.

Output layout mirrors the replay cache builder:

    out_root/
      manifest.json
      seed_1234/
        geometry.npz
        manifest.json
        training/
          manifest.json
          records_00000.pkl
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.array_search.action_filters import ActionFilters
from src.array_search.labels import label_records
from src.array_search.records import flush_records
from src.array_search.scenarios import generate_initial_arrays
from src.array_search.self_play import HeuristicPolicy, RandomPolicy, run_array_self_play
from src.array_search.state_adapter import arrays_to_obs
from src.graph_training.geometry_cache import build_geometry_cache_from_observations
from src.graph_training.sparse_geometry_cache import DEFAULT_SHIP_BUCKETS
from src.graph_training.state import EDGE_FEATURE_NAMES, PLANET_FEATURE_NAMES


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def geometry_observations(scenario, steps: list[int]) -> list[dict[str, Any]]:
    return [
        arrays_to_obs(
            scenario,
            seat=0,
            rel_step=int(step),
            owners=scenario.owners,
            ships=scenario.ships,
            schedule={},
        )
        for step in steps
    ]


def mean_or_zero(values: list[int | float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def build_one(args: argparse.Namespace, seed: int, out_root: Path) -> dict[str, Any]:
    scenario_id = f"seed_{int(seed)}"
    scenario_dir = out_root / scenario_id
    if scenario_dir.exists() and args.overwrite:
        shutil.rmtree(scenario_dir)
    if scenario_dir.exists() and any(scenario_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{scenario_dir} exists; pass --overwrite or choose a new output root")
    training_dir = scenario_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    scenario = generate_initial_arrays(seed, num_players=args.num_players)
    steps = list(range(0, max(1, args.horizon_turns + args.geometry_extra_turns), max(1, args.geometry_stride)))
    if steps[-1] != args.horizon_turns + args.geometry_extra_turns:
        steps.append(args.horizon_turns + args.geometry_extra_turns)

    t0 = time.perf_counter()
    geometry_cache = build_geometry_cache_from_observations(
        geometry_observations(scenario, steps),
        parse_ints(args.ship_buckets),
        max_steps=args.max_engine_steps,
        keep_only_eta_improvements=not args.no_eta_prune,
        min_eta_improvement=args.min_eta_improvement,
        max_chase_ratio=args.geometry_max_chase_ratio,
        metadata={
            "scenario_id": scenario_id,
            "seed": int(seed),
            "num_players": int(args.num_players),
            "steps": steps,
            "ship_buckets": parse_ints(args.ship_buckets),
        },
    )
    geometry_seconds = time.perf_counter() - t0
    geometry_cache.metadata["build_seconds"] = geometry_seconds
    geometry_cache.metadata["entry_count"] = int(geometry_cache.reachable.size)
    geometry_cache.metadata["reachable_count"] = int(geometry_cache.reachable.sum())
    geometry_cache.save(scenario_dir / "geometry.npz")

    policy = HeuristicPolicy() if args.policy == "heuristic" else RandomPolicy(seed=seed)
    t1 = time.perf_counter()
    result = run_array_self_play(
        scenario=scenario,
        geometry_cache=geometry_cache,
        policies={args.policy: policy},
        horizon_turns=args.horizon_turns,
        candidate_limit=args.candidate_limit,
        max_launches=args.max_launches,
        beam_width=args.beam_width,
        max_same_target=args.max_same_target,
        include_support=args.include_support,
        filters=ActionFilters(max_launches_per_turn=args.max_launches),
        max_chase_ratio=args.candidate_max_chase_ratio,
        chase_eta_allow=args.chase_eta_allow,
        scenario_id=scenario_id,
    )
    records = result["records"]
    if args.label_records:
        label_records(records, horizons=tuple(parse_ints(args.label_horizons)), primary_horizon=args.primary_horizon)
    training_seconds = time.perf_counter() - t1

    shards = []
    for shard_idx, start in enumerate(range(0, len(records), max(1, args.shard_size))):
        shard = records[start : start + args.shard_size]
        shards.append(flush_records(training_dir / f"records_{shard_idx:05d}.pkl", shard))

    candidate_counts = [len(r.get("candidates", [])) for r in records]
    action_set_counts = [len(r.get("action_sets", [])) for r in records]
    training_manifest = {
        "record_count": len(records),
        "shards": shards,
        "wall_seconds": training_seconds,
        "records_per_second": len(records) / training_seconds if training_seconds > 0 else 0.0,
        "candidates": {"mean": mean_or_zero(candidate_counts), "max": max(candidate_counts) if candidate_counts else 0},
        "action_sets": {"mean": mean_or_zero(action_set_counts), "max": max(action_set_counts) if action_set_counts else 0},
        "reject_counts": result["reject_counts"],
        "labelled": bool(args.label_records),
    }
    with open(training_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(training_manifest, f, indent=2, sort_keys=True)

    scenario_manifest = {
        "scenario_id": scenario_id,
        "seed": int(seed),
        "num_players": int(args.num_players),
        "geometry": {
            "file": "geometry.npz",
            "steps": len(geometry_cache.steps),
            "planets": len(geometry_cache.planet_ids),
            "ship_buckets": [int(x) for x in geometry_cache.ship_buckets.tolist()],
            "entry_count": int(geometry_cache.reachable.size),
            "reachable_count": int(geometry_cache.reachable.sum()),
            "build_seconds": geometry_seconds,
        },
        "training": training_manifest,
        "planet_feature_names": list(PLANET_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
    }
    with open(scenario_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(scenario_manifest, f, indent=2, sort_keys=True)
    return scenario_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--horizon-turns", type=int, default=50)
    parser.add_argument("--geometry-stride", type=int, default=1)
    parser.add_argument("--geometry-extra-turns", type=int, default=120)
    parser.add_argument("--ship-buckets", default=",".join(str(x) for x in DEFAULT_SHIP_BUCKETS))
    parser.add_argument("--max-engine-steps", type=int, default=160)
    parser.add_argument("--no-eta-prune", action="store_true")
    parser.add_argument("--min-eta-improvement", type=float, default=1.0)
    parser.add_argument("--geometry-max-chase-ratio", type=float, default=0.0)
    parser.add_argument("--candidate-limit", type=int, default=80)
    parser.add_argument("--max-launches", type=int, default=10)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--max-same-target", type=int, default=3)
    parser.add_argument("--candidate-max-chase-ratio", type=float, default=0.0)
    parser.add_argument("--chase-eta-allow", type=int, default=12)
    parser.add_argument("--include-support", action="store_true")
    parser.add_argument("--policy", choices=["random", "heuristic"], default="random")
    parser.add_argument("--label-records", action="store_true")
    parser.add_argument("--label-horizons", default="30,60,120")
    parser.add_argument("--primary-horizon", type=int, default=60)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds)) if args.num_seeds else parse_ints(args.seeds)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    scenarios = []
    for seed in seeds:
        print(f"scenario seed_{seed}: build")
        scenarios.append(build_one(args, seed, out_root))
        print(f"scenario seed_{seed}: records={scenarios[-1]['training']['record_count']}")

    root_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "scenario_count": len(scenarios),
        "record_count": sum(int(s["training"]["record_count"]) for s in scenarios),
        "wall_seconds": time.perf_counter() - start,
        "scenarios": [
            {
                "scenario_id": s["scenario_id"],
                "seed": s["seed"],
                "records": s["training"]["record_count"],
                "geometry_file": f"{s['scenario_id']}/geometry.npz",
                "training_dir": f"{s['scenario_id']}/training",
            }
            for s in scenarios
        ],
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(root_manifest, f, indent=2, sort_keys=True)
    print(f"Scenario cache root: {out_root}")
    print(f"Records: {root_manifest['record_count']}")


if __name__ == "__main__":
    main()
