"""Benchmark geometry-cache strategies for array-search self-play.

This script compares:

* dense precompute at one or more step strides,
* lazy exact lookup, where geometry entries are solved only when candidate
  generation asks for them.

The output is intentionally compact JSON so runs can be compared over time.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.array_search.scenarios import generate_initial_arrays
from src.array_search.self_play import HeuristicPolicy, RandomPolicy, run_array_self_play
from src.array_search.state_adapter import arrays_to_obs
from src.graph_training.geometry_cache import build_geometry_cache_from_observations
from src.graph_training.sparse_geometry_cache import DEFAULT_SHIP_BUCKETS, SparseGeometryCache


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


@dataclass
class StrategyResult:
    strategy: str
    build_seconds: float
    game_seconds_mean: float
    game_seconds_first: float
    game_seconds_warm_mean: float
    game_seconds_p95: float
    records_per_second_mean: float
    lookup_count: int | None = None
    solve_count: int | None = None
    cache_hits: int | None = None
    cache_size: int | None = None
    reachable_count: int | None = None


def dense_geometry(scenario, *, stride: int, horizon_turns: int, extra_turns: int, buckets: list[int], max_steps: int):
    steps = list(range(0, int(horizon_turns) + int(extra_turns) + 1, max(1, int(stride))))
    observations = [
        arrays_to_obs(scenario, seat=0, rel_step=step, owners=scenario.owners, ships=scenario.ships, schedule={})
        for step in steps
    ]
    return build_geometry_cache_from_observations(
        observations,
        buckets,
        max_steps=int(max_steps),
        keep_only_eta_improvements=True,
        metadata={"benchmark": True, "stride": int(stride)},
    )


def run_games(args: argparse.Namespace, scenario, geometry, *, policy_kind: str) -> tuple[list[float], list[float]]:
    times: list[float] = []
    rates: list[float] = []
    for repeat in range(int(args.repeats)):
        policy = HeuristicPolicy() if policy_kind == "heuristic" else RandomPolicy(seed=args.seed + repeat)
        t0 = time.perf_counter()
        result = run_array_self_play(
            scenario=scenario,
            geometry_cache=geometry,
            policies={policy.policy_id: policy},
            horizon_turns=args.horizon_turns,
            candidate_limit=args.candidate_limit,
            max_launches=args.max_launches,
            beam_width=args.beam_width,
            max_same_target=args.max_same_target,
            include_support=args.include_support,
            keep_records=args.keep_records,
        )
        elapsed = time.perf_counter() - t0
        records = len(result.get("records", [])) if args.keep_records else args.horizon_turns * args.num_players
        times.append(elapsed)
        rates.append(float(records) / elapsed if elapsed > 0 else 0.0)
    return times, rates


def summarize_game_times(times: list[float], rates: list[float]) -> tuple[float, float, float]:
    if not times:
        return 0.0, 0.0, 0.0
    p95_idx = min(len(times) - 1, math.ceil(0.95 * len(times)) - 1)
    return (
        float(statistics.fmean(times)),
        float(sorted(times)[p95_idx]),
        float(statistics.fmean(rates)) if rates else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--horizon-turns", type=int, default=20)
    parser.add_argument("--geometry-extra-turns", type=int, default=40)
    parser.add_argument("--ship-buckets", default=",".join(str(x) for x in DEFAULT_SHIP_BUCKETS))
    parser.add_argument("--max-engine-steps", type=int, default=80)
    parser.add_argument("--strides", default="1,5,10")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=40)
    parser.add_argument("--max-launches", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=12)
    parser.add_argument("--max-same-target", type=int, default=3)
    parser.add_argument("--include-support", action="store_true")
    parser.add_argument("--policy", choices=["random", "heuristic"], default="heuristic")
    parser.add_argument("--keep-records", action="store_true")
    parser.add_argument("--amortize-runs", default="1,10,100,1000,10000")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    buckets = parse_ints(args.ship_buckets)
    scenario = generate_initial_arrays(args.seed, num_players=args.num_players)
    results: list[StrategyResult] = []

    for stride in ([] if args.skip_dense else parse_ints(args.strides)):
        t0 = time.perf_counter()
        geometry = dense_geometry(
            scenario,
            stride=stride,
            horizon_turns=args.horizon_turns,
            extra_turns=args.geometry_extra_turns,
            buckets=buckets,
            max_steps=args.max_engine_steps,
        )
        build_seconds = time.perf_counter() - t0
        times, rates = run_games(args, scenario, geometry, policy_kind=args.policy)
        mean_s, p95_s, mean_rate = summarize_game_times(times, rates)
        results.append(
            StrategyResult(
                strategy=f"dense_stride_{stride}",
                build_seconds=build_seconds,
                game_seconds_mean=mean_s,
                game_seconds_first=times[0] if times else 0.0,
                game_seconds_warm_mean=float(statistics.fmean(times[1:])) if len(times) > 1 else 0.0,
                game_seconds_p95=p95_s,
                records_per_second_mean=mean_rate,
                reachable_count=int(geometry.reachable.sum()),
                cache_size=int(geometry.reachable.size),
            )
        )

    lazy = SparseGeometryCache(scenario, buckets, max_steps=args.max_engine_steps)
    t0 = time.perf_counter()
    times, rates = run_games(args, scenario, lazy, policy_kind=args.policy)
    build_seconds = time.perf_counter() - t0 - sum(times)
    mean_s, p95_s, mean_rate = summarize_game_times(times, rates)
    results.append(
        StrategyResult(
            strategy="lazy_exact",
            build_seconds=max(0.0, build_seconds),
            game_seconds_mean=mean_s,
            game_seconds_first=times[0] if times else 0.0,
            game_seconds_warm_mean=float(statistics.fmean(times[1:])) if len(times) > 1 else 0.0,
            game_seconds_p95=p95_s,
            records_per_second_mean=mean_rate,
            lookup_count=lazy.lookup_count,
            solve_count=lazy.solve_count,
            cache_hits=lazy.cache_hits,
            cache_size=lazy.entry_count,
        )
    )

    amortize_runs = parse_ints(args.amortize_runs)
    payload: dict[str, Any] = {
        "args": vars(args),
        "results": [asdict(r) for r in results],
        "amortized_seconds_per_game": {
            str(runs): {
                r.strategy: (
                    (r.build_seconds + r.game_seconds_first + max(0, runs - 1) * r.game_seconds_warm_mean)
                    / max(runs, 1)
                    if r.strategy == "lazy_exact" and r.game_seconds_warm_mean > 0.0
                    else (r.build_seconds / max(runs, 1)) + r.game_seconds_mean
                )
                for r in results
            }
            for runs in amortize_runs
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
