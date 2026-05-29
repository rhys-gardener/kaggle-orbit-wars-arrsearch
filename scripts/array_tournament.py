"""Array-only tournament harness for ranker checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.array_search.ranker import load_ranker
from src.array_search.scenarios import generate_initial_arrays
from src.array_search.self_play import HeuristicPolicy, RankerPolicy, seat_scores, run_array_self_play
from src.array_search.state_adapter import arrays_to_obs
from src.graph_training.geometry_cache import build_geometry_cache_from_observations
from src.graph_training.sparse_geometry_cache import DEFAULT_SHIP_BUCKETS


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_geometry(scenario, *, stride: int, horizon_turns: int, buckets: list[int]):
    steps = list(range(0, horizon_turns + 121, max(1, stride)))
    if steps[-1] != horizon_turns + 120:
        steps.append(horizon_turns + 120)
    observations = [
        arrays_to_obs(scenario, seat=0, rel_step=step, owners=scenario.owners, ships=scenario.ships, schedule={})
        for step in steps
    ]
    return build_geometry_cache_from_observations(
        observations,
        buckets,
        keep_only_eta_improvements=True,
        metadata={"seed": int(scenario.seed), "tournament": True},
    )


def policy_from_arg(text: str, *, policy_id: str, device: str):
    if text == "heuristic":
        return HeuristicPolicy(policy_id=policy_id)
    model = load_ranker(text, map_location=device)
    return RankerPolicy(model=model, policy_id=policy_id, device=device)


def play_game(args: argparse.Namespace, seed: int, policy_a, policy_b) -> dict[str, Any]:
    scenario = generate_initial_arrays(seed, num_players=args.num_players)
    geometry = build_geometry(
        scenario,
        stride=args.geometry_stride,
        horizon_turns=args.horizon_turns,
        buckets=parse_ints(args.ship_buckets),
    )
    result = run_array_self_play(
        scenario=scenario,
        geometry_cache=geometry,
        policies={"A": policy_a, "B": policy_b},
        horizon_turns=args.horizon_turns,
        candidate_limit=args.candidate_limit,
        max_launches=args.max_launches,
        beam_width=args.beam_width,
        max_same_target=args.max_same_target,
        include_support=args.include_support,
        keep_records=False,
    )
    scores = seat_scores(
        result["owners"],
        result["ships"],
        result["production"],
        args.num_players,
        active_mask=result.get("active_mask"),
    )
    policy_scores = {"A": 0.0, "B": 0.0}
    for seat, values in scores.items():
        pid = "A" if ((seat + seed) % 2) == 0 else "B"
        policy_scores[pid] += values["production"] * 1000.0 + values["ships"]
    winner = "A" if policy_scores["A"] > policy_scores["B"] else "B"
    return {"seed": int(seed), "seat_scores": scores, "policy_scores": policy_scores, "winner": winner}


def run_tournament(args: argparse.Namespace) -> dict[str, Any]:
    policy_a = policy_from_arg(args.policy_a, policy_id="A", device=args.device)
    policy_b = policy_from_arg(args.policy_b, policy_id="B", device=args.device)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds)) if args.num_seeds else parse_ints(args.seeds)
    games = [play_game(args, seed, policy_a, policy_b) for seed in seeds]
    wins_a = sum(1 for game in games if game["winner"] == "A")
    prod_leads = [game["policy_scores"]["A"] - game["policy_scores"]["B"] for game in games]
    return {
        "games": games,
        "n_games": len(games),
        "wins_a": wins_a,
        "wins_b": len(games) - wins_a,
        "win_rate_a": wins_a / max(len(games), 1),
        "mean_policy_score_lead_a": sum(prod_leads) / max(len(prod_leads), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-a", required=True, help="Checkpoint path or 'heuristic'.")
    parser.add_argument("--policy-b", required=True, help="Checkpoint path or 'heuristic'.")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--horizon-turns", type=int, default=300)
    parser.add_argument("--geometry-stride", type=int, default=1)
    parser.add_argument("--ship-buckets", default=",".join(str(x) for x in DEFAULT_SHIP_BUCKETS))
    parser.add_argument("--candidate-limit", type=int, default=80)
    parser.add_argument("--max-launches", type=int, default=10)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--max-same-target", type=int, default=3)
    parser.add_argument("--include-support", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    result = run_tournament(args)
    print(json.dumps({k: v for k, v in result.items() if k != "games"}, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
