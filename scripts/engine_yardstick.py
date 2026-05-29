"""Real-engine yardstick: a trained ranker vs the kaggle_ender heuristic.

This is the *absolute* benchmark the self-play acceptance gate cannot provide:
it runs the real kaggle_environments engine (kaggle_sim=True) against a fixed,
non-lineage opponent (the >1000-leaderboard heuristic), so a rising score here
means real strength, not self-referential drift.

Requires: uv pip install "kaggle-environments>=1.28.0" --no-deps   (per CLAUDE.md)

Examples:
    uv run python scripts/engine_yardstick.py --checkpoint runs/<run>/checkpoints/ranker_A_round_0050.pt \
        --opponent kaggle_ender --num-seeds 8 --num-players 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.array_ranker_agent import make_agent
from sim import make_env


def _state_field(state: Any, name: str) -> Any:
    """kaggle_environments per-agent state is dict- or attribute-like."""
    if hasattr(state, name):
        return getattr(state, name)
    try:
        return state[name]
    except (KeyError, TypeError, IndexError):
        return None


def build_opponent(spec: str, *, device: str, candidate_limit: int, max_launches: int) -> Callable[[Any], Any]:
    if spec == "kaggle_ender":
        from agents.kaggle_ender import agent as ender_agent

        return ender_agent
    if spec == "random":
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent  # type: ignore

        return random_agent
    # Otherwise treat the spec as another ranker checkpoint path.
    return make_agent(spec, device=device, candidate_limit=candidate_limit, max_launches=max_launches)


def _seat_assignment(num_players: int, rotation: int) -> list[int]:
    """Seats the agent-under-test occupies for a given rotation (rest = opponent).

    Mirrors the training parity convention: rotation 0 -> even seats, 1 -> odd."""
    return [seat for seat in range(num_players) if (seat % 2) == (rotation % 2)]


def play_game(
    agent_fn: Callable[[Any], Any],
    opponent_fn: Callable[[Any], Any],
    *,
    seed: int,
    num_players: int,
    rotation: int,
    episode_steps: int,
) -> dict[str, Any]:
    agent_seats = set(_seat_assignment(num_players, rotation))
    agent_list = [agent_fn if seat in agent_seats else opponent_fn for seat in range(num_players)]
    env = make_env(
        configuration={"seed": int(seed), "episodeSteps": int(episode_steps)},
        kaggle_sim=True,
    )
    env.run(agent_list)
    final = env.steps[-1]
    agent_reward = 0.0
    opp_reward = 0.0
    statuses = []
    for seat in range(num_players):
        reward = _state_field(final[seat], "reward")
        statuses.append(_state_field(final[seat], "status"))
        reward = float(reward) if reward is not None else 0.0
        if seat in agent_seats:
            agent_reward += reward
        else:
            opp_reward += reward
    return {
        "seed": int(seed),
        "rotation": int(rotation),
        "agent_seats": sorted(agent_seats),
        "agent_reward": agent_reward,
        "opp_reward": opp_reward,
        "lead": agent_reward - opp_reward,
        "agent_win": bool(agent_reward > opp_reward),
        "statuses": statuses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Ranker .pt for the agent under test.")
    parser.add_argument("--opponent", default="kaggle_ender", help="'kaggle_ender' | 'random' | a ranker .pt path.")
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=200_000, help="Held out from training seeds.")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--seat-rotations", type=int, default=2, help="Position swaps per seed to neutralize bias.")
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--candidate-limit", type=int, default=80)
    parser.add_argument("--max-launches", type=int, default=10)
    parser.add_argument("--include-support", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    agent_fn = make_agent(
        args.checkpoint,
        device=args.device,
        candidate_limit=args.candidate_limit,
        max_launches=args.max_launches,
        include_support=args.include_support,
    )
    opponent_fn = build_opponent(
        args.opponent,
        device=args.device,
        candidate_limit=args.candidate_limit,
        max_launches=args.max_launches,
    )

    games: list[dict[str, Any]] = []
    rotations = max(1, min(int(args.seat_rotations), int(args.num_players)))
    for offset in range(int(args.num_seeds)):
        seed = int(args.seed_start) + offset
        for rotation in range(rotations):
            game = play_game(
                agent_fn,
                opponent_fn,
                seed=seed,
                num_players=int(args.num_players),
                rotation=rotation,
                episode_steps=int(args.episode_steps),
            )
            games.append(game)
            print(
                f"seed={game['seed']} rot={game['rotation']} "
                f"agent={game['agent_reward']:.1f} opp={game['opp_reward']:.1f} "
                f"{'WIN' if game['agent_win'] else 'loss'}"
            )

    n = max(1, len(games))
    wins = sum(1 for g in games if g["agent_win"])
    leads = [g["lead"] for g in games]
    summary = {
        "checkpoint": args.checkpoint,
        "opponent": args.opponent,
        "n_games": len(games),
        "agent_win_rate": wins / n,
        "mean_lead": sum(leads) / n,
        "min_lead": min(leads) if leads else 0.0,
        "max_lead": max(leads) if leads else 0.0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "games": games}, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
