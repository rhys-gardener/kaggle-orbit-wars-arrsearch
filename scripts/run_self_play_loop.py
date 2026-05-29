"""Concurrent two-ranker array self-play loop."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.array_search.labels import label_records
from src.array_search.mcts_teacher import MCTSTeacherConfig, apply_mcts_teacher
from src.array_search.ranker import CandidateRanker, load_ranker, save_ranker, train_ranker_epoch
from src.array_search.records import load_record_shards
from src.array_search.scenarios import generate_initial_arrays
from src.array_search.self_play import RankerPolicy, run_array_self_play, seat_scores, seed_everything
from src.array_search.state_adapter import arrays_to_obs
from src.array_search.training_log import TrainingLogger
from src.graph_training.geometry_cache import build_geometry_cache_from_observations
from src.graph_training.sparse_geometry_cache import DEFAULT_SHIP_BUCKETS, SparseGeometryCache


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def resolve_geometry_cache_dir(args) -> Path:
    """Shared, cross-run geometry cache location (override with --geometry-cache-dir)."""
    if args.geometry_cache_dir:
        return Path(args.geometry_cache_dir)
    return ROOT / "runs" / "_geometry_cache"


def build_geometry(scenario, args, *, cache_dir: Path | None = None):
    if args.geometry_mode == "sparse":
        cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            bucket_tag = "_".join(str(x) for x in parse_ints(args.ship_buckets))
            # Key on every input that changes the geometry so a shared cache
            # never serves a stale map: seed + player count + engine step budget
            # + ship buckets. (max_steps is baked into the .npz and silently
            # overrides the run's value on load, so it MUST be in the filename.)
            cache_path = cache_dir / (
                f"seed_{int(scenario.seed)}"
                f"_np_{int(scenario.num_players)}"
                f"_steps_{int(args.max_engine_steps)}"
                f"_b_{bucket_tag}_sparse.npz"
            )
            if cache_path.exists():
                return SparseGeometryCache.load(cache_path, scenario), cache_path, True
        return (
            SparseGeometryCache(
                scenario,
                parse_ints(args.ship_buckets),
                max_steps=args.max_engine_steps,
                metadata={"loop": True, "seed": int(scenario.seed)},
            ),
            cache_path,
            False,
        )

    steps = list(range(0, args.horizon_turns + args.geometry_extra_turns + 1, max(1, args.geometry_stride)))
    observations = [
        arrays_to_obs(scenario, seat=0, rel_step=step, owners=scenario.owners, ships=scenario.ships, schedule={})
        for step in steps
    ]
    return (
        build_geometry_cache_from_observations(
            observations,
            parse_ints(args.ship_buckets),
            max_steps=args.max_engine_steps,
            keep_only_eta_improvements=True,
            metadata={"loop": True, "seed": int(scenario.seed)},
        ),
        None,
        False,
    )


def trim_buffer(buffer: list[dict], max_size: int) -> None:
    if len(buffer) > max_size:
        del buffer[: len(buffer) - max_size]


def collect_record_paths(text: str) -> list[Path]:
    if not text:
        return []
    out: list[Path] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        if path.is_dir():
            out.extend(sorted(path.glob("*.pkl")))
        else:
            out.append(path)
    return out


def _latest_common_ranker_round(ckpt_dir: Path) -> tuple[int, Path, Path] | None:
    a_paths = sorted(ckpt_dir.glob("ranker_A_round_*.pt"))
    b_paths = sorted(ckpt_dir.glob("ranker_B_round_*.pt"))
    a_by_round = {_round_from_checkpoint_name(path): path for path in a_paths}
    b_by_round = {_round_from_checkpoint_name(path): path for path in b_paths}
    common = sorted(r for r in a_by_round if r is not None and r in b_by_round)
    if not common:
        return None
    latest = int(common[-1])
    return latest, a_by_round[latest], b_by_round[latest]


def _round_from_checkpoint_name(path: Path) -> int | None:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return None


def _file_size(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return int(path.stat().st_size)


def _memory_bytes() -> dict[str, int] | None:
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    process = psutil.Process(os.getpid())
    info = process.memory_info()
    return {
        "rss": int(getattr(info, "rss", 0)),
        "vms": int(getattr(info, "vms", 0)),
    }


def _geometry_stats(geometry) -> dict[str, int]:
    return {
        "entry_count": int(getattr(geometry, "entry_count", 0)),
        "lookup_count": int(getattr(geometry, "lookup_count", 0)),
        "solve_count": int(getattr(geometry, "solve_count", 0)),
        "cache_hits": int(getattr(geometry, "cache_hits", 0)),
    }


def _seat_policy_map(num_players: int, seed: int) -> dict[int, str]:
    return {seat: ("A" if ((seat + seed) % 2) == 0 else "B") for seat in range(num_players)}


def _policy_score_summary(scores: dict[int, dict[str, float]], seat_policy: dict[int, str]) -> dict[str, object]:
    production = {"A": 0.0, "B": 0.0}
    ships = {"A": 0.0, "B": 0.0}
    total = {"A": 0.0, "B": 0.0}
    for seat, values in scores.items():
        policy_id = seat_policy[int(seat)]
        production[policy_id] += float(values["production"])
        ships[policy_id] += float(values["ships"])
        total[policy_id] += float(values["production"]) * 1000.0 + float(values["ships"])
    delta = float(total["A"] - total["B"])
    return {
        "policy_production": production,
        "policy_ships": ships,
        "policy_score": total,
        "policy_score_delta_a_minus_b": delta,
        "policy_winner": "A" if delta > 0.0 else ("B" if delta < 0.0 else "draw"),
    }


def evaluate_challenger(
    challenger_path: Path,
    pool: list[Path],
    *,
    args: argparse.Namespace,
    seed_base: int,
) -> tuple[float, float, int]:
    if not pool:
        return 1.0, 0.0, 0
    challenger = load_ranker(challenger_path, map_location=args.device)
    wins = 0
    games = 0
    leads: list[float] = []
    for opponent_idx, opponent_path in enumerate(pool):
        opponent = load_ranker(opponent_path, map_location=args.device)
        for seed_offset in range(args.eval_seeds):
            seed = int(seed_base + opponent_idx * args.eval_seeds + seed_offset)
            scenario = generate_initial_arrays(seed, num_players=args.num_players)
            geometry, _cache_path, _cache_loaded = build_geometry(
                scenario, args, cache_dir=resolve_geometry_cache_dir(args)
            )
            result = run_array_self_play(
                scenario=scenario,
                geometry_cache=geometry,
                policies={
                    "A": RankerPolicy(
                        model=challenger,
                        policy_id="challenger",
                        device=args.device,
                        max_launches=args.max_launches,
                    ),
                    "B": RankerPolicy(
                        model=opponent,
                        policy_id="incumbent",
                        device=args.device,
                        max_launches=args.max_launches,
                    ),
                },
                horizon_turns=args.eval_horizon_turns,
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
            lead = policy_scores["A"] - policy_scores["B"]
            leads.append(float(lead))
            wins += int(lead > 0.0)
            games += 1
    return wins / max(games, 1), sum(leads) / max(len(leads), 1), games


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seeds-per-round", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--horizon-turns", type=int, default=50)
    parser.add_argument(
        "--explore-temperature",
        type=float,
        default=0.0,
        help="Decision-time exploration for training policies (0 = greedy). Eval stays greedy.",
    )
    parser.add_argument(
        "--long-horizon-fraction",
        type=float,
        default=0.0,
        help="Fraction of games rolled to --long-horizon-turns for late-game state coverage.",
    )
    parser.add_argument("--long-horizon-turns", type=int, default=250)
    parser.add_argument("--geometry-mode", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--geometry-stride", type=int, default=1)
    parser.add_argument("--geometry-extra-turns", type=int, default=120)
    parser.add_argument(
        "--geometry-cache-dir",
        default="",
        help="Shared geometry cache dir reused across runs (default: runs/_geometry_cache).",
    )
    parser.add_argument("--ship-buckets", default=",".join(str(x) for x in DEFAULT_SHIP_BUCKETS))
    parser.add_argument("--max-engine-steps", type=int, default=160)
    parser.add_argument("--candidate-limit", type=int, default=80)
    parser.add_argument("--max-launches", type=int, default=10)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--max-same-target", type=int, default=3)
    parser.add_argument("--include-support", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-seeds", type=int, default=2)
    parser.add_argument("--eval-horizon-turns", type=int, default=120)
    parser.add_argument("--accept-win-rate", type=float, default=0.55)
    parser.add_argument("--mcts-teacher-rate", type=float, default=0.0)
    parser.add_argument("--mcts-root-action-sets", type=int, default=32)
    parser.add_argument(
        "--mcts-opponent-samples",
        type=int,
        default=3,
        help="Opponent continuations sampled per root action (teacher cost scales with this).",
    )
    parser.add_argument("--mcts-depth", type=int, default=2, help="Active-play turns before passive accrual; must be >= 2.")
    parser.add_argument("--mcts-aggregate", choices=["mean", "min"], default="mean")
    parser.add_argument("--mcts-rollout-horizon", type=int, default=30)
    parser.add_argument("--replay-prior-checkpoint", default="")
    parser.add_argument("--init-checkpoint-a", default="")
    parser.add_argument("--init-checkpoint-b", default="")
    parser.add_argument("--replay-records", default="")
    parser.add_argument("--resume-from-dir", default="")
    parser.add_argument("--round-start", type=int, default=1)
    parser.add_argument("--append-log", action="store_true")
    parser.add_argument("--map-pool-size", type=int, default=32)
    # Default 1: visit a distinct seed each game and cycle the pool across rounds.
    # Self-play policies are deterministic with weights frozen within a round, so
    # repeating the same seed inside a round just replays an identical game. Set
    # >1 only with a stochastic/exploring policy where repeats add variety.
    parser.add_argument("--games-per-map", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if int(args.mcts_depth) < 2:
        parser.error("--mcts-depth must be >= 2 (the teacher is policy-response only)")

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_round = None
    if args.resume_from_dir:
        latest = _latest_common_ranker_round(Path(args.resume_from_dir) / "checkpoints")
        if latest is None:
            raise FileNotFoundError(f"no matching A/B ranker checkpoints under {args.resume_from_dir}")
        resume_round, resume_a, resume_b = latest
        if not args.init_checkpoint_a:
            args.init_checkpoint_a = str(resume_a)
        if not args.init_checkpoint_b:
            args.init_checkpoint_b = str(resume_b)
        if int(args.round_start) <= 1:
            args.round_start = int(resume_round) + 1
        args.append_log = True

    logger = TrainingLogger(out_dir / "training.jsonl", append=bool(args.append_log))
    logger.log_msg(
        "run_start",
        out_dir=out_dir,
        args=vars(args),
        resume_round=resume_round,
        memory_bytes=_memory_bytes(),
    )

    if args.init_checkpoint_a or args.init_checkpoint_b:
        if not args.init_checkpoint_a or not args.init_checkpoint_b:
            raise ValueError("--init-checkpoint-a and --init-checkpoint-b must be provided together")
        model_a = load_ranker(args.init_checkpoint_a, map_location=args.device)
        model_b = load_ranker(args.init_checkpoint_b, map_location=args.device)
        logger.log_msg(
            "loaded_policy_checkpoints",
            checkpoint_a=args.init_checkpoint_a,
            checkpoint_b=args.init_checkpoint_b,
        )
    elif args.replay_prior_checkpoint:
        model_a = load_ranker(args.replay_prior_checkpoint, map_location=args.device)
        model_b = load_ranker(args.replay_prior_checkpoint, map_location=args.device)
        logger.log_msg("loaded_replay_prior", checkpoint=args.replay_prior_checkpoint)
    else:
        model_a = CandidateRanker()
        model_b = CandidateRanker()
        logger.log_msg("initialized_random_rankers")
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=args.lr)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=args.lr)
    buffer_a: list[dict] = []
    buffer_b: list[dict] = []
    replay_paths = collect_record_paths(args.replay_records)
    replay_load_t0 = time.perf_counter()
    replay_records = load_record_shards(replay_paths)
    logger.log_msg(
        "replay_records_loaded",
        paths=len(replay_paths),
        records=len(replay_records),
        seconds=time.perf_counter() - replay_load_t0,
        memory_bytes=_memory_bytes(),
    )
    incumbent_pool: list[Path] = []

    try:
        final_round = int(args.round_start) + int(args.rounds) - 1
        games_per_map = max(1, int(args.games_per_map))
        map_pool_size = max(1, int(args.map_pool_size))
        planned_seeds = {
            args.seed_start + (((r - 1) * args.seeds_per_round + off) // games_per_map) % map_pool_size
            for r in range(int(args.round_start), final_round + 1)
            for off in range(args.seeds_per_round)
        }
        logger.log_msg(
            "map_coverage",
            distinct_maps=len(planned_seeds),
            map_pool_size=map_pool_size,
            games_per_map=games_per_map,
            single_map_warning=len(planned_seeds) <= 1,
        )
        if len(planned_seeds) <= 1:
            print(
                "WARNING: this run trains on a single map "
                f"(seed {sorted(planned_seeds)[0]}). Lower --games-per-map or raise "
                "--rounds/--seeds-per-round for map diversity.",
                file=sys.stderr,
            )
        for round_idx in range(int(args.round_start), final_round + 1):
            round_t0 = time.perf_counter()
            round_records: list[dict] = []
            round_deltas: list[float] = []
            round_wins = {"A": 0, "B": 0, "draw": 0}
            logger.log_msg(
                "round_start",
                iter=round_idx,
                total_rounds=final_round,
                buffer_a=len(buffer_a),
                buffer_b=len(buffer_b),
                replay_records=len(replay_records),
                memory_bytes=_memory_bytes(),
            )
            for offset in range(args.seeds_per_round):
                game_t0 = time.perf_counter()
                global_game_idx = (round_idx - 1) * args.seeds_per_round + offset
                map_pool_size = max(1, int(args.map_pool_size))
                games_per_map = max(1, int(args.games_per_map))
                seed = args.seed_start + ((global_game_idx // games_per_map) % map_pool_size)
                # Late-game state coverage: a deterministic subset of games rolls
                # far past the default horizon so the buffer sees turns 50..500.
                is_long = False
                if float(args.long_horizon_fraction) > 0.0:
                    stride = max(1, round(1.0 / float(args.long_horizon_fraction)))
                    is_long = (global_game_idx % stride) == 0
                game_horizon = int(args.long_horizon_turns) if is_long else int(args.horizon_turns)
                scenario = generate_initial_arrays(seed, num_players=args.num_players)
                geometry_t0 = time.perf_counter()
                geometry, geometry_path, geometry_loaded = build_geometry(
                    scenario, args, cache_dir=resolve_geometry_cache_dir(args)
                )
                logger.log_msg(
                    "geometry_ready",
                    iter=round_idx,
                    game_offset=offset,
                    seed=seed,
                    mode=args.geometry_mode,
                    cache_loaded=geometry_loaded,
                    cache_path=geometry_path,
                    cache_bytes=_file_size(geometry_path),
                    seconds=time.perf_counter() - geometry_t0,
                    **_geometry_stats(geometry),
                )
                self_play_t0 = time.perf_counter()
                result = run_array_self_play(
                    scenario=scenario,
                    geometry_cache=geometry,
                    policies={
                        "A": RankerPolicy(
                            model=model_a,
                            policy_id="A",
                            device=args.device,
                            max_launches=args.max_launches,
                            temperature=float(args.explore_temperature),
                            seed=args.seed + global_game_idx * 2 + 1,
                        ),
                        "B": RankerPolicy(
                            model=model_b,
                            policy_id="B",
                            device=args.device,
                            max_launches=args.max_launches,
                            temperature=float(args.explore_temperature),
                            seed=args.seed + global_game_idx * 2 + 2,
                        ),
                    },
                    horizon_turns=game_horizon,
                    candidate_limit=args.candidate_limit,
                    max_launches=args.max_launches,
                    beam_width=args.beam_width,
                    max_same_target=args.max_same_target,
                    include_support=args.include_support,
                    scenario_id=f"round_{round_idx}_seed_{seed}",
                )
                self_play_seconds = time.perf_counter() - self_play_t0
                if geometry_path is not None and hasattr(geometry, "save"):
                    save_t0 = time.perf_counter()
                    geometry.save(geometry_path)
                    logger.log_msg(
                        "geometry_saved",
                        iter=round_idx,
                        game_offset=offset,
                        seed=seed,
                        cache_path=geometry_path,
                        cache_bytes=_file_size(geometry_path),
                        seconds=time.perf_counter() - save_t0,
                        **_geometry_stats(geometry),
                    )
                records = result["records"]
                label_t0 = time.perf_counter()
                label_records(records)
                label_seconds = time.perf_counter() - label_t0
                applied_a = 0
                applied_b = 0
                teacher_seconds = 0.0
                if args.mcts_teacher_rate > 0.0:
                    teacher_t0 = time.perf_counter()
                    teacher_config = MCTSTeacherConfig(
                        root_action_sets=args.mcts_root_action_sets,
                        opponent_samples=args.mcts_opponent_samples,
                        depth=args.mcts_depth,
                        rollout_horizon=args.mcts_rollout_horizon,
                        aggregate=args.mcts_aggregate,
                    )
                    records_a = [r for r in records if r.get("source", {}).get("policy_id") == "A"]
                    records_b = [r for r in records if r.get("source", {}).get("policy_id") == "B"]
                    applied_a = apply_mcts_teacher(
                        records_a,
                        rate=args.mcts_teacher_rate,
                        seed=args.seed + round_idx * 1000 + offset,
                        model=model_a,
                        device=args.device,
                        config=teacher_config,
                        scenario=scenario,
                        geometry_cache=geometry,
                        candidate_limit=args.candidate_limit,
                        max_launches=args.max_launches,
                        beam_width=args.beam_width,
                        max_same_target=args.max_same_target,
                        include_support=args.include_support,
                    )
                    applied_b = apply_mcts_teacher(
                        records_b,
                        rate=args.mcts_teacher_rate,
                        seed=args.seed + round_idx * 1000 + offset + 500,
                        model=model_b,
                        device=args.device,
                        config=teacher_config,
                        scenario=scenario,
                        geometry_cache=geometry,
                        candidate_limit=args.candidate_limit,
                        max_launches=args.max_launches,
                        beam_width=args.beam_width,
                        max_same_target=args.max_same_target,
                        include_support=args.include_support,
                    )
                    teacher_seconds = time.perf_counter() - teacher_t0
                    logger.log_msg(
                        "mcts_teacher",
                        iter=round_idx,
                        seed=seed,
                        applied_a=applied_a,
                        applied_b=applied_b,
                        depth=args.mcts_depth,
                        root_action_sets=args.mcts_root_action_sets,
                        rollout_horizon=args.mcts_rollout_horizon,
                        opponent_samples=args.mcts_opponent_samples,
                        aggregate=args.mcts_aggregate,
                        mode="policy_response",
                    )
                round_records.extend(records)
                scores = seat_scores(
                    result["owners"],
                    result["ships"],
                    result["production"],
                    args.num_players,
                    active_mask=result.get("active_mask"),
                )
                winner = max(scores, key=lambda seat: (scores[seat]["production"], scores[seat]["ships"]))
                seat_policy = _seat_policy_map(args.num_players, seed)
                policy_summary = _policy_score_summary(scores, seat_policy)
                round_deltas.append(float(policy_summary["policy_score_delta_a_minus_b"]))
                round_wins[str(policy_summary["policy_winner"])] += 1
                logger.log_game(
                    iter=round_idx,
                    seed=seed,
                    seat_policy=seat_policy,
                    production_by_seat={seat: scores[seat]["production"] for seat in scores},
                    ships_by_seat={seat: scores[seat]["ships"] for seat in scores},
                    winner_seat=int(winner),
                    horizon_turns=game_horizon,
                    is_long_horizon=bool(is_long),
                    explore_temperature=float(args.explore_temperature),
                    **policy_summary,
                    game_offset=offset,
                    records=len(records),
                    self_play_seconds=self_play_seconds,
                    label_seconds=label_seconds,
                    teacher_seconds=teacher_seconds,
                    mcts_applied_a=applied_a,
                    mcts_applied_b=applied_b,
                    total_game_seconds=time.perf_counter() - game_t0,
                    geometry_stats=_geometry_stats(geometry),
                    reject_counts=result.get("reject_counts", {}),
                )

            buffer_a.extend(r for r in round_records if r.get("source", {}).get("policy_id") == "A")
            buffer_b.extend(r for r in round_records if r.get("source", {}).get("policy_id") == "B")
            # Keep buffers in insertion order so trim_buffer evicts the oldest
            # (FIFO) records. SGD-order shuffling happens on a per-epoch training
            # copy below, so it never disturbs the buffer's chronology.
            trim_buffer(buffer_a, args.buffer_size)
            trim_buffer(buffer_b, args.buffer_size)
            logger.log_msg(
                "round_score",
                iter=round_idx,
                games=len(round_deltas),
                mean_policy_score_delta_a_minus_b=(
                    sum(round_deltas) / len(round_deltas) if round_deltas else 0.0
                ),
                min_policy_score_delta_a_minus_b=min(round_deltas) if round_deltas else 0.0,
                max_policy_score_delta_a_minus_b=max(round_deltas) if round_deltas else 0.0,
                wins_a=round_wins["A"],
                wins_b=round_wins["B"],
                draws=round_wins["draw"],
            )
            logger.log_msg(
                "train_start",
                iter=round_idx,
                round_records=len(round_records),
                buffer_a=len(buffer_a),
                buffer_b=len(buffer_b),
                replay_records=len(replay_records),
                train_rows_a=len(buffer_a) + len(replay_records),
                train_rows_b=len(buffer_b) + len(replay_records),
                train_epochs=args.train_epochs,
                memory_bytes=_memory_bytes(),
            )

            stats_a = None
            stats_b = None
            train_t0 = time.perf_counter()
            for epoch_idx in range(1, args.train_epochs + 1):
                epoch_t0 = time.perf_counter()
                # Shuffle a fresh per-epoch copy so replay rows are interleaved
                # with self-play rows (not always trailing) and SGD order varies
                # across epochs, while the underlying buffers stay FIFO-ordered.
                rows_a = buffer_a + replay_records
                rows_b = buffer_b + replay_records
                random.shuffle(rows_a)
                random.shuffle(rows_b)
                stats_a = train_ranker_epoch(model_a, rows_a, opt_a, device=args.device)
                stats_b = train_ranker_epoch(model_b, rows_b, opt_b, device=args.device)
                logger.log_msg(
                    "train_epoch",
                    iter=round_idx,
                    epoch=epoch_idx,
                    seconds=time.perf_counter() - epoch_t0,
                    loss_a=stats_a.loss,
                    loss_b=stats_b.loss,
                    records_a=stats_a.records_used,
                    records_b=stats_b.records_used,
                    memory_bytes=_memory_bytes(),
                )
            assert stats_a is not None and stats_b is not None
            train_seconds = time.perf_counter() - train_t0
            logger.log_iter(
                iter=round_idx,
                policy="A",
                loss=stats_a.loss,
                loss_listwise=stats_a.loss_listwise,
                loss_pairwise=stats_a.loss_pairwise,
                n_records=stats_a.records_used,
                mean_candidates_per_turn=stats_a.mean_candidates_per_turn,
                train_seconds=train_seconds,
                buffer_records=len(buffer_a),
                replay_records=len(replay_records),
            )
            logger.log_iter(
                iter=round_idx,
                policy="B",
                loss=stats_b.loss,
                loss_listwise=stats_b.loss_listwise,
                loss_pairwise=stats_b.loss_pairwise,
                n_records=stats_b.records_used,
                mean_candidates_per_turn=stats_b.mean_candidates_per_turn,
                train_seconds=train_seconds,
                buffer_records=len(buffer_b),
                replay_records=len(replay_records),
            )

            path_a = ckpt_dir / f"ranker_A_round_{round_idx:04d}.pt"
            path_b = ckpt_dir / f"ranker_B_round_{round_idx:04d}.pt"
            ckpt_t0 = time.perf_counter()
            save_ranker(path_a, model_a, extra={"round": round_idx, "policy": "A"})
            save_ranker(path_b, model_b, extra={"round": round_idx, "policy": "B"})
            logger.log_msg(
                "checkpoint_saved",
                iter=round_idx,
                path_a=path_a,
                path_b=path_b,
                bytes_a=_file_size(path_a),
                bytes_b=_file_size(path_b),
                seconds=time.perf_counter() - ckpt_t0,
            )

            if round_idx % max(1, args.eval_every) == 0:
                # Snapshot the pool so A and B are both judged against the same
                # incumbents; otherwise B would face A's just-accepted checkpoint.
                eval_pool = list(incumbent_pool)
                for policy_name, path in (("A", path_a), ("B", path_b)):
                    win_rate, mean_lead, n_games = evaluate_challenger(
                        path,
                        eval_pool,
                        args=args,
                        seed_base=args.seed_start + 100_000 + round_idx * 1000 + (0 if policy_name == "A" else 500),
                    )
                    accepted = win_rate >= float(args.accept_win_rate)
                    logger.log_eval(
                        iter=round_idx,
                        challenger=policy_name,
                        opponent="incumbent_pool",
                        n_games=n_games,
                        win_rate=win_rate,
                        mean_production_lead=mean_lead,
                        accepted=accepted,
                        pool=[str(p) for p in incumbent_pool],
                    )
                    if accepted:
                        incumbent_pool.append(path)
                incumbent_pool = incumbent_pool[-8:]
            logger.log_msg(
                "round_done",
                iter=round_idx,
                seconds=time.perf_counter() - round_t0,
                buffer_a=len(buffer_a),
                buffer_b=len(buffer_b),
                replay_records=len(replay_records),
                memory_bytes=_memory_bytes(),
            )
    finally:
        logger.close()
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
