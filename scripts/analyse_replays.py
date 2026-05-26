"""Empirical action-distribution analysis over top Kaggle leaderboard replays.

Pulls a sample of replay episodes from the Orbit Wars daily Kaggle datasets,
extracts every launch by every *winning* seat, resolves target/eta via the
same physics the training pipeline uses, and emits quantile statistics that
inform the default values of ``ActionFilters`` (gap #2 of the array-search
plan's "Open questions").

Replays are **not** used as training data — only as a source of empirical
defaults. The user-confirmed reference memory permits this; see
``reference_orbit_wars_episodes`` and CLAUDE.md.

Usage:
    KAGGLE_USERNAME=<u> KAGGLE_KEY=<k> uv run python scripts/analyse_replays.py \
        --dates 2026-05-24,2026-05-25 \
        --per-date 100

Outputs:
    docs/replay_action_stats.md   — human-readable report
    docs/replay_action_stats.json — machine-readable raw quantiles
    replays/sample/*.json         — cached downloads (gitignored)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Bootstrap repo root onto sys.path so we can import src.physics from a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.physics import fleet_speed, intercept  # noqa: E402


REPLAY_CACHE = ROOT / "replays" / "sample"
OUT_MD = ROOT / "docs" / "replay_action_stats.md"
OUT_JSON = ROOT / "docs" / "replay_action_stats.json"


@dataclass
class Launch:
    episode_id: str
    seat: int
    won: bool
    turn: int
    source_pid: int
    ships: int
    eta: int
    target_pid: int | None
    source_garrison_at_launch: float
    source_production: float
    target_owner_rel: str  # "mine", "neutral", "enemy"
    target_production: float


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def list_dataset_files(slug: str) -> list[str]:
    """List filenames in a Kaggle dataset via the CLI."""
    files: list[str] = []
    page_token: str | None = None
    while True:
        cmd = ["uv", "run", "kaggle", "datasets", "files", slug]
        if page_token:
            cmd.extend(["--page-token", page_token])
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        if result.returncode != 0:
            raise RuntimeError(f"kaggle list failed: {result.stderr}")
        next_token = None
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("Next Page Token = "):
                next_token = stripped.removeprefix("Next Page Token = ").strip()
                continue
            if stripped.startswith("name") or stripped.startswith("---"):
                continue
            parts = stripped.split()
            if not parts:
                continue
            name = parts[0]
            if name.endswith(".json"):
                files.append(name)
        if not next_token:
            break
        page_token = next_token
    return files


def download_episode(slug: str, filename: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename
    if target.exists() and target.stat().st_size > 0:
        return target
    cmd = [
        "uv", "run", "kaggle", "datasets", "download",
        "-f", filename, "-p", str(dest_dir), slug,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"download {filename} failed: {result.stderr}")
    if not target.exists():
        raise FileNotFoundError(f"expected {target} after download")
    return target


def _resolve_target(action_angle: float, source: list, ships: int,
                    planets: list[list], av: float,
                    comet_ids: set[int], raw_comets: list) -> tuple[int, int] | None:
    """Find which planet a fleet with the given launch angle would hit.

    Mirrors ``infer_fleet_events`` but starts from the *launch* state rather
    than a mid-flight fleet.
    """
    sx, sy = float(source[2]), float(source[3])
    best_pid = None
    best_eta = None
    best_diff = 0.35
    for planet in planets:
        if int(planet[0]) == int(source[0]):
            continue
        tx, ty, eta = intercept(sx, sy, planet, av, ships, comet_ids, raw_comets)
        if tx is None or eta is None:
            continue
        predicted = math.atan2(float(ty) - sy, float(tx) - sx)
        diff = abs((predicted - action_angle + math.pi) % (2.0 * math.pi) - math.pi)
        if diff < best_diff:
            best_pid = int(planet[0])
            best_eta = int(eta)
            best_diff = diff
    if best_pid is None:
        return None
    return best_pid, best_eta


def extract_launches(episode: dict[str, Any]) -> list[Launch]:
    eid = str(episode.get("id", "?"))
    rewards = episode.get("rewards") or []
    statuses = episode.get("statuses") or []
    # Winning seats: rewards[i] is max non-negative
    winners = set()
    if rewards:
        max_r = max(r for r in rewards if r is not None)
        for i, r in enumerate(rewards):
            if r is not None and r == max_r and statuses[i] != "ERROR":
                winners.add(i)

    launches: list[Launch] = []
    steps = episode.get("steps") or []
    for turn_idx, step in enumerate(steps):
        for seat, seat_state in enumerate(step):
            actions = seat_state.get("action") or []
            if not actions:
                continue
            obs = seat_state.get("observation") or {}
            planets = obs.get("planets") or []
            if not planets:
                continue
            av = float(obs.get("angular_velocity") or 0.035)
            comet_ids = set(int(x) for x in (obs.get("comet_planet_ids") or []))
            raw_comets = obs.get("comets") or []
            planet_by_id = {int(p[0]): p for p in planets}
            for action in actions:
                try:
                    src_pid = int(action[0])
                    angle = float(action[1])
                    ships = int(action[2])
                except (TypeError, IndexError, ValueError):
                    continue
                source = planet_by_id.get(src_pid)
                if source is None or ships <= 0:
                    continue
                source_garrison = float(source[5])
                source_prod = float(source[6])
                resolved = _resolve_target(angle, source, ships, planets, av, comet_ids, raw_comets)
                if resolved is None:
                    target_pid, eta = None, None
                else:
                    target_pid, eta = resolved
                target_owner_rel = "?"
                target_prod = 0.0
                if target_pid is not None:
                    tgt = planet_by_id.get(target_pid)
                    if tgt is not None:
                        tgt_owner = int(tgt[1])
                        if tgt_owner == seat:
                            target_owner_rel = "mine"
                        elif tgt_owner == -1:
                            target_owner_rel = "neutral"
                        else:
                            target_owner_rel = "enemy"
                        target_prod = float(tgt[6])
                launches.append(Launch(
                    episode_id=eid,
                    seat=seat,
                    won=(seat in winners),
                    turn=turn_idx,
                    source_pid=src_pid,
                    ships=ships,
                    eta=int(eta) if eta is not None else -1,
                    target_pid=target_pid,
                    source_garrison_at_launch=source_garrison,
                    source_production=source_prod,
                    target_owner_rel=target_owner_rel,
                    target_production=target_prod,
                ))
    return launches


def _quantiles(values: list[float], qs=(0.05, 0.25, 0.50, 0.75, 0.95)) -> dict[str, float]:
    if not values:
        return {f"p{int(q*100):02d}": float("nan") for q in qs}
    sorted_vals = sorted(values)
    out = {}
    for q in qs:
        idx = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
        out[f"p{int(q*100):02d}"] = float(sorted_vals[idx])
    return out


def _per_turn_launches_per_seat(episodes: list[dict[str, Any]]) -> list[int]:
    """For each (episode, seat, turn), count actions; only winning seats."""
    counts: list[int] = []
    for ep in episodes:
        rewards = ep.get("rewards") or []
        if not rewards:
            continue
        max_r = max(rewards)
        winners = {i for i, r in enumerate(rewards) if r == max_r}
        for step in ep.get("steps") or []:
            for seat, seat_state in enumerate(step):
                if seat not in winners:
                    continue
                actions = seat_state.get("action") or []
                counts.append(len(actions))
    return counts


def _multi_source_target_rate(launches: list[Launch]) -> float:
    """Fraction of (episode, seat, turn) cells where ≥2 launches share a target."""
    grouped: dict[tuple[str, int, int], list[int | None]] = defaultdict(list)
    for L in launches:
        if not L.won:
            continue
        grouped[(L.episode_id, L.seat, L.turn)].append(L.target_pid)
    if not grouped:
        return 0.0
    multi = 0
    total = 0
    for targets in grouped.values():
        total += 1
        c = Counter(t for t in targets if t is not None)
        if c and max(c.values()) >= 2:
            multi += 1
    return multi / max(total, 1)


def _histogram(values: list[float], bins: list[float]) -> list[int]:
    counts = [0] * (len(bins) - 1)
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break
        else:
            if v >= bins[-1]:
                counts[-1] += 1
    return counts


def _format_histogram(values: list[float], bins: list[float], width: int = 40, fmt: str = "{:>5.0f}") -> str:
    if not values:
        return "(no data)"
    counts = _histogram(values, bins)
    peak = max(counts) if counts else 1
    lines = []
    for i, c in enumerate(counts):
        lo = bins[i]
        hi = bins[i + 1] if i + 1 < len(bins) else float("inf")
        bar_len = int(width * c / max(peak, 1))
        if hi == float("inf"):
            label = f"[{fmt.format(lo)}+{' ' * (len(fmt.format(lo)) + 1)})"
        else:
            label = f"[{fmt.format(lo)}, {fmt.format(hi)})"
        lines.append(f"  {label}  {c:>6} {'#' * bar_len}")
    return "\n".join(lines)


def render_report(launches: list[Launch], episodes: list[dict], episode_ids: list[str]) -> tuple[str, dict]:
    winning = [L for L in launches if L.won]
    losing = [L for L in launches if not L.won]

    small_threshold = 25
    small_winning = [L for L in winning if L.ships < small_threshold]
    big_winning = [L for L in winning if L.ships >= small_threshold]

    ships_w = [L.ships for L in winning]
    ships_l = [L.ships for L in losing]
    eta_w = [L.eta for L in winning if L.eta >= 0]
    eta_w_small = [L.eta for L in small_winning if L.eta >= 0]
    eta_w_big = [L.eta for L in big_winning if L.eta >= 0]
    pct_garrison = [
        L.ships / max(L.source_garrison_at_launch, 1.0)
        for L in winning if L.source_garrison_at_launch > 0
    ]

    per_turn_counts = _per_turn_launches_per_seat(episodes)
    nonzero_per_turn = [c for c in per_turn_counts if c > 0]

    multi_src_rate = _multi_source_target_rate(launches)

    target_owner_breakdown = Counter(L.target_owner_rel for L in winning)
    total_winning = max(len(winning), 1)
    defensive_share = target_owner_breakdown.get("mine", 0) / total_winning
    neutral_share = target_owner_breakdown.get("neutral", 0) / total_winning
    enemy_share = target_owner_breakdown.get("enemy", 0) / total_winning

    raw = {
        "n_episodes": len(episodes),
        "episode_ids": episode_ids[:50],
        "n_winning_launches": len(winning),
        "n_losing_launches": len(losing),
        "ships_winning": _quantiles(ships_w),
        "ships_losing": _quantiles(ships_l),
        "eta_winning": _quantiles(eta_w),
        "eta_winning_small_fleets_lt25": _quantiles(eta_w_small),
        "eta_winning_big_fleets_ge25": _quantiles(eta_w_big),
        "ships_as_pct_source_garrison": _quantiles(pct_garrison),
        "launches_per_seat_turn_all": _quantiles([float(c) for c in per_turn_counts]),
        "launches_per_seat_turn_nonzero": _quantiles([float(c) for c in nonzero_per_turn]),
        "multi_source_target_rate": multi_src_rate,
        "target_owner_share": {
            "mine_defensive": defensive_share,
            "neutral_expansion": neutral_share,
            "enemy_attack": enemy_share,
        },
    }

    # Markdown report
    ships_hist = _format_histogram(
        ships_w, [0, 3, 5, 8, 15, 25, 50, 100, 200, 500, 1000]
    )
    eta_hist = _format_histogram(
        eta_w, [0, 5, 10, 15, 20, 25, 35, 50, 70, 100, 150]
    )
    pct_hist = _format_histogram(
        pct_garrison,
        [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.00, 1.50, 2.50, 5.00, 10.0],
        fmt="{:>5.2f}",
    )
    launches_hist = _format_histogram(
        [float(c) for c in per_turn_counts], list(range(0, 11))
    )

    proposed = {
        "min_ships_per_launch": max(1, int(raw["ships_winning"]["p05"])),
        "min_ships_pct_of_source": round(raw["ships_as_pct_source_garrison"]["p05"], 3),
        "max_eta_unconditional": int(math.ceil(raw["eta_winning"]["p95"])),
        "small_fleet_threshold": small_threshold,
        "small_fleet_eta_cap": int(math.ceil(raw["eta_winning_small_fleets_lt25"]["p95"])) if eta_w_small else None,
        "max_launches_per_turn": int(math.ceil(raw["launches_per_seat_turn_nonzero"]["p95"])),
        "multi_source_bonus_evidence": multi_src_rate,
    }
    raw["proposed_defaults"] = proposed

    lines = []
    lines.append("# Replay action-distribution analysis")
    lines.append("")
    lines.append("Empirical quantiles of launch decisions made by **winning** seats across")
    lines.append(f"{len(episodes)} Kaggle leaderboard replays from the dates analysed. Drives the")
    lines.append("default values in `ActionFilters` (see `docs/plans/array-search-initiative.md`).")
    lines.append("")
    lines.append("Replays are *not* training data — only a source of heuristic defaults.")
    lines.append("")
    lines.append("## Proposed `ActionFilters` defaults")
    lines.append("")
    lines.append("```python")
    lines.append("@dataclass(frozen=True)")
    lines.append("class ActionFilters:")
    for k, v in proposed.items():
        if v is None:
            lines.append(f"    {k} = None  # insufficient data")
        else:
            lines.append(f"    {k} = {v!r}")
    lines.append("```")
    lines.append("")
    lines.append("Derivation: each cap is the P95 quantile of the relevant winning-seat")
    lines.append("distribution; each minimum is P05. Tight enough to reject obviously bad")
    lines.append("candidates without clipping anything top agents actually do.")
    lines.append("")
    lines.append("## Sample sizes")
    lines.append("")
    lines.append(f"- Episodes parsed: **{len(episodes)}**")
    lines.append(f"- Winning-seat launches: **{len(winning):,}**")
    lines.append(f"- Losing-seat launches: **{len(losing):,}**")
    lines.append(f"- Per-(seat, turn) cells (winners): **{len(per_turn_counts):,}**")
    lines.append("")
    lines.append("## Ships per launch (winning seats)")
    lines.append("")
    lines.append(f"- Quantiles: p05={raw['ships_winning']['p05']:.0f}, p25={raw['ships_winning']['p25']:.0f}, "
                 f"p50={raw['ships_winning']['p50']:.0f}, p75={raw['ships_winning']['p75']:.0f}, "
                 f"p95={raw['ships_winning']['p95']:.0f}")
    lines.append(f"- Compare losing: p50={raw['ships_losing']['p50']:.0f}, p95={raw['ships_losing']['p95']:.0f}")
    lines.append("")
    lines.append("```")
    lines.append(ships_hist)
    lines.append("```")
    lines.append("")
    lines.append("## Ships as fraction of source garrison at launch")
    lines.append("")
    lines.append(f"- p05={raw['ships_as_pct_source_garrison']['p05']:.3f}, "
                 f"p25={raw['ships_as_pct_source_garrison']['p25']:.3f}, "
                 f"p50={raw['ships_as_pct_source_garrison']['p50']:.3f}, "
                 f"p75={raw['ships_as_pct_source_garrison']['p75']:.3f}, "
                 f"p95={raw['ships_as_pct_source_garrison']['p95']:.3f}")
    lines.append("")
    lines.append("```")
    lines.append(pct_hist)
    lines.append("```")
    lines.append("")
    lines.append("## Launch ETA (winning seats)")
    lines.append("")
    lines.append(f"- All: p50={raw['eta_winning']['p50']:.0f}, p75={raw['eta_winning']['p75']:.0f}, "
                 f"p95={raw['eta_winning']['p95']:.0f}")
    if eta_w_small:
        lines.append(f"- Small fleets (<25 ships): p50={raw['eta_winning_small_fleets_lt25']['p50']:.0f}, "
                     f"p95={raw['eta_winning_small_fleets_lt25']['p95']:.0f}")
    if eta_w_big:
        lines.append(f"- Big fleets (≥25 ships):   p50={raw['eta_winning_big_fleets_ge25']['p50']:.0f}, "
                     f"p95={raw['eta_winning_big_fleets_ge25']['p95']:.0f}")
    lines.append("")
    lines.append("```")
    lines.append(eta_hist)
    lines.append("```")
    lines.append("")
    lines.append("## Launches per seat per turn (winning seats only)")
    lines.append("")
    lines.append(f"- All turns: p50={raw['launches_per_seat_turn_all']['p50']:.0f}, "
                 f"p75={raw['launches_per_seat_turn_all']['p75']:.0f}, "
                 f"p95={raw['launches_per_seat_turn_all']['p95']:.0f}")
    if nonzero_per_turn:
        lines.append(f"- Non-empty turns only: p50={raw['launches_per_seat_turn_nonzero']['p50']:.0f}, "
                     f"p95={raw['launches_per_seat_turn_nonzero']['p95']:.0f}")
    lines.append("")
    lines.append("```")
    lines.append(launches_hist)
    lines.append("```")
    lines.append("")
    lines.append("## Multi-source attack rate")
    lines.append("")
    lines.append(f"Of (episode, seat, turn) cells with ≥1 launch by a winning seat, **{multi_src_rate:.1%}**")
    lines.append("contained two or more launches hitting the same target on the same turn.")
    lines.append("(High → multi-source coordination is a real signal worth biasing toward.)")
    lines.append("")
    lines.append("## Target-owner mix (winning launches)")
    lines.append("")
    lines.append(f"- Attacks on enemy planets: **{enemy_share:.1%}**")
    lines.append(f"- Captures on neutral planets: **{neutral_share:.1%}**")
    lines.append(f"- Defensive launches (target = my planet): **{defensive_share:.1%}**")
    lines.append("")
    lines.append("(If defensive_share is meaningful (>5%), the candidate ranker already")
    lines.append("sees these via `target_owner_rel` — no special candidate kind needed.)")

    return "\n".join(lines) + "\n", raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="2026-05-24,2026-05-25",
                    help="Comma-separated list of YYYY-MM-DD daily datasets")
    ap.add_argument("--per-date", type=int, default=100,
                    help="Number of episodes to sample per date")
    ap.add_argument("--seed", type=int, default=42, help="Sampling RNG seed")
    ap.add_argument("--reuse-existing", action="store_true",
                    help="Use only already-downloaded episode files in replays/sample/")
    args = ap.parse_args()

    _load_env()

    REPLAY_CACHE.mkdir(parents=True, exist_ok=True)

    if args.reuse_existing:
        episode_files = sorted(REPLAY_CACHE.glob("*.json"))
        print(f"Reusing {len(episode_files)} cached episodes")
    else:
        rng = random.Random(args.seed)
        episode_files: list[Path] = []
        for date in args.dates.split(","):
            date = date.strip()
            slug = f"kaggle/orbit-wars-episodes-{date}"
            print(f"Listing {slug}…", flush=True)
            all_files = list_dataset_files(slug)
            print(f"  found {len(all_files)} files", flush=True)
            sample = rng.sample(all_files, k=min(args.per_date, len(all_files)))
            for i, fname in enumerate(sample):
                path = download_episode(slug, fname, REPLAY_CACHE)
                episode_files.append(path)
                if (i + 1) % 10 == 0:
                    print(f"  [{date}] downloaded {i+1}/{len(sample)}", flush=True)

    print(f"\nParsing {len(episode_files)} episodes…", flush=True)
    episodes = []
    episode_ids = []
    all_launches: list[Launch] = []
    bad = 0
    for i, p in enumerate(episode_files):
        try:
            with open(p) as f:
                ep = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            bad += 1
            continue
        episodes.append(ep)
        episode_ids.append(str(ep.get("id", p.stem)))
        all_launches.extend(extract_launches(ep))
        if (i + 1) % 25 == 0:
            print(f"  parsed {i+1}/{len(episode_files)}; launches so far: {len(all_launches):,}", flush=True)
    if bad:
        print(f"  skipped {bad} unreadable files")

    report_md, raw = render_report(all_launches, episodes, episode_ids)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(report_md, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
