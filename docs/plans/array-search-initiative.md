# Array-Search Initiative

Date: 2026-05-26
Repo: `kaggle-orbit-wars-arrsearch` (forked direction from `kaggle-orbit-wars-rg`)

## Why a new repo

The original repo (`kaggle-orbit-wars-rg`) explored heuristic agents, then
evolved scorers (CMA-ES over a 22-weight linear scorer), then MLP rankers, then
a v17 search/value-guided policy. That track plateaued around the mid-leaderboard
and is hard to push without expensive engine-in-the-loop training.

This repo takes a different bet: **pure array transformations, no env in the
training hot path**. The goal is to train far more cheaply by:

1. Generating initial states as arrays (real-distribution maps via
   `sim.make_env` at step 0 only — never calling `.step()`).
2. Precomputing geometry (planet positions, ETAs, hit planets) in closed form
   from each map's orbit parameters.
3. Rolling state forward with array_env (production accrual, scheduled arrivals,
   combat resolution) — no engine, no fast_env, no kaggle_environments.
4. Generating candidate launches + action sets from cached geometry — no engine
   trajectory solver in the loop.
5. Driving rollouts via multi-agent self-play (each seat picks its own action
   set), so state distribution improves as agents do.

The hypothesis: with the env removed from the inner loop, we can scale to ~10k
states per minute on a laptop, ~1M with a bit of batching, and run the
cache→train→re-cache loop in hours instead of days.

## What we keep from the old repo

Carried forward:

- `main.py` — current submission baseline (v17 scored/ranker). Source of physics
  helpers (`fleet_speed`, `intercept`, `sun_blocked`, `is_orbiting`,
  `_solve_engine_angle`, `_trajectory_first_hit`) that the geometry cache uses.
  In the long run we'll extract these into `src/physics.py` and let
  `main.py` import from there, but the migration is `main.py`-as-is to keep
  the diff minimal.
- `src/graph_training/` — state extraction, action generation, beam search,
  array_env, geometry cache. These are the right primitives for this track;
  what changes is how we *drive* them.
- `sim/` — only kept as the source of real-distribution initial states.
  Never called inside the training loop.
- `orbit-wars/` — Kaggle env reference docs (README + agents.md). Useful for
  consulting game rules without leaving the repo.
- `envs/orbit_wars_mini/` — small wrapper env, useful for sanity checks.

Dropped:

- Replay JSONs, replay-driven pipelines, all `tmp_v17_*` outputs, old weights,
  the v17 / path-to-the-top / value-guided plans, all `agents/` except the
  physics dependency in `main.py`.
- All scripts that operate on replays or run env-in-the-loop training.
  Tournament and evolve scripts will be re-introduced later if needed for
  evaluation against the old baseline.

## Pipeline

```
random seed
  └─ sim.make_env(seed) → step-0 obs (real map distribution, never .step())
     └─ initial arrays: owners0, ships0, production, planet_xy0, radii, av, comets
        ├─ build geometry cache: (step, src, tgt, ship_bucket) → angle, eta, hit_id
        └─ self-play rollout loop:
           for turn in range(horizon):
              for seat in players:
                 ctx = build_graph_state_from_arrays(seat, owners, ships, schedule, step)
                 candidates = generate_cached_launch_candidates(ctx, geometry_cache)
                 action_sets = generate_action_sets(ctx, candidates)
                 record = (ctx, candidates, action_sets)        ← cached training row
                 chosen = policy[seat](record)                   ← fixed heuristic / ranker
                 schedule_action_set(record, chosen, owners, ships, schedule)
              # advance arrays one turn
              production accrual; resolve_combat(schedule[turn])
```

All four green boxes are already array-only; the only env touch is the step-0
obs read.

## What's already in this repo

From `src/graph_training/`:

- [state.py](../../src/graph_training/state.py) — `build_graph_state(obs)`,
  planet/edge feature extraction, inferred fleet events.
- [actions.py](../../src/graph_training/actions.py) —
  `generate_launch_candidates` (engine path) and
  `generate_cached_launch_candidates` (geometry-cache path).
- [search.py](../../src/graph_training/search.py) — beam search over
  candidates → whole-turn action sets.
- [array_env.py](../../src/graph_training/array_env.py) — `initial_arrays`,
  `schedule_action_set`, `_resolve_combat`, `rollout_cached_action_set`,
  `score_arrays`. Pure ndarray rollout.
- [geometry_cache.py](../../src/graph_training/geometry_cache.py) — closed-form
  precomputation over `(step, source, target, ship_bucket)`.

## Gaps to fill (build order)

1. **Random initial-scenario generator.** Use `sim.make_env(seed).reset()` and
   read the step-0 obs. No `.step()` calls. Wrap the result as the array tuple
   `(planet_xy0, radii, owners0, ships0, production, angular_velocity, comets)`.
   File: `src/array_search/scenarios.py`.

2. **Arrays → GraphState adapter.** Today `build_graph_state(obs)` only
   consumes a kaggle-shaped obs dict. We need either a sibling
   `build_graph_state_from_arrays(...)` or an `arrays_to_obs(...)` helper that
   reconstructs the obs dict the existing function expects (planets, fleets,
   angular_velocity, comet_ids, raw_comets, step). The latter is smaller (one
   helper, no parallel feature-extraction code path) and safer to ship first.
   File: `src/array_search/state_adapter.py`.

3. **Multi-seat per-turn step in array_env.** Today `schedule_action_set`
   handles one seat's launches. Extend with a `step_multi_seat(...)` that
   takes a list of `(seat, chosen_action_set)` tuples, schedules all
   simultaneously, accrues production, and resolves combat at the next turn.
   File: extend `src/graph_training/array_env.py`.

4. **End-to-end driver: `scripts/build_graph_array_scenarios.py`.** For each
   seed: build initial arrays → geometry cache → self-play rollout → save
   one record per seat per turn into scenario folder
   (`geometry.npz`, `training/records_*.pkl`, `manifest.json`). Mirror the
   layout that `build_graph_scenario_caches.py` in the old repo wrote, so
   downstream loaders are reusable.

5. **Rollout-based labeller.** For each cached record, evaluate each
   `action_set` by `rollout_cached_action_set` at multiple horizons (e.g.
   H=30, 60, 120) and store production-at-horizon, production-lead, ship-share.
   Plan §"Scoring Philosophy" defines the priority order; see
   [array_env.py:153](../../src/graph_training/array_env.py#L153).
   File: `src/array_search/labels.py`.

6. **Small ranker.** Pointwise or listwise model over
   `(graph_state, candidate features → score)`. Start with a tiny MLP over
   the existing 20 planet features + 26 edge features. Train on pairwise
   losses derived from the rollout labels. PyTorch is already in pyproject.
   File: `src/array_search/ranker.py`, training driver
   `scripts/train_ranker.py`.

7. **Self-improvement loop.** Replace fixed-policy self-play with
   ranker-vs-ranker self-play, re-cache, re-train. Cadence: cache N states,
   train K epochs, evaluate ranker vs previous-best in a small array-only
   tournament, accept if win-rate ≥ 55%. File:
   `scripts/run_self_play_loop.py`.

## Self-play multi-agent setup

For each scenario, each seat runs an independent policy. Options:

- **Fixed self-play**: same policy for all seats. Cheapest. Generates correlated
  states but is fine for the first round.
- **Diversified self-play**: each seat samples from a pool of policies
  (current ranker, previous-best ranker, random valid, heuristic). Better
  state distribution.
- **League-style**: keep a rotating cast of past rankers and sample from them.
  Defer until step 7 is working.

Each `(seat, turn)` becomes one training record. A 4-player game with horizon
H produces 4·H records. Map diversity comes from seeds. Hitting 10k records:
~50 seeds × 50 turns × 4 seats. Hitting 1M: ~5k seeds × 50 turns × 4 seats,
which is still cheap when the loop is array-only.

## Scoring philosophy

Carried verbatim from the old graph-training plan because it still applies:

1. Maximize owned production at horizon.
2. Break ties on production lead over strongest enemy.
3. Then ship share.
4. Penalize extinction / home loss / immediate recapture risk.

Implemented in [array_env.py:153](../../src/graph_training/array_env.py#L153)
(`score_arrays`).

## Risks and watch-outs

- **Map distribution drift.** Hand-rolled random maps may not match what
  Kaggle generates. Mitigation: source initial obs from `sim.make_env(seed)`
  at step 0 only. This preserves the real distribution without env in the
  training loop.
- **Inbound fleet events.** `build_graph_state` infers fleet destinations from
  ray-angle matching. If we synthesize fleets in flight during rollouts,
  the inference must produce the same destination we scheduled. Test:
  after `schedule_action_set`, the next-turn `build_graph_state` must list
  the same target.
- **Comet handling.** Comets are in `obs["comets"][i]["paths"]`. The geometry
  cache supports this, but the array adapter must thread comet paths through
  consistently across turns.
- **Reward hacking by rollout horizon.** Production-at-H=30 favours opportunistic
  captures the model might not be able to hold. Mitigate with multi-horizon
  labels (H ∈ {30, 60, 120}) and a survival penalty.
- **No engine ground-truth.** Once we're in pure-array land, bugs in
  `_resolve_combat` or `intercept` will train into the model invisibly.
  Mitigation: parity tests against the real engine for a sample of
  (state, action) → next-state transitions before we trust the loop.

## Submission path

This repo can still produce a `main.py` Kaggle submission. The current
`main.py` is a working v17-style baseline; once the array-search ranker is
trained, it gets embedded into a new `main.py` the same way the v17 ranker is
embedded today (look for `_v17_embedded_ranker` in `main.py`). Submission
process is unchanged — see [CLAUDE.md](../../CLAUDE.md).

## Open questions for the next session

1. Do we want to refactor physics out of `main.py` into `src/physics.py`
   before building the new pipeline, or after? Argument for *before*:
   `main.py` is 1500+ lines of submission code we don't want to drag through
   imports. Argument for *after*: refactor risk on a code path we know works.
2. Choose a starting horizon (30/60/120) and a ranker architecture (linear /
   MLP / tiny graph net) for step 6.
3. What does "evaluate ranker vs previous-best" look like in array-only land?
   We need an array-only tournament harness; the kaggle-env tournament from
   the old repo doesn't transfer cleanly.
