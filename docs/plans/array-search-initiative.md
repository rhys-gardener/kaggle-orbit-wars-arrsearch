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
  `_solve_engine_angle`, `_trajectory_first_hit`). PR-1 extracts a copy of
  these into `src/physics.py` for the training pipeline to import; `main.py`
  retains its own copies (Kaggle requires a self-contained submission file).
  A parity test asserts the two copies stay in agreement.
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

6. **Small ranker — candidate-as-row, listwise.** Pointwise MLP over a single
   candidate's feature vector → scalar score. **The number of planets P never
   enters the model dimension**; each candidate (an (src, tgt, ships) launch)
   becomes one input row of ~24 features, and the model is invariant to P and
   K (number of candidates per turn). Training is listwise: per `(seat, turn)`
   record, softmax over the K candidate scores, cross-entropy against the
   rollout-best candidate (with H=60 as the primary label, H=30/120 as
   auxiliaries). Auxiliary pairwise hinge against bottom-quartile candidates
   stabilises early training. Architecture: `Linear(m, 64) → ReLU →
   Linear(64, 64) → ReLU → Linear(64, 1)`. PyTorch is in pyproject.
   Inference per turn: score every candidate, greedy-pack into an action set
   within the ship budget (matches `search.py`'s composition logic).
   Files: `src/array_search/features.py` (candidate row construction),
   `src/array_search/ranker.py`, training driver `scripts/train_ranker.py`.

   Per-candidate feature columns (rough cut — finalise in PR-3):
   - **Target/edge**: `valid_mask`, `eta`, `distance`, `ship_cost_min`,
     `ship_cost_125pct`, `ship_cost_full`, `target_ships_now`, `target_prod`,
     `target_owner_rel` (mine/neutral/enemy one-hots), `target_radius`,
     `target_garrison_at_eta`, `friendly_ships_arriving_before_eta`,
     `enemy_ships_arriving_before_eta`, `enemy_min_eta_to_target`,
     `target_prod_rank`,
     `future_prod_value = target_prod * max(0, MAX_TURNS - step - eta)`,
     `path_clear (actual_hit_id == target_id)`,
     `target_orbit_radius`, `target_phase_at_eta (sin/cos)`.
   - **Defence-specific**: `target_is_my_threatened_planet` (target_owner_mine
     AND target_under_threat). Added because replay analysis shows 72% of
     winning launches target the agent's own planets — a dedicated feature
     gives the model a direct head into the defence decision.
   - **Source**: `source_ships`, `source_prod`, `source_under_threat`.
   - **Filter-tag flags** (from `ActionFilters` lax tier): `is_below_typical_min_ships`,
     `is_high_eta_launch`. Marginal candidates that pass the strict cuts but
     are unusual; the model learns when to use them.
   - **Global (broadcast across rows of one turn)**: `turn_remaining`,
     `my_planet_share`, `my_prod_share`, `num_players`.

7. **Self-improvement loop — concurrent two-ranker self-play.** No fixed
   policy in the data path. Maintain two policy networks π_A, π_B, each with
   its own optimiser and replay buffer. Every game: seats 0/2 use π_A, seats
   1/3 use π_B (rotate per seed to neutralise positional bias). Each network
   trains only on its own seats' records, so they diverge stochastically and
   never collapse to a single self-imitating fixed point.

   **Cold-start**: both networks initialised from random weights. Untrained
   networks still emit a preference over candidates; the rollout labeller
   provides ground truth from turn 0. First 2-3 rounds are noisy but the
   loop is self-bootstrapping — no heuristic baseline in the data path.

   **Acceptance gate**: every K rounds, snapshot π_A and π_B into an
   **incumbent pool**. Challenger plays the pool (sampled opponents);
   accept-as-new-best at win-rate ≥ 55% against the pool average. The
   "previous-best" framing from earlier drafts of this plan is replaced by a
   small pool to avoid overfitting to one specific opponent.

   File: `scripts/run_self_play_loop.py`. Tournament harness:
   `scripts/array_tournament.py` (~200 lines: N seeds × seat rotations ×
   T=300 turns, scored by `score_arrays`).

## Self-play multi-agent setup

The data path never uses a fixed policy. Two networks, π_A and π_B, train
simultaneously off the same games: in a 4-player game seats 0/2 are driven by
π_A, seats 1/3 by π_B; seat assignment is rotated per seed. Each network
trains only on its own seats' records.

This replaces the older "fixed self-play → diversified → league-style"
progression. Cold-start is from random weights; first rounds are noisy but
self-bootstrapping. An **incumbent pool** of past snapshots provides
opponents for the acceptance gate (see step 7), not the training distribution.

Each `(seat, turn)` becomes one training record tagged with `policy_id` (A or
B) so simultaneous training can route correctly. A 4-player game with horizon
H produces 4·H records (2·H per network). Map diversity comes from seeds.
Hitting 10k records: ~50 seeds × 50 turns × 4 seats. Hitting 1M: ~5k seeds.

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

## Resolved decisions (carried out of the 2026-05-26 scoping session)

1. **Physics extraction order**: copy (not move) the 6 helpers into
   `src/physics.py` **before** the new pipeline. `main.py` keeps its copies
   (Kaggle requires self-contained submission). Parity test in
   `tests/test_physics_parity.py` gates drift. Done in PR-1.
2. **Rollout horizon**: label every action set at H ∈ {30, 60, 120}. **Train
   against H=60 as the primary head**, with 30 and 120 as auxiliaries.
3. **Ranker architecture**: candidate-as-row pointwise MLP, listwise softmax
   loss. See gap 6 for the feature columns. P (planet count) does not enter
   the model dimension.
4. **Self-play training**: two networks trained simultaneously (π_A, π_B),
   never a fixed policy. Cold-start from random weights. See "Self-play
   multi-agent setup" and gap 7.
5. **Tournament gate**: array-only `scripts/array_tournament.py`, ~200 lines,
   no kaggle env. Acceptance threshold 55% against an incumbent pool, not
   a single previous-best.

## Action filters (data-driven, set 2026-05-26)

Derived from 100 May 24-25 leaderboard replays (41k winning-seat launches).
Full analysis: [docs/replay_action_stats.md](../replay_action_stats.md).
Two surprises overturned my initial intuition: (a) small fleets travel
*further* than big ones in winning play (P95 eta of <25-ship fleets = 70;
of ≥25-ship fleets = 28), and (b) **72% of winning launches target the
agent's own planets** — defence is the dominant action type.

### Strict filter (hard reject from candidate space)
- `eta > 80` — margin above winner P95=57. No useful launches happen here.
- `ships < 1` — invalid.
- `ships ≤ 2 AND eta > 40` — excludes "tiny lob across the board"; rarely
  occurs in winning play and tends to be wasted production.

### Lax filter (admit + feature flag)
- `ships ≤ 3` → `is_below_typical_min_ships` (~13% of winning launches).
- `eta > 50` → `is_high_eta_launch` (~7% of winning launches).

These feed into the per-candidate row so the ranker can learn when the
unusual cases actually win.

### Action-set composition
- `max_launches_per_turn = 10` (hard cap). P95 of non-empty turns = 11;
  going much above is noise.
- `multi_source_bonus = 0.15` — soft additive bonus during action-set
  composition when ≥2 candidates already in the set share a target. 22.4%
  of winning multi-launch turns coordinate this way.

### Dropped
- `min_ships_per_launch` as a hard threshold — winner P05 is 1 ship, and
  ~28% of winning launches are <8 ships. Hard-filtering them would block
  legitimate strategy.
- `min_ships_pct_of_source` — noisy because production accrues between turns
  (P75 of `ships/garrison_at_launch` = 10.5×). Drop the knob entirely.
- "Defensive candidate kind" — the candidate space already contains
  my→my launches via the existing edge loop in `state.py`. The model
  learns defence via `target_owner_rel` and the new
  `target_is_my_threatened_planet` flag.

### Ablation hatch
All filter values live in a `frozen` dataclass; setting `max_eta_strict =
999, max_launches_per_turn = 99, …` reproduces the unfiltered baseline.
After the ranker is trained, run one round with the relaxed config to
test whether the model wants to break any of these rules.

## Open questions for later sessions
2. **PR-1 schedule schema bump** — agreed shape:
   `schedule[absolute_arrival_turn] → list[(source_id, target_id, owner, ships, launch_turn)]`.
   Existing `_resolve_combat` only cares about `(target_idx, owner, ships)`,
   so the bump is additive; `state_adapter.arrays_to_obs` is the consumer
   that reads `source_id` and `launch_turn` to synthesise in-flight fleets.
3. **Step-0 planet snapshot** stored alongside the array tuple so we can
   derive each planet's `phase0` without redundant state. Decided yes.
