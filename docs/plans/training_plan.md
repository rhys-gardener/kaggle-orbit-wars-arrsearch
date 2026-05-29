# Array-Search Training Plan

Status: implemented v1, ready for replay-initialized self-play calibration.

## Summary

The current training stack now has the intended four-stage shape:

1. Sparse exact geometry per map, persisted and reused across repeated games.
2. Replay bootstrap records/checkpoint from `replays/sample`.
3. MCTS-style teacher labels over valid action sets.
4. Two-policy self-play league with replay initialization, self-play labels, and sampled teacher labels.

The core bet remains unchanged: use the engine only for initial observations and replay ingestion, then train through array rollouts and exact cached geometry.

## Implemented

### Sparse geometry
- Production cache: `src/graph_training/sparse_geometry_cache.py`.
- Interface: `.lookup(...)`, `.save(...)`, `.load(...)`, compatible with cached candidate generation.
- Default buckets:
  `1,2,4,8,16,24,32,40,48,56,64,80,96,112,128,160,192,224,256`.
- `scripts/run_self_play_loop.py` now defaults to `--geometry-mode sparse` and writes per-seed cache files under `<out-dir>/geometry/`.

### Replay bootstrap
- Script: `scripts/build_replay_bootstrap.py`.
- Reads replay JSONs from `replays/sample`.
- Builds graph records, matches replay launches to generated candidates, and injects exact replay candidates when bucketed candidates miss the replay move.
- Trains an optional initial ranker checkpoint and writes reusable `.pkl` record shards.

### MCTS teacher v1
- Module: `src/array_search/mcts_teacher.py`.
- Operates over whole-turn action sets, not raw launches.
- Uses rollout value plus optional ranker prior to choose teacher labels.
- Policy-response only: `--mcts-depth >= 2` is required (`< 2` is rejected). The
  redundant passive root-rollout mode was removed — the base labeller
  (`labels.label_record`) already passive-rolls every action set for free, so the
  teacher exists solely to add active future play the base loop cannot produce.
- `--mcts-opponent-samples N` (now live; was reserved) branches the rollout over
  N stochastic opponent continuations (opponents play the learned policy with
  `softmax_temperature` exploration) and aggregates with `--mcts-aggregate`
  (`mean` = expected value, `min` = worst-case/adversarial). N=1 collapses to a
  single greedy line.
- Emits labels in the same candidate-positive shape used by the ranker.

### Self-play league
- Runner: `scripts/run_self_play_loop.py`.
- Two trainable policies A/B, assigned across seats with seed-rotated parity.
- Supports replay initialization via `--replay-prior-checkpoint`.
- Supports replay rows in the training buffer via `--replay-records`, but this is currently too expensive for the full 200-replay shard set without sampling.
- Supports sampled teacher labels via `--mcts-teacher-rate`.
- Supports map reuse via `--map-pool-size` and `--games-per-map`.
- Logs flushed JSONL progress rows for run start, geometry load/save, game score deltas, MCTS labels, train timing, checkpoint saves, and round summaries.

## Caveats

- The MCTS teacher is policy-response with stochastic opponent-sampling branching
  (`opponent_samples` continuations aggregated by mean/min). It is still not a
  full UCT tree: future turns are played by the current learned policy (sampled),
  not selected by UCB over an expanded tree, and there is no backprop/tree reuse.
  A full UCT implementation remains future work.
- Sparse geometry is exact but can have a cold-start cost on the first game for a map. Reusing maps amortizes this, and the cache persists as `.npz`.
- Replay bootstrap currently uses downloaded/local replay JSONs only. More replays can be downloaded later through the dataset manifest/Kaggle CLI path.
- Replay bootstrap currently accumulates all records in memory before writing shards and only writes its manifest at the end. The 200-replay bootstrap run produced 23,025 records across 90 shards, about 2.84 GB, and took about 11.8 hours. Before larger replay runs, add incremental shard flushing plus a `progress.jsonl`/CSV log with replay index, elapsed time, records found, shard writes, and memory use.
- Self-play currently mixes replay records by loading/training against the full replay buffer each iteration. With the 200-replay bootstrap, the first smoke used about 22.9k rows per policy update and became the dominant cost. Before larger league runs, add replay-buffer sampling/capping per policy update, plus logging for replay/self-play/MCTS row counts.
- `round_score` is an A-vs-B relative signal, not an absolute strength measure. Read it over a full balanced seed window, because individual maps can still favour one seat parity.
- Rollout labels score fixed first-turn action sets without future launches. Policy-response MCTS depth does simulate future launches for teacher-labelled records, but ordinary rollout labels still do not. Passive rollout labels also do not yet apply future comet active/inactive masks inside `rollout_cached_action_set`; self-play state stepping and policy-response teacher stepping handle comet masks.
- Submission-time action generation is not implemented here. The trained model will still need a compact online candidate/refinement path in `main.py`.

## Run Sequence

### 1. Replay bootstrap calibration

Start with a moderate local sample:

```bash
uv run python scripts/build_replay_bootstrap.py \
  --out-dir runs/replay_bootstrap_calib_001 \
  --max-files 10 \
  --max-turns 250 \
  --candidate-limit 80 \
  --max-launches 10 \
  --beam-width 32 \
  --train-epochs 3 \
  --shard-size 256
```

Expected outputs:
- `runs/replay_bootstrap_calib_001/records/*.pkl`
- `runs/replay_bootstrap_calib_001/checkpoints/replay_bootstrap_ranker.pt`
- `runs/replay_bootstrap_calib_001/manifest.json`

### 2. Sparse self-play + teacher calibration

Use the 200-replay checkpoint as the initial policy prior, but do not include
the full replay record directory until replay-buffer sampling/capping is added.
Track improvement using `round_score.mean_policy_score_delta_a_minus_b`,
`wins_a`, `wins_b`, and the per-policy `iter` loss rows.

```bash
uv run python scripts/run_self_play_loop.py \
  --out-dir runs/sparse_mcts_200prior_calib_001 \
  --rounds 50 \
  --seeds-per-round 1 \
  --num-players 4 \
  --horizon-turns 50 \
  --geometry-mode sparse \
  --candidate-limit 40 \
  --max-launches 6 \
  --beam-width 12 \
  --max-engine-steps 100 \
  --mcts-teacher-rate 0.05 \
  --mcts-root-action-sets 8 \
  --mcts-depth 2 \
  --mcts-opponent-samples 3 \
  --mcts-aggregate mean \
  --mcts-rollout-horizon 25 \
  --explore-temperature 0.5 \
  --long-horizon-fraction 0.15 \
  --long-horizon-turns 250 \
  --replay-prior-checkpoint runs/replay_bootstrap_200_001/checkpoints/replay_bootstrap_ranker.pt \
  --map-pool-size 8 \
  --games-per-map 6 \
  --eval-every 99
```

Expected outputs:
- `runs/sparse_mcts_200prior_calib_001/training.jsonl`
- `runs/sparse_mcts_200prior_calib_001/checkpoints/*.pt`
- `runs/sparse_mcts_200prior_calib_001/geometry/*.npz`

Live monitor:

```powershell
Get-Content runs/sparse_mcts_200prior_calib_001/training.jsonl -Tail 40 -Wait
```

### 3. Scale after inspection

Only scale after checking:
- records per second,
- sparse geometry cold/warm behavior,
- teacher label count,
- loss trend for A/B,
- `round_score` A-vs-B deltas and win counts,
- candidate counts per turn,
- whether games have meaningful production/ship movement by horizon 20.

Likely next scale:

```bash
--rounds 50 --seeds-per-round 2 --horizon-turns 50 \
--map-pool-size 16 --games-per-map 20 --mcts-teacher-rate 0.05
```

For a heavier teacher calibration, prefer an explicit depth setting so the cost
is visible in the command:

```bash
--mcts-teacher-rate 0.05 --mcts-root-action-sets 8 \
--mcts-depth 2 --mcts-rollout-horizon 50
```

## Verification

Current verification completed:

- `uv run python -m compileall src scripts tests`
- `uv run pytest` -> 69 passed
- Replay bootstrap smoke generated 43 replay records and a checkpoint.
- Sparse/MCTS self-play smoke generated sparse geometry, teacher log rows, and A/B checkpoints.
- Replay-buffer smoke confirmed replay rows are included in A/B training.

Key regression coverage:
- sparse geometry save/load,
- replay exact candidate injection,
- MCTS teacher labels are reachable,
- comet path parity,
- physics parity,
- state adapter parity,
- multi-seat array stepping.
