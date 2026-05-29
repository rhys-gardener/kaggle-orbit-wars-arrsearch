# Replay action-distribution analysis

Empirical quantiles of launch decisions made by **winning** seats across
200 Kaggle leaderboard replays from the dates analysed. Drives the
default values in `ActionFilters` (see `docs/plans/array-search-initiative.md`).

Replays are *not* training data — only a source of heuristic defaults.

## Proposed `ActionFilters` defaults

```python
@dataclass(frozen=True)
class ActionFilters:
    min_ships_per_launch = 1
    min_ships_pct_of_source = 0.061
    max_eta_unconditional = 52
    small_fleet_threshold = 25
    small_fleet_eta_cap = 61
    max_launches_per_turn = 25
    multi_source_bonus_evidence = 0.29156900087899207
```

Derivation: each cap is the P95 quantile of the relevant winning-seat
distribution; each minimum is P05. Tight enough to reject obviously bad
candidates without clipping anything top agents actually do.

## Sample sizes

- Episodes parsed: **200**
- Winning-seat launches: **142,403**
- Losing-seat launches: **38,666**
- Per-(seat, turn) cells (winners): **44,257**

## Ships per launch (winning seats)

- Quantiles: p05=1, p25=4, p50=12, p75=44, p95=473
- Compare losing: p50=19, p95=101

```
  [    0,     3)   25663 ########################################
  [    3,     5)   16281 #########################
  [    5,     8)   16159 #########################
  [    8,    15)   17582 ###########################
  [   15,    25)   14545 ######################
  [   25,    50)   19727 ##############################
  [   50,   100)   14195 ######################
  [  100,   200)    6775 ##########
  [  200,   500)    4585 #######
  [  500,  1000)    6891 ##########
```

## Ships as fraction of source garrison at launch

- p05=0.061, p25=0.636, p50=1.000, p75=6.400, p95=44.000

```
  [ 0.00,  0.05)    6315 #####
  [ 0.05,  0.10)    2935 ##
  [ 0.10,  0.20)    4661 ####
  [ 0.20,  0.35)    8018 #######
  [ 0.35,  0.50)    5803 #####
  [ 0.50,  0.70)   10464 #########
  [ 0.70,  0.85)    5425 #####
  [ 0.85,  1.00)    5791 #####
  [ 1.00,  1.50)   28853 ###########################
  [ 1.50,  2.50)    9704 #########
  [ 2.50,  5.00)   12075 ###########
  [ 5.00, 10.00)   42162 ########################################
```

## Launch ETA (winning seats)

- All: p50=10, p75=20, p95=52
- Small fleets (<25 ships): p50=12, p95=61
- Big fleets (≥25 ships):   p50=5, p95=26

```
  [    0,     5)   25972 ########################
  [    5,    10)   41905 ########################################
  [   10,    15)   25781 ########################
  [   15,    20)   11421 ##########
  [   20,    25)    7559 #######
  [   25,    35)   11554 ###########
  [   35,    50)    9349 ########
  [   50,    70)    5710 #####
  [   70,   100)    2433 ##
  [  100,   150)      49 
```

## Launches per seat per turn (winning seats only)

- All turns: p50=1, p75=3, p95=18
- Non-empty turns only: p50=2, p95=25

```
  [    0,     1)   16923 ########################################
  [    1,     2)    8839 ####################
  [    2,     3)    5077 ############
  [    3,     4)    2679 ######
  [    4,     5)    1579 ###
  [    5,     6)    4131 #########
  [    6,     7)     538 #
  [    7,     8)     360 
  [    8,     9)     277 
  [    9,    10)    3854 #########
```

## Multi-source attack rate

Of (episode, seat, turn) cells with ≥1 launch by a winning seat, **29.2%**
contained two or more launches hitting the same target on the same turn.
(High → multi-source coordination is a real signal worth biasing toward.)

## Target-owner mix (winning launches)

- Attacks on enemy planets: **13.2%**
- Captures on neutral planets: **6.4%**
- Defensive launches (target = my planet): **80.0%**

(If defensive_share is meaningful (>5%), the candidate ranker already
sees these via `target_owner_rel` — no special candidate kind needed.)
