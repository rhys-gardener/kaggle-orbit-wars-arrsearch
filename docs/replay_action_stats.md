# Replay action-distribution analysis

Empirical quantiles of launch decisions made by **winning** seats across
100 Kaggle leaderboard replays from the dates analysed. Drives the
default values in `ActionFilters` (see `docs/plans/array-search-initiative.md`).

Replays are *not* training data — only a source of heuristic defaults.

## Proposed `ActionFilters` defaults

```python
@dataclass(frozen=True)
class ActionFilters:
    min_ships_per_launch = 1
    min_ships_pct_of_source = 0.062
    max_eta_unconditional = 57
    small_fleet_threshold = 25
    small_fleet_eta_cap = 70
    max_launches_per_turn = 11
    multi_source_bonus_evidence = 0.22385008517887564
```

Derivation: each cap is the P95 quantile of the relevant winning-seat
distribution; each minimum is P05. Tight enough to reject obviously bad
candidates without clipping anything top agents actually do.

## Sample sizes

- Episodes parsed: **100**
- Winning-seat launches: **41,166**
- Losing-seat launches: **22,674**
- Per-(seat, turn) cells (winners): **23,091**

## Ships per launch (winning seats)

- Quantiles: p05=1, p25=5, p50=20, p75=55, p95=747
- Compare losing: p50=9, p95=88

```
  [    0,     3)    5336 #############################
  [    3,     5)    3421 ##################
  [    5,     8)    4168 #######################
  [    8,    15)    4736 ##########################
  [   15,    25)    4888 ##########################
  [   25,    50)    7245 ########################################
  [   50,   100)    5407 #############################
  [  100,   200)    2331 ############
  [  200,   500)    1195 ######
  [  500,  1000)    2439 #############
```

## Ships as fraction of source garrison at launch

- p05=0.062, p25=0.750, p50=2.581, p75=10.500, p95=58.500

```
  [ 0.00,  0.05)    1773 ####
  [ 0.05,  0.10)     876 ##
  [ 0.10,  0.20)    1055 ##
  [ 0.20,  0.35)    1800 ####
  [ 0.35,  0.50)    1432 ###
  [ 0.50,  0.70)    2798 ######
  [ 0.70,  0.85)    1519 ###
  [ 0.85,  1.00)    1520 ###
  [ 1.00,  1.50)    5312 ############
  [ 1.50,  2.50)    2344 #####
  [ 2.50,  5.00)    3693 ########
  [ 5.00, 10.00)   17008 ########################################
```

## Launch ETA (winning seats)

- All: p50=10, p75=22, p95=57
- Small fleets (<25 ships): p50=14, p95=70
- Big fleets (≥25 ships):   p50=6, p95=28

```
  [    0,     5)    8081 ###########################
  [    5,    10)   11676 ########################################
  [   10,    15)    6015 ####################
  [   15,    20)    3725 ############
  [   20,    25)    2404 ########
  [   25,    35)    3693 ############
  [   35,    50)    2564 ########
  [   50,    70)    1709 #####
  [   70,   100)    1058 ###
  [  100,   150)      72 
```

## Launches per seat per turn (winning seats only)

- All turns: p50=1, p75=2, p95=6
- Non-empty turns only: p50=2, p95=11

```
  [    0,     1)   11338 ########################################
  [    1,     2)    4502 ###############
  [    2,     3)    2306 ########
  [    3,     4)    1155 ####
  [    4,     5)     637 ##
  [    5,     6)    1817 ######
  [    6,     7)     206 
  [    7,     8)     177 
  [    8,     9)     141 
  [    9,    10)     812 ##
```

## Multi-source attack rate

Of (episode, seat, turn) cells with ≥1 launch by a winning seat, **22.4%**
contained two or more launches hitting the same target on the same turn.
(High → multi-source coordination is a real signal worth biasing toward.)

## Target-owner mix (winning launches)

- Attacks on enemy planets: **17.6%**
- Captures on neutral planets: **10.0%**
- Defensive launches (target = my planet): **72.0%**

(If defensive_share is meaningful (>5%), the candidate ranker already
sees these via `target_owner_rel` — no special candidate kind needed.)
