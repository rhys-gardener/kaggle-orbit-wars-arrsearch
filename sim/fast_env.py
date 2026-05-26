"""FastEnv: byte-equivalent Orbit Wars simulator that calls the official
kaggle_environments interpreter() directly.

What we skip vs `kaggle_environments.make("orbit_wars")`:
  - `structify` deepcopy of state on every tick
  - `process_schema` validation on every action and state
  - stdout/stderr redirection per tick (StringIO setup)
  - per-tick agent wrappers, timeout enforcement
  - schema-driven step state mutation

What we keep:
  - the exact same interpreter() — same code path, same RNG, same outputs
  - the same step-counter / done / status semantics
  - per-step snapshots of planets+fleets (cheap shallow copies) so callers
    that scan `env.steps[::stride]` still work

Drop-in usage:
    from sim import make_env
    env = make_env(configuration={"episodeSteps": 500})
    env.run([agent_a, agent_b])
    final = env.steps[-1]
    reward = final[0].reward
"""
from __future__ import annotations

from typing import Any, Callable

from sim.interpreter_fast import interpreter


class _AttrDict(dict):
    """Dict that also supports attribute access — mimics kaggle's `structify`.

    The interpreter touches state both ways (`state[i].observation.planets`
    in interpreter code, `obs.get("planets", [])` in agent code), so we need
    both protocols on the same object.
    """

    __slots__ = ()

    def __getattr__(self, k):  # type: ignore[override]
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):  # type: ignore[override]
        self[k] = v

    def __delattr__(self, k):  # type: ignore[override]
        try:
            del self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def _make_initial_state(num_players: int) -> list[_AttrDict]:
    states: list[_AttrDict] = []
    for _ in range(num_players):
        obs = _AttrDict()
        obs.remainingOverageTime = 60.0
        s = _AttrDict()
        s.observation = obs
        s.action = None
        s.status = "ACTIVE"
        s.reward = 0.0
        s.info = {}
        states.append(s)
    return states


def _clone_action(action):
    if action is None:
        return None
    if isinstance(action, list):
        out = []
        for item in action:
            out.append(list(item) if isinstance(item, (list, tuple)) else item)
        return out
    return action


def _clone_comets(comets: list) -> list[dict]:
    """Copy mutable comet bookkeeping while sharing read-only path arrays."""
    return [
        {
            "planet_ids": list(g.get("planet_ids", [])),
            "paths": g.get("paths", []),
            "path_index": int(g.get("path_index", -1)),
        }
        for g in (comets or [])
    ]


def _copy_shared_observation_fields(src_obs) -> dict[str, Any]:
    return {
        "planets": [list(p) for p in src_obs.get("planets", [])],
        "fleets": [list(f) for f in src_obs.get("fleets", [])],
        "initial_planets": [list(p) for p in src_obs.get("initial_planets", [])],
        "comets": _clone_comets(src_obs.get("comets", [])),
        "comet_planet_ids": list(src_obs.get("comet_planet_ids", [])),
        "step": int(src_obs.get("step", 0) or 0),
        "angular_velocity": src_obs.get("angular_velocity", 0.0),
        "next_fleet_id": int(src_obs.get("next_fleet_id", 0) or 0),
    }


def _sync_observation_views(state: list[_AttrDict]) -> None:
    """Keep per-player observations aligned with player-0 shared state."""
    if not state:
        return
    obs0 = state[0].observation
    for i, s in enumerate(state):
        obs = s.observation
        obs.planets = obs0.planets
        obs.fleets = obs0.fleets
        obs.initial_planets = obs0.initial_planets
        obs.comets = obs0.comets
        obs.comet_planet_ids = obs0.comet_planet_ids
        obs.angular_velocity = obs0.angular_velocity
        obs.next_fleet_id = obs0.next_fleet_id
        obs.step = int(obs0.get("step", 0) or 0)
        obs.player = i


def _clone_state(state: list[_AttrDict]) -> list[_AttrDict]:
    """Deep-copy a live or snapshot state for branch-isolated rollouts."""
    if not state:
        return []
    shared = _copy_shared_observation_fields(state[0].observation)
    cloned: list[_AttrDict] = []
    for i, src_state in enumerate(state):
        src_obs = src_state.observation
        obs = _AttrDict()
        obs.planets = shared["planets"]
        obs.fleets = shared["fleets"]
        obs.initial_planets = shared["initial_planets"]
        obs.comets = shared["comets"]
        obs.comet_planet_ids = shared["comet_planet_ids"]
        obs.step = shared["step"]
        obs.angular_velocity = shared["angular_velocity"]
        obs.next_fleet_id = shared["next_fleet_id"]
        obs.player = i
        obs.remainingOverageTime = src_obs.get("remainingOverageTime", 60.0)

        dst_state = _AttrDict()
        dst_state.observation = obs
        dst_state.action = _clone_action(src_state.get("action", None))
        dst_state.status = src_state.get("status", "ACTIVE")
        dst_state.reward = src_state.get("reward", 0.0)
        dst_state.info = dict(src_state.get("info", {}) or {})
        cloned.append(dst_state)
    return cloned


def _snapshot_step(state: list[_AttrDict]) -> list[_AttrDict]:
    """Create a cheap per-tick snapshot of `state`.

    The interpreter mutates planet and fleet rows in place. Without a copy,
    every entry in `env.steps` would point at the final state. We shallow-
    copy each planet/fleet row (7-element lists) which is fast and matches
    what callers actually inspect.

    Status / reward are scalars and get copied by value.
    """
    obs0 = state[0].observation
    planets_snap = [p[:] for p in obs0.planets]
    fleets_snap = [f[:] for f in obs0.fleets]
    initial_planets_snap = [p[:] for p in obs0.initial_planets]
    # Comets share path arrays which are read-only post-spawn; copy outer.
    comets_snap = _clone_comets(obs0.comets)
    comet_pids_snap = list(obs0.comet_planet_ids)
    step_val = obs0.step
    av = obs0.angular_velocity
    next_fleet_id = obs0.next_fleet_id

    snap_states: list[_AttrDict] = []
    for i, s in enumerate(state):
        sobs = _AttrDict()
        sobs.planets = planets_snap
        sobs.fleets = fleets_snap
        sobs.comets = comets_snap
        sobs.comet_planet_ids = comet_pids_snap
        sobs.step = step_val
        sobs.angular_velocity = av
        sobs.next_fleet_id = next_fleet_id
        sobs.player = i
        sobs.remainingOverageTime = s.observation.get("remainingOverageTime", 60.0)
        sobs.initial_planets = initial_planets_snap
        ss = _AttrDict()
        ss.observation = sobs
        ss.action = _clone_action(s.action)
        ss.status = s.status
        ss.reward = s.reward
        ss.info = dict(s.get("info", {}) or {})
        snap_states.append(ss)
    return snap_states


def _plain_state(state: list[_AttrDict]) -> dict[str, Any]:
    """Return a compact serializable state record for offline search data."""
    if not state:
        raise ValueError("cannot snapshot an empty state")
    obs0 = state[0].observation
    return {
        "observation": {
            "planets": [list(p) for p in obs0.planets],
            "fleets": [list(f) for f in obs0.fleets],
            "initial_planets": [list(p) for p in obs0.initial_planets],
            "comets": _clone_comets(obs0.comets),
            "comet_planet_ids": list(obs0.comet_planet_ids),
            "step": int(obs0.step),
            "angular_velocity": obs0.angular_velocity,
            "next_fleet_id": int(obs0.next_fleet_id),
        },
        "players": [
            {
                "action": _clone_action(s.get("action", None)),
                "status": s.get("status", "ACTIVE"),
                "reward": s.get("reward", 0.0),
                "info": dict(s.get("info", {}) or {}),
                "remainingOverageTime": s.observation.get("remainingOverageTime", 60.0),
            }
            for s in state
        ],
    }


def _state_from_plain(record: dict[str, Any]) -> list[_AttrDict]:
    obs_record = record["observation"]
    players = record["players"]
    shared_obs = _AttrDict()
    shared_obs.planets = [list(p) for p in obs_record.get("planets", [])]
    shared_obs.fleets = [list(f) for f in obs_record.get("fleets", [])]
    shared_obs.initial_planets = [list(p) for p in obs_record.get("initial_planets", [])]
    shared_obs.comets = _clone_comets(obs_record.get("comets", []))
    shared_obs.comet_planet_ids = list(obs_record.get("comet_planet_ids", []))
    shared_obs.step = int(obs_record.get("step", 0) or 0)
    shared_obs.angular_velocity = obs_record.get("angular_velocity", 0.0)
    shared_obs.next_fleet_id = int(obs_record.get("next_fleet_id", 0) or 0)

    state: list[_AttrDict] = []
    for i, player_record in enumerate(players):
        obs = _AttrDict()
        obs.planets = shared_obs.planets
        obs.fleets = shared_obs.fleets
        obs.initial_planets = shared_obs.initial_planets
        obs.comets = shared_obs.comets
        obs.comet_planet_ids = shared_obs.comet_planet_ids
        obs.step = shared_obs.step
        obs.angular_velocity = shared_obs.angular_velocity
        obs.next_fleet_id = shared_obs.next_fleet_id
        obs.player = i
        obs.remainingOverageTime = player_record.get("remainingOverageTime", 60.0)

        s = _AttrDict()
        s.observation = obs
        s.action = _clone_action(player_record.get("action", None))
        s.status = player_record.get("status", "ACTIVE")
        s.reward = player_record.get("reward", 0.0)
        s.info = dict(player_record.get("info", {}) or {})
        state.append(s)
    return state


class FastEnv:
    """Minimal byte-equivalent Orbit Wars environment.

    Public surface chosen to match the bits of kaggle's `Environment` that
    the rest of this repo actually uses: `.run`, `.step`, `.reset`, `.steps`,
    `.configuration`, `.info`, `.done`, `.state`.
    """

    def __init__(self, configuration: dict | None = None, info: dict | None = None,
                 record_snapshots: bool = True):
        cfg = _AttrDict()
        # Defaults sourced from orbit_wars.json
        cfg.episodeSteps = 500
        cfg.shipSpeed = 6.0
        cfg.cometSpeed = 4.0
        cfg.seed = None
        cfg.actTimeout = 1.0
        cfg.runTimeout = 9999.0
        cfg.agentTimeout = 2.0
        cfg.maxLogLength = 10000
        if configuration:
            for k, v in configuration.items():
                cfg[k] = v
        self.configuration = cfg
        self.info: dict = info if info is not None else {}
        self.state: list[_AttrDict] | None = None
        self.steps: list[list[_AttrDict]] = []
        self.done: bool = False
        self._record_snapshots = record_snapshots
        self._current_step: int = -1

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------
    def reset(self, num_agents: int) -> list[_AttrDict]:
        """Run the interpreter's init phase. Mirrors kaggle's reset() so the
        interpreter sees the same conditions it does in real games:
          - all statuses set to INACTIVE so env.done is True during init
          - interpreter populates obs0.planets etc.
          - step is set to 0 (kaggle: `0 if self.done else len(self.steps)`)
          - statuses restored to ACTIVE for the first real tick
        """
        self.state = _make_initial_state(num_agents)
        for s in self.state:
            s.status = "INACTIVE"
        interpreter(self.state, self)
        self.state[0].observation.step = 0
        _sync_observation_views(self.state)
        for s in self.state:
            s.status = "ACTIVE"
        self.done = False
        self._current_step = 0
        # First entry in `.steps` is the post-init snapshot (kaggle convention).
        self.steps = [_snapshot_step(self.state) if self._record_snapshots else self.state]
        return self.state

    def step(self, actions: list[Any]) -> list[_AttrDict]:
        if self.state is None:
            raise RuntimeError("Call reset(num_agents) before step().")
        if self.done:
            raise RuntimeError("Environment is done; reset before stepping again.")
        current_step = int(self.state[0].observation.get("step", self._current_step) or 0)
        for i, action in enumerate(actions):
            self.state[i].action = action if action is not None else []

        interpreter(self.state, self)

        # Replicate kaggle's post-interpreter bookkeeping (core.py:602, 272).
        # A mid-game clone may not have historical snapshots, so derive the
        # next turn from an explicit current step rather than len(self.steps).
        self.state[0].observation.step = current_step + 1
        self._current_step = current_step + 1
        _sync_observation_views(self.state)

        if self.state[0].observation.step >= self.configuration.episodeSteps - 1:
            for s in self.state:
                if s.status in ("ACTIVE", "INACTIVE"):
                    s.status = "DONE"

        self.done = all(s.status != "ACTIVE" for s in self.state)
        self.steps.append(_snapshot_step(self.state) if self._record_snapshots else self.state)
        return self.state

    def run(self, agents: list[Callable[[Any], Any] | None]) -> list[list[_AttrDict]]:
        """Run a full game. `agents` are callables that map obs -> list of moves.

        Agent exceptions are caught and the offending agent is marked ERROR
        (matching kaggle's `__run_interpreter_prod` semantics: a crashing
        agent loses but the game continues for others).
        """
        if self.state is None or self.done:
            self.reset(len(agents))
        if len(self.state) != len(agents):  # type: ignore[arg-type]
            raise ValueError(f"Expected {len(self.state)} agents, got {len(agents)}.")  # type: ignore[arg-type]

        while not self.done:
            actions: list[Any] = []
            for i, ag in enumerate(agents):
                if self.state[i].status != "ACTIVE" or ag is None:  # type: ignore[index]
                    actions.append(None)
                    continue
                obs = self.state[i].observation  # type: ignore[index]
                try:
                    actions.append(ag(obs))
                except Exception:
                    actions.append(None)
                    self.state[i].status = "ERROR"  # type: ignore[index]
                    self.state[i].reward = None  # type: ignore[index]
            self.step(actions)
        return self.steps

    # ------------------------------------------------------------------
    # Mid-game fork helpers for offline policy-improvement search
    # ------------------------------------------------------------------
    def clone(self, record_snapshots: bool | None = None,
              preserve_history: bool = True) -> "FastEnv":
        """Return a branch-isolated copy that can continue from this turn.

        Mutable game state is deep-copied; comet path coordinate arrays are
        shared because the interpreter treats them as read-only and only
        mutates group metadata such as `planet_ids` and `path_index`.
        """
        if self.state is None:
            raise RuntimeError("Cannot clone before reset().")
        keep_snapshots = self._record_snapshots if record_snapshots is None else record_snapshots
        clone = FastEnv(
            configuration=dict(self.configuration),
            info=dict(self.info),
            record_snapshots=keep_snapshots,
        )
        clone.state = _clone_state(self.state)
        clone.done = self.done
        clone._current_step = int(clone.state[0].observation.step)
        _sync_observation_views(clone.state)

        if preserve_history and self.steps:
            if keep_snapshots and self._record_snapshots:
                clone.steps = [_clone_state(step) for step in self.steps]
            else:
                count = max(len(self.steps), clone._current_step + 1)
                clone.steps = [clone.state for _ in range(count)]
        else:
            clone._seed_steps_from_current()

        expected = clone._current_step + 1
        if len(clone.steps) < expected:
            if keep_snapshots:
                clone.steps.extend(_snapshot_step(clone.state) for _ in range(expected - len(clone.steps)))
            else:
                clone.steps.extend(clone.state for _ in range(expected - len(clone.steps)))
        elif len(clone.steps) > expected:
            clone.steps = clone.steps[:expected]
        return clone

    def to_snapshot(self) -> dict[str, Any]:
        """Serialize the current state without full replay history."""
        if self.state is None:
            raise RuntimeError("Cannot snapshot before reset().")
        return {
            "configuration": dict(self.configuration),
            "info": dict(self.info),
            "done": self.done,
            "current_step": int(self.state[0].observation.step),
            "state": _plain_state(self.state),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any],
                      record_snapshots: bool = True) -> "FastEnv":
        env = cls(
            configuration=snapshot.get("configuration", {}),
            info=dict(snapshot.get("info", {}) or {}),
            record_snapshots=record_snapshots,
        )
        env.state = _state_from_plain(snapshot["state"])
        env.done = bool(snapshot.get("done", False))
        env._current_step = int(snapshot.get(
            "current_step",
            env.state[0].observation.get("step", 0),
        ) or 0)
        env.state[0].observation.step = env._current_step
        _sync_observation_views(env.state)
        env._seed_steps_from_current()
        return env

    @classmethod
    def from_state(cls, state: list[_AttrDict], configuration: dict | None = None,
                   info: dict | None = None, done: bool = False,
                   record_snapshots: bool = True) -> "FastEnv":
        env = cls(configuration=configuration, info=info, record_snapshots=record_snapshots)
        env.state = _clone_state(state)
        env.done = done
        env._current_step = int(env.state[0].observation.get("step", 0) or 0)
        _sync_observation_views(env.state)
        env._seed_steps_from_current()
        return env

    def _seed_steps_from_current(self) -> None:
        if self.state is None:
            self.steps = []
            return
        count = int(self.state[0].observation.get("step", 0) or 0) + 1
        if self._record_snapshots:
            self.steps = [_snapshot_step(self.state) for _ in range(count)]
        else:
            self.steps = [self.state for _ in range(count)]


def make_env(configuration: dict | None = None, info: dict | None = None,
             record_snapshots: bool = True, kaggle_sim: bool = False, **_ignored):
    """Factory matching `kaggle_environments.make("orbit_wars", ...)` shape.

    Pass `kaggle_sim=True` to get the real kaggle env (e.g. for parity tests
    or replays). Default uses FastEnv.
    """
    if kaggle_sim:
        from kaggle_environments import make
        kwargs = {}
        if configuration is not None:
            kwargs["configuration"] = configuration
        if info is not None:
            kwargs["info"] = info
        return make("orbit_wars", debug=False, **kwargs)
    return FastEnv(configuration=configuration, info=info, record_snapshots=record_snapshots)
