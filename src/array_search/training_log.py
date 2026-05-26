"""Append-only JSONL logger for the array-search training loop.

One line per event. Each line is a JSON object with a ``"t"`` (event type)
field plus event-specific fields and an automatic ``"ts"`` ISO timestamp.

Reading the log:

    import pandas as pd
    df = pd.read_json("logs/run_xyz.jsonl", lines=True)
    df[df["t"] == "iter"].plot(x="iter", y="loss")

Event types:
    iter          per-training-iteration loss/grad stats
    game          one self-play game outcome
    eval          tournament evaluation vs incumbent pool
    filter_stats  rejection counts from ActionFilters
    msg           free-form annotation / error / checkpoint pointer
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any


class TrainingLogger:
    """Thread-unsafe, line-buffered JSONL logger.

    Open one per training run. Safe to call ``close()`` more than once.
    """

    def __init__(self, path: Path | str, *, append: bool = True, flush_each: bool = True):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._f = open(self._path, mode, encoding="utf-8")
        self._flush_each = flush_each

    @property
    def path(self) -> Path:
        return self._path

    def _emit(self, event_type: str, **fields: Any) -> None:
        if self._f.closed:
            raise RuntimeError("TrainingLogger is closed")
        payload: dict[str, Any] = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "t": event_type,
        }
        for k, v in fields.items():
            payload[k] = _json_safe(v)
        self._f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        if self._flush_each:
            self._f.flush()

    # -- Convenience wrappers; callers can pass extra **kwargs ad-hoc -----------

    def log_iter(
        self,
        *,
        iter: int,
        policy: str,
        loss: float,
        loss_listwise: float | None = None,
        loss_pairwise: float | None = None,
        grad_norm: float | None = None,
        n_records: int | None = None,
        mean_candidates_per_turn: float | None = None,
        **extra: Any,
    ) -> None:
        self._emit(
            "iter",
            iter=int(iter),
            policy=str(policy),
            loss=float(loss),
            loss_listwise=loss_listwise,
            loss_pairwise=loss_pairwise,
            grad_norm=grad_norm,
            n_records=n_records,
            mean_candidates_per_turn=mean_candidates_per_turn,
            **extra,
        )

    def log_game(
        self,
        *,
        iter: int,
        seed: int,
        seat_policy: dict[int, str],
        production_by_seat: dict[int, float],
        ships_by_seat: dict[int, float],
        winner_seat: int | None,
        horizon_turns: int,
        **extra: Any,
    ) -> None:
        self._emit(
            "game",
            iter=int(iter),
            seed=int(seed),
            seat_policy={str(k): v for k, v in seat_policy.items()},
            production_by_seat={str(k): float(v) for k, v in production_by_seat.items()},
            ships_by_seat={str(k): float(v) for k, v in ships_by_seat.items()},
            winner_seat=winner_seat,
            horizon_turns=int(horizon_turns),
            **extra,
        )

    def log_eval(
        self,
        *,
        iter: int,
        challenger: str,
        opponent: str,
        n_games: int,
        win_rate: float,
        mean_production_lead: float,
        spearman_vs_rollout_label: float | None = None,
        **extra: Any,
    ) -> None:
        self._emit(
            "eval",
            iter=int(iter),
            challenger=str(challenger),
            opponent=str(opponent),
            n_games=int(n_games),
            win_rate=float(win_rate),
            mean_production_lead=float(mean_production_lead),
            spearman_vs_rollout_label=spearman_vs_rollout_label,
            **extra,
        )

    def log_filter_stats(
        self,
        *,
        iter: int,
        n_pre_filter: int,
        n_post_filter: int,
        reject_counts: dict[str, int],
        **extra: Any,
    ) -> None:
        self._emit(
            "filter_stats",
            iter=int(iter),
            n_pre_filter=int(n_pre_filter),
            n_post_filter=int(n_post_filter),
            reject_counts={str(k): int(v) for k, v in reject_counts.items()},
            reject_rate=(
                float((n_pre_filter - n_post_filter) / n_pre_filter) if n_pre_filter else 0.0
            ),
            **extra,
        )

    def log_msg(self, message: str, **extra: Any) -> None:
        self._emit("msg", message=str(message), **extra)

    # -- lifecycle -----------------------------------------------------------------

    def close(self) -> None:
        if not self._f.closed:
            self._f.flush()
            self._f.close()

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _json_safe(value: Any) -> Any:
    """Coerce numpy scalars / Path objects into JSON-native types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    # numpy fallback (don't import numpy at module level)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Convenience reader. For analysis, prefer ``pandas.read_json(..., lines=True)``."""
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
