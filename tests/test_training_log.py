"""Tests for the JSONL training logger."""

from __future__ import annotations

import json

import numpy as np

from src.array_search.training_log import TrainingLogger, read_jsonl


def test_writes_jsonl_line_per_event(tmp_path):
    log_path = tmp_path / "run.jsonl"
    with TrainingLogger(log_path) as log:
        log.log_iter(iter=1, policy="A", loss=1.23, grad_norm=0.5)
        log.log_iter(iter=2, policy="A", loss=1.18, grad_norm=0.4)
        log.log_msg("checkpoint saved", path="weights/A_2.pt")

    rows = read_jsonl(log_path)
    assert len(rows) == 3
    assert rows[0]["t"] == "iter"
    assert rows[0]["iter"] == 1
    assert rows[0]["policy"] == "A"
    assert rows[2]["t"] == "msg"
    assert rows[2]["message"] == "checkpoint saved"
    assert rows[2]["path"] == "weights/A_2.pt"
    # Every row gets a timestamp.
    for r in rows:
        assert "ts" in r


def test_log_game(tmp_path):
    log_path = tmp_path / "games.jsonl"
    with TrainingLogger(log_path) as log:
        log.log_game(
            iter=4,
            seed=12345,
            seat_policy={0: "A", 1: "B"},
            production_by_seat={0: 47.0, 1: 41.0},
            ships_by_seat={0: 240.0, 1: 180.0},
            winner_seat=0,
            horizon_turns=300,
        )
    rows = read_jsonl(log_path)
    assert rows[0]["t"] == "game"
    assert rows[0]["winner_seat"] == 0
    # Dict keys are stringified for JSON compatibility.
    assert rows[0]["seat_policy"] == {"0": "A", "1": "B"}


def test_log_eval_with_optional_field(tmp_path):
    log_path = tmp_path / "eval.jsonl"
    with TrainingLogger(log_path) as log:
        log.log_eval(
            iter=10,
            challenger="A",
            opponent="pool",
            n_games=50,
            win_rate=0.58,
            mean_production_lead=4.2,
            spearman_vs_rollout_label=0.71,
        )
    rows = read_jsonl(log_path)
    assert rows[0]["win_rate"] == 0.58
    assert rows[0]["spearman_vs_rollout_label"] == 0.71


def test_log_filter_stats_computes_reject_rate(tmp_path):
    log_path = tmp_path / "filters.jsonl"
    with TrainingLogger(log_path) as log:
        log.log_filter_stats(
            iter=1,
            n_pre_filter=1000,
            n_post_filter=820,
            reject_counts={"eta_gt_strict": 100, "off_target_hit": 80},
        )
    rows = read_jsonl(log_path)
    assert rows[0]["n_pre_filter"] == 1000
    assert rows[0]["n_post_filter"] == 820
    assert abs(rows[0]["reject_rate"] - 0.18) < 1e-9


def test_handles_numpy_scalars(tmp_path):
    """Numpy scalars must serialise; we shouldn't crash on them."""
    log_path = tmp_path / "np.jsonl"
    with TrainingLogger(log_path) as log:
        log.log_iter(
            iter=1, policy="A",
            loss=float(np.float32(0.5)),
            grad_norm=float(np.float32(0.25)),
            n_records=int(np.int32(4096)),
        )
    rows = read_jsonl(log_path)
    assert rows[0]["loss"] == 0.5
    assert rows[0]["n_records"] == 4096


def test_append_mode_preserves_existing(tmp_path):
    log_path = tmp_path / "run.jsonl"
    with TrainingLogger(log_path, append=False) as log:
        log.log_iter(iter=1, policy="A", loss=1.0)
    with TrainingLogger(log_path, append=True) as log:
        log.log_iter(iter=2, policy="A", loss=0.9)
    rows = read_jsonl(log_path)
    assert [r["iter"] for r in rows] == [1, 2]


def test_close_is_idempotent(tmp_path):
    log_path = tmp_path / "x.jsonl"
    log = TrainingLogger(log_path)
    log.log_msg("hi")
    log.close()
    log.close()  # second close should not raise


def test_lines_are_valid_json(tmp_path):
    """Each emitted line must be standalone-parseable as JSON."""
    log_path = tmp_path / "x.jsonl"
    with TrainingLogger(log_path) as log:
        log.log_iter(iter=1, policy="A", loss=1.0)
        log.log_msg("hello world", extra_field=[1, 2, 3])
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                json.loads(line)  # raises on malformed JSON
