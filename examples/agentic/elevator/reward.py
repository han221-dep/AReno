"""Reward function for the elevator-dispatch tool-call example.

The reward function replays the model's action string with :func:`game.play`,
returns a scalar shaped by delivered passengers, mean wait, and invalid-action
rate, and writes the full metric dict onto ``source_record["metrics"]`` so
downstream metrics and tests can assert the emitted fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

DELIVERED_VALUE = 1.0
WAIT_WEIGHT = 0.01
INVALID_PENALTY = 0.1
COVERAGE_CAP = 0.3


def reward_fn(record: Any) -> float:
    """Score one completion by replaying its ``dispatch`` action string."""

    source = record.source_record
    building = game.normalize_building(source["building"])
    actions = _tool_actions(record)
    metrics = _play(building, actions, source)
    source["metrics"] = metrics
    return _scalar_reward(metrics, actions, building)


def _play(building: dict[str, Any], actions: str, source: dict) -> dict:
    max_steps = int(source.get("max_steps", game.DEFAULT_MAX_STEPS))
    return game.play(building, actions, max_steps=max_steps)


def _scalar_reward(metrics: dict, actions: str, building: dict) -> float:
    """Float reward shaped like delivered passengers minus wait and invalids.

    Empty or unparseable action strings map to ``-1.0`` -- mirroring the other
    agentic examples' failing-score convention. When nobody is delivered the
    score is loosened off the old flat ``-1.0`` floor with a small bounded
    ``coverage`` shaping term: a no-op policy still cannot win, but an episode
    that ran closer to the full arrival horizon is no longer tied with one that
    ended blind before anyone showed up. Group advantage is only non-zero when
    within-group scores differ, and a flat ``-1.0`` for every miss leaves
    GRPO/GSPO with no gradient to start from.
    """

    if not actions:
        return -1.0
    delivered = float(metrics["delivered_passengers"])
    wait_term = WAIT_WEIGHT * float(metrics["mean_wait"])
    invalid_term = INVALID_PENALTY * float(metrics["invalid_rate"])
    reward = DELIVERED_VALUE * delivered - wait_term - invalid_term
    if delivered > 0:
        return reward
    coverage = _coverage_signal(metrics, building)
    shaped = coverage - invalid_term
    return min(-1.0 + shaped, -0.1)


def _coverage_signal(metrics: dict, building: dict) -> float:
    """Cold-start shaping for episodes that delivered no one.

    A *continuous* credit for how far the episode's clock ran toward the full
    arrival horizon, scaled to ``[0, COVERAGE_CAP]``. The horizon is the *last*
    scheduled arrival (not the first): the goal is to keep dispatching until
    every passenger has shown up, and a sequence that ends before the first
    arrival can never serve anyone. Unlike a binary threshold, the linear ramp
    makes two short-but-different sequences compare differently, so the
    within-group advantage is non-zero even when nobody is delivered --
    GRPO/GSPO then has a gradient to start from.
    """

    arrivals = building.get("arrivals") or []
    last_arrival_tick = (
        max(int(e.get("tick", 0)) for e in arrivals) if arrivals else 0
    )
    horizon = max(1, last_arrival_tick + 1)
    n_attempts = int(metrics.get("n_attempts", 0))
    return min(COVERAGE_CAP, COVERAGE_CAP * n_attempts / horizon)


def _tool_actions(record: Any) -> str:
    """Extract the final ``dispatch.actions`` string from a reward record."""

    for call in record.tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "dispatch":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return ""
        if isinstance(arguments, dict):
            return _clean_actions(arguments.get("actions"))
    # Fall back to a <dispatch> tag if the policy answered in text instead of a tool call.
    completion = getattr(record, "completion", "") or ""
    return game.parse_action_sequence(completion) if completion else ""


def _clean_actions(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).upper()
    return "".join(ch for ch in text if ch in game.ACTIONS)
