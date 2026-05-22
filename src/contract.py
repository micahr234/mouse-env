"""Rollout contract types (v1 core) for mouse-envs ↔ mouse-core."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class RewardDict(TypedDict):
    """Per-step reward payload."""

    step: float
    episodic: float


class RolloutStepCore(TypedDict):
    """Required logical fields for one env at one step (single-env view)."""

    env_name: str
    episode_index: int
    step_index: int
    action: dict[str, Any]
    observation: dict[str, Any]
    done: int
    reward: RewardDict


# Canonical keys for action / observation dicts (v1 discrete-first).
ACTION_KEY_DISCRETE = "discrete"
ACTION_KEY_CONTINUOUS = "continuous"

OBS_KEY_DISCRETE = "discrete"
OBS_KEY_CONTINUOUS = "continuous"
OBS_KEY_IMAGE = "image"

REWARD_KEY_STEP = "step"
REWARD_KEY_EPISODIC = "episodic"

DONE_RUNNING = 0
DONE_TERMINATED = 1
DONE_TRUNCATED = 2

# Legacy info keys still emitted by the wrapper stack (deprecation path).
LEGACY_INFO_KEYS = frozenset(
    {
        "env_name",
        "env_idx",
        "episode_step",
        "global_step",
        "xformed_reward",
        "done",
        "episode_length",
        "episode_cum_reward",
        "metadata_q_star",
    }
)

CORE_INFO_KEYS_V1 = frozenset(
    {
        "env_name",
        "episode_index",
        "step_index",
        "action",
        "observation",
        "done",
        "reward",
    }
)


class RolloutExtensions(TypedDict, total=False):
    """Optional per-step extensions (batched in vector envs)."""

    metadata_q_star: Any
    episode_length: Any
    episode_cum_reward: Any
    ns_params: dict[str, Any]
