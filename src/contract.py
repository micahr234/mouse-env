"""Rollout contract types (v1 core) for mouse-env ↔ mouse-core."""

from __future__ import annotations

from typing import Any, TypedDict

CONTRACT_VERSION = 1


class RewardDict(TypedDict):
    """Per-step reward payload."""

    step: float
    episodic: float


class RolloutStepCore(TypedDict):
    """Required logical fields for one env at one step (single-env view)."""

    env_id: str
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

# v1 core keys expected in vector info (batched).
CORE_INFO_KEYS_V1 = frozenset(
    {
        "env_id",
        "episode_index",
        "step_index",
        "action",
        "observation",
        "done",
        "reward",
    }
)

# Legacy info keys still emitted by the wrapper stack (migration).
LEGACY_INFO_KEYS = frozenset(
    {
        "env_name",  # same string as env_id today; prefer env_id
        "env_idx",
        "episode_step",  # 1-based; replaced by step_index (0-based)
        "global_step",
        "xformed_reward",  # moved to reward["episodic"]
        "done",
        "episode_length",
        "episode_cum_reward",
        "metadata_q_star",
    }
)

EXTENSIONS_INFO_KEY = "extensions"


class RolloutExtensions(TypedDict, total=False):
    """Optional per-step extensions (batched in vector envs)."""

    metadata_q_star: Any
    episode_length: Any
    episode_cum_reward: Any
    ns_params: dict[str, Any]
