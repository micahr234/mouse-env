"""Environment configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

OBSERVATION_KINDS = ("continuous", "discrete", "image")


@dataclass
class EnvConfig:
    """Configuration for building a vector environment via :func:`mouse_envs.make_vector_env`."""

    group_id: str
    seed: int
    num_envs: int
    max_episode_steps: int | None
    kwargs: dict | None = None
    render: bool = False
    q_star_source: dict[str, Any] | None = None
    env_fn: Callable[[], Any] | None = None
    observation_kind: str | None = None
    observation_indices: list[int] | None = None
    reward_scale: float = 1.0
    reward_shift: float = 0.0
    group_ids: list[str] | None = None
    reset_reward: float = 0.0
