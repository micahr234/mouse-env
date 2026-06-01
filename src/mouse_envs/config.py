"""Environment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EnvConfig:
    """Configuration for building a vector environment via :func:`mouse_envs.make_vector_env`."""

    group_id: str
    seed: int
    num_envs: int
    max_episode_steps: int | None
    kwargs: dict | None = None
    render: bool = False
    non_stationary_params: dict | None = None
    q_star_source: dict[str, Any] | None = None
    atari_preprocessing: bool | None = None
    atari_preprocessing_kwargs: dict | None = None
    observation_indices: list[int] | None = None
    reward_scale: float = 1.0
    reward_shift: float = 0.0
    group_ids: list[str] | None = None
    reset_reward: float = 0.0
