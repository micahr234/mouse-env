"""Shared vector-env builder base."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from mouse.envs.stack import build_vector_env_stack


class ConstructionSeedWrapper(gym.Wrapper):
    """Control construction-time and per-episode reset seeds for custom MDPs."""

    def __init__(self, env_fn: Callable[[int], gym.Env], seed: int):
        super().__init__(env_fn(seed))
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        return self.env.reset(seed=int(self._rng.integers(0, 2**31)), options=options)


@dataclass
class VectorEnvBuildParams:
    """Arguments shared by all vector env backends."""

    env_id: str
    seed: int
    num_envs: int
    max_steps_per_episode: int
    env_kwargs: dict[str, Any] | None = None
    render: bool = False
    env_name: str | None = None
    observation_indices: list[int] | None = None
    reward_scale: float = 1.0
    reward_shift: float = 0.0
    q_star_source: dict[str, Any] | None = None
    obs_key: str = "observation"


class VectorEnvBuilder(ABC):
    """Build a wrapped vector env from per-env factory callables."""

    def __init__(self, params: VectorEnvBuildParams):
        if params.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {params.num_envs}.")
        self.params = params

    @abstractmethod
    def make_single_env(self, index: int) -> gym.Env:
        """Create one underlying env for parallel index ``index``."""

    def env_fns(self) -> list[Callable[[], gym.Env]]:
        return [lambda i=i: self.make_single_env(i) for i in range(self.params.num_envs)]

    def build(self) -> gym.vector.VectorEnv:
        p = self.params
        return build_vector_env_stack(
            env_fns=self.env_fns(),
            env_id=p.env_id,
            env_name=p.env_name if p.env_name is not None else p.env_id,
            seed=p.seed,
            max_steps_per_episode=p.max_steps_per_episode,
            obs_key=p.obs_key,
            reward_scale=p.reward_scale,
            reward_shift=p.reward_shift,
            q_star_source=p.q_star_source,
        )
