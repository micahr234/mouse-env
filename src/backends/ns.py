"""NS-Gym vector env builder."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from mouse.envs.backends.base import VectorEnvBuildParams, VectorEnvBuilder
from mouse.envs.ns_gym import make_ns_env
from mouse.envs.stack import ObservationSliceWrapper


class NSVectorEnvBuilder(VectorEnvBuilder):
    """Build non-stationary vector envs; each parallel env owns independent NS dynamics."""

    def __init__(
        self,
        params: VectorEnvBuildParams,
        *,
        non_stationary_params: dict[str, Any],
    ):
        super().__init__(params)
        self._non_stationary_params = non_stationary_params

    def make_single_env(self, index: int) -> gym.Env:
        del index
        p = self.params
        env = make_ns_env(
            env_id=p.env_id,
            non_stationary_params=self._non_stationary_params,
            max_steps_per_episode=p.max_steps_per_episode,
            env_kwargs=p.env_kwargs,
            render=p.render,
        )
        if p.observation_indices is not None:
            env = ObservationSliceWrapper(env=env, indices=p.observation_indices)
        return env


def NSVectorEnv(
    env_id: str,
    non_stationary_params: dict[str, Any],
    seed: int,
    max_steps_per_episode: int,
    num_envs: int = 1,
    env_kwargs: dict[str, Any] | None = None,
    render: bool = False,
    env_name: str | None = None,
    observation_indices: list[int] | None = None,
    reward_scale: float = 1.0,
    reward_shift: float = 0.0,
    q_star_source: dict[str, Any] | None = None,
) -> gym.vector.VectorEnv:
    """Create a non-stationary vector env. See :class:`NSVectorEnvBuilder`."""
    params = VectorEnvBuildParams(
        env_id=env_id,
        seed=seed,
        num_envs=num_envs,
        max_steps_per_episode=max_steps_per_episode,
        env_kwargs=env_kwargs,
        render=render,
        env_name=env_name,
        observation_indices=observation_indices,
        reward_scale=reward_scale,
        reward_shift=reward_shift,
        q_star_source=q_star_source,
        obs_key="observation",
    )
    return NSVectorEnvBuilder(params, non_stationary_params=non_stationary_params).build()
