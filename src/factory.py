"""Unified vector environment factory."""

from __future__ import annotations

from typing import Any, overload

import gymnasium as gym

from mouse.envs.config import EnvConfig, resolve_q_star_source_for_env
from mouse.envs.routing import is_ns_gym_env, normalize_env_id


@overload
def make_vector_env(config: EnvConfig, /) -> gym.vector.VectorEnv: ...


@overload
def make_vector_env(
    env_id: str,
    seed: int,
    max_steps_per_episode: int | None,
    /,
    *,
    num_envs: int = 1,
    env_kwargs: dict[str, Any] | None = None,
    render: bool = False,
    env_name: str | None = None,
    non_stationary_params: dict[str, Any] | None = None,
    env_type: str | None = None,
    atari_preprocessing: bool = False,
    atari_preprocessing_kwargs: dict[str, Any] | None = None,
    observation_indices: list[int] | None = None,
    reward_scale: float = 1.0,
    reward_shift: float = 0.0,
    q_star_source: dict[str, Any] | None = None,
) -> gym.vector.VectorEnv: ...


def make_vector_env(
    config_or_env_id: EnvConfig | str,
    seed: int | None = None,
    max_steps_per_episode: int | None = None,
    /,
    *,
    num_envs: int = 1,
    env_kwargs: dict[str, Any] | None = None,
    render: bool = False,
    env_name: str | None = None,
    non_stationary_params: dict[str, Any] | None = None,
    env_type: str | None = None,
    atari_preprocessing: bool = False,
    atari_preprocessing_kwargs: dict[str, Any] | None = None,
    observation_indices: list[int] | None = None,
    reward_scale: float = 1.0,
    reward_shift: float = 0.0,
    q_star_source: dict[str, Any] | None = None,
) -> gym.vector.VectorEnv:
    """Create a vector env from :class:`EnvConfig` or explicit keyword arguments."""
    if isinstance(config_or_env_id, EnvConfig):
        return _make_vector_env_from_config(config_or_env_id)
    if seed is None or max_steps_per_episode is None:
        raise TypeError(
            "make_vector_env(env_id, seed, max_steps_per_episode, ...) requires "
            "env_id, seed, and max_steps_per_episode when not passing EnvConfig."
        )
    return _make_vector_env_impl(
        env_id=config_or_env_id,
        seed=seed,
        max_steps_per_episode=max_steps_per_episode,
        num_envs=num_envs,
        env_kwargs=env_kwargs,
        render=render,
        env_name=env_name,
        non_stationary_params=non_stationary_params,
        env_type=env_type,
        atari_preprocessing=atari_preprocessing,
        atari_preprocessing_kwargs=atari_preprocessing_kwargs,
        observation_indices=observation_indices,
        reward_scale=reward_scale,
        reward_shift=reward_shift,
        q_star_source=q_star_source,
    )


def _make_vector_env_from_config(config: EnvConfig) -> gym.vector.VectorEnv:
    b = config.build_config
    return _make_vector_env_impl(
        env_id=b.env_id,
        seed=b.seed,
        max_steps_per_episode=b.max_episode_steps,
        num_envs=b.num_envs,
        env_kwargs=b.kwargs,
        render=b.render,
        env_name=b.env_name,
        non_stationary_params=b.non_stationary_params,
        env_type=b.env_type,
        atari_preprocessing=bool(b.atari_preprocessing) if b.atari_preprocessing is not None else False,
        atari_preprocessing_kwargs=b.atari_preprocessing_kwargs,
        observation_indices=b.observation_indices,
        reward_scale=b.reward_scale,
        reward_shift=b.reward_shift,
        q_star_source=b.q_star_source,
    )


def _make_vector_env_impl(
    *,
    env_id: str,
    seed: int,
    max_steps_per_episode: int | None,
    num_envs: int,
    env_kwargs: dict[str, Any] | None,
    render: bool,
    env_name: str | None,
    non_stationary_params: dict[str, Any] | None,
    env_type: str | None,
    atari_preprocessing: bool,
    atari_preprocessing_kwargs: dict[str, Any] | None,
    observation_indices: list[int] | None,
    reward_scale: float,
    reward_shift: float,
    q_star_source: dict[str, Any] | None,
) -> gym.vector.VectorEnv:
    if max_steps_per_episode is None:
        raise ValueError(
            "max_steps_per_episode is required (used to normalise xformed_reward). "
            "Set max_episode_steps in the env config."
        )
    resolved_env_id = normalize_env_id(env_id)
    resolved_q_star_source = resolve_q_star_source_for_env(env_id, q_star_source)
    if is_ns_gym_env(env_id=env_id, non_stationary_params=non_stationary_params, env_type=env_type):
        ns_params = non_stationary_params
        if ns_params is None and env_id.startswith("NS-"):
            ns_params = {}
        if ns_params is None:
            raise ValueError("non_stationary_params is required for NS-Gym env.")
        from mouse.envs.backends.ns import NSVectorEnv

        return NSVectorEnv(
            env_id=resolved_env_id,
            non_stationary_params=ns_params,
            seed=seed,
            num_envs=num_envs,
            max_steps_per_episode=max_steps_per_episode,
            env_kwargs=env_kwargs,
            render=render,
            env_name=env_name,
            observation_indices=observation_indices,
            reward_scale=reward_scale,
            reward_shift=reward_shift,
            q_star_source=resolved_q_star_source,
        )
    from mouse.envs.backends.plain import PlainVectorEnv

    return PlainVectorEnv(
        env_id=resolved_env_id,
        seed=seed,
        num_envs=num_envs,
        max_steps_per_episode=max_steps_per_episode,
        env_kwargs=env_kwargs,
        render=render,
        env_name=env_name,
        atari_preprocessing=atari_preprocessing,
        atari_preprocessing_kwargs=atari_preprocessing_kwargs,
        observation_indices=observation_indices,
        reward_scale=reward_scale,
        reward_shift=reward_shift,
        q_star_source=resolved_q_star_source,
    )
