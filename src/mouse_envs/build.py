"""Build environments from :class:`EnvConfig`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym

from mouse_envs.config import EnvConfig
from mouse_envs.experts.action_star import apply_q_star_source_env_kwargs
from mouse_envs.env_ids import PROCEDURAL_FROZENLAKE_ENV_ID, SYNTHETIC_ENV_ID
from mouse_envs.format import MouseEnv, _InnerEnv
from mouse_envs.wrappers import (
    ConstructionSeedWrapper,
    ObservationSliceWrapper,
    build_env_stack,
)


def _require_env_id(env_id: str) -> None:
    if not env_id:
        raise ValueError(
            "id is required on EnvConfig but was not set. "
            "Provide a non-empty id (e.g. 'CartPole-v1')."
        )


def _resolve_names(name_base: str, num_envs: int) -> list[str]:
    if not name_base:
        raise ValueError(
            "name must be non-empty when provided on EnvConfig. "
            "Omit it to use id as the environment name base."
        )
    return [f"{name_base}#{i}" for i in range(num_envs)]


def make_env(configs: EnvConfig | list[EnvConfig]) -> MouseEnv:
    """Create a :class:`MouseEnv` from one or more :class:`EnvConfig` objects.

    Pass a single :class:`EnvConfig` for the common single-env case, or a list to
    combine multiple heterogeneous environments into one :class:`MouseEnv`. Envs are
    stepped sequentially; outputs are never concatenated across configs.

    Usage — single env::

        env = make_env(EnvConfig(id="CartPole-v1", seed=0, num_envs=4, max_episode_steps=500))
        for _ in range(1000):
            [(outputs, metrics)] = env.step(env.sample_random_inputs())

    Usage — multiple envs::

        env = make_env([
            EnvConfig(id="CartPole-v1", seed=0, num_envs=4, max_episode_steps=500),
            EnvConfig(id="MountainCar-v0", seed=1, num_envs=2, max_episode_steps=200),
        ])
        for _ in range(1000):
            results = env.step(env.sample_random_inputs())
            for outputs, metrics in results:
                ...
    """
    if isinstance(configs, EnvConfig):
        configs = [configs]
    inner_envs = [_make_inner_env(cfg) for cfg in configs]
    return MouseEnv(inner_envs)


def _make_inner_env(config: EnvConfig) -> _InnerEnv:
    if config.max_episode_steps is None:
        raise ValueError(
            "max_episode_steps is required (used to normalise xformed_reward). "
            "Set max_episode_steps in the env config."
        )
    if config.num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {config.num_envs}.")

    _require_env_id(config.id)
    name_base = config.id if config.name is None else config.name
    resolved_names = _resolve_names(name_base, config.num_envs)
    resolved_q_star_source = (
        dict(config.q_star_source) if config.q_star_source is not None else None
    )

    gym_env = _build_plain_vector_env(
        config=config,
        resolved_name=resolved_names[0],
        resolved_q_star_source=resolved_q_star_source,
    )

    return _InnerEnv(gym_env, resolved_names, reset_reward=config.reset_reward)


def _prepare_plain_env_kwargs(config: EnvConfig) -> dict[str, Any]:
    env_kwargs = dict(config.kwargs or {})
    env_kwargs = apply_q_star_source_env_kwargs(
        env_id=config.id,
        env_kwargs=env_kwargs,
        q_star_source=config.q_star_source,
    )
    if config.id == PROCEDURAL_FROZENLAKE_ENV_ID:
        from mouse_envs.worlds.procedural_frozenlake import ensure_procedural_frozenlake_registered

        ensure_procedural_frozenlake_registered()
        random_map_wrapper_raw = env_kwargs.pop("random_map_wrapper", None)
        if isinstance(random_map_wrapper_raw, dict):
            env_kwargs.update(dict(random_map_wrapper_raw))
        elif random_map_wrapper_raw is not None:
            raise ValueError("env_kwargs.random_map_wrapper must be a dict when provided.")
    elif config.id == SYNTHETIC_ENV_ID:
        from mouse_envs.worlds.synthetic import ensure_synthetic_env_registered

        ensure_synthetic_env_registered()
    if config.render and "render_mode" not in env_kwargs:
        env_kwargs["render_mode"] = "human"
    return env_kwargs


def _make_plain_single_env(
    config: EnvConfig,
    index: int,
    *,
    env_kwargs: dict[str, Any],
) -> gym.Env:
    mdp_seed = config.seed + index
    seeded_at_construction = config.id in (SYNTHETIC_ENV_ID, PROCEDURAL_FROZENLAKE_ENV_ID)

    def env_fn(s: int) -> gym.Env:
        if config.env_fn is not None:
            return config.env_fn()
        kw = dict(env_kwargs)
        if seeded_at_construction:
            kw["seed"] = s
        return gym.make(
            config.id,
            max_episode_steps=config.max_episode_steps,
            **kw,
        )

    env = ConstructionSeedWrapper(env_fn, seed=mdp_seed)
    if config.observation_indices is not None:
        env = ObservationSliceWrapper(env=env, indices=config.observation_indices)
    return env


def _build_plain_vector_env(
    *,
    config: EnvConfig,
    resolved_name: str,
    resolved_q_star_source: dict[str, Any] | None,
) -> gym.vector.VectorEnv:
    env_kwargs = {} if config.env_fn is not None else _prepare_plain_env_kwargs(config)
    max_episode_steps = config.max_episode_steps
    assert max_episode_steps is not None

    if config.env_fn is None and config.id in (SYNTHETIC_ENV_ID, PROCEDURAL_FROZENLAKE_ENV_ID):
        clean_kwargs = dict(env_kwargs)
        clean_kwargs.pop("seed", None)
    else:
        clean_kwargs = env_kwargs

    env_fns: list[Callable[[], gym.Env]] = [
        lambda i=i: _make_plain_single_env(
            config,
            i,
            env_kwargs=clean_kwargs,
        )
        for i in range(config.num_envs)
    ]

    return build_env_stack(
        env_fns=env_fns,
        env_id=config.id,
        name=resolved_name,
        seed=config.seed,
        max_steps_per_episode=max_episode_steps,
        observation_kind=config.observation_kind,
        reward_scale=config.reward_scale,
        reward_shift=config.reward_shift,
        q_star_source=resolved_q_star_source,
    )
