"""Build environments from :class:`EnvConfig`."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from mouse_envs.config import EnvConfig
from mouse_envs.experts.action_star import apply_q_star_source_env_kwargs
from mouse_envs.env_ids import PROCEDURAL_FROZENLAKE_ENV_ID, SYNTHETIC_ENV_ID
from mouse_envs.format import MouseEnv, _EnvInstance
from mouse_envs.wrappers import (
    ObservationSliceWrapper,
    SeedStreamWrapper,
    build_single_env,
)


def _require_env_id(env_id: str) -> None:
    if not env_id:
        raise ValueError(
            "id is required on EnvConfig but was not set. "
            "Provide a non-empty id (e.g. 'CartPole-v1')."
        )


def make_env(configs: EnvConfig | list[EnvConfig]) -> MouseEnv:
    """Create a :class:`MouseEnv` from one or more :class:`EnvConfig` objects.

    Pass a single :class:`EnvConfig` for the common single-env case, or a list to
    combine multiple heterogeneous environments into one :class:`MouseEnv`. Each
    ``EnvConfig.num_envs=N`` expands to N independent single-env instances. All
    env instances are stepped sequentially; outputs form a flat list indexed by
    env.

    Usage — single env::

        env = make_env(EnvConfig(id="CartPole-v1", seed=0, num_envs=4, episodes_per_task=5))
        for _ in range(1000):
            inputs = env.sample_random_inputs()
            outputs = env.step(inputs)

    Usage — multiple envs::

        env = make_env([
            EnvConfig(id="CartPole-v1", seed=0, num_envs=2, episodes_per_task=5),
            EnvConfig(id="MountainCar-v0", seed=1, num_envs=3, episodes_per_task=5),
        ])
        for _ in range(1000):
            outputs = env.step(env.sample_random_inputs())
            cartpole_outs = outputs[:2]
            mountaincar_outs = outputs[2:]
    """
    if isinstance(configs, EnvConfig):
        configs = [configs]
    env_instances: list[_EnvInstance] = []
    for cfg in configs:
        env_instances.extend(_make_env_instances(cfg))
    return MouseEnv(env_instances)


def _make_env_instances(config: EnvConfig) -> list[_EnvInstance]:
    """Expand one :class:`EnvConfig` into a flat list of single-env instances."""
    if config.num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {config.num_envs}.")

    _require_env_id(config.id)
    name_base = config.id if config.name is None else config.name
    resolved_q_star_source = (
        dict(config.q_star_source) if config.q_star_source is not None else None
    )

    env_kwargs = {} if config.env_fn is not None else _prepare_plain_env_kwargs(config)

    env_instances: list[_EnvInstance] = []
    for i in range(config.num_envs):
        name = f"{name_base}_{i}"

        env = build_single_env(
            env_fn=lambda idx=i: _make_plain_single_env(config, idx, env_kwargs=env_kwargs),
            env_id=config.id,
            observation_kind=config.observation_kind,
            q_star_source=resolved_q_star_source,
        )
        env_instances.append(
            _EnvInstance(
                env=env,
                name=name,
                reset_reward=config.reset_reward,
                episode_reset_options=config.episode_reset_options,
                task_reset_options=config.task_reset_options,
                reward_scale=config.reward_scale,
                reward_shift=config.reward_shift,
                episodes_per_task=config.episodes_per_task,
            )
        )
    return env_instances


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
    reset_seed = (config.reset_seed if config.reset_seed is not None else config.seed) + index

    def env_fn() -> gym.Env:
        if config.env_fn is not None:
            return config.env_fn()
        kw = dict(env_kwargs)
        return gym.make(config.id, **kw)

    env = SeedStreamWrapper(
        env_fn,
        reset_seed=reset_seed,
    )
    if config.observation_indices is not None:
        env = ObservationSliceWrapper(env=env, indices=config.observation_indices)
    return env
