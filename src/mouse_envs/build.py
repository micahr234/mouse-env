"""Build environments from :class:`EnvConfig`."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from mouse_envs.config import EnvConfig
from mouse_envs.experts.action_star import apply_q_star_source_env_kwargs
from mouse_envs.env_ids import PROCEDURAL_FROZENLAKE_ENV_ID, SYNTHETIC_ENV_ID
from mouse_envs.format import MouseEnv, _Slot
from mouse_envs.wrappers import (
    ConstructionSeedWrapper,
    ObservationSliceWrapper,
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
    ``EnvConfig.num_envs=N`` expands to N independent single-env slots. All slots
    are stepped sequentially; outputs form a flat list indexed by slot.

    Usage — single env::

        env = make_env(EnvConfig(id="CartPole-v1", seed=0, num_envs=4, episodes_per_task=5))
        for _ in range(1000):
            inputs = env.sample_random_inputs()
            outputs, metrics = env.step(inputs)

    Usage — multiple envs::

        env = make_env([
            EnvConfig(id="CartPole-v1", seed=0, num_envs=2, episodes_per_task=5),
            EnvConfig(id="MountainCar-v0", seed=1, num_envs=3, episodes_per_task=5),
        ])
        for _ in range(1000):
            outputs, metrics = env.step(env.sample_random_inputs())
            cartpole_outs = outputs[:2]
            mountaincar_outs = outputs[2:]
    """
    if isinstance(configs, EnvConfig):
        configs = [configs]
    slots: list[_Slot] = []
    for cfg in configs:
        slots.extend(_make_slots(cfg))
    return MouseEnv(slots)


def _make_slots(config: EnvConfig) -> list[_Slot]:
    """Expand one :class:`EnvConfig` into a flat list of :class:`_Slot` instances."""
    if config.num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {config.num_envs}.")

    _require_env_id(config.id)
    name_base = config.id if config.name is None else config.name
    resolved_q_star_source = (
        dict(config.q_star_source) if config.q_star_source is not None else None
    )

    env_kwargs = {} if config.env_fn is not None else _prepare_plain_env_kwargs(config)

    if config.env_fn is None and config.id in (SYNTHETIC_ENV_ID, PROCEDURAL_FROZENLAKE_ENV_ID):
        clean_kwargs = dict(env_kwargs)
        clean_kwargs.pop("seed", None)
    else:
        clean_kwargs = env_kwargs

    slots: list[_Slot] = []
    for i in range(config.num_envs):
        name = f"{name_base}_{i}"
        slot_seed = config.seed + i

        env = build_single_env(
            env_fn=lambda idx=i: _make_plain_single_env(config, idx, env_kwargs=clean_kwargs),
            env_id=config.id,
            name=name,
            seed=slot_seed,
            observation_kind=config.observation_kind,
            q_star_source=resolved_q_star_source,
        )
        slots.append(
            _Slot(
                env=env,
                name=name,
                reset_reward=config.reset_reward,
                reward_scale=config.reward_scale,
                reward_shift=config.reward_shift,
                episodes_per_task=config.episodes_per_task,
            )
        )
    return slots


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
        return gym.make(config.id, **kw)

    env = ConstructionSeedWrapper(env_fn, seed=mdp_seed)
    if config.observation_indices is not None:
        env = ObservationSliceWrapper(env=env, indices=config.observation_indices)
    return env
