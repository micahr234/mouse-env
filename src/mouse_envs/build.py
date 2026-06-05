"""Build vector environments from :class:`EnvConfig`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym

from mouse_envs.config import EnvConfig
from mouse_envs.experts.action_star import apply_q_star_source_env_kwargs
from mouse_envs.env_ids import PROCEDURAL_FROZENLAKE_ENV_ID, SYNTHETIC_ENV_ID
from mouse_envs.format import MouseVectorEnv
from mouse_envs.wrappers import (
    ConstructionSeedWrapper,
    ObservationSliceWrapper,
    build_vector_env_stack,
)


def _resolve_group_ids(group_id: str, num_envs: int, group_ids: list[str] | None) -> list[str]:
    if group_ids is not None:
        if len(group_ids) != num_envs:
            raise ValueError(
                f"group_ids has {len(group_ids)} entries but num_envs={num_envs}. "
                f"Provide exactly one id per vector index."
            )
        seen: dict[str, list[int]] = {}
        for i, gid in enumerate(group_ids):
            seen.setdefault(gid, []).append(i)
        duplicates = {k: v for k, v in seen.items() if len(v) > 1}
        if duplicates:
            raise ValueError(
                f"group_id must be unique per vector index — found duplicates: {duplicates}. "
                f"Use a unique suffix per index (e.g. '{group_id}#0', '{group_id}#1')."
            )
        return list(group_ids)

    if not group_id:
        raise ValueError(
            "group_id is required on EnvConfig but was not set. "
            "Provide a non-empty group_id (e.g. 'CartPole-v1')."
        )

    if num_envs == 1:
        return [group_id]

    return [f"{group_id}#{i}" for i in range(num_envs)]


def make_vector_env(config: EnvConfig) -> MouseVectorEnv:
    """Create a vector env from :class:`EnvConfig`."""
    if config.max_episode_steps is None:
        raise ValueError(
            "max_episode_steps is required (used to normalise xformed_reward). "
            "Set max_episode_steps in the env config."
        )
    if config.num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {config.num_envs}.")

    resolved_group_id = config.group_id
    resolved_q_star_source = (
        dict(config.q_star_source) if config.q_star_source is not None else None
    )
    resolved_group_ids = _resolve_group_ids(resolved_group_id, config.num_envs, config.group_ids)

    gym_env = _build_plain_vector_env(
        config=config,
        resolved_group_id=resolved_group_id,
        resolved_q_star_source=resolved_q_star_source,
        resolved_group_ids=resolved_group_ids,
    )

    return MouseVectorEnv(
        gym_env,
        resolved_group_ids,
        reset_reward=config.reset_reward,
    )


def _prepare_plain_env_kwargs(config: EnvConfig) -> dict[str, Any]:
    env_kwargs = dict(config.kwargs or {})
    env_kwargs = apply_q_star_source_env_kwargs(
        env_id=config.group_id,
        env_kwargs=env_kwargs,
        q_star_source=config.q_star_source,
    )
    if config.group_id == PROCEDURAL_FROZENLAKE_ENV_ID:
        from mouse_envs.worlds.procedural_frozenlake import ensure_procedural_frozenlake_registered

        ensure_procedural_frozenlake_registered()
        random_map_wrapper_raw = env_kwargs.pop("random_map_wrapper", None)
        if isinstance(random_map_wrapper_raw, dict):
            env_kwargs.update(dict(random_map_wrapper_raw))
        elif random_map_wrapper_raw is not None:
            raise ValueError("env_kwargs.random_map_wrapper must be a dict when provided.")
    elif config.group_id == SYNTHETIC_ENV_ID:
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
    seeded_at_construction = config.group_id in (SYNTHETIC_ENV_ID, PROCEDURAL_FROZENLAKE_ENV_ID)

    def env_fn(s: int) -> gym.Env:
        if config.env_fn is not None:
            return config.env_fn()
        kw = dict(env_kwargs)
        if seeded_at_construction:
            kw["seed"] = s
        return gym.make(
            config.group_id,
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
    resolved_group_id: str,
    resolved_q_star_source: dict[str, Any] | None,
    resolved_group_ids: list[str],
) -> gym.vector.VectorEnv:
    env_kwargs = {} if config.env_fn is not None else _prepare_plain_env_kwargs(config)
    max_episode_steps = config.max_episode_steps
    assert max_episode_steps is not None

    if config.env_fn is None and config.group_id in (SYNTHETIC_ENV_ID, PROCEDURAL_FROZENLAKE_ENV_ID):
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

    return build_vector_env_stack(
        env_fns=env_fns,
        group_id=resolved_group_id,
        seed=config.seed,
        max_steps_per_episode=max_episode_steps,
        observation_kind=config.observation_kind,
        reward_scale=config.reward_scale,
        reward_shift=config.reward_shift,
        q_star_source=resolved_q_star_source,
        group_ids=resolved_group_ids,
    )
