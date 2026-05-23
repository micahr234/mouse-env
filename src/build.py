"""Build vector environments from :class:`EnvConfig`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym

from mouse.envs.config import (
    EnvConfig,
    is_ns_gym_env,
    normalize_group_id,
    resolve_q_star_source_for_env,
)
from mouse.envs.custom.action_star import apply_q_star_source_env_kwargs
from mouse.envs.custom.atari import (
    ensure_ale_registered,
    is_ale_env,
    wrap_atari_preprocessing,
)
from mouse.envs.custom.ns_gym import make_ns_env
from mouse.envs.env_ids import CUSTOM_FROZENLAKE_ENV_ID, SYNTHETIC_ENV_ID
from mouse.envs.format import MouseVectorEnv
from mouse.envs.wrappers import (
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

    resolved_group_id = normalize_group_id(config.group_id)
    resolved_q_star_source = resolve_q_star_source_for_env(config.group_id, config.q_star_source)
    resolved_group_ids = _resolve_group_ids(resolved_group_id, config.num_envs, config.group_ids)

    if is_ns_gym_env(config.group_id, config.non_stationary_params):
        gym_env = _build_ns_vector_env(
            config=config,
            resolved_group_id=resolved_group_id,
            resolved_q_star_source=resolved_q_star_source,
            resolved_group_ids=resolved_group_ids,
        )
    else:
        gym_env = _build_plain_vector_env(
            config=config,
            resolved_group_id=resolved_group_id,
            resolved_q_star_source=resolved_q_star_source,
            resolved_group_ids=resolved_group_ids,
        )

    return MouseVectorEnv(gym_env, resolved_group_ids)


def _prepare_plain_env_kwargs(config: EnvConfig, *, atari_preprocessing: bool) -> dict[str, Any]:
    env_kwargs = dict(config.kwargs or {})
    env_kwargs = apply_q_star_source_env_kwargs(
        env_id=config.group_id,
        env_kwargs=env_kwargs,
        q_star_source=config.q_star_source,
    )
    if config.group_id == CUSTOM_FROZENLAKE_ENV_ID:
        from mouse.envs.custom.frozenlake import ensure_custom_frozenlake_registered

        ensure_custom_frozenlake_registered()
        random_map_wrapper_raw = env_kwargs.pop("random_map_wrapper", None)
        if isinstance(random_map_wrapper_raw, dict):
            env_kwargs.update(dict(random_map_wrapper_raw))
        elif random_map_wrapper_raw is not None:
            raise ValueError("env_kwargs.random_map_wrapper must be a dict when provided.")
    elif config.group_id == SYNTHETIC_ENV_ID:
        from mouse.envs.custom.synthetic import ensure_synthetic_env_registered

        ensure_synthetic_env_registered()
    if config.render and "render_mode" not in env_kwargs:
        env_kwargs["render_mode"] = "human"
    if is_ale_env(config.group_id):
        ensure_ale_registered()
    if is_ale_env(config.group_id) and atari_preprocessing:
        env_kwargs["frameskip"] = 1
    if config.observation_indices is not None and is_ale_env(config.group_id):
        raise ValueError("observation_indices is not supported for ALE (Atari) envs.")
    return env_kwargs


def _mdp_seed_for_index(config: EnvConfig, index: int) -> int:
    if config.group_id in (SYNTHETIC_ENV_ID, CUSTOM_FROZENLAKE_ENV_ID):
        return int(config.seed) + index
    return config.seed + index


def _make_plain_single_env(
    config: EnvConfig,
    index: int,
    *,
    env_kwargs: dict[str, Any],
    atari_preprocessing: bool,
) -> gym.Env:
    mdp_seed = _mdp_seed_for_index(config, index)
    seeded_at_construction = config.group_id in (SYNTHETIC_ENV_ID, CUSTOM_FROZENLAKE_ENV_ID)
    use_preprocessing = is_ale_env(config.group_id) and atari_preprocessing

    def env_fn(s: int) -> gym.Env:
        kw = dict(env_kwargs)
        if seeded_at_construction:
            kw["seed"] = s
        env = gym.make(
            config.group_id,
            max_episode_steps=config.max_episode_steps,
            **kw,
        )
        return wrap_atari_preprocessing(
            env,
            enabled=use_preprocessing,
            preprocessing_kwargs=config.atari_preprocessing_kwargs,
        )

    env = ConstructionSeedWrapper(env_fn, seed=mdp_seed)
    if config.observation_indices is not None:
        env = ObservationSliceWrapper(env=env, indices=config.observation_indices)
    return env


def _make_ns_single_env(config: EnvConfig) -> gym.Env:
    env = make_ns_env(
        env_id=config.group_id,
        non_stationary_params=config.non_stationary_params or {},
        max_steps_per_episode=config.max_episode_steps,
        env_kwargs=config.kwargs,
        render=config.render,
    )
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
    atari_preprocessing = bool(config.atari_preprocessing) if config.atari_preprocessing is not None else False
    env_kwargs = _prepare_plain_env_kwargs(config, atari_preprocessing=atari_preprocessing)

    if config.group_id in (SYNTHETIC_ENV_ID, CUSTOM_FROZENLAKE_ENV_ID):
        clean_kwargs = dict(env_kwargs)
        clean_kwargs.pop("seed", None)
    else:
        clean_kwargs = env_kwargs

    env_fns: list[Callable[[], gym.Env]] = [
        lambda i=i: _make_plain_single_env(
            config,
            i,
            env_kwargs=clean_kwargs,
            atari_preprocessing=atari_preprocessing,
        )
        for i in range(config.num_envs)
    ]

    obs_key = "observation_image" if is_ale_env(config.group_id) else "observation"
    return build_vector_env_stack(
        env_fns=env_fns,
        group_id=resolved_group_id,
        seed=config.seed,
        max_steps_per_episode=config.max_episode_steps,
        obs_key=obs_key,
        reward_scale=config.reward_scale,
        reward_shift=config.reward_shift,
        q_star_source=resolved_q_star_source,
        group_ids=resolved_group_ids,
    )


def _build_ns_vector_env(
    *,
    config: EnvConfig,
    resolved_group_id: str,
    resolved_q_star_source: dict[str, Any] | None,
    resolved_group_ids: list[str],
) -> gym.vector.VectorEnv:
    env_fns: list[Callable[[], gym.Env]] = [
        lambda: _make_ns_single_env(config) for _ in range(config.num_envs)
    ]
    return build_vector_env_stack(
        env_fns=env_fns,
        group_id=resolved_group_id,
        seed=config.seed,
        max_steps_per_episode=config.max_episode_steps,
        obs_key="observation",
        reward_scale=config.reward_scale,
        reward_shift=config.reward_shift,
        q_star_source=resolved_q_star_source,
        group_ids=resolved_group_ids,
    )
