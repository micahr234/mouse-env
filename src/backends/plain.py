"""Plain Gymnasium and custom env vector builder."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from mouse.envs.action_star import apply_q_star_source_env_kwargs
from mouse.envs.backends.atari import (
    ensure_ale_registered,
    is_ale_env,
    wrap_atari_preprocessing,
)
from mouse.envs.backends.base import (
    ConstructionSeedWrapper,
    VectorEnvBuildParams,
    VectorEnvBuilder,
)
from mouse.envs.env_ids import CUSTOM_FROZENLAKE_ENV_ID, SYNTHETIC_ENV_ID
from mouse.envs.stack import ObservationSliceWrapper


class PlainVectorEnvBuilder(VectorEnvBuilder):
    """Build vector envs for standard Gymnasium, Atari, and custom tabular envs."""

    def __init__(
        self,
        params: VectorEnvBuildParams,
        *,
        atari_preprocessing: bool = False,
        atari_preprocessing_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(params)
        self._atari_preprocessing = atari_preprocessing
        self._atari_preprocessing_kwargs = atari_preprocessing_kwargs
        self._env_kwargs = self._prepare_env_kwargs()

    def _prepare_env_kwargs(self) -> dict[str, Any]:
        p = self.params
        env_kwargs = dict(p.env_kwargs or {})
        env_kwargs = apply_q_star_source_env_kwargs(
            env_id=p.env_id,
            env_kwargs=env_kwargs,
            q_star_source=p.q_star_source,
        )
        if p.env_id == CUSTOM_FROZENLAKE_ENV_ID:
            from mouse.envs.frozenlake import ensure_custom_frozenlake_registered

            ensure_custom_frozenlake_registered()
            random_map_wrapper_raw = env_kwargs.pop("random_map_wrapper", None)
            if isinstance(random_map_wrapper_raw, dict):
                env_kwargs.update(dict(random_map_wrapper_raw))
            elif random_map_wrapper_raw is not None:
                raise ValueError(
                    "env_kwargs.random_map_wrapper must be a dict when provided."
                )
        elif p.env_id == SYNTHETIC_ENV_ID:
            from mouse.envs.synthetic import ensure_synthetic_env_registered

            ensure_synthetic_env_registered()
        if p.render and "render_mode" not in env_kwargs:
            env_kwargs["render_mode"] = "human"
        if is_ale_env(p.env_id):
            ensure_ale_registered()
        if is_ale_env(p.env_id) and self._atari_preprocessing:
            env_kwargs["frameskip"] = 1
        if p.observation_indices is not None and is_ale_env(p.env_id):
            raise ValueError("observation_indices is not supported for ALE (Atari) envs.")
        return env_kwargs

    def _mdp_seed_for_index(self, index: int) -> int:
        p = self.params
        if p.env_id in (SYNTHETIC_ENV_ID, CUSTOM_FROZENLAKE_ENV_ID):
            return int(p.seed) + index
        return p.seed + index

    def make_single_env(self, index: int) -> gym.Env:
        p = self.params
        mdp_seed = self._mdp_seed_for_index(index)
        seeded_at_construction = p.env_id in (SYNTHETIC_ENV_ID, CUSTOM_FROZENLAKE_ENV_ID)
        use_preprocessing = is_ale_env(p.env_id) and self._atari_preprocessing

        def env_fn(s: int) -> gym.Env:
            kw = dict(self._env_kwargs)
            if seeded_at_construction:
                kw["seed"] = s
            env = gym.make(p.env_id, max_episode_steps=p.max_steps_per_episode, **kw)
            return wrap_atari_preprocessing(
                env,
                enabled=use_preprocessing,
                preprocessing_kwargs=self._atari_preprocessing_kwargs,
            )

        env = ConstructionSeedWrapper(env_fn, seed=mdp_seed)
        if p.observation_indices is not None:
            env = ObservationSliceWrapper(env=env, indices=p.observation_indices)
        return env


def PlainVectorEnv(
    env_id: str,
    seed: int,
    max_steps_per_episode: int,
    num_envs: int = 1,
    env_kwargs: dict[str, Any] | None = None,
    render: bool = False,
    env_name: str | None = None,
    atari_preprocessing: bool = False,
    atari_preprocessing_kwargs: dict[str, Any] | None = None,
    observation_indices: list[int] | None = None,
    reward_scale: float = 1.0,
    reward_shift: float = 0.0,
    q_star_source: dict[str, Any] | None = None,
) -> gym.vector.VectorEnv:
    """Create a plain (non-NS-Gym) vector env. See :class:`PlainVectorEnvBuilder`."""
    requested_obs_key = "observation_image" if is_ale_env(env_id) else "observation"
    if env_id in (SYNTHETIC_ENV_ID, CUSTOM_FROZENLAKE_ENV_ID):
        clean_kwargs = dict(env_kwargs or {})
        clean_kwargs.pop("seed", None)
    else:
        clean_kwargs = env_kwargs
    params = VectorEnvBuildParams(
        env_id=env_id,
        seed=seed,
        num_envs=num_envs,
        max_steps_per_episode=max_steps_per_episode,
        env_kwargs=clean_kwargs,
        render=render,
        env_name=env_name,
        observation_indices=observation_indices,
        reward_scale=reward_scale,
        reward_shift=reward_shift,
        q_star_source=q_star_source,
        obs_key=requested_obs_key,
    )
    return PlainVectorEnvBuilder(
        params,
        atari_preprocessing=atari_preprocessing,
        atari_preprocessing_kwargs=atari_preprocessing_kwargs,
    ).build()
