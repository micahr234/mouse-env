"""Environment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_SB3_Q_STAR_CARTPOLE: dict[str, Any] = {
    "provider": "sb3_rl_zoo",
    "algo": "ppo",
    "repo_id": "sb3/ppo-CartPole-v1",
    "filename": "ppo-CartPole-v1.zip",
    "deterministic": True,
}


def normalize_group_id(group_id: str) -> str:
    """Strip the legacy ``NS-`` prefix before passing the id to ``gym.make``."""
    if group_id.startswith("NS-"):
        return group_id[3:]
    return group_id


def is_ns_gym_env(
    _group_id: str,
    non_stationary_params: dict[str, Any] | None,
) -> bool:
    """Return ``True`` when the env should use the NS-Gym backend."""
    return bool(non_stationary_params)


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

    def build(self):
        """Build a vector env from this config (alias for ``make_vector_env(self)``)."""
        from mouse_envs.build import make_vector_env

        return make_vector_env(self)

    @classmethod
    def cartpole(
        cls,
        *,
        seed: int = 0,
        num_envs: int = 1,
        max_episode_steps: int = 500,
        q_star_source: dict[str, Any] | None | object = ...,
        observation_indices: list[int] | None = None,
        **kwargs: Any,
    ) -> EnvConfig:
        """Preset for ``CartPole-v1`` with SB3 PPO Q* by default.

        Pass ``q_star_source=None`` explicitly to disable expert metadata.
        """
        if q_star_source is ...:
            resolved_q_star: dict[str, Any] | None = dict(DEFAULT_SB3_Q_STAR_CARTPOLE)
        else:
            resolved_q_star = q_star_source
        return cls(
            group_id="CartPole-v1",
            seed=seed,
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            q_star_source=resolved_q_star,
            observation_indices=observation_indices,
            **kwargs,
        )

    @classmethod
    def ns_cartpole(
        cls,
        *,
        seed: int = 0,
        num_envs: int = 1,
        max_episode_steps: int = 500,
        non_stationary_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> EnvConfig:
        """Preset for non-stationary ``CartPole-v1`` with optional physics schedules."""
        params = non_stationary_params if non_stationary_params is not None else {}
        return cls(
            group_id="CartPole-v1",
            seed=seed,
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            non_stationary_params=params,
            **kwargs,
        )

    @classmethod
    def procedural_frozenlake(
        cls,
        *,
        seed: int = 0,
        num_envs: int = 1,
        max_episode_steps: int = 200,
        env_kwargs: dict[str, Any] | None = None,
        q_star_source: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> EnvConfig:
        """Preset for ``Procedural-FrozenLake-v1``."""
        return cls(
            group_id="Procedural-FrozenLake-v1",
            seed=seed,
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            kwargs=env_kwargs,
            q_star_source=q_star_source or {"provider": "metadata_q_star"},
            **kwargs,
        )

    @classmethod
    def synthetic(
        cls,
        *,
        seed: int = 0,
        num_envs: int = 1,
        max_episode_steps: int = 200,
        env_kwargs: dict[str, Any] | None = None,
        q_star_source: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> EnvConfig:
        """Preset for ``SyntheticEnv-v1``."""
        return cls(
            group_id="SyntheticEnv-v1",
            seed=seed,
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            kwargs=env_kwargs,
            q_star_source=q_star_source or {"provider": "metadata_q_star"},
            **kwargs,
        )

    @classmethod
    def atari(
        cls,
        env_id: str = "ALE/Pong-v5",
        *,
        seed: int = 0,
        num_envs: int = 4,
        max_episode_steps: int = 10000,
        frame_skip: int = 4,
        screen_size: int = 84,
        noop_max: int = 30,
        **kwargs: Any,
    ) -> EnvConfig:
        """Preset for ALE Atari envs — common ``AtariPreprocessing`` defaults only."""
        return cls(
            group_id=env_id,
            seed=seed,
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            atari_preprocessing=True,
            atari_preprocessing_kwargs={
                "frame_skip": frame_skip,
                "screen_size": screen_size,
                "noop_max": noop_max,
            },
            **kwargs,
        )


def resolve_q_star_source_for_env(
    _group_id: str,
    q_star_source: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the effective ``q_star_source`` config for an env."""
    if q_star_source:
        return dict(q_star_source)
    return None
