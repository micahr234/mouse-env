"""Environment and rollout configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mouse.envs.routing import normalize_env_id

DEFAULT_SB3_Q_STAR_CARTPOLE: dict[str, Any] = {
    "provider": "sb3_rl_zoo",
    "algo": "ppo",
    "repo_id": "sb3/ppo-CartPole-v1",
    "filename": "ppo-CartPole-v1.zip",
    "deterministic": True,
}

_DEFAULT_SB3_Q_STAR_SOURCE_BY_BASE_ENV_ID: dict[str, dict[str, Any]] = {
    "CartPole-v1": DEFAULT_SB3_Q_STAR_CARTPOLE,
}


@dataclass
class EnvBuildConfig:
    """Parameters used to construct a ``gym.vector.VectorEnv``."""

    env_id: str
    seed: int
    num_envs: int
    max_episode_steps: int | None
    kwargs: dict[str, Any] | None = None
    render: bool = False
    non_stationary_params: dict[str, Any] | None = None
    env_type: str | None = None
    atari_preprocessing: bool | None = None
    atari_preprocessing_kwargs: dict[str, Any] | None = None
    observation_indices: list[int] | None = None
    reward_scale: float = 1.0
    reward_shift: float = 0.0
    q_star_source: dict[str, Any] | None = None
    env_name: str | None = None


@dataclass
class RolloutConfig:
    """Parameters consumed by MOUSE runners (not by ``gym.make``)."""

    num_steps: int | None = None
    action_source_loop_prob_schedule: tuple[tuple[int | float, float], ...] | None = None
    action_source_episode_prob_schedule: tuple[tuple[int | float, float], ...] | None = None
    action_source: str = "random"
    action_source_temperature: float = 1.0
    split: str | None = "train"
    clear_history: bool = False


@dataclass
class EnvConfig:
    """Configuration for a single vector environment used during rollout or deployment.

    Pass an instance to :func:`mouse.envs.make_vector_env` or use
    :meth:`build` to construct the env directly.

    For construction-only fields see :class:`EnvBuildConfig`; for runner fields see
    :class:`RolloutConfig` via :attr:`build_config` and :attr:`rollout_config`.
    """

    env_id: str
    seed: int
    num_envs: int
    max_episode_steps: int | None
    kwargs: dict | None = None
    render: bool = False
    non_stationary_params: dict | None = None
    num_steps: int | None = None
    action_source_loop_prob_schedule: tuple[tuple[int | float, float], ...] | None = None
    action_source_episode_prob_schedule: tuple[tuple[int | float, float], ...] | None = None
    q_star_source: dict[str, Any] | None = None
    action_source: str = "random"
    action_source_temperature: float = 1.0
    split: str | None = "train"
    env_type: str | None = None
    atari_preprocessing: bool | None = None
    atari_preprocessing_kwargs: dict | None = None
    observation_indices: list[int] | None = None
    reward_scale: float = 1.0
    reward_shift: float = 0.0
    clear_history: bool = False

    @property
    def build_config(self) -> EnvBuildConfig:
        """Env construction fields as a dedicated config object."""
        return EnvBuildConfig(
            env_id=self.env_id,
            seed=self.seed,
            num_envs=self.num_envs,
            max_episode_steps=self.max_episode_steps,
            kwargs=self.kwargs,
            render=self.render,
            non_stationary_params=self.non_stationary_params,
            env_type=self.env_type,
            atari_preprocessing=self.atari_preprocessing,
            atari_preprocessing_kwargs=self.atari_preprocessing_kwargs,
            observation_indices=self.observation_indices,
            reward_scale=self.reward_scale,
            reward_shift=self.reward_shift,
            q_star_source=self.q_star_source,
        )

    @property
    def rollout_config(self) -> RolloutConfig:
        """Runner-only fields as a dedicated config object."""
        return RolloutConfig(
            num_steps=self.num_steps,
            action_source_loop_prob_schedule=self.action_source_loop_prob_schedule,
            action_source_episode_prob_schedule=self.action_source_episode_prob_schedule,
            action_source=self.action_source,
            action_source_temperature=self.action_source_temperature,
            split=self.split,
            clear_history=self.clear_history,
        )

    def build(self):
        """Build a vector env from this config (alias for ``make_vector_env(self)``)."""
        from mouse.envs.factory import make_vector_env

        return make_vector_env(self)

    @classmethod
    def cartpole(
        cls,
        *,
        seed: int = 0,
        num_envs: int = 1,
        max_episode_steps: int = 500,
        q_star_source: dict[str, Any] | None = None,
        observation_indices: list[int] | None = None,
        **kwargs: Any,
    ) -> EnvConfig:
        """Preset for ``CartPole-v1`` with SB3 PPO Q* by default."""
        return cls(
            env_id="CartPole-v1",
            seed=seed,
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            q_star_source=q_star_source if q_star_source is not None else dict(DEFAULT_SB3_Q_STAR_CARTPOLE),
            observation_indices=observation_indices,
            env_type="plain",
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
        """Preset for ``NS-CartPole-v1`` with optional physics schedules."""
        params = non_stationary_params if non_stationary_params is not None else {}
        return cls(
            env_id="NS-CartPole-v1",
            seed=seed,
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            non_stationary_params=params,
            env_type="ns_gym",
            **kwargs,
        )

    @classmethod
    def frozenlake(
        cls,
        *,
        seed: int = 0,
        num_envs: int = 1,
        max_episode_steps: int = 200,
        env_kwargs: dict[str, Any] | None = None,
        q_star_source: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> EnvConfig:
        """Preset for ``Custom-FrozenLake-v1`` with metadata Q*."""
        return cls(
            env_id="Custom-FrozenLake-v1",
            seed=seed,
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            kwargs=env_kwargs,
            q_star_source=q_star_source or {"provider": "metadata_q_star"},
            env_type="plain",
            **kwargs,
        )


def resolve_q_star_source_for_env(
    env_id: str,
    q_star_source: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the effective ``q_star_source`` config for an env."""
    if q_star_source:
        return dict(q_star_source)
    base = normalize_env_id(env_id)
    default = _DEFAULT_SB3_Q_STAR_SOURCE_BY_BASE_ENV_ID.get(base)
    return dict(default) if default is not None else None
