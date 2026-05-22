"""Environment factory and wrappers for NS-Gym and plain Gymnasium envs.

Provides:
- EnvConfig: dataclass for a single test/rollout environment
- make_vector_env: unified factory returning NSVectorEnv or PlainVectorEnv
- NS-Gym: non-stationary classic control (CartPole, MountainCar, etc.) with configurable param schedules
- Plain: standard vector-obs envs (same IDs) or Atari (ALE/*) with optional image obs
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class EnvConfig:
    env_id: str
    seed: int
    num_envs: int
    max_episode_steps: int | None
    kwargs: dict | None
    render: bool
    non_stationary_params: dict | None
    # Deploy only: steps to collect for this env.
    num_steps: int | None
    # Piecewise-linear schedule over global loop steps; None = constant 1.0. Prefix before first knot uses 1.0.
    action_source_loop_prob_schedule: tuple[tuple[int | float, float], ...] | None
    # Optional schedule over *episode* timestep (``episode_step`` in rollout data) at test time.
    # Composes with the global schedule as P_effective = P_loop * P_episode.
    action_source_episode_prob_schedule: tuple[tuple[int | float, float], ...] | None
    # Optional env-side expert metadata source (used to emit metadata_q_star).
    q_star_source: dict[str, Any] | None
    # Runner action source: "random" | "q_star" | "learned_sp" | "learned_dqn" | "learned_sv" | "learned_vec_dqn".
    action_source: str
    # Sampling temperature used by the selected policy source.
    # temperature == 0: deterministic argmax; temperature > 0: softmax sampling.
    action_source_temperature: float
    split: str | None  # "train" or "eval"; default "train"
    env_type: str | None  # "ns_gym" | "plain"; optional
    atari_preprocessing: bool | None  # if True, wrap ALE env with gymnasium.wrappers.AtariPreprocessing
    atari_preprocessing_kwargs: dict | None  # optional overrides (noop_max, frame_skip, screen_size, etc.)
    observation_indices: list[int] | None  # if set, wrap with ObservationSliceWrapper (e.g. [0, 2] for CartPole position+angle only)
    reward_scale: float = 1.0  # env reward becomes reward * reward_scale + reward_shift (after each step)
    reward_shift: float = 0.0
    # Test rollout: when True, each RunnerTest.run() resets model context to tokens from that run only (full stream kept for dataset save).
    clear_history: bool = False

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing

from mouse.envs.base import _BaseVectorEnv, ObservationSliceWrapper
from mouse.envs.ns_gym import NSVectorEnv, is_ns_gym_env, normalize_env_id
from mouse.envs.action_star import (
    apply_q_star_source_env_kwargs,
)
from mouse.envs.frozenlake import (
    CUSTOM_FROZENLAKE_ENV_ID,
    ensure_custom_frozenlake_registered,
)
from mouse.envs.synthetic import (
    SYNTHETIC_ENV_ID,
    ensure_synthetic_env_registered,
)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

# When YAML omits ``q_star_source``, :func:`make_vector_env` fills these (key = normalize_env_id).
_DEFAULT_SB3_Q_STAR_SOURCE_BY_BASE_ENV_ID: dict[str, dict[str, Any]] = {
    "CartPole-v1": {
        "provider": "sb3_rl_zoo",
        "algo": "ppo",
        "repo_id": "sb3/ppo-CartPole-v1",
        "filename": "ppo-CartPole-v1.zip",
        "deterministic": True,
    },
}


def resolve_q_star_source_for_env(
    env_id: str,
    q_star_source: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Use YAML ``q_star_source`` when provided; otherwise env-specific defaults if defined."""
    if q_star_source:
        return dict(q_star_source)
    base = normalize_env_id(env_id)
    default = _DEFAULT_SB3_Q_STAR_SOURCE_BY_BASE_ENV_ID.get(base)
    return dict(default) if default is not None else None


# -----------------------------------------------------------------------------
# Plain envs: standard vector-obs and Atari (image obs)
# -----------------------------------------------------------------------------


def _is_ale_env(env_id: str) -> bool:
    """True if env_id is an Atari (ALE) env that uses image observations."""
    return env_id.startswith("ALE/")


def _ensure_ale_registered() -> None:
    """Import and register ale_py with Gymnasium so gym.make('ALE/...') works."""
    try:
        import ale_py  # noqa: F401
        gym.register_envs(ale_py)
    except ImportError as e:
        raise ImportError(
            "ALE (Atari) envs require the ale_py package. Install with: pip install 'gymnasium[atari]'"
        ) from e


class ConstructionSeedWrapper(gym.Wrapper):
    """Constructs an env with a seed and auto-generates deterministic reset seeds.

    Takes a factory callable ``env_fn(seed: int) -> gym.Env`` so the seed is
    available both at construction time (for map/MDP generation in custom envs)
    and at reset time (via an internal RNG that draws a fresh seed each episode).
    Any seed passed to reset() is ignored — reproducibility is fully owned by
    this wrapper.
    """

    def __init__(self, env_fn: Callable[[int], gym.Env], seed: int):
        super().__init__(env_fn(seed))
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        return self.env.reset(seed=int(self._rng.integers(0, 2**31)), options=options)


class PlainVectorEnv(_BaseVectorEnv):
    """Vector env for plain Gymnasium environments (no ns_gym). Same interface as NSVectorEnv.

    Supports: (1) standard envs (e.g. CartPole-v1) with vector observation; (2) Atari (ALE/*)
    with observation_image = env's native observation (or AtariPreprocessing output if enabled).
    """

    def __init__(
        self,
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
    ):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
        env_kwargs = dict(env_kwargs or {})
        env_kwargs = apply_q_star_source_env_kwargs(
            env_id=env_id,
            env_kwargs=env_kwargs,
            q_star_source=q_star_source,
        )
        if env_id == CUSTOM_FROZENLAKE_ENV_ID:
            ensure_custom_frozenlake_registered()
            random_map_wrapper_raw = env_kwargs.pop("random_map_wrapper", None)
            if random_map_wrapper_raw is None:
                pass
            elif isinstance(random_map_wrapper_raw, dict):
                # Backward compatible config shape:
                # kwargs.random_map_wrapper.<arg> -> kwargs.<arg>
                env_kwargs.update(dict(random_map_wrapper_raw))
            else:
                raise ValueError(
                    "env_kwargs.random_map_wrapper must be a dict when provided."
                )
        elif env_id == SYNTHETIC_ENV_ID:
            ensure_synthetic_env_registered()
        if render and "render_mode" not in env_kwargs:
            env_kwargs["render_mode"] = "human"
        if _is_ale_env(env_id):
            _ensure_ale_registered()
        obs_key = "observation_image" if _is_ale_env(env_id) else "observation"
        use_preprocessing = _is_ale_env(env_id) and atari_preprocessing
        if use_preprocessing:
            env_kwargs["frameskip"] = 1  # AtariPreprocessing requires base env frameskip=1
        if observation_indices is not None and _is_ale_env(env_id):
            raise ValueError("observation_indices is not supported for ALE (Atari) envs.")

        # Envs that accept a constructor-level seed for map/MDP generation.
        _seeded_at_construction = env_id in (SYNTHETIC_ENV_ID, CUSTOM_FROZENLAKE_ENV_ID)
        if _seeded_at_construction:
            # `seed` from make_vector_env (EnvConfig.seed in runner_test / loop) is authoritative.
            # Ignore any duplicate under env kwargs so YAML `kwargs.seed` cannot override it.
            env_kwargs.pop("seed", None)
            mdp_base_seed = int(seed)
        else:
            mdp_base_seed = seed

        def make_env(mdp_seed: int):
            def env_fn(s: int) -> gym.Env:
                kw = dict(env_kwargs)
                if _seeded_at_construction:
                    kw["seed"] = s
                env = gym.make(env_id, max_episode_steps=max_steps_per_episode, **kw)
                if use_preprocessing:
                    env = AtariPreprocessing(env, **(atari_preprocessing_kwargs or {}))
                return env

            env = ConstructionSeedWrapper(env_fn, seed=mdp_seed)
            if observation_indices is not None:
                env = ObservationSliceWrapper(env=env, indices=observation_indices)
            return env

        # Every instance gets seed+i so each env is independently but reproducibly seeded.
        env_fns = [(lambda s=mdp_base_seed + i: make_env(s)) for i in range(num_envs)]

        super().__init__(
            env_fns=env_fns,
            env_id=env_id,
            env_name=env_name if env_name is not None else env_id,
            num_envs=num_envs,
            render=render,
            seed=seed,
            max_steps_per_episode=max_steps_per_episode,
            obs_key=obs_key,
            reward_scale=reward_scale,
            reward_shift=reward_shift,
            q_star_source=q_star_source,
        )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def make_vector_env(
    env_id: str,
    seed: int,
    max_steps_per_episode: int | None,
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
) -> NSVectorEnv | PlainVectorEnv:
    """Create a vector env runner: NS-Gym (non-stationary) or plain (standard/Atari) based on config.

    When ``q_star_source`` is omitted, built-in defaults apply for some env ids (e.g. ``CartPole-v1``
    / ``NS-CartPole-v1`` uses the SB3 PPO checkpoint from the Hugging Face rl-zoo).
    """
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
