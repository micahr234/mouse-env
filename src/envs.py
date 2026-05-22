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
    """Configuration for a single vector environment used during rollout or deployment.

    Passed directly to :func:`make_vector_env`, which reads all fields and constructs
    the appropriate environment type (plain Gymnasium, Atari, NS-Gym, or custom).

    Attributes:
        env_id: Gymnasium env id (e.g. ``"CartPole-v1"``, ``"NS-CartPole-v1"``,
            ``"ALE/Pong-v5"``, ``"Custom-FrozenLake-v1"``).
        seed: Master seed. Each parallel env gets an offset seed (``seed + i``).
        num_envs: Number of parallel envs in the ``SyncVectorEnv``.
        max_episode_steps: Step budget per episode; used to normalise ``xformed_reward``.
            Required — pass ``None`` only when the env's own ``TimeLimit`` wrapper should
            control truncation (not recommended; ``xformed_reward`` will be incorrect).
        kwargs: Extra keyword arguments forwarded to ``gym.make``. For custom envs
            (FrozenLake / SyntheticEnv) this controls map generation parameters.
        render: When ``True``, sets ``render_mode="human"`` on the underlying env.
        non_stationary_params: Scheduler + update-function config for each physics
            parameter to vary (NS-Gym envs only). See :func:`mouse.envs.ns_gym.create_ns_gym_update_functions`.
        num_steps: Deploy-only: number of steps to collect for this env before stopping.
        action_source_loop_prob_schedule: Piecewise-linear probability schedule over
            *global loop steps* (``(step, prob)`` knots). ``None`` = constant 1.0. Steps
            before the first knot use 1.0.
        action_source_episode_prob_schedule: Optional probability schedule over the
            *episode step* (``episode_step`` in rollout data). Composes with the loop
            schedule as ``P_effective = P_loop * P_episode``. Used at test time.
        q_star_source: Expert metadata source config. Keys depend on the provider:

            - ``{"provider": "metadata_q_star"}`` — read ``q_star`` directly from env info
              (custom envs with ``emit_q_star=True``).
            - ``{"provider": "sb3_rl_zoo", "algo": "ppo", "repo_id": "...", "filename": "..."}``
              — load an SB3 policy from the Hugging Face rl-zoo.
            - ``{"provider": "hf_q_table", "repo_id": "...", "filename": "..."}``
              — load a tabular Q-table pickle from the Hub.

            When ``None``, built-in defaults are used for some envs (e.g. CartPole-v1
            gets the SB3 PPO checkpoint automatically).
        action_source: Policy to select actions during rollout. One of
            ``"random"``, ``"q_star"``, ``"learned_sp"``, ``"learned_dqn"``,
            ``"learned_sv"``, ``"learned_vec_dqn"``.
        action_source_temperature: Sampling temperature for the action source.
            ``0.0`` = deterministic argmax; ``> 0`` = softmax sampling.
        split: Dataset split tag, e.g. ``"train"`` or ``"eval"``. Default is ``"train"``.
        env_type: Routing hint: ``"ns_gym"`` or ``"plain"``. When ``None``, routing is
            inferred from the env id (``NS-`` prefix → NS-Gym).
        atari_preprocessing: When ``True``, wraps ALE envs with
            ``gymnasium.wrappers.AtariPreprocessing`` (grayscale, frame skip, resize).
        atari_preprocessing_kwargs: Optional overrides for ``AtariPreprocessing``
            (e.g. ``noop_max``, ``frame_skip``, ``screen_size``).
        observation_indices: When set, slices the observation vector to these indices
            using :class:`~mouse.envs.base.ObservationSliceWrapper`. Not supported for
            Atari envs.
        reward_scale: Multiply each step reward by this factor before adding ``reward_shift``.
            ``episode_cum_reward`` in info always reflects the raw (unscaled) return.
        reward_shift: Add this constant to each step reward after applying ``reward_scale``.
        clear_history: Test-rollout flag. When ``True``, the model context is reset to
            tokens from the current run only (the full stream is still saved to the dataset).
    """

    env_id: str
    seed: int
    num_envs: int
    max_episode_steps: int | None
    kwargs: dict | None
    render: bool
    non_stationary_params: dict | None
    num_steps: int | None
    action_source_loop_prob_schedule: tuple[tuple[int | float, float], ...] | None
    action_source_episode_prob_schedule: tuple[tuple[int | float, float], ...] | None
    q_star_source: dict[str, Any] | None
    action_source: str
    action_source_temperature: float
    split: str | None
    env_type: str | None
    atari_preprocessing: bool | None
    atari_preprocessing_kwargs: dict | None
    observation_indices: list[int] | None
    reward_scale: float = 1.0
    reward_shift: float = 0.0
    clear_history: bool = False

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing

from mouse.envs.base import ObservationSliceWrapper, build_vector_env_stack
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
    """Return the effective ``q_star_source`` config for an env.

    Uses the caller-supplied config when provided; falls back to built-in defaults
    for known env ids (e.g. ``CartPole-v1`` gets the SB3 PPO checkpoint from the
    Hugging Face rl-zoo). Returns ``None`` when no source is configured.
    """
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
    """Wraps a factory callable so that construction-time and reset-time seeds are controlled.

    Takes a factory callable ``env_fn(seed: int) -> gym.Env`` so the seed is
    available both at construction time (for map/MDP generation in custom envs)
    and at reset time via an internal RNG that draws a fresh reproducible seed
    each episode. Any seed passed to ``reset()`` is ignored — reproducibility is
    fully owned by this wrapper.

    Args:
        env_fn: Callable that takes an integer seed and returns a ``gym.Env``.
        seed: Seed used both for construction and as the RNG seed for episode resets.
    """

    def __init__(self, env_fn: Callable[[int], gym.Env], seed: int):
        super().__init__(env_fn(seed))
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        return self.env.reset(seed=int(self._rng.integers(0, 2**31)), options=options)


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
    """Create a plain (non-NS-Gym) vector env wrapping any standard or Atari Gymnasium env.

    Supports two env families:

    1. **Standard discrete-obs envs** (CartPole, MountainCar, FrozenLake, …) — obs key is
       ``"observation"`` or ``"observation_discrete"`` depending on the observation space dtype.
    2. **Atari (ALE/*)** — obs key is ``"observation_image"``; optionally applies
       ``gymnasium.wrappers.AtariPreprocessing`` when ``atari_preprocessing=True``.

    Custom envs registered in this package (``Custom-FrozenLake-v1``,
    ``Custom-SyntheticEnv-v1``) are handled transparently; their construction-time seed
    is set from ``seed`` and any ``kwargs.seed`` is ignored to prevent accidental overrides.

    Call ``env.reset()`` once before the first ``env.step()``. All step metadata
    (``episode_step``, ``done``, ``xformed_reward``, etc.) is injected into ``info``
    by the wrapper stack.

    Args:
        env_id: Gymnasium environment id. Use ``"ALE/..."`` for Atari.
        seed: Master seed. Each of the ``num_envs`` parallel envs gets ``seed + i``.
        max_steps_per_episode: Passed to ``gym.make(max_episode_steps=...)`` and used
            to normalise ``xformed_reward``.
        num_envs: Number of parallel envs in the underlying ``SyncVectorEnv``.
        env_kwargs: Extra keyword arguments forwarded to ``gym.make``.
        render: Set ``render_mode="human"`` on the underlying env.
        env_name: Name string injected into ``info["env_name"]``. Defaults to ``env_id``.
        atari_preprocessing: Wrap ALE envs with ``AtariPreprocessing`` (grayscale,
            frame skip, screen resize, no-op starts).
        atari_preprocessing_kwargs: Override kwargs for ``AtariPreprocessing``
            (e.g. ``frame_skip``, ``screen_size``, ``noop_max``).
        observation_indices: Slice the observation vector to these indices via
            :class:`~mouse.envs.base.ObservationSliceWrapper`. Not supported for Atari.
        reward_scale: Multiply rewards by this scalar (``episode_cum_reward`` is unaffected).
        reward_shift: Add this constant to each scaled reward.
        q_star_source: Expert policy source config; see :class:`EnvConfig` for details.

    Returns:
        A fully wrapped ``gym.vector.VectorEnv`` with the standard wrapper stack applied.

    Raises:
        ValueError: If ``num_envs < 1``, or ``observation_indices`` is combined with an
            Atari env, or the action space is continuous (not supported).
    """
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
    requested_obs_key = "observation_image" if _is_ale_env(env_id) else "observation"
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

    def make_env(mdp_seed: int) -> gym.Env:
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

    return build_vector_env_stack(
        env_fns=env_fns,
        env_id=env_id,
        env_name=env_name if env_name is not None else env_id,
        seed=seed,
        max_steps_per_episode=max_steps_per_episode,
        obs_key=requested_obs_key,  # build_vector_env_stack resolves against actual obs space
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
) -> gym.vector.VectorEnv:
    """Create a vector env: routes to NS-Gym or plain (standard/Atari) based on the env id.

    This is the main public factory. Routing is determined by the ``NS-`` prefix on ``env_id``
    (or the ``env_type`` override). The returned env is always a ``gym.vector.VectorEnv`` with
    the full wrapper stack applied — call ``env.reset()`` once before the first ``env.step()``.

    When ``q_star_source`` is omitted, built-in defaults apply for some env ids:
    ``CartPole-v1`` / ``NS-CartPole-v1`` automatically use the SB3 PPO checkpoint from
    the Hugging Face rl-zoo.

    Args:
        env_id: Gymnasium env id. Prefix with ``NS-`` for non-stationary variants
            (e.g. ``"NS-CartPole-v1"``). Use ``"ALE/..."`` for Atari.
        seed: Master seed passed to all parallel envs (each gets ``seed + i``).
        max_steps_per_episode: Episode step budget; required for ``xformed_reward``
            normalisation. Pass ``None`` only to inherit the env's own ``TimeLimit``.
        num_envs: Number of parallel envs in the ``SyncVectorEnv``.
        env_kwargs: Extra keyword arguments forwarded to ``gym.make``.
        render: Set ``render_mode="human"`` on the underlying env.
        env_name: Override string for ``info["env_name"]``. Defaults to ``env_id``.
        non_stationary_params: Parameter schedule config for NS-Gym envs. See
            :func:`~mouse.envs.ns_gym.create_ns_gym_update_functions` for the format.
            Required when routing to NS-Gym (either via ``NS-`` prefix or ``env_type="ns_gym"``).
        env_type: Explicit routing override: ``"ns_gym"`` or ``"plain"``. When ``None``,
            the ``NS-`` prefix on ``env_id`` determines routing.
        atari_preprocessing: Apply ``AtariPreprocessing`` to ALE envs (grayscale,
            frame skip, screen resize).
        atari_preprocessing_kwargs: Override kwargs for ``AtariPreprocessing``.
        observation_indices: Slice the observation vector to these indices. Not supported
            for Atari envs.
        reward_scale: Multiply rewards by this scalar after each step.
        reward_shift: Add this constant to each scaled reward.
        q_star_source: Expert metadata source config. See :class:`EnvConfig` for details.

    Returns:
        A fully wrapped ``gym.vector.VectorEnv`` ready for rollout collection.

    Raises:
        ValueError: If ``max_steps_per_episode`` is ``None``, or NS-Gym routing is
            triggered but ``non_stationary_params`` is absent.
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
