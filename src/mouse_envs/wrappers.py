"""Single-env wrappers and stack factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium.core import ObservationWrapper


# -----------------------------------------------------------------------------
# Observation helpers
# -----------------------------------------------------------------------------


class ObservationSliceWrapper(ObservationWrapper):
    """Slice a ``Box`` observation vector to a subset of indices."""

    def __init__(self, env: gym.Env, indices: list[int]):
        if not indices:
            raise ValueError("observation_indices must be non-empty.")
        super().__init__(env)
        self._indices = np.array(indices, dtype=np.intp)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box):
            raise ValueError(
                f"ObservationSliceWrapper requires Box observation space, got {type(space).__name__}."
            )
        low = np.asarray(space.low).flatten()
        high = np.asarray(space.high).flatten()
        if len(low) != len(high) or max(self._indices) >= len(low):
            raise ValueError(
                f"observation_indices {indices} out of range for space shape {low.shape}."
            )
        self.observation_space = gym.spaces.Box(
            low=low[self._indices],
            high=high[self._indices],
            dtype=getattr(space, "dtype", np.float32),
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        flat = np.asarray(observation).flatten()
        return flat[self._indices].astype(self.observation_space.dtype)


def _is_discrete_like(space: gym.Space) -> bool:
    if isinstance(space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)):
        return True
    if isinstance(space, gym.spaces.Tuple):
        return all(_is_discrete_like(s) for s in space.spaces)
    if isinstance(space, gym.spaces.Dict):
        return all(_is_discrete_like(s) for s in space.spaces.values())
    if isinstance(space, gym.spaces.Box):
        obs_dtype = np.dtype(space.dtype)
        return bool(np.issubdtype(obs_dtype, np.integer) or np.issubdtype(obs_dtype, np.bool_))
    return False


def resolve_obs_key(env: gym.Env, observation_kind: str | None = None) -> str:
    """Return the internal routing key for this env's observation channel.

    Used internally to select the flat observation field written to each result record
    (``observation_discrete``, ``observation_continuous``, or ``observation_image``).
    ``observation_kind`` (``"continuous"``, ``"discrete"``, or ``"image"``) forces a
    channel explicitly; ``None`` auto-detects from the observation space. Auto-detection
    cannot recognise image spaces (a uint8 ``Box`` that otherwise looks discrete), so
    image envs must set ``observation_kind="image"``.
    """
    if observation_kind == "image":
        return "observation_image"
    if observation_kind == "discrete":
        return "observation_discrete"
    if observation_kind == "continuous":
        return "observation"
    if observation_kind is not None:
        raise ValueError(
            f"observation_kind must be one of 'continuous', 'discrete', 'image', or None; "
            f"got {observation_kind!r}."
        )
    if _is_discrete_like(env.observation_space):
        return "observation_discrete"
    return "observation"


# -----------------------------------------------------------------------------
# Single-env wrappers
# -----------------------------------------------------------------------------


class EnvIdentityWrapper(gym.Wrapper):
    """Expose ``obs_key``, ``action_dim``, ``env_seed``, and ``sample_random_input()``."""

    def __init__(
        self,
        env: gym.Env,
        name: str,
        env_seed: int,
        obs_key: str,
    ):
        super().__init__(env)
        self.name = name
        self.env_seed = int(env_seed)
        self.obs_key = obs_key
        self._initial_reset_done = False

    @property
    def action_dim(self) -> int:
        space = self.action_space
        if isinstance(space, gym.spaces.Discrete):
            return int(space.n)
        if isinstance(space, gym.spaces.Box):
            return int(np.prod(space.shape))
        if isinstance(space, gym.spaces.MultiDiscrete):
            return len(space.nvec)
        return int(getattr(space, "n", 0))

    def sample_random_input(self) -> np.ndarray:
        return np.asarray(self.action_space.sample())

    def reset(self, **kwargs: Any):
        if not self._initial_reset_done:
            self._initial_reset_done = True
        return self.env.reset(**kwargs)


class QStarWrapper(gym.Wrapper):
    """Inject expert Q-values into ``info["env_q_star"]`` after each step and reset."""

    def __init__(
        self,
        env: gym.Env,
        env_id: str,
        q_star_source: dict[str, Any],
        obs_key: str,
    ):
        from mouse_envs.experts.action_star import build_q_star_source_adapter

        super().__init__(env)
        self._adapter = build_q_star_source_adapter(
            env_id=env_id,
            q_star_source=q_star_source,
            obs_key=obs_key,
            single_observation_space=env.observation_space,
        )
        self._action_dim = int(getattr(env, "action_dim", 0))
        self._continuous = isinstance(env.action_space, gym.spaces.Box)

    @property
    def obs_key(self) -> str:
        return cast(Any, self.env).obs_key

    @property
    def env_seed(self) -> int:
        return cast(Any, self.env).env_seed

    @property
    def action_dim(self) -> int:
        return cast(Any, self.env).action_dim

    def sample_random_input(self) -> np.ndarray:
        return cast(Any, self.env).sample_random_input()

    def _action_star_to_q_star(self, ast: Any) -> np.ndarray:
        """Convert a single expert action into the ``env_q_star`` representation."""
        if self._continuous:
            from mouse_envs.experts.action_star import action_star_to_continuous_q_star

            ast_batch = np.asarray(ast, dtype=np.float64).reshape(1, -1)
            return action_star_to_continuous_q_star(
                actions=ast_batch, num_envs=1, action_dim=self._action_dim
            )[0]
        ast_arr = np.asarray(ast, dtype=np.int64).reshape(-1)
        from mouse_envs.experts.action_star import action_star_to_one_hot_q_star

        return action_star_to_one_hot_q_star(actions=ast_arr, num_actions=self._action_dim)[0]

    def _obs_as_batch(self, obs: Any) -> np.ndarray:
        """Return obs with a leading batch dimension for the adapter interface."""
        if isinstance(obs, dict):
            # Dict obs: use first value; most adapters do not support dict obs
            arr = np.asarray(next(iter(obs.values())))
        else:
            arr = np.asarray(obs)
        return arr[np.newaxis]

    def _attach(self, obs: Any, info: dict[str, Any], *, done: bool) -> dict[str, Any]:
        if self._adapter is None:
            return info
        done_mask = np.array([done], dtype=np.bool_)
        obs_batch = self._obs_as_batch(obs)

        q_star = self._adapter.q_star_from_infos(infos=info, num_envs=1)
        if q_star is None:
            q_star = self._adapter.q_star_from_observation(obs=obs_batch, done_mask=done_mask)
        if q_star is None and not self._continuous:
            q_star = self._adapter.q_star_from_action_star_infos(
                infos=info, num_envs=1, num_actions=self._action_dim
            )
        if q_star is None:
            ast = self._adapter.action_star_from_observation(obs=obs_batch, done_mask=done_mask)
            if ast is not None:
                q_star = self._action_star_to_q_star(ast)
        if q_star is not None:
            info = dict(info)
            q_arr = np.asarray(q_star, dtype=np.float64)
            if q_arr.ndim == 2 and q_arr.shape[0] == 1:
                q_arr = q_arr[0]
            info["env_q_star"] = q_arr
        return info

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        return obs, self._attach(obs, info, done=False)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated) or bool(truncated)
        return obs, reward, terminated, truncated, self._attach(obs, info, done=done)


class SeedStreamWrapper(gym.Wrapper):
    """Control mouse-env's internal reset stream."""

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        *,
        reset_seed: int | None,
    ):
        super().__init__(env_fn())
        self._reset_rng = np.random.default_rng(reset_seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is None:
            seed = int(self._reset_rng.integers(0, 2**31))
        return self.env.reset(seed=seed, options=options)


# -----------------------------------------------------------------------------
# Stack factory
# -----------------------------------------------------------------------------


def build_single_env(
    env_fn: Callable[[], gym.Env],
    env_id: str,
    name: str,
    seed: int,
    observation_kind: str | None = None,
    q_star_source: dict[str, Any] | None = None,
) -> gym.Env:
    """Build the single-env wrapper stack around one ``gym.Env`` factory call."""
    env = env_fn()
    resolved_obs_key = resolve_obs_key(env, observation_kind)
    env = EnvIdentityWrapper(env, name=name, env_seed=seed, obs_key=resolved_obs_key)
    if q_star_source is not None:
        env = QStarWrapper(
            env, env_id=env_id, q_star_source=q_star_source, obs_key=resolved_obs_key
        )
    return env
