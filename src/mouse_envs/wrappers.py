"""Single-env wrappers and stack factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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


# -----------------------------------------------------------------------------
# Single-env wrappers
# -----------------------------------------------------------------------------


def _action_dim_for_space(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))
    if isinstance(space, gym.spaces.MultiDiscrete):
        return len(space.nvec)
    return int(getattr(space, "n", 0))


class QStarWrapper(gym.Wrapper):
    """Inject expert Q-values into ``info["q_star"]`` after each step and reset."""

    def __init__(
        self,
        env: gym.Env,
        env_id: str,
        q_star_source: dict[str, Any],
    ):
        from mouse_envs.experts.action_star import build_q_star_source_adapter

        super().__init__(env)
        self._adapter = build_q_star_source_adapter(
            env_id=env_id,
            q_star_source=q_star_source,
            single_observation_space=env.observation_space,
        )
        self._action_dim = _action_dim_for_space(env.action_space)
        self._continuous = isinstance(env.action_space, gym.spaces.Box)

    def _action_star_to_q_star(self, ast: Any) -> np.ndarray:
        """Convert a single expert action into the ``q_star`` representation."""
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
            info["q_star"] = q_arr
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
    q_star_source: dict[str, Any] | None = None,
) -> gym.Env:
    """Build the single-env wrapper stack around one ``gym.Env`` factory call."""
    env = env_fn()
    if q_star_source is not None:
        env = QStarWrapper(env, env_id=env_id, q_star_source=q_star_source)
    return env
