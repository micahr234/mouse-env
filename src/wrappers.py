"""Vector env wrappers, observation helpers, and stack factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObservationWrapper
from gymnasium.vector import SyncVectorEnv

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


def resolve_obs_key(env: gym.vector.VectorEnv, requested: str = "observation") -> str:
    """Return the canonical observation-dict key for this env's observation space."""
    if requested == "observation_image":
        return requested
    if _is_discrete_like(env.single_observation_space):
        return "observation_discrete"
    return requested


# -----------------------------------------------------------------------------
# Vector env wrappers
# -----------------------------------------------------------------------------


def _is_reset_frame(info: dict[str, Any], num_envs: int) -> np.ndarray:
    """Return per-env mask for SyncVectorEnv ``NEXT_STEP`` autoreset frames."""
    raw = info.get("is_reset_frame")
    if raw is None:
        return np.zeros(num_envs, dtype=np.bool_)
    return np.asarray(raw, dtype=np.bool_)


class AutoresetFrameWrapper(gym.vector.VectorWrapper):
    """Mark ``SyncVectorEnv`` autoreset observations in ``info["is_reset_frame"]``.

    With ``AutoresetMode.NEXT_STEP``, a sub-env that finished on the previous step is
    reset on the next ``step()`` call without applying the action. That transition has
    zero reward and ``terminated/truncated == False``.
    """

    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__(env)
        self._prev_dones = np.zeros(env.num_envs, dtype=np.bool_)

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._prev_dones[:] = False
        info = dict(info)
        info["is_reset_frame"] = np.zeros(self.num_envs, dtype=np.bool_)
        return obs, info

    def step(self, actions: Any):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        dones = np.asarray(terminated, dtype=np.bool_) | np.asarray(truncated, dtype=np.bool_)
        is_reset_frame = self._prev_dones & ~dones
        self._prev_dones = dones
        info = dict(info)
        info["is_reset_frame"] = is_reset_frame
        return obs, reward, terminated, truncated, info


class _RewardTransformWrapper(gym.vector.VectorWrapper):
    """Scale and shift rewards: ``r_out = r * scale + shift``."""

    def __init__(self, env: gym.vector.VectorEnv, scale: float, shift: float):
        super().__init__(env)
        self._scale = float(scale)
        self._shift = float(shift)

    def step(self, actions: Any):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        reward = np.asarray(reward, dtype=np.float64) * self._scale + self._shift
        return obs, reward, terminated, truncated, info


class EpisodeStatisticsWrapper(gym.vector.VectorWrapper):
    """Track per-episode length and cumulative reward; inject into ``info`` at episode boundaries."""

    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__(env)
        n = env.num_envs
        self._episode_length = np.zeros(n, dtype=np.int64)
        self._episode_return = np.zeros(n, dtype=np.float64)
        self._prev_dones = np.zeros(n, dtype=np.bool_)

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        n = self.num_envs
        self._episode_length[:] = 0
        self._episode_return[:] = 0.0
        self._prev_dones[:] = False
        info = dict(info)
        info["episode_length"] = np.full(n, np.nan, dtype=np.float64)
        info["episode_cum_reward"] = np.full(n, np.nan, dtype=np.float64)
        return obs, info

    def step(self, actions: Any):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        dones = np.asarray(terminated, dtype=np.bool_) | np.asarray(truncated, dtype=np.bool_)
        is_reset_frame = _is_reset_frame(info, self.num_envs)
        self._episode_length[self._prev_dones] = 0
        self._episode_return[self._prev_dones] = 0.0
        step_mask = ~is_reset_frame
        self._episode_length[step_mask] += 1
        self._episode_return[step_mask] += np.asarray(reward, dtype=np.float64)[step_mask]
        episode_length_out = np.full(self.num_envs, np.nan, dtype=np.float64)
        episode_return_out = np.full(self.num_envs, np.nan, dtype=np.float64)
        episode_length_out[dones] = self._episode_length[dones].astype(np.float64)
        episode_return_out[dones] = self._episode_return[dones]
        info = dict(info)
        info["episode_length"] = episode_length_out
        info["episode_cum_reward"] = episode_return_out
        self._prev_dones = dones
        return obs, reward, terminated, truncated, info


class StepCounterWrapper(gym.vector.VectorWrapper):
    """Track per-env step counters and inject them into ``info``."""

    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__(env)
        n = env.num_envs
        self._episode_time = np.zeros(n, dtype=np.int64)
        self._episode_index = np.zeros(n, dtype=np.int64)
        self._prev_dones = np.zeros(n, dtype=np.bool_)

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._episode_time[:] = 0
        self._prev_dones[:] = False
        info = dict(info)
        info["episode_time"] = self._episode_time.copy()
        info["episode_index"] = self._episode_index.copy()
        return obs, info

    def step(self, actions: Any):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        dones = np.asarray(terminated, dtype=np.bool_) | np.asarray(truncated, dtype=np.bool_)
        is_reset_frame = _is_reset_frame(info, self.num_envs)
        self._episode_time[self._prev_dones] = 0
        self._episode_index[self._prev_dones] += 1
        self._episode_time[~is_reset_frame] += 1
        info = dict(info)
        info["episode_time"] = self._episode_time.copy()
        info["episode_index"] = self._episode_index.copy()
        self._prev_dones = dones
        return obs, reward, terminated, truncated, info


class XformedRewardWrapper(gym.vector.VectorWrapper):
    """Compute and inject a normalised reward signal into ``info``."""

    def __init__(self, env: gym.vector.VectorEnv, max_steps: int):
        super().__init__(env)
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        self._max_steps = float(max_steps)
        n = env.num_envs
        self._episode_reward_sum = np.zeros(n, dtype=np.float64)
        self._prev_dones = np.zeros(n, dtype=np.bool_)

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._episode_reward_sum[:] = 0.0
        self._prev_dones[:] = False
        info = dict(info)
        info["xformed_reward"] = np.zeros(self.num_envs, dtype=np.float64)
        return obs, info

    def step(self, actions: Any):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        dones = np.asarray(terminated, dtype=np.bool_) | np.asarray(truncated, dtype=np.bool_)
        is_reset_frame = _is_reset_frame(info, self.num_envs)
        self._episode_reward_sum[self._prev_dones] = 0.0
        r = np.asarray(reward, dtype=np.float64)
        self._episode_reward_sum[~is_reset_frame] += r[~is_reset_frame]
        episode_time = np.asarray(info["episode_time"], dtype=np.float64)
        xformed = (self._episode_reward_sum + (episode_time - 1.0) * r) / self._max_steps
        xformed[is_reset_frame] = 0.0
        info = dict(info)
        info["xformed_reward"] = xformed
        self._prev_dones = dones
        return obs, reward, terminated, truncated, info


class DoneEncodingWrapper(gym.vector.VectorWrapper):
    """Encode episode termination status as an integer into ``info["done"]``."""

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["done"] = np.zeros(self.num_envs, dtype=np.int64)
        return obs, info

    def step(self, actions: Any):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        terminated = np.asarray(terminated, dtype=np.bool_)
        truncated = np.asarray(truncated, dtype=np.bool_)
        done_int = np.zeros(self.num_envs, dtype=np.int64)
        done_int[truncated] = 2
        done_int[terminated] = 1
        info = dict(info)
        info["done"] = done_int
        return obs, reward, terminated, truncated, info


class EnvIdentityWrapper(gym.vector.VectorWrapper):
    """Inject environment identity into ``info`` and expose convenience attributes."""

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        group_id: str,
        env_seed: int,
        obs_key: str,
        group_ids: list[str] | None = None,
    ):
        super().__init__(env)
        if group_ids is not None:
            if len(group_ids) != env.num_envs:
                raise ValueError(
                    f"group_ids has {len(group_ids)} entries but num_envs={env.num_envs}."
                )
            self._group_id_arr = np.array(group_ids)
        else:
            self._group_id_arr = np.full((env.num_envs,), group_id)
        self._env_idx_arr = np.arange(env.num_envs, dtype=np.int64)
        self.env_seed = int(env_seed)
        self.obs_key = obs_key

    @property
    def action_dim(self) -> int:
        space = self.single_action_space
        if isinstance(space, gym.spaces.Discrete):
            return int(space.n)
        if isinstance(space, gym.spaces.Box):
            return int(np.prod(space.shape))
        if isinstance(space, gym.spaces.MultiDiscrete):
            return len(space.nvec)
        return int(getattr(space, "n", 0))

    def sample_random_actions(self) -> np.ndarray:
        return np.asarray(self.action_space.sample())

    def _inject(self, info: dict[str, Any]) -> dict[str, Any]:
        info = dict(info)
        info["group_id"] = self._group_id_arr.copy()
        info["env_idx"] = self._env_idx_arr.copy()
        return info

    def reset(self, **kwargs: Any):
        if "seed" not in kwargs:
            kwargs["seed"] = self.env_seed
        obs, info = self.env.reset(**kwargs)
        return obs, self._inject(info)

    def step(self, actions: Any):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return obs, reward, terminated, truncated, self._inject(info)


class QStarWrapper(gym.vector.VectorWrapper):
    """Inject expert Q-values into ``info["metadata_q_star"]`` after each step and reset."""

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        group_id: str,
        q_star_source: dict[str, Any],
        obs_key: str,
    ):
        from mouse.envs.custom.action_star import build_q_star_source_adapter

        super().__init__(env)
        self._adapter = build_q_star_source_adapter(
            env_id=group_id,
            q_star_source=q_star_source,
            obs_key=obs_key,
            single_observation_space=env.single_observation_space,
        )
        self._action_dim = int(getattr(env.single_action_space, "n", 0))

    @property
    def obs_key(self) -> str:
        return self.env.obs_key

    @property
    def env_seed(self) -> int:
        return self.env.env_seed

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    def sample_random_actions(self) -> np.ndarray:
        return self.env.sample_random_actions()

    def _attach(
        self,
        obs: Any,
        info: dict[str, Any],
        done_mask: np.ndarray | None,
    ) -> dict[str, Any]:
        if self._adapter is None:
            return info
        q_star = self._adapter.q_star_from_infos(infos=info, num_envs=self.num_envs)
        if q_star is None:
            q_star = self._adapter.q_star_from_observation(
                obs=np.asarray(obs), done_mask=done_mask
            )
        if q_star is None:
            q_star = self._adapter.q_star_from_action_star_infos(
                infos=info, num_envs=self.num_envs, num_actions=self._action_dim
            )
        if q_star is None:
            ast = self._adapter.action_star_from_observation(
                obs=np.asarray(obs), done_mask=done_mask
            )
            if ast is not None:
                ast_arr = np.asarray(ast, dtype=np.int64).reshape(-1)
                if ast_arr.shape[0] != self.num_envs:
                    raise ValueError(
                        f"expert policy returned shape {ast_arr.shape}, "
                        f"expected first dim {self.num_envs}."
                    )
                from mouse.envs.custom.action_star import action_star_to_one_hot_q_star

                q_star = action_star_to_one_hot_q_star(
                    actions=ast_arr, num_actions=self._action_dim
                )
        if q_star is not None:
            info = dict(info)
            info["metadata_q_star"] = np.asarray(q_star, dtype=np.float64)
        return info

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        return obs, self._attach(obs, info, done_mask=None)

    def step(self, actions: Any):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        dones = np.asarray(terminated, dtype=np.bool_) | np.asarray(truncated, dtype=np.bool_)
        return obs, reward, terminated, truncated, self._attach(obs, info, done_mask=dones)


class ConstructionSeedWrapper(gym.Wrapper):
    """Control construction-time and per-episode reset seeds for custom MDPs."""

    def __init__(self, env_fn: Callable[[int], gym.Env], seed: int):
        super().__init__(env_fn(seed))
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        return self.env.reset(seed=int(self._rng.integers(0, 2**31)), options=options)


# -----------------------------------------------------------------------------
# Stack factory
# -----------------------------------------------------------------------------


def build_vector_env_stack(
    env_fns: list,
    group_id: str,
    seed: int,
    max_steps_per_episode: int,
    obs_key: str = "observation",
    reward_scale: float = 1.0,
    reward_shift: float = 0.0,
    q_star_source: dict[str, Any] | None = None,
    group_ids: list[str] | None = None,
) -> gym.vector.VectorEnv:
    """Compose the standard vector-env wrapper stack around a ``SyncVectorEnv``."""
    env: gym.vector.VectorEnv = SyncVectorEnv(
        env_fns,
        copy=True,
        observation_mode="different",
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )

    env = AutoresetFrameWrapper(env)

    resolved_obs_key = resolve_obs_key(env, requested=obs_key)

    env = EpisodeStatisticsWrapper(env)
    env = _RewardTransformWrapper(env, scale=reward_scale, shift=reward_shift)
    env = StepCounterWrapper(env)
    env = XformedRewardWrapper(env, max_steps=max_steps_per_episode)
    env = DoneEncodingWrapper(env)
    env = EnvIdentityWrapper(
        env, group_id=group_id, env_seed=seed, obs_key=resolved_obs_key, group_ids=group_ids
    )
    if q_star_source is not None:
        env = QStarWrapper(
            env, group_id=group_id, q_star_source=q_star_source, obs_key=resolved_obs_key
        )
    return env
