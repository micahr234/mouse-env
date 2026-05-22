"""Shared base classes for vector env runners."""

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv
from gymnasium.core import ObservationWrapper

from mouse.envs.action_star import (
    action_star_to_one_hot_q_star,
    build_q_star_source_adapter,
)


class ObservationSliceWrapper(ObservationWrapper):
    """Slice the observation to a subset of indices and update observation_space.

    Useful for making CartPole partially observable: use indices [0, 2] to keep
    only cart position and pole angle, removing linear and angular velocity.
    """

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


class _BaseVectorEnv:
    """Shared logic for vector env runners: SyncVectorEnv setup, step loop, and discrete-action API."""

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        env_id: str,
        env_name: str,
        num_envs: int,
        render: bool,
        seed: int,
        max_steps_per_episode: int,
        obs_key: str = "observation",
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
        q_star_source: dict[str, Any] | None = None,
    ):
        self.env_id = env_id
        self.env_name = env_name
        self.num_envs = num_envs
        self.render = render
        self.env_seed = seed
        self._obs_key = obs_key
        self.env = SyncVectorEnv(
            env_fns,
            copy=True,
            observation_mode="different",
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
        )
        self.action_space = self.env.action_space
        self.single_action_space = self.env.single_action_space
        self._obs_key = self._resolve_obs_key(obs_key)
        if isinstance(self.single_action_space, gym.spaces.Box):
            raise ValueError("Only discrete action spaces are supported.")
        self.action_dim = int(getattr(self.single_action_space, "n", 0))
        self.init = True
        if max_steps_per_episode <= 0:
            raise ValueError(f"max_steps_per_episode must be positive, got {max_steps_per_episode}")
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.reward_scale = float(reward_scale)
        self.reward_shift = float(reward_shift)
        self._q_star_source = build_q_star_source_adapter(
            env_id=env_id,
            q_star_source=q_star_source,
            obs_key=self._obs_key,
            single_observation_space=self.env.single_observation_space,
        )

    def _resolve_obs_key(self, requested_obs_key: str) -> str:
        """Pick canonical observation key from env observation space."""
        def _is_discrete_like(space: gym.Space) -> bool:
            if isinstance(
                space,
                (gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary),
            ):
                return True
            if isinstance(space, gym.spaces.Tuple):
                return all(_is_discrete_like(s) for s in space.spaces)
            if isinstance(space, gym.spaces.Dict):
                return all(_is_discrete_like(s) for s in space.spaces.values())
            if isinstance(space, gym.spaces.Box):
                obs_dtype = np.dtype(space.dtype)
                return bool(
                    np.issubdtype(obs_dtype, np.integer)
                    or np.issubdtype(obs_dtype, np.bool_)
                )
            return False

        if requested_obs_key == "observation_image":
            return requested_obs_key
        obs_space = self.env.single_observation_space
        if _is_discrete_like(obs_space):
            return "observation_discrete"
        return requested_obs_key

    def _format_observation_fields(self, obs: Any) -> dict[str, np.ndarray]:
        """Convert raw env observations to store schema fields."""
        if self._obs_key == "observation_image":
            return {"observation_image": np.asarray(obs)}
        if self._obs_key == "observation_discrete":
            # Gymnasium vector envs may emit tuple-structured discrete observations
            # as (feature, env) instead of (env, feature). Canonicalize to [N, ...].
            if isinstance(obs, (tuple, list)) and len(obs) > 0:
                parts = [np.asarray(x, dtype=np.int64) for x in obs]
                if all(p.ndim >= 1 and p.shape[0] == self.num_envs for p in parts):
                    return {"observation_discrete": np.stack(parts, axis=-1)}

            arr = np.asarray(obs, dtype=np.int64)
            if arr.ndim == 0:
                arr = np.full((self.num_envs,), int(arr), dtype=np.int64)
                return {"observation_discrete": arr}
            if arr.ndim >= 2 and arr.shape[0] != self.num_envs and arr.shape[1] == self.num_envs:
                arr = np.swapaxes(arr, 0, 1)
            return {"observation_discrete": arr.ravel()}
        return {"observation": np.asarray(obs)}

    def _attach_expert_metadata(
        self,
        data: dict[str, Any],
        infos: dict[str, Any],
        obs_fields: dict[str, np.ndarray],
        done_mask: np.ndarray | None,
    ) -> None:
        """Attach ``metadata_q_star`` only.

        Uses env/tabular Q when available, else ``predict_q``, else one-hot encoding of
        ``action_star`` from infos or from ``predict``. Discrete hints are never exposed as a
        separate ``metadata_action_star`` field.

        Labels refer to the **same** observation as ``obs_fields`` (post-reset or post-step).
        """
        if self._q_star_source is None:
            return
        q_star = self._q_star_source.q_star_from_infos(
            infos=infos,
            num_envs=self.num_envs,
        )
        if q_star is None and self._obs_key in obs_fields:
            q_star = self._q_star_source.q_star_from_observation(
                obs=np.asarray(obs_fields[self._obs_key]),
                done_mask=done_mask,
            )
        if q_star is None:
            q_star = self._q_star_source.q_star_from_action_star_infos(
                infos=infos,
                num_envs=self.num_envs,
                num_actions=self.action_dim,
            )
        if q_star is None and self._obs_key in obs_fields:
            ast = self._q_star_source.action_star_from_observation(
                obs=np.asarray(obs_fields[self._obs_key]),
                done_mask=done_mask,
            )
            if ast is not None:
                ast_arr = np.asarray(ast, dtype=np.int64).reshape(-1)
                if ast_arr.shape[0] != self.num_envs:
                    raise ValueError(
                        "expert policy produced a discrete action batch with shape "
                        f"{ast_arr.shape}, expected first dim {self.num_envs}."
                    )
                q_star = action_star_to_one_hot_q_star(
                    actions=ast_arr,
                    num_actions=self.action_dim,
                )
        if q_star is not None:
            data["metadata_q_star"] = np.asarray(q_star, dtype=np.float64)

    def sample_random_actions(self) -> np.ndarray:
        return np.asarray(self.action_space.sample(), dtype=np.int64)

    def step(
        self, actions: np.ndarray | None = None
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Step all envs. Returns ``(data, metrics)``.

        When ``q_star_source`` is configured, ``data`` may include ``metadata_q_star``
        (true Q from infos / ``predict_q``, or one-hot Q from discrete ``action_star`` /
        ``predict``). Test runners read this key on ``last_data`` for ``action_source='q_star'``.
        """
        if self.init:
            self.init = False
            self.episode_step = np.zeros((self.num_envs,), dtype=np.int64)
            self.global_steps = np.zeros((self.num_envs,), dtype=np.int64)
            self.episode_reward_sum = np.zeros((self.num_envs,), dtype=np.float64)
            self.episode_reward_sum_raw = np.zeros((self.num_envs,), dtype=np.float64)
            actions = self.sample_random_actions()
            obs, infos = self.env.reset(seed=self.env_seed)
            self.dones = np.zeros((self.num_envs,), dtype=np.bool_)
            data = {
                "env_name": np.full((self.num_envs,), self.env_name),
                "env_idx": np.arange(self.num_envs),
                "global_step": self.global_steps,
                "episode_step": self.episode_step,
                "action": actions,
                "reward": np.zeros((self.num_envs,), dtype=np.float64),
                "xformed_reward": np.zeros((self.num_envs,), dtype=np.float64),
                "done": np.zeros((self.num_envs,), dtype=np.int64),
            }
            obs_fields = self._format_observation_fields(obs)
            data.update(obs_fields)
            data.update(
                {
                    f"metadata_{k}": v
                    for k, v in infos.items()
                    if not k.startswith("_")
                }
            )
            self._attach_expert_metadata(
                data=data,
                infos=infos,
                obs_fields=obs_fields,
                done_mask=None,
            )
            metrics = {
                "episode_length": np.full((self.num_envs,), np.nan, dtype=np.float64),
                "episode_cum_reward": np.full((self.num_envs,), np.nan, dtype=np.float64),
            }
            if self.render:
                self.env.render()
            return data, metrics

        else:
            self.episode_step += 1
            self.episode_step[self.dones] = 0
            self.episode_reward_sum[self.dones] = 0.0
            self.episode_reward_sum_raw[self.dones] = 0.0
            self.global_steps += 1
            if actions is None:
                actions = self.sample_random_actions()
            else:
                actions = np.asarray(actions, dtype=np.int64)
            obs, raw_rewards, terminations, truncations, infos = self.env.step(actions)
            terminations = np.asarray(terminations, dtype=np.bool_)
            truncations = np.asarray(truncations, dtype=np.bool_)
            self.dones = terminations | truncations
            # 0=not done, 1=terminal (natural end), 2=truncated (time limit / external)
            # Termination takes priority: if both fire simultaneously, we record terminal.
            done_int = np.zeros(self.num_envs, dtype=np.int64)
            done_int[truncations] = 2
            done_int[terminations] = 1  # overwrites 2 where both are set
            raw_rewards = np.asarray(raw_rewards, dtype=np.float64)
            ss_rewards = raw_rewards * self.reward_scale + self.reward_shift
            self.episode_reward_sum += ss_rewards
            self.episode_reward_sum_raw += raw_rewards
            xformed_rewards = (self.episode_reward_sum + (self.episode_step.astype(np.float64) - 1.0) * ss_rewards) / self.max_steps_per_episode
            data = {
                "env_name": np.full((self.num_envs,), self.env_name),
                "env_idx": np.arange(self.num_envs),
                "global_step": self.global_steps,
                "episode_step": self.episode_step,
                "action": actions,
                "reward": ss_rewards,
                "xformed_reward": xformed_rewards,
                "done": done_int,
            }
            obs_fields = self._format_observation_fields(obs)
            data.update(obs_fields)
            data.update(
                {
                    f"metadata_{k}": v
                    for k, v in infos.items()
                    if not k.startswith("_")
                }
            )
            self._attach_expert_metadata(
                data=data,
                infos=infos,
                obs_fields=obs_fields,
                done_mask=self.dones,
            )
            metrics = {
                "episode_length": np.full((self.num_envs,), np.nan, dtype=np.float64),
                "episode_cum_reward": np.full((self.num_envs,), np.nan, dtype=np.float64),
            }
            metrics["episode_length"][self.dones] = self.episode_step[self.dones].astype(np.float64)
            metrics["episode_cum_reward"][self.dones] = self.episode_reward_sum_raw[self.dones]
            if self.render:
                self.env.render()
            return data, metrics

    def close(self) -> None:
        self.env.close()
