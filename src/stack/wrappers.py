"""Vector env wrappers and stack factory."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv

from mouse.envs.stack.obs import resolve_obs_key

# -----------------------------------------------------------------------------
# Vector env wrappers
# -----------------------------------------------------------------------------


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
    """Track per-episode length and cumulative reward; inject into ``info`` at episode boundaries.

    Must be placed *innermost* (directly around ``SyncVectorEnv``) so that it sees the
    raw (unscaled) rewards before any reward-transform wrapper. This ensures
    ``info["episode_cum_reward"]`` always reflects the true environment return.

    Injected info keys:

    - ``"episode_length"`` — ``float64[num_envs]``: episode length in steps, ``NaN``
      for envs that did not finish this step.
    - ``"episode_cum_reward"`` — ``float64[num_envs]``: cumulative raw reward for the
      episode that just ended, ``NaN`` for running envs.

    Both keys are also present (all ``NaN``) after ``reset()``.
    """

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
        # Reset accumulators for envs that started a new episode this step
        # (i.e. that were done on the previous step, given NEXT_STEP autoreset).
        self._episode_length[self._prev_dones] = 0
        self._episode_return[self._prev_dones] = 0.0
        self._episode_length += 1
        self._episode_return += np.asarray(reward, dtype=np.float64)
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
    """Track per-env step counters and inject them into ``info``.

    Injected info keys:

    - ``"episode_step"`` — ``int64[num_envs]``: step index within the current episode
      (starts at 1 on the first step; resets to 1 at the start of each new episode).
    - ``"global_step"`` — ``int64[num_envs]``: monotonically increasing count of all
      steps taken by each env since the last ``reset()``.

    Both keys are set to zeros in ``reset()`` info as a pre-step baseline.
    ``XformedRewardWrapper`` depends on ``"episode_step"`` from this wrapper and must
    be placed *outside* it in the stack.
    """

    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__(env)
        n = env.num_envs
        self._episode_step = np.zeros(n, dtype=np.int64)
        self._global_step = np.zeros(n, dtype=np.int64)
        self._prev_dones = np.zeros(n, dtype=np.bool_)

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._episode_step[:] = 0
        self._global_step[:] = 0
        self._prev_dones[:] = False
        info = dict(info)
        info["episode_step"] = self._episode_step.copy()
        info["global_step"] = self._global_step.copy()
        return obs, info

    def step(self, actions: Any):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        dones = np.asarray(terminated, dtype=np.bool_) | np.asarray(truncated, dtype=np.bool_)
        self._episode_step[self._prev_dones] = 0
        self._episode_step += 1
        self._global_step += 1
        info = dict(info)
        info["episode_step"] = self._episode_step.copy()
        info["global_step"] = self._global_step.copy()
        self._prev_dones = dones
        return obs, reward, terminated, truncated, info


class XformedRewardWrapper(gym.vector.VectorWrapper):
    """Compute and inject a normalised reward signal into ``info``.

    The formula is:

    .. code-block:: text

        xformed_reward = (episode_reward_sum + (episode_step - 1) * r) / max_steps

    This approximates the *average* reward up to and including the current step,
    normalised by the episode length budget. It provides a smoother training signal
    than raw rewards for agents that attend over step histories.

    Tracks scaled (post-transform) rewards. Reads ``info["episode_step"]`` injected
    by :class:`StepCounterWrapper`, which must therefore be *inside* this wrapper.

    Injected info key:

    - ``"xformed_reward"`` — ``float64[num_envs]``: normalised reward for this step.

    Args:
        env: The env to wrap.
        max_steps: Episode step budget used as the normalisation denominator.

    Raises:
        ValueError: If ``max_steps <= 0``.
    """

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
        self._episode_reward_sum[self._prev_dones] = 0.0
        r = np.asarray(reward, dtype=np.float64)
        self._episode_reward_sum += r
        episode_step = np.asarray(info["episode_step"], dtype=np.float64)
        xformed = (self._episode_reward_sum + (episode_step - 1.0) * r) / self._max_steps
        info = dict(info)
        info["xformed_reward"] = xformed
        self._prev_dones = dones
        return obs, reward, terminated, truncated, info


class DoneEncodingWrapper(gym.vector.VectorWrapper):
    """Encode episode termination status as an integer into ``info["done"]``.

    Gymnasium exposes two Boolean flags (``terminated``, ``truncated``). This wrapper
    collapses them into a single integer for compact storage in rollout datasets:

    - ``0`` — episode is still running.
    - ``1`` — episode terminated naturally (reward signal is valid).
    - ``2`` — episode was truncated by the time limit (no terminal reward).

    When both ``terminated`` and ``truncated`` are ``True`` simultaneously, ``1``
    takes priority.

    Injected info key:

    - ``"done"`` — ``int64[num_envs]``: termination code for this step.
    """

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
        done_int[terminated] = 1  # overwrites 2 where both fire
        info = dict(info)
        info["done"] = done_int
        return obs, reward, terminated, truncated, info


class EnvIdentityWrapper(gym.vector.VectorWrapper):
    """Inject environment identity into ``info`` and expose convenience attributes.

    This is the outermost wrapper in the standard stack (before the optional
    :class:`QStarWrapper`). It is the primary interface through which runner code
    interacts with the env object.

    Injected info keys:

    - ``"env_name"`` — ``str[num_envs]``: the environment name string.
    - ``"env_idx"`` — ``int64[num_envs]``: integer index of each parallel env (0 … N-1).

    Attributes:
        env_seed (int): Seed used for seeding ``reset()`` automatically.
        obs_key (str): Canonical observation key — one of ``"observation"``,
            ``"observation_discrete"``, or ``"observation_image"``.
        action_dim (int): Number of discrete actions in the action space.

    Args:
        env: The env to wrap.
        env_name: Name string injected into ``info["env_name"]`` each step.
        env_seed: Seed used for automatic seeding when ``reset()`` is called
            without an explicit ``seed`` argument.
        obs_key: Canonical observation key for this env (already resolved by
            :func:`resolve_obs_key`).
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        env_name: str,
        env_seed: int,
        obs_key: str,
    ):
        super().__init__(env)
        self._env_name_arr = np.full((env.num_envs,), env_name)
        self._env_idx_arr = np.arange(env.num_envs, dtype=np.int64)
        self.env_seed = int(env_seed)
        self.obs_key = obs_key

    @property
    def action_dim(self) -> int:
        """Number of discrete actions (``action_space.n``)."""
        return int(getattr(self.single_action_space, "n", 0))

    def sample_random_actions(self) -> np.ndarray:
        """Sample a random action for each parallel env.

        Returns:
            ``int64`` array of shape ``(num_envs,)`` with uniformly sampled actions.
        """
        return np.asarray(self.action_space.sample(), dtype=np.int64)

    def _inject(self, info: dict[str, Any]) -> dict[str, Any]:
        info = dict(info)
        info["env_name"] = self._env_name_arr.copy()
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
    """Inject expert Q-values into ``info["metadata_q_star"]`` after each step and reset.

    Uses an :class:`~mouse.envs.action_star.ExpertPolicyAdapter` built from ``q_star_source``
    to derive Q-values. Sources are tried in priority order until one succeeds:

    1. ``q_star`` key already present in env info (custom envs with ``emit_q_star=True``).
    2. ``predict_q(obs)`` on the external policy (e.g. DQN from SB3).
    3. ``action_star`` key in env info → one-hot encoded Q-values.
    4. ``external_policy.predict(obs)`` → one-hot encoded Q-values.

    If no source produces values, ``metadata_q_star`` is omitted silently.

    Injected info key:

    - ``"metadata_q_star"`` — ``float64[num_envs, action_dim]``: expert Q-values,
      shape ``(num_envs, action_dim)``.

    Args:
        env: The env to wrap.
        env_id: Environment id string used to select the correct adapter.
        q_star_source: Expert source config dict; see :class:`~mouse.envs.config.EnvConfig` for keys.
        obs_key: Canonical observation key used by the adapter when querying the policy.
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        env_id: str,
        q_star_source: dict[str, Any],
        obs_key: str,
    ):
        from mouse.envs.action_star import build_q_star_source_adapter

        super().__init__(env)
        self._adapter = build_q_star_source_adapter(
            env_id=env_id,
            q_star_source=q_star_source,
            obs_key=obs_key,
            single_observation_space=env.single_observation_space,
        )
        self._action_dim = int(getattr(env.single_action_space, "n", 0))

    @property
    def obs_key(self) -> str:
        """Delegated from :class:`EnvIdentityWrapper` when present."""
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
                from mouse.envs.action_star import action_star_to_one_hot_q_star

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


# -----------------------------------------------------------------------------
# Stack factory
# -----------------------------------------------------------------------------


def build_vector_env_stack(
    env_fns: list,
    env_id: str,
    env_name: str,
    seed: int,
    max_steps_per_episode: int,
    obs_key: str = "observation",
    reward_scale: float = 1.0,
    reward_shift: float = 0.0,
    q_star_source: dict[str, Any] | None = None,
) -> gym.vector.VectorEnv:
    """Compose the standard vector-env wrapper stack around a ``SyncVectorEnv``.

    Wrapper order (inner → outer):

    .. code-block:: text

        SyncVectorEnv
          → EpisodeStatisticsWrapper   raw-reward episode stats
          → _RewardTransformWrapper    reward * scale + shift
          → StepCounterWrapper         episode_step, global_step in info
          → XformedRewardWrapper       xformed_reward in info
          → DoneEncodingWrapper        done as {0, 1, 2} in info
          → EnvIdentityWrapper         env_name, env_idx, seed, obs_key, action_dim
          → QStarWrapper               metadata_q_star (only when q_star_source is set)

    The returned env exposes the standard ``(obs, reward, terminated, truncated, info)``
    step API. Call ``env.reset()`` once before the first ``env.step()``; ``reset()``
    seeds automatically from the stored ``env_seed`` on :class:`EnvIdentityWrapper`.

    Args:
        env_fns: List of zero-argument callables, each returning a single ``gym.Env``.
            Length determines ``num_envs``.
        env_id: Environment id passed to :class:`QStarWrapper` for adapter selection.
        env_name: Name string injected into ``info["env_name"]`` each step.
        seed: Seed stored on :class:`EnvIdentityWrapper` for automatic ``reset()`` seeding.
        max_steps_per_episode: Passed to :class:`XformedRewardWrapper` as the normalisation
            denominator.
        obs_key: Observation key hint; resolved against the actual observation space by
            :func:`resolve_obs_key` (discrete spaces override to ``"observation_discrete"``).
        reward_scale: Reward multiplier applied by ``_RewardTransformWrapper``.
        reward_shift: Reward offset applied after scaling.
        q_star_source: Expert source config. When non-``None``, :class:`QStarWrapper` is
            added as the outermost layer.

    Returns:
        A fully wrapped ``gym.vector.VectorEnv``.

    Raises:
        ValueError: If the action space is continuous (``Box``); only discrete actions
            are supported.
    """
    env: gym.vector.VectorEnv = SyncVectorEnv(
        env_fns,
        copy=True,
        observation_mode="different",
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )
    if isinstance(env.single_action_space, gym.spaces.Box):
        raise ValueError("Only discrete action spaces are supported.")

    # Resolve the canonical obs key now that we have the observation space.
    resolved_obs_key = resolve_obs_key(env, requested=obs_key)

    # Episode stats must be innermost so they see raw (unscaled) rewards.
    env = EpisodeStatisticsWrapper(env)
    env = _RewardTransformWrapper(env, scale=reward_scale, shift=reward_shift)
    env = StepCounterWrapper(env)
    # XformedRewardWrapper reads episode_step from StepCounterWrapper (inner).
    env = XformedRewardWrapper(env, max_steps=max_steps_per_episode)
    env = DoneEncodingWrapper(env)
    env = EnvIdentityWrapper(env, env_name=env_name, env_seed=seed, obs_key=resolved_obs_key)
    if q_star_source is not None:
        env = QStarWrapper(env, env_id=env_id, q_star_source=q_star_source, obs_key=resolved_obs_key)
    return env
