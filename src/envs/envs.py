"""Environment factory and wrappers for NS-Gym and plain Gymnasium envs.

Provides:
- EnvConfig: dataclass for a single test/rollout environment
- make_vector_env: unified factory returning NSVectorEnv or PlainVectorEnv
- NS-Gym: non-stationary classic control (CartPole, MountainCar, etc.) with configurable param schedules
- Plain: standard vector-obs envs (same IDs) or Atari (ALE/*) with optional image obs
"""

from dataclasses import dataclass
from typing import Any, Callable, cast


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
from gymnasium.vector import SyncVectorEnv
from gymnasium.core import ObservationWrapper
from gymnasium.wrappers import AtariPreprocessing
from ns_gym.schedulers import (
    ContinuousScheduler,
    DiscreteScheduler,
    MemorylessScheduler,
    PeriodicScheduler,
    RandomScheduler,
)
from ns_gym.update_functions import (
    DeterministicTrend,
    DistributionCyclicUpdate,
    DistributionDecrementUpdate,
    DistributionIncrementUpdate,
    DistributionNoUpdate,
    DistributionStepWiseUpdate,
    ExponentialDecay,
    GeometricProgression,
    IncrementUpdate,
    NoUpdate,
    OscillatingUpdate,
    RandomWalk,
    RandomWalkWithDrift,
    RandomWalkWithDriftAndTrend,
    StepWiseUpdate,
)
from ns_gym.wrappers import NSClassicControlWrapper, NSCliffWalkingWrapper, NSFrozenLakeWrapper

from mouse.envs.action_star import (
    action_star_to_one_hot_q_star,
    apply_q_star_source_env_kwargs,
    build_q_star_source_adapter,
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


def is_ns_gym_env(
    env_id: str,
    non_stationary_params: dict[str, Any] | None,
    env_type: str | None,
) -> bool:
    """True iff env_id uses the NS- prefix (prefix-only NS routing)."""
    _ = non_stationary_params
    _ = env_type
    return env_id.startswith("NS-")


def normalize_env_id(env_id: str) -> str:
    """Strip routing prefixes before calling gym.make(...)."""
    if env_id.startswith("NS-"):
        return env_id[3:]
    return env_id


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
# NS-Gym: wrappers, update functions, single env, vector env
# -----------------------------------------------------------------------------

_NS_INFO_KEYS = ("Ground Truth Env Change", "Ground Truth Delta Change")

_NS_WRAPPER = {
    "CartPole-v1": NSClassicControlWrapper,
    "Acrobot-v1": NSClassicControlWrapper,
    "MountainCar-v0": NSClassicControlWrapper,
    "MountainCarContinuous-v0": NSClassicControlWrapper,
    "Pendulum-v1": NSClassicControlWrapper,
    "FrozenLake-v1": NSFrozenLakeWrapper,
    "CliffWalking-v1": NSCliffWalkingWrapper,
}

# Envs whose tunable params are non-scalar (e.g. transition-prob lists for grid worlds).
# ns_gym's base._get_delta_change does `updated_param - param`, which fails for lists,
# so delta_change_notification must be disabled for these.
_NS_NO_DELTA_CHANGE = {"FrozenLake-v1", "CliffWalking-v1"}

# Concrete scheduler types we instantiate; used for runtime check before passing to update functions
_scheduler_types = (
    ContinuousScheduler,
    DiscreteScheduler,
    MemorylessScheduler,
    PeriodicScheduler,
    RandomScheduler,
)


def _filter_ns_info(info: dict[str, Any]) -> dict[str, Any]:
    """Return only ground-truth env change and delta change from env info."""
    return {k: info.get(k, {}) for k in _NS_INFO_KEYS}


def extract_ns_params(infos: dict[str, Any]) -> dict[str, Any]:
    """Extract non-stationary parameters from env info (Ground Truth Env Change and Ground Truth Delta Change)."""
    ns_params = {}
    env_change = infos.get("Ground Truth Env Change", {})
    delta_change = infos.get("Ground Truth Delta Change", {})
    for k, flag in env_change.items():
        if not k.startswith("_"):
            flag = np.asarray(flag)
            quantity = np.asarray(delta_change.get(k, 0))
            ns_params[f"{k}_flag"] = flag
            ns_params[k] = quantity
    return ns_params



class NSGymInterfaceWrapper(gym.Wrapper):
    """Makes an NS-Gym env look like a standard Gymnasium env: observation is only the state, info is extracted NS params.

    Wraps NS-Gym environments so that:
    - observation: only the observation state entry (obs["state"] from the inner env).
    - info: extracted non-stationary params (same shape as extract_ns_params), i.e. keys like "{param}_flag" and "{param}".

    Other observation dict keys and raw "Ground Truth *" info keys are not exposed; extraction runs inside the wrapper.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        inner_obs_space = env.observation_space
        if isinstance(inner_obs_space, gym.spaces.Dict) and "state" in inner_obs_space.spaces:
            self.observation_space = inner_obs_space["state"]
        else:
            self.observation_space = inner_obs_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        state = obs["state"] if isinstance(obs, dict) and "state" in obs else obs
        info = extract_ns_params(_filter_ns_info(info))
        return state, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = obs["state"] if isinstance(obs, dict) and "state" in obs else obs
        info = extract_ns_params(_filter_ns_info(info))
        return state, reward, terminated, truncated, info


def create_ns_gym_update_functions(ns_gym_config: dict[str, Any]) -> dict[str, Any]:
    """Build param update functions for ns_gym from config.

    ns_gym_config must be a plain dict (e.g. from config loader); nested
    scheduler_kwargs and update_kwargs must be plain dicts as well.
    """
    if not isinstance(ns_gym_config, dict):
        raise TypeError(
            f"ns_gym_config must be a dict, got {type(ns_gym_config).__name__}."
        )
    param_update_functions = {}
    for param_name, update_config in ns_gym_config.items():
        if not isinstance(update_config, dict):
            raise TypeError(
                f"ns_gym_config[{param_name!r}] must be a dict, got {type(update_config).__name__}."
            )
        scheduler_type = update_config["scheduler"]
        update_func_type = update_config["update_function"]
        scheduler_kwargs_raw = update_config.get("scheduler_kwargs", {}) or {}
        update_kwargs_raw = update_config.get("update_kwargs", {}) or {}
        if not isinstance(scheduler_kwargs_raw, dict):
            raise TypeError(
                f"ns_gym_config[{param_name!r}].scheduler_kwargs must be a plain dict, got {type(scheduler_kwargs_raw).__name__}."
            )
        if not isinstance(update_kwargs_raw, dict):
            raise TypeError(
                f"ns_gym_config[{param_name!r}].update_kwargs must be a plain dict, got {type(update_kwargs_raw).__name__}."
            )
        scheduler_kwargs = scheduler_kwargs_raw
        update_kwargs = update_kwargs_raw
        if scheduler_type == "continuous":
            scheduler = ContinuousScheduler(**scheduler_kwargs)
        elif scheduler_type == "periodic":
            scheduler = PeriodicScheduler(**scheduler_kwargs)
        elif scheduler_type == "random":
            scheduler = RandomScheduler(**scheduler_kwargs)
        elif scheduler_type == "discrete":
            scheduler_kwargs = dict(scheduler_kwargs)
            event_list = scheduler_kwargs.pop("event_list", [])
            scheduler = DiscreteScheduler(set(event_list), **scheduler_kwargs)
        elif scheduler_type == "memoryless":
            scheduler = MemorylessScheduler(**scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        if not isinstance(scheduler, _scheduler_types):
            raise TypeError(
                f"Expected a scheduler instance (one of {[t.__name__ for t in _scheduler_types]}), "
                f"got {type(scheduler).__name__}"
            )
        sched: Any = cast(Any, scheduler)
        if update_func_type == "increment":
            update_func = IncrementUpdate(sched, **update_kwargs)
        elif update_func_type == "random_walk":
            update_func = RandomWalk(sched, **update_kwargs)
        elif update_func_type == "no_update":
            update_func = NoUpdate(sched)
        elif update_func_type == "deterministic_trend":
            update_func = DeterministicTrend(sched, **update_kwargs)
        elif update_func_type == "exponential_decay":
            update_func = ExponentialDecay(sched, **update_kwargs)
        elif update_func_type == "geometric_progression":
            update_func = GeometricProgression(sched, **update_kwargs)
        elif update_func_type == "oscillating":
            update_func = OscillatingUpdate(sched, **update_kwargs)
        elif update_func_type == "random_walk_with_drift":
            update_func = RandomWalkWithDrift(sched, **update_kwargs)
        elif update_func_type == "random_walk_with_drift_and_trend":
            update_func = RandomWalkWithDriftAndTrend(sched, **update_kwargs)
        elif update_func_type == "step_wise":
            update_func = StepWiseUpdate(sched, **update_kwargs)
        elif update_func_type == "distribution_decrement":
            update_func = DistributionDecrementUpdate(sched, **update_kwargs)
        elif update_func_type == "distribution_increment":
            update_func = DistributionIncrementUpdate(sched, **update_kwargs)
        elif update_func_type == "distribution_step_wise":
            update_func = DistributionStepWiseUpdate(sched, **update_kwargs)
        elif update_func_type == "distribution_cyclic":
            update_func = DistributionCyclicUpdate(sched, **update_kwargs)
        elif update_func_type == "distribution_no_update":
            update_func = DistributionNoUpdate(sched)
        else:
            raise ValueError(f"Unknown update function type: {update_func_type}")
        param_update_functions[param_name] = update_func
    return param_update_functions


def make_ns_env(
    env_id: str,
    non_stationary_params: dict[str, Any],
    max_steps_per_episode: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
    render: bool = False,
) -> gym.Env:
    """Create one non-stationary environment with persistent RNG update functions."""
    env_kwargs = env_kwargs or {}
    if render and "render_mode" not in env_kwargs:
        env_kwargs = {**env_kwargs, "render_mode": "human"}

    param_update_functions = create_ns_gym_update_functions(non_stationary_params)
    base_env = gym.make(
        env_id,
        max_episode_steps=max_steps_per_episode,
        **env_kwargs,
    )
    wrapper_cls = _NS_WRAPPER.get(env_id)
    if wrapper_cls is None:
        raise ValueError(f"No NS-Gym wrapper registered for env_id={env_id!r}. Known: {list(_NS_WRAPPER)}")
    return NSGymInterfaceWrapper(wrapper_cls(  # type: ignore
        base_env,
        param_update_functions,
        change_notification=True,
        delta_change_notification=env_id not in _NS_NO_DELTA_CHANGE,
    ))


# -----------------------------------------------------------------------------
# Observation slicing (e.g. CartPole: position + angle only, no velocities)
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Common base for vector env runners
# -----------------------------------------------------------------------------


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


class NSVectorEnv(_BaseVectorEnv):
    """Persistent non-stationary vector env runner with step state."""

    def __init__(
        self,
        env_id: str,
        non_stationary_params: dict[str, Any],
        seed: int,
        max_steps_per_episode: int,
        num_envs: int = 1,
        env_kwargs: dict[str, Any] | None = None,
        render: bool = False,
        env_name: str | None = None,
        observation_indices: list[int] | None = None,
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
        q_star_source: dict[str, Any] | None = None,
    ):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
        env_kwargs = env_kwargs or {}

        def make_env():
            env = make_ns_env(
                env_id=env_id,
                non_stationary_params=non_stationary_params,
                max_steps_per_episode=max_steps_per_episode,
                env_kwargs=env_kwargs,
                render=render,
            )
            if observation_indices is not None:
                env = ObservationSliceWrapper(env=env, indices=observation_indices)
            return env

        super().__init__(
            env_fns=[make_env] * num_envs,
            env_id=env_id,
            env_name=env_name if env_name is not None else env_id,
            num_envs=num_envs,
            render=render,
            seed=seed,
            max_steps_per_episode=max_steps_per_episode,
            obs_key="observation",
            reward_scale=reward_scale,
            reward_shift=reward_shift,
            q_star_source=q_star_source,
        )


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
