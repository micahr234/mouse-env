"""Public step API and rollout contract types for mouse-env ↔ mouse-core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Required, TypedDict, cast

import gymnasium as gym
import numpy as np
import torch

ACTION_KEY = "action"
OBS_KEY = "observation"

TIME_KEY = "time"

DONE_RUNNING             = 0
DONE_EPISODE_TERMINATED  = 1
DONE_EPISODE_TRUNCATED   = 2
DONE_TASK_TERMINATED     = 3
DONE_TASK_TRUNCATED      = 4


def _torch_dtype_for_np_dtype(dtype: Any) -> torch.dtype:
    """Map a numpy dtype to the closest torch dtype available."""
    dt = np.dtype(dtype)
    dtype_map = {
        np.dtype(np.bool_): torch.bool,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.int16): torch.int16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
    }
    if dt in dtype_map:
        return dtype_map[dt]
    if np.issubdtype(dt, np.floating):
        return torch.float32
    if np.issubdtype(dt, np.integer) or np.issubdtype(dt, np.bool_):
        return torch.int64
    return torch.float32


def _torch_dtype_for_space(space: gym.Space) -> torch.dtype:
    """Map a Gymnasium space dtype to the torch dtype used to store its samples."""
    raw = getattr(space, "dtype", None)
    if raw is None:
        return torch.float32
    return _torch_dtype_for_np_dtype(raw)


@dataclass
class FieldSpec:
    """Describes one field in an output or input dict.

    ``dtype`` is the Python/torch type of the value (e.g. ``torch.float32``,
    ``torch.int64``, ``int``, ``float``, ``np.float64``). ``shape`` is the tensor
    shape as a tuple; ``()`` for scalars and plain Python primitives.
    """

    dtype: torch.dtype | type
    shape: tuple[int, ...]


@dataclass
class OutputSpec:
    """Mirrors the output dict: one attribute per key in ``outputs[i]``.

    ``observation`` is a single :class:`FieldSpec` for standard observation spaces, or
    a ``dict[str, FieldSpec]`` for ``gym.spaces.Dict`` observation spaces.

    Every key from the underlying Gymnasium ``info`` dict is forwarded verbatim as
    ``info_<key>`` in the step output. No env-specific filtering is applied.
    """

    time: FieldSpec
    observation: FieldSpec | dict[str, FieldSpec]
    reward: FieldSpec
    done: FieldSpec
    episode_index: FieldSpec
    task_index: FieldSpec


@dataclass
class InputSpec:
    """Mirrors the input dict: one attribute per key in ``inputs[i]``.

    ``action`` describes the single ``"action"`` tensor. Its ``dtype`` and shape
    mirror the underlying Gymnasium action space where possible.
    """

    action: FieldSpec


class StepOutput(TypedDict, total=False):
    """All per-env fields for one step (single-env view, ``outputs[i]``).

    Tensor fields are ``torch.Tensor``; other fields are plain Python types.
    The ``observation`` field is a tensor for ordinary observation spaces, or a
    ``dict[str, torch.Tensor]`` for ``gym.spaces.Dict`` observation spaces.

    Every key from the underlying Gymnasium ``info`` dict is forwarded as
    ``info_<key>``. For example, ``info["env_q_star"]`` appears as
    ``outputs[i]["info_env_q_star"]``, ``info["map"]`` as
    ``outputs[i]["info_map"]``, and ``info["ns_params"]`` as
    ``outputs[i]["info_ns_params"]``.
    """

    time: Required[torch.Tensor]
    observation: torch.Tensor | dict[str, torch.Tensor]
    reward: Required[torch.Tensor]
    done: Required[torch.Tensor]
    episode_index: Required[int]
    task_index: Required[int]


class MetricsTracker:
    """Accumulates per-env episode statistics; attached to :class:`MouseEnv` as ``.tracker``.

    ``MouseEnv.step()`` feeds completed-episode results automatically. Call
    :meth:`clear` to wipe all accumulated data (e.g. between evaluation runs).

    Attributes
    ----------
    episode_cum_rewards:
        Per-env list of raw (unscaled) cumulative rewards for every episode
        completed since the last :meth:`clear` call. Empty lists until an
        episode finishes in that env instance.
    episode_lengths:
        Per-env list of episode step counts for every completed episode since
        the last :meth:`clear` call.
    """

    def __init__(self, num_envs: int) -> None:
        self._num_envs = num_envs
        self._episode_cum_rewards: list[list[float]] = [[] for _ in range(num_envs)]
        self._episode_lengths: list[list[float]] = [[] for _ in range(num_envs)]

    def _record(self, env_index: int, cum_reward: float, length: float) -> None:
        self._episode_cum_rewards[env_index].append(cum_reward)
        self._episode_lengths[env_index].append(length)

    def clear(self) -> None:
        """Wipe all accumulated episode data for every env instance."""
        self._episode_cum_rewards = [[] for _ in range(self._num_envs)]
        self._episode_lengths = [[] for _ in range(self._num_envs)]

    @property
    def episode_cum_rewards(self) -> list[list[float]]:
        """Per-env lists of raw cumulative rewards for completed episodes."""
        return self._episode_cum_rewards

    @property
    def episode_lengths(self) -> list[list[float]]:
        """Per-env lists of episode lengths (step counts) for completed episodes."""
        return self._episode_lengths


class _EnvInstance:
    """Internal: wraps a single ``gym.Env`` with the Mouse step protocol.

    Each env instance manages its own episode state — episode time, index, and
    cumulative rewards — and implements the two-frame boundary sequence: a
    terminal step (``done=1/2``) followed by a reset frame (``done=0``,
    ``time=0``) on the next ``step()`` call, with the user's action on the
    reset-frame call silently ignored.
    """

    def __init__(
        self,
        env: gym.Env,
        name: str,
        *,
        reset_reward: float = 0.0,
        episode_reset_options: dict | None = None,
        task_reset_options: dict | None = None,
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
        episodes_per_task: int,
    ):
        self._env = env
        self._name = name
        self._reset_reward = float(reset_reward)
        self._episode_reset_options = dict(episode_reset_options or {})
        self._task_reset_options = dict(task_reset_options or {})
        self._reward_scale = float(reward_scale)
        self._reward_shift = float(reward_shift)
        self._episodes_per_task = int(episodes_per_task)

        # Episode state
        self._needs_initial_reset = True
        self._autoreset_pending = False
        self._task_done_pending = False
        self._episode_time = 0
        self._episode_index = 0
        self._task_episode_count = 0  # episodes completed in current task
        self._task_index = 0
        self._episode_cum_reward = 0.0

        # Spec
        self._obs_channel, self._obs_dtypes, self._output_spec, self._input_spec = (
            self._build_specs()
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_spec(self) -> OutputSpec:
        return self._output_spec

    @property
    def input_spec(self) -> InputSpec:
        return self._input_spec

    def _build_specs(
        self,
    ) -> tuple[
        str | None,
        dict[str, torch.dtype],
        OutputSpec,
        InputSpec,
    ]:
        obs_space = self._env.observation_space
        act_space = self._env.action_space

        # --- observation side ---
        if isinstance(obs_space, gym.spaces.Dict):
            obs_dtypes: dict[str, torch.dtype] = {
                key: _torch_dtype_for_space(sub)
                for key, sub in obs_space.spaces.items()
            }
            single_channel = None
            obs_field: FieldSpec | dict[str, FieldSpec] = {
                key: FieldSpec(
                    dtype=obs_dtypes[key],
                    shape=tuple(getattr(sub, "shape", ()) or ()),
                )
                for key, sub in obs_space.spaces.items()
            }
        else:
            obs_torch_dtype = _torch_dtype_for_space(obs_space)
            obs_dtypes = {OBS_KEY: obs_torch_dtype}
            single_channel = OBS_KEY
            obs_shape = tuple(getattr(obs_space, "shape", ()) or ())
            obs_field = FieldSpec(dtype=obs_torch_dtype, shape=obs_shape)

        # --- action side ---
        act_torch_dtype = _torch_dtype_for_space(act_space)
        if isinstance(act_space, gym.spaces.Discrete):
            act_shape: tuple[int, ...] = ()
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            act_shape = (len(act_space.nvec),)
        else:
            act_shape = tuple(getattr(act_space, "shape", ()) or ())

        output_spec = OutputSpec(
            time=FieldSpec(dtype=torch.int64, shape=()),
            observation=obs_field,
            reward=FieldSpec(dtype=torch.float32, shape=()),
            done=FieldSpec(dtype=torch.int64, shape=()),
            episode_index=FieldSpec(dtype=int, shape=()),
            task_index=FieldSpec(dtype=int, shape=()),
        )
        input_spec = InputSpec(action=FieldSpec(dtype=act_torch_dtype, shape=act_shape))
        return single_channel, obs_dtypes, output_spec, input_spec

    def _action_tensor(self, value: Any, *, dtype: torch.dtype) -> torch.Tensor:
        arr = np.asarray(value).flatten()
        if arr.size == 1:
            return torch.tensor(arr.item(), dtype=dtype)
        return torch.tensor(arr, dtype=dtype)

    def sample_random_input(self) -> dict:
        """Sample a random action as a ``dict`` with a flat ``"action"`` key."""
        raw = self._env.action_space.sample()
        act_dtype = cast(torch.dtype, self._input_spec.action.dtype)
        return {ACTION_KEY: self._action_tensor(raw, dtype=act_dtype)}

    def _require_input(self, input_dict: Any) -> np.ndarray:
        """Extract and validate the ``"action"`` key from an input dict."""
        if not isinstance(input_dict, dict):
            raise ValueError(
                f"input must be a dict with an '{ACTION_KEY}' entry, "
                f"got {type(input_dict).__name__}."
            )
        if ACTION_KEY not in input_dict:
            raise ValueError(
                f"input must contain the '{ACTION_KEY}' key; "
                f"got keys {sorted(input_dict.keys())}."
            )
        value = cast(Any, input_dict)[ACTION_KEY]
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    def _prepare_action(self, action_np: np.ndarray) -> Any:
        """Convert a numpy action array to the format expected by ``gym.Env.step``."""
        space = self._env.action_space
        if isinstance(space, gym.spaces.Discrete):
            return int(np.asarray(action_np).reshape(-1)[0])
        if isinstance(space, gym.spaces.MultiDiscrete):
            dtype = getattr(space, "dtype", np.int64)
            return np.asarray(action_np, dtype=dtype).reshape(-1)
        dtype = getattr(space, "dtype", None)
        arr = np.asarray(action_np, dtype=dtype) if dtype is not None else np.asarray(action_np)
        return arr.reshape(getattr(space, "shape", ()) or ())

    def _reward_tensor(self, raw_reward: Any) -> torch.Tensor:
        """Return the reward tensor, applying shaping only when explicitly configured."""
        if self._reward_scale == 1.0 and self._reward_shift == 0.0:
            return torch.as_tensor(raw_reward)
        shaped_reward = float(raw_reward) * self._reward_scale + self._reward_shift
        return torch.tensor(shaped_reward, dtype=torch.float32)

    def _obs_entry(self, obs: Any) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Build observation field(s) from a single-env observation."""
        if isinstance(obs, dict):
            return {OBS_KEY: {k: torch.as_tensor(np.asarray(v)) for k, v in obs.items()}}
        channel = cast(str, self._obs_channel)
        return {channel: torch.as_tensor(np.asarray(obs))}

    def _reset_options_for_boundary(self, *, task_start: bool) -> dict[str, Any]:
        options = dict(self._episode_reset_options)
        if task_start:
            options.update(self._task_reset_options)
        return options

    def _do_reset(self, *, task_start: bool) -> tuple[dict, None]:
        """Call env.reset() and return the reset-frame output; no episode result."""
        reset_options = self._reset_options_for_boundary(task_start=task_start)
        reset_kwargs = {"options": reset_options} if reset_options else {}
        obs, info = self._env.reset(**reset_kwargs)
        self._episode_time = 0
        self._episode_cum_reward = 0.0

        output: dict = {
            TIME_KEY: torch.tensor(0, dtype=torch.int64),
            "reward": torch.tensor(self._reset_reward, dtype=torch.float32),
            "done": torch.tensor(DONE_RUNNING, dtype=torch.int64),
            "episode_index": self._episode_index,
            "task_index": self._task_index,
        }
        output.update(self._obs_entry(obs))

        if isinstance(info, dict):
            for key, value in info.items():
                output[f"info_{key}"] = value
        elif info is not None:
            output["info"] = info

        return output, None

    def step(self, input_dict: dict) -> tuple[dict, tuple[float, float] | None]:
        """Step this env instance; return ``(output, episode_result)``.

        ``episode_result`` is ``(cum_reward, length)`` when the episode ended on this
        step, or ``None`` otherwise (including reset frames).
        """
        if self._needs_initial_reset:
            self._needs_initial_reset = False
            return self._do_reset(task_start=True)

        if self._autoreset_pending:
            self._autoreset_pending = False
            self._episode_index += 1
            task_start = False
            if self._task_done_pending:
                self._task_done_pending = False
                self._task_index += 1
                self._task_episode_count = 0
                task_start = True
            else:
                self._task_episode_count += 1
            return self._do_reset(task_start=task_start)

        # Regular step — validate and unpack input
        action_np = self._require_input(input_dict)
        action = self._prepare_action(action_np)
        obs, raw_reward, terminated, truncated, info = self._env.step(action)

        # Track raw cumulative reward (unscaled) for tracker
        raw_reward_f = float(raw_reward)
        self._episode_cum_reward += raw_reward_f

        self._episode_time += 1

        # Determine done code — codes 3/4 fire when this episode is the last in the task.
        # episodes_per_task == 0 means unlimited: task boundary never fires automatically.
        task_done = self._episodes_per_task > 0 and (
            self._task_episode_count + 1 == self._episodes_per_task
        )
        if terminated:
            done = DONE_TASK_TERMINATED if task_done else DONE_EPISODE_TERMINATED
        elif truncated:
            done = DONE_TASK_TRUNCATED if task_done else DONE_EPISODE_TRUNCATED
        else:
            done = DONE_RUNNING

        output: dict = {
            TIME_KEY: torch.tensor(self._episode_time, dtype=torch.int64),
            "reward": self._reward_tensor(raw_reward),
            "done": torch.tensor(done, dtype=torch.int64),
            "episode_index": self._episode_index,
            "task_index": self._task_index,
        }
        output.update(self._obs_entry(obs))

        if isinstance(info, dict):
            for key, value in info.items():
                output[f"info_{key}"] = value
        elif info is not None:
            output["info"] = info

        episode_result: tuple[float, float] | None
        if done != DONE_RUNNING:
            episode_result = (self._episode_cum_reward, float(self._episode_time))
            self._autoreset_pending = True
            self._task_done_pending = task_done
        else:
            episode_result = None

        return output, episode_result

    def render(self) -> list:
        """Return rendered frames from this env instance."""
        frames = self._env.render()
        if frames is None:
            return []
        if isinstance(frames, (list, tuple)):
            return list(frames)
        return [frames]

    def close(self) -> None:
        self._env.close()


class MouseEnv(gym.Env):
    """A flat list of independent env instances, each built from one :class:`EnvConfig`.

    Use :func:`mouse_envs.make_env` with a single :class:`EnvConfig` or a
    ``list[EnvConfig]`` to construct. Each config creates one independent env
    instance.

    ``step`` and ``sample_random_inputs`` use a flat structure indexed by env.
    ``inputs[i]`` is the input dict for the i-th env instance. ``step`` returns a
    flat ``list[dict]`` of outputs — one per env instance.

    Episode statistics are accumulated automatically in :attr:`tracker`
    (:class:`MetricsTracker`). Call ``env.tracker.clear()`` to reset the accumulated
    data between evaluation runs.

    ``MouseEnv`` subclasses :class:`gymnasium.Env` to expose Gymnasium spaces, but it
    intentionally keeps the Mouse rollout protocol. Public ``reset()`` raises
    ``NotImplementedError``; call ``step()`` only. The first ``step()`` after construction
    performs an internal reset for each env instance and returns initial observations
    with ``done == 0`` and ``time == 0``; inputs on that call are ignored. The step after
    any episode terminates or truncates is also a reset frame: the user's action is
    ignored and the first observation of the new episode is returned.

    Every ``outputs[i]`` contains:
        time (int64 tensor)       — step index within the episode (0-based)
        observation (tensor/dict) — the observation emitted by the env
        reward (tensor)           — raw env reward, unless reward_scale/reward_shift are set
        done (int64 tensor)       — 0=running, 1=episode terminated, 2=episode truncated,
                                    3=task terminated, 4=task truncated
        episode_index (int)       — episode counter for this env instance
        task_index (int)          — task counter for this env instance
        info_<key> (any)          — every key from the Gymnasium info dict is forwarded as
                                    ``info_<key>``. For example, ``info["env_q_star"]``
                                    from a Q* wrapper appears as ``info_env_q_star``,
                                    ``info["map"]`` as ``info_map``, ``info["ns_params"]``
                                    as ``info_ns_params``.

    Introspect the full output and input contracts via ``env.output_specs[i]`` and
    ``env.input_specs[i]``, which are :class:`OutputSpec` and :class:`InputSpec`
    dataclasses (one per env instance).
    """

    def __init__(self, env_instances: list[_EnvInstance]) -> None:
        if not env_instances:
            raise ValueError("MouseEnv requires at least one env instance.")
        self._env_instances = env_instances
        self._tracker = MetricsTracker(len(env_instances))

    @property
    def tracker(self) -> MetricsTracker:
        """Episode-statistics tracker; accumulates results from every completed episode.

        Call ``env.tracker.clear()`` to wipe accumulated data between evaluation runs.
        """
        return self._tracker

    @property
    def num_envs(self) -> int:
        """Total number of independent env instances."""
        return len(self._env_instances)

    @property
    def names(self) -> tuple[str, ...]:
        """All env instance names."""
        return tuple(env.name for env in self._env_instances)

    @property
    def output_specs(self) -> list[OutputSpec]:
        """One :class:`OutputSpec` per env instance."""
        return [env.output_spec for env in self._env_instances]

    @property
    def input_specs(self) -> list[InputSpec]:
        """One :class:`InputSpec` per env instance."""
        return [env.input_spec for env in self._env_instances]

    @property
    def action_space(self) -> gym.spaces.Tuple:
        """Gymnasium tuple action space, one subspace per env instance."""
        return gym.spaces.Tuple(tuple(env._env.action_space for env in self._env_instances))

    @property
    def observation_space(self) -> gym.spaces.Tuple:
        """Gymnasium tuple observation space, one subspace per env instance."""
        return gym.spaces.Tuple(tuple(env._env.observation_space for env in self._env_instances))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Mouse rollouts reset internally; use ``step()`` instead."""
        raise NotImplementedError(
            "MouseEnv does not support public reset(); call step() to use the "
            "reset-free Mouse rollout protocol."
        )

    def sample_random_inputs(self) -> list[dict]:
        """Sample random inputs for every env instance.

        Returns a flat ``list[dict]`` — one dict per env instance. Pass the result
        directly to ``step()``.
        """
        return [env.sample_random_input() for env in self._env_instances]

    def step(self, inputs: list[dict]) -> list[dict]:
        """Step all env instances sequentially and return outputs.

        ``inputs[i]`` is the input dict for env instance ``i``. Returns a flat
        ``list[dict]`` — one output dict per env instance. On the first call and on
        any call immediately after an episode ends, the corresponding input is
        ignored and a reset frame is returned instead.

        Completed-episode statistics are recorded automatically into
        :attr:`tracker`. Call ``env.tracker.clear()`` to reset between runs.
        """
        if not isinstance(inputs, list):
            raise ValueError(
                f"inputs must be a list with one dict per env instance; got {type(inputs).__name__}."
            )
        if len(inputs) != self.num_envs:
            raise ValueError(
                f"inputs must contain exactly {self.num_envs} entries, got {len(inputs)}."
            )
        all_outputs: list[dict] = []
        for i, (env, inp) in enumerate(zip(self._env_instances, inputs)):
            output, episode_result = env.step(inp)
            all_outputs.append(output)
            if episode_result is not None:
                cum_reward, length = episode_result
                self._tracker._record(i, cum_reward, length)
        return all_outputs

    def render(self) -> list:
        """Return rendered frames from all env instances, flattened into one list.

        Requires ``render_mode="rgb_array"`` (pass via ``EnvConfig.kwargs``).
        """
        frames: list = []
        for env in self._env_instances:
            frames.extend(env.render())
        return frames

    def close(self) -> None:
        """Close all env instances."""
        for env in self._env_instances:
            env.close()
