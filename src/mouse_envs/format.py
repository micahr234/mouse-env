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


def _torch_dtype_for_space(space: gym.Space) -> torch.dtype:
    """Map a Gymnasium space to the torch dtype used to store its samples.

    Integer/boolean spaces (``Discrete``, ``MultiDiscrete``, ``MultiBinary``, and
    integer ``Box``) are stored as ``int64``; floating spaces as ``float32``. The
    dtype is read from the space itself, never inferred from a channel/key name.
    """
    raw = getattr(space, "dtype", None)
    dt = np.dtype(raw) if raw is not None else np.dtype(np.float32)
    if np.issubdtype(dt, np.floating):
        return torch.float32
    return torch.int64


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
    a ``dict[str, FieldSpec]`` for ``gym.spaces.Dict`` observation spaces (where each
    subspace key appears directly on the output dict rather than under an
    ``"observation"`` key).

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

    ``action`` describes the single ``"action"`` tensor. Its ``dtype`` signals the
    action kind: ``torch.int64`` for discrete spaces, ``torch.float32`` for
    continuous (``Box``) spaces.
    """

    action: FieldSpec


class StepOutput(TypedDict, total=False):
    """All per-env fields for one step (single-env view, ``outputs[i]``).

    Tensor fields are ``torch.Tensor``; other fields are plain Python types.
    The ``observation`` field is a flat tensor (not a nested dict). For
    ``gym.spaces.Dict`` observation spaces, the subspace keys appear directly on the
    output dict instead.

    Every key from the underlying Gymnasium ``info`` dict is forwarded as
    ``info_<key>``. For example, ``info["env_q_star"]`` appears as
    ``outputs[i]["info_env_q_star"]``, ``info["map"]`` as
    ``outputs[i]["info_map"]``, and ``info["ns_params"]`` as
    ``outputs[i]["info_ns_params"]``.
    """

    time: Required[torch.Tensor]
    observation: torch.Tensor
    reward: Required[torch.Tensor]
    done: Required[torch.Tensor]
    episode_index: Required[int]
    task_index: Required[int]


class MetricsTracker:
    """Accumulates per-slot episode statistics; attached to :class:`MouseEnv` as ``.tracker``.

    ``MouseEnv.step()`` feeds completed-episode results automatically. Call
    :meth:`clear` to wipe all accumulated data (e.g. between evaluation runs).

    Attributes
    ----------
    episode_cum_rewards:
        Per-slot list of raw (unscaled) cumulative rewards for every episode
        completed since the last :meth:`clear` call. Empty lists until an
        episode finishes in that slot.
    episode_lengths:
        Per-slot list of episode step counts for every completed episode since
        the last :meth:`clear` call.
    """

    def __init__(self, num_slots: int) -> None:
        self._num_slots = num_slots
        self._episode_cum_rewards: list[list[float]] = [[] for _ in range(num_slots)]
        self._episode_lengths: list[list[float]] = [[] for _ in range(num_slots)]

    def _record(self, slot: int, cum_reward: float, length: float) -> None:
        self._episode_cum_rewards[slot].append(cum_reward)
        self._episode_lengths[slot].append(length)

    def clear(self) -> None:
        """Wipe all accumulated episode data for every slot."""
        self._episode_cum_rewards = [[] for _ in range(self._num_slots)]
        self._episode_lengths = [[] for _ in range(self._num_slots)]

    @property
    def episode_cum_rewards(self) -> list[list[float]]:
        """Per-slot lists of raw cumulative rewards for completed episodes."""
        return self._episode_cum_rewards

    @property
    def episode_lengths(self) -> list[list[float]]:
        """Per-slot lists of episode lengths (step counts) for completed episodes."""
        return self._episode_lengths


class _Slot:
    """Internal: wraps a single ``gym.Env`` with the Mouse step protocol.

    Each slot manages its own episode state — episode time, index, and cumulative
    rewards — and implements the two-frame boundary sequence: a terminal step
    (``done=1/2``) followed by a reset frame (``done=0``, ``time=0``) on the next
    ``step()`` call, with the user's action on the reset-frame call silently ignored.
    """

    def __init__(
        self,
        env: gym.Env,
        name: str,
        *,
        reset_reward: float = 0.0,
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
        episodes_per_task: int,
    ):
        self._env = env
        self._name = name
        self._reset_reward = float(reset_reward)
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
    def obs_key(self) -> str:
        return getattr(self._env, "obs_key", OBS_KEY)

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
            if self.obs_key == "observation_discrete":
                obs_torch_dtype = torch.int64
            elif self.obs_key == "observation_image":
                obs_torch_dtype = torch.float32
            else:
                obs_torch_dtype = torch.float32
            obs_dtypes = {OBS_KEY: obs_torch_dtype}
            single_channel = OBS_KEY
            obs_shape = tuple(getattr(obs_space, "shape", ()) or ())
            obs_field = FieldSpec(dtype=obs_torch_dtype, shape=obs_shape)

        # --- action side ---
        if isinstance(act_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
            act_torch_dtype = torch.int64
            if isinstance(act_space, gym.spaces.Discrete):
                act_shape: tuple[int, ...] = ()
            else:
                act_shape = (len(act_space.nvec),)
        else:
            act_torch_dtype = torch.float32
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
        raw = cast(Any, self._env).sample_random_input()
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
            return np.asarray(action_np).reshape(-1).astype(np.int64)
        return np.asarray(action_np, dtype=np.float32).reshape(
            getattr(space, "shape", ()) or ()
        )

    def _obs_entry(self, obs: Any) -> dict[str, torch.Tensor]:
        """Build observation field(s) from a single-env observation."""
        if isinstance(obs, dict):
            return {
                k: torch.tensor(np.asarray(v), dtype=self._obs_dtypes[k])
                for k, v in obs.items()
            }
        channel = cast(str, self._obs_channel)
        return {channel: torch.tensor(np.asarray(obs), dtype=self._obs_dtypes[channel])}

    def _do_reset(self) -> tuple[dict, None]:
        """Call env.reset() and return the reset-frame output; no episode result."""
        obs, info = self._env.reset()
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
        """Step this slot; return ``(output, episode_result)`` for the single env.

        ``episode_result`` is ``(cum_reward, length)`` when the episode ended on this
        step, or ``None`` otherwise (including reset frames).
        """
        if self._needs_initial_reset:
            self._needs_initial_reset = False
            return self._do_reset()

        if self._autoreset_pending:
            self._autoreset_pending = False
            self._episode_index += 1
            if self._task_done_pending:
                self._task_done_pending = False
                self._task_index += 1
                self._task_episode_count = 0
            else:
                self._task_episode_count += 1
            return self._do_reset()

        # Regular step — validate and unpack input
        action_np = self._require_input(input_dict)
        action = self._prepare_action(action_np)
        obs, raw_reward, terminated, truncated, info = self._env.step(action)

        # Track raw cumulative reward (unscaled) for tracker
        raw_reward_f = float(raw_reward)
        self._episode_cum_reward += raw_reward_f

        # Compute scaled/shifted reward
        shaped_reward = raw_reward_f * self._reward_scale + self._reward_shift

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
            "reward": torch.tensor(shaped_reward, dtype=torch.float32),
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
        """Return rendered frames from this slot."""
        frames = self._env.render()
        if frames is None:
            return []
        if isinstance(frames, (list, tuple)):
            return list(frames)
        return [frames]

    def close(self) -> None:
        self._env.close()


class MouseEnv:
    """A flat list of independent environment slots, each built from one :class:`EnvConfig`.

    Use :func:`mouse_envs.make_env` with a single :class:`EnvConfig` or a
    ``list[EnvConfig]`` to construct. ``EnvConfig.num_envs=N`` creates N independent
    slots — equivalent to specifying N separate single-env configs.

    ``step`` and ``sample_random_inputs`` use a flat structure indexed by slot.
    ``inputs[i]`` is the input dict for the i-th slot. ``step`` returns a flat
    ``list[dict]`` of outputs — one per slot.

    Episode statistics are accumulated automatically in :attr:`tracker`
    (:class:`MetricsTracker`). Call ``env.tracker.clear()`` to reset the accumulated
    data between evaluation runs.

    There is no public ``reset()`` — call ``step()`` only. The first ``step()`` after
    construction performs an internal reset for each slot and returns initial
    observations with ``done == 0`` and ``time == 0``; inputs on that call are ignored.
    The step after any episode terminates or truncates is also a reset frame: the user's
    action is ignored and the first observation of the new episode is returned.

    Every ``outputs[i]`` contains:
        time (int64 tensor)       — step index within the episode (0-based)
        observation (tensor)      — the observation tensor
        reward (float32 tensor)   — scaled/shifted per-step reward (raw × scale + shift)
        done (int64 tensor)       — 0=running, 1=episode terminated, 2=episode truncated,
                                    3=task terminated, 4=task truncated
        episode_index (int)       — episode counter for this slot
        task_index (int)          — task counter for this slot
        info_<key> (any)          — every key from the Gymnasium info dict is forwarded as
                                    ``info_<key>``. For example, ``info["env_q_star"]``
                                    from a Q* wrapper appears as ``info_env_q_star``,
                                    ``info["map"]`` as ``info_map``, ``info["ns_params"]``
                                    as ``info_ns_params``.

    Introspect the full output and input contracts via ``env.output_specs[i]`` and
    ``env.input_specs[i]``, which are :class:`OutputSpec` and :class:`InputSpec`
    dataclasses (one per slot).
    """

    def __init__(self, slots: list[_Slot]) -> None:
        if not slots:
            raise ValueError("MouseEnv requires at least one slot.")
        self._slots = slots
        self._tracker = MetricsTracker(len(slots))

    @property
    def tracker(self) -> MetricsTracker:
        """Episode-statistics tracker; accumulates results from every completed episode.

        Call ``env.tracker.clear()`` to wipe accumulated data between evaluation runs.
        """
        return self._tracker

    @property
    def num_envs(self) -> int:
        """Total number of independent slots."""
        return len(self._slots)

    @property
    def names(self) -> tuple[str, ...]:
        """All slot names."""
        return tuple(s.name for s in self._slots)

    @property
    def output_specs(self) -> list[OutputSpec]:
        """One :class:`OutputSpec` per slot."""
        return [s.output_spec for s in self._slots]

    @property
    def input_specs(self) -> list[InputSpec]:
        """One :class:`InputSpec` per slot."""
        return [s.input_spec for s in self._slots]

    def sample_random_inputs(self) -> list[dict]:
        """Sample random inputs for every slot.

        Returns a flat ``list[dict]`` — one dict per slot. Pass the result directly
        to ``step()``.
        """
        return [s.sample_random_input() for s in self._slots]

    def step(self, inputs: list[dict]) -> list[dict]:
        """Step all slots sequentially and return outputs.

        ``inputs[i]`` is the input dict for slot ``i``. Returns a flat
        ``list[dict]`` — one output dict per slot. On the first call and on any
        call immediately after an episode ends, the corresponding slot's input is
        ignored and a reset frame is returned instead.

        Completed-episode statistics are recorded automatically into
        :attr:`tracker`. Call ``env.tracker.clear()`` to reset between runs.
        """
        all_outputs: list[dict] = []
        for i, (slot, inp) in enumerate(zip(self._slots, inputs)):
            output, episode_result = slot.step(inp)
            all_outputs.append(output)
            if episode_result is not None:
                cum_reward, length = episode_result
                self._tracker._record(i, cum_reward, length)
        return all_outputs

    def render(self) -> list:
        """Return rendered frames from all slots, flattened into one list.

        Requires ``render_mode="rgb_array"`` (pass via ``EnvConfig.kwargs``).
        """
        frames: list = []
        for s in self._slots:
            frames.extend(s.render())
        return frames

    def close(self) -> None:
        """Close all slots."""
        for s in self._slots:
            s.close()
