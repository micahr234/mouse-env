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

DONE_RUNNING = 0
DONE_TERMINATED = 1
DONE_TRUNCATED = 2


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

    Optional fields (``q_star``, ``ns_params``) are ``None`` when not configured.
    """

    time: FieldSpec
    observation: FieldSpec | dict[str, FieldSpec]
    reward: FieldSpec
    done: FieldSpec
    episode_index: FieldSpec
    reward_episodic: FieldSpec
    q_star: FieldSpec | None
    ns_params: FieldSpec | None


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
    """

    time: Required[torch.Tensor]
    observation: torch.Tensor
    reward: Required[torch.Tensor]
    done: Required[torch.Tensor]
    episode_index: Required[int]
    reward_episodic: Required[float]
    q_star: Any
    ns_params: Any


class RolloutMetrics(TypedDict):
    """Per-env episode statistics for the current step (``metrics[i]`` view)."""

    episode_cum_reward: list[float]
    episode_length: list[float]


class _InnerEnv:
    """Internal: wraps a single Gymnasium VectorEnv with the Mouse step protocol."""

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        names: list[str],
        *,
        reset_reward: float = 0.0,
    ):
        self._env = env
        self._names = tuple(names)
        self._needs_initial_reset = True
        self._reset_reward = float(reset_reward)
        self._obs_channel, self._obs_dtypes, self._output_spec, self._input_spec = (
            self._build_specs()
        )

    def _build_specs(
        self,
    ) -> tuple[
        str | None,
        dict[str, torch.dtype],
        OutputSpec,
        InputSpec,
    ]:
        """Build observation schema and both spec objects once at construction.

        Returns ``(single_channel, obs_dtypes, output_spec, input_spec)`` where
        ``single_channel`` is the lone obs key for non-Dict spaces (``None`` for Dict
        spaces) and ``obs_dtypes`` maps each obs output key to its torch dtype.
        """
        obs_space = self._env.single_observation_space
        act_space = self._env.single_action_space

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

        # --- q_star spec: only present when QStarWrapper is in the stack ---
        action_dim = int(getattr(self._env, "action_dim", 0))
        q_star_field: FieldSpec | None = None
        if action_dim > 0 and self._has_q_star_wrapper():
            q_star_field = FieldSpec(dtype=np.float64, shape=(action_dim,))

        output_spec = OutputSpec(
            time=FieldSpec(dtype=torch.int64, shape=()),
            observation=obs_field,
            reward=FieldSpec(dtype=torch.float32, shape=()),
            done=FieldSpec(dtype=torch.int64, shape=()),
            episode_index=FieldSpec(dtype=int, shape=()),
            reward_episodic=FieldSpec(dtype=float, shape=()),
            q_star=q_star_field,
            ns_params=None,
        )
        input_spec = InputSpec(
            action=FieldSpec(dtype=act_torch_dtype, shape=act_shape)
        )
        return single_channel, obs_dtypes, output_spec, input_spec

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def single_observation_space(self):
        return self._env.single_observation_space

    @property
    def single_action_space(self):
        return self._env.single_action_space

    @property
    def obs_key(self) -> str:
        """Forward obs_key from the inner EnvIdentityWrapper if available."""
        return getattr(self._env, "obs_key", "observation")

    @property
    def action_dim(self) -> int:
        """Forward action_dim from the inner EnvIdentityWrapper if available."""
        return getattr(self._env, "action_dim", 0)

    @property
    def output_spec(self) -> OutputSpec:
        return self._output_spec

    @property
    def input_spec(self) -> InputSpec:
        return self._input_spec

    def _has_q_star_wrapper(self) -> bool:
        """Return True if a QStarWrapper is present anywhere in the wrapper stack."""
        from mouse_envs.wrappers import QStarWrapper

        current: Any = self._env
        while current is not None:
            if isinstance(current, QStarWrapper):
                return True
            current = getattr(current, "env", None)
        return False

    def _action_tensor(self, value: Any, *, dtype: torch.dtype) -> torch.Tensor:
        arr = np.asarray(value).flatten()
        if arr.size == 1:
            return torch.tensor(arr.item(), dtype=dtype)
        return torch.tensor(arr, dtype=dtype)

    def sample_random_inputs(self) -> list[dict]:
        """Sample random actions as ``list[dict]`` with a flat ``"action"`` key."""
        raw = cast(Any, self._env).sample_random_inputs()
        act_dtype = cast(torch.dtype, self._input_spec.action.dtype)
        inputs: list[dict] = []
        for i in range(self.num_envs):
            inputs.append({ACTION_KEY: self._action_tensor(raw[i], dtype=act_dtype)})
        return inputs

    def step(self, inputs: list[dict]) -> tuple[list[dict], list[dict]]:
        """Step all parallel slots; return ``(outputs, metrics)``."""
        if self._needs_initial_reset:
            self._needs_initial_reset = False
            obs, info = self._env.reset()
            return self._build_records(obs, info, is_reset=True)
        else:
            raw_inputs = self._unpack_inputs(inputs)
            obs, reward, _terminated, _truncated, info = self._env.step(raw_inputs)
            return self._build_records(obs, info, reward=reward, is_reset=False)

    def render(self) -> list:
        """Return rendered frames from all parallel slots."""
        frames = self._env.render()
        if frames is None:
            return []
        if isinstance(frames, (list, tuple)):
            return list(frames)
        return [frames]

    def close(self) -> None:
        self._env.close()

    def _require_input(self, input_record: Any, index: int) -> np.ndarray:
        """Extract and validate the ``"action"`` key from an input dict."""
        if not isinstance(input_record, dict):
            raise ValueError(
                f"inputs[{index}] must be a dict with an '{ACTION_KEY}' entry, "
                f"got {type(input_record).__name__}."
            )
        if ACTION_KEY not in input_record:
            raise ValueError(
                f"inputs[{index}] must contain the '{ACTION_KEY}' key; "
                f"got keys {sorted(input_record.keys())}."
            )
        value = cast(Any, input_record)[ACTION_KEY]
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    def _unpack_inputs(self, inputs: list[dict]) -> np.ndarray:
        space = self._env.single_action_space
        raw_list = [self._require_input(td, i) for i, td in enumerate(inputs)]
        if isinstance(space, gym.spaces.Discrete):
            return np.asarray(
                [np.asarray(a).reshape(-1)[0] for a in raw_list], dtype=np.int64
            )
        if isinstance(space, gym.spaces.MultiDiscrete):
            return np.stack(
                [np.asarray(a).reshape(-1) for a in raw_list]
            ).astype(np.int64)
        raw = np.stack(
            [np.asarray(a).reshape(-1) for a in raw_list]
        ).astype(np.float32)
        return raw.reshape((self.num_envs, *(getattr(space, "shape", ()) or ())))

    def _obs_for_index(self, obs: Any, i: int) -> dict[str, torch.Tensor]:
        """Build observation field(s) for slot index ``i``."""
        if isinstance(obs, dict):
            return {
                k: torch.tensor(np.asarray(v[i]), dtype=self._obs_dtypes[k])
                for k, v in obs.items()
            }
        channel = cast(str, self._obs_channel)
        return {
            channel: torch.tensor(np.asarray(obs[i]), dtype=self._obs_dtypes[channel])
        }

    def _build_metrics(self, info: dict, *, empty_episode_stats: bool = False) -> list[dict]:
        if empty_episode_stats:
            return [
                {"episode_cum_reward": [], "episode_length": []}
                for _ in range(self.num_envs)
            ]
        cum_reward = (
            np.asarray(info["episode_cum_reward"], dtype=np.float64)
            if "episode_cum_reward" in info
            else np.full(self.num_envs, np.nan, dtype=np.float64)
        )
        length = (
            np.asarray(info["episode_length"], dtype=np.float64)
            if "episode_length" in info
            else np.full(self.num_envs, np.nan, dtype=np.float64)
        )
        return [
            {
                "episode_cum_reward": (
                    [float(cum_reward[i])] if not np.isnan(cum_reward[i]) else []
                ),
                "episode_length": (
                    [float(length[i])] if not np.isnan(length[i]) else []
                ),
            }
            for i in range(self.num_envs)
        ]

    def _ns_params_for_env(self, ns_params: Any, i: int) -> Any:
        if isinstance(ns_params, list):
            return ns_params[i]
        return ns_params

    def _reset_frame_mask(self, info: dict, *, is_reset: bool) -> np.ndarray:
        if is_reset:
            return np.ones(self.num_envs, dtype=np.bool_)
        raw = info.get("is_reset_frame")
        if raw is None:
            return np.zeros(self.num_envs, dtype=np.bool_)
        return np.asarray(raw, dtype=np.bool_)

    def _build_records(
        self,
        obs: Any,
        info: dict,
        *,
        reward: Any = None,
        is_reset: bool,
    ) -> tuple[list[dict], list[dict]]:
        reset_mask = self._reset_frame_mask(info, is_reset=is_reset)

        if is_reset:
            reward_arr = np.full(self.num_envs, self._reset_reward, dtype=np.float32)
            xformed_arr = np.zeros(self.num_envs, dtype=np.float64)
            done_arr = np.full(self.num_envs, DONE_RUNNING, dtype=np.int64)
        else:
            reward_arr = np.asarray(reward, dtype=np.float32)
            xformed_arr = np.asarray(info["xformed_reward"], dtype=np.float64)
            done_arr = np.asarray(info["done"], dtype=np.int64)
            reward_arr[reset_mask] = self._reset_reward
            xformed_arr[reset_mask] = 0.0
            done_arr[reset_mask] = DONE_RUNNING

        episode_index = np.asarray(info["episode_index"], dtype=np.int64)
        q_star = (
            np.asarray(info["metadata_q_star"], dtype=np.float64)
            if "metadata_q_star" in info
            else None
        )
        ns_params = info.get("ns_params")

        outputs: list[dict] = []
        for i in range(self.num_envs):
            entry: dict = {
                TIME_KEY: torch.tensor(
                    int(info["episode_time"][i]), dtype=torch.int64
                ),
                "reward": torch.tensor(float(reward_arr[i]), dtype=torch.float32),
                "done": torch.tensor(int(done_arr[i]), dtype=torch.int64),
                "episode_index": int(episode_index[i]),
                "reward_episodic": float(xformed_arr[i]),
            }
            entry.update(self._obs_for_index(obs, i))
            if q_star is not None:
                entry["q_star"] = q_star[i]
            if ns_params is not None:
                entry["ns_params"] = self._ns_params_for_env(ns_params, i)
            outputs.append(entry)

        metrics = self._build_metrics(info, empty_episode_stats=is_reset)
        return outputs, metrics


class MouseEnv:
    """A sequential list of environments, each built from one :class:`EnvConfig`.

    Use :func:`mouse_envs.make_env` with a single :class:`EnvConfig` or a
    ``list[EnvConfig]`` to construct. A single-config call is the degenerate case
    with one inner env.

    ``step`` and ``sample_random_inputs`` always use a nested structure indexed by
    env. ``inputs_per_env[i]`` is the ``list[dict]`` for the i-th inner env.
    ``step`` returns ``[(outputs, metrics), ...]``, one tuple per inner env.

    Each inner env's parallel slots are stepped by a ``SyncVectorEnv``; the inner
    envs are iterated sequentially. There is no public ``reset()`` — call ``step()``
    only. The first ``step()`` performs an internal reset for each inner env.

    Call ``step()`` only — there is no public ``reset()``. The first ``step()`` after
    construction performs an internal reset and returns initial observations with the
    configured reset-frame ``reward`` and ``done == 0``; inputs on that call are
    ignored.

    Every ``outputs[i]`` (within one inner env's result) contains:
        time (int64 tensor)       — step index within the episode (0-based)
        observation (tensor)      — the observation tensor
        reward (float32 tensor)   — raw per-step reward
        done (int64 tensor)       — 0=running, 1=terminated, 2=truncated
        episode_index (int)       — episode counter for this parallel slot
        reward_episodic (float)   — normalised training signal
        q_star (optional)         — float64[action_dim] expert Q-values when configured
        ns_params (optional)      — surfaced when an env wrapper sets info["ns_params"]

    Introspect the full output and input contracts via ``env.output_specs[i]`` and
    ``env.input_specs[i]``, which are :class:`OutputSpec` and :class:`InputSpec`
    dataclasses.
    """

    def __init__(self, inner_envs: list[_InnerEnv]) -> None:
        if not inner_envs:
            raise ValueError("MouseEnv requires at least one inner env.")
        self._inners = inner_envs

    @property
    def num_envs(self) -> int:
        """Total parallel slots across all inner envs."""
        return sum(e.num_envs for e in self._inners)

    @property
    def names(self) -> tuple[str, ...]:
        """All env slot names, flattened across inner envs."""
        result: list[str] = []
        for e in self._inners:
            result.extend(e.names)
        return tuple(result)

    @property
    def output_specs(self) -> list[OutputSpec]:
        """One :class:`OutputSpec` per inner env."""
        return [e.output_spec for e in self._inners]

    @property
    def input_specs(self) -> list[InputSpec]:
        """One :class:`InputSpec` per inner env."""
        return [e.input_spec for e in self._inners]

    def sample_random_inputs(self) -> list[list[dict]]:
        """Sample random inputs for every inner env.

        Returns ``list[list[dict]]`` — the outer list is indexed by inner env,
        the inner list by parallel slot. Pass the result directly to ``step()``.
        """
        return [e.sample_random_inputs() for e in self._inners]

    def step(
        self, inputs_per_env: list[list[dict]]
    ) -> list[tuple[list[dict], list[dict]]]:
        """Step all inner envs sequentially.

        ``inputs_per_env[i]`` is the input ``list[dict]`` for the i-th inner env
        (one dict per parallel slot). Returns ``[(outputs, metrics), ...]``, one
        tuple per inner env — outputs are never concatenated across envs.
        """
        return [
            e.step(inputs)
            for e, inputs in zip(self._inners, inputs_per_env)
        ]

    def render(self) -> list:
        """Return rendered frames from all inner envs, flattened into one list.

        Requires ``render_mode="rgb_array"`` (pass via ``EnvConfig.kwargs``).
        """
        frames: list = []
        for e in self._inners:
            frames.extend(e.render())
        return frames

    def close(self) -> None:
        """Close all inner envs."""
        for e in self._inners:
            e.close()
