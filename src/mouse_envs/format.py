"""Public step API and rollout contract types for mouse-env ↔ mouse-core."""

from __future__ import annotations

from typing import Any, Required, TypedDict, cast

import gymnasium as gym
import numpy as np
import torch

ACTION_KEY_DISCRETE = "discrete"
ACTION_KEY_CONTINUOUS = "continuous"

OBS_KEY_DISCRETE = "discrete"
OBS_KEY_CONTINUOUS = "continuous"
OBS_KEY_IMAGE = "image"

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


class RolloutResult(TypedDict, total=False):
    """All per-env fields for one step (single-env view, ``result[i]``).

    Tensor fields are ``torch.Tensor``; other fields are plain Python types.
    """

    time: Required[torch.Tensor]
    observation: Required[dict[str, Any]]
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


class MouseVectorEnv:
    """Wraps a Gymnasium vector env and returns (result, metrics).

    ``names`` identifies the sub-envs by vector index. ``name`` returns the first
    name, which is convenient for ``num_envs == 1``. ``result`` is a list of length
    ``num_envs``. Each ``result[i]`` is a dict containing the full per-step record
    for environment ``i``.

    Call ``step()`` only — there is no public ``reset()``. The first ``step()`` after
    construction performs an internal reset and returns initial observations with the
    configured reset-frame ``reward`` and ``done == 0``; actions passed on that call
    are ignored.
    Subsequent ``step()`` calls apply actions normally. Finished sub-envs are
    auto-reset by the inner ``SyncVectorEnv`` (``AutoresetMode.NEXT_STEP``) on the
    next step; that autoreset frame uses the configured reset reward and ``done == 0``,
    like the initial reset. Episode boundaries appear as non-zero ``done`` on the
    terminal transition.

    Every ``result[i]`` contains:
        time (int64 tensor)      — step index within the episode (0-based)
        observation (dict)       — tensors: "discrete", "continuous", and/or "image",
                                   each keeping its native shape (images stay 2-D/3-D)
        reward (float32 tensor)  — raw per-step reward; reset default on reset frames
        done (int64 tensor)      — 0=running, 1=terminated, 2=truncated; 0 on reset frames
        episode_index (int)      — episode counter for this parallel env
        reward_episodic (float)  — normalised training signal; 0.0 on reset frames
        q_star (optional)        — float64[action_dim] expert Q-values when configured
        ns_params (optional)     — surfaced when an env wrapper sets info["ns_params"]

    Both sides of the contract are dicts: each ``result[i]["observation"]`` is a
    dict keyed by channel (``"discrete"``, ``"continuous"``, and/or ``"image"``),
    and each action input must be a dict too. Pass ``list[dict]`` where every
    ``actions[i]["action"]`` is a dict with a ``"discrete"`` or ``"continuous"``
    tensor; a bare tensor is rejected.

    ``metrics`` uses the same env index as ``result`` (``metrics[i]``):
        episode_cum_reward: list[float]   — empty unless env ``i`` finished on this step
        episode_length:     list[float]   — one value per finish on this step

    ``time`` is 0-based within the episode. Internal ``info["episode_time"]`` from
    ``EpisodeTrackingWrapper`` is 1-based after the first real step; ``MouseVectorEnv``
    maps this at the public boundary. Initial reset records have ``time == 0``.
    """

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
        self._obs_channel, self._obs_dtypes = self._build_obs_schema()

    def _build_obs_schema(self) -> tuple[str | None, dict[str, torch.dtype]]:
        """Record the output observation key(s) and their dtypes from the space.

        Computed once at construction so ``_obs_for_index`` never inspects key names
        at runtime. Returns ``(single_channel, dtypes)`` where ``single_channel`` is
        the lone output key for non-dict spaces (``None`` for ``Dict`` spaces) and
        ``dtypes`` maps each output key to its stored torch dtype.
        """
        space = self._env.single_observation_space
        if isinstance(space, gym.spaces.Dict):
            dtypes = {
                key: _torch_dtype_for_space(sub) for key, sub in space.spaces.items()
            }
            return None, dtypes
        if self.obs_key == "observation_discrete":
            return OBS_KEY_DISCRETE, {OBS_KEY_DISCRETE: torch.int64}
        if self.obs_key == "observation_image":
            return OBS_KEY_IMAGE, {OBS_KEY_IMAGE: torch.float32}
        return OBS_KEY_CONTINUOUS, {OBS_KEY_CONTINUOUS: torch.float32}

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def name(self) -> str:
        return self._names[0]

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

    def _action_tensor(self, value: Any, *, dtype: torch.dtype) -> torch.Tensor:
        arr = np.asarray(value).flatten()
        if arr.size == 1:
            return torch.tensor(arr.item(), dtype=dtype)
        return torch.tensor(arr, dtype=dtype)

    def sample_random_actions(self) -> list[dict]:
        """Sample random actions as ``list[dict]`` with ``action`` dict keys."""
        raw = cast(Any, self._env).sample_random_actions()
        space = self._env.single_action_space
        actions: list[dict] = []
        for i in range(self.num_envs):
            if isinstance(space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
                action = {"discrete": self._action_tensor(raw[i], dtype=torch.int64)}
            else:
                action = {"continuous": self._action_tensor(raw[i], dtype=torch.float32)}
            actions.append({"action": action})
        return actions

    def step(self, actions: list[dict]) -> tuple[list[dict], list[dict]]:
        """Step all envs; return ``(result, metrics)``.

        On the first call after construction, performs an internal reset and returns
        initial observations (actions are ignored). Otherwise applies ``actions`` to
        all parallel envs.
        """
        if self._needs_initial_reset:
            self._needs_initial_reset = False
            obs, info = self._env.reset()
            return self._build_records(obs, info, is_reset=True)
        else:
            raw_actions = self._unpack_actions(actions)
            obs, reward, _terminated, _truncated, info = self._env.step(raw_actions)
            return self._build_records(obs, info, reward=reward, is_reset=False)

    def close(self) -> None:
        self._env.close()

    def _action_dict(self, action_record: dict[str, Any], index: int) -> dict[str, Any]:
        """Return the action dict for env ``index``, enforcing the dict contract.

        Each ``actions[i]["action"]`` must be a dict keyed by action type
        (``"discrete"`` or ``"continuous"``); a bare tensor is rejected.
        """
        if not isinstance(action_record, dict):
            raise ValueError(
                f"actions[{index}] must be a dict with an 'action' entry, "
                f"got {type(action_record).__name__}."
            )
        try:
            entry = action_record["action"]
        except KeyError as exc:
            raise ValueError(
                f"actions[{index}] is missing the required 'action' entry."
            ) from exc
        if not isinstance(entry, dict):
            raise ValueError(
                f"actions[{index}]['action'] must be a dict keyed by action type "
                f"('{ACTION_KEY_DISCRETE}' or '{ACTION_KEY_CONTINUOUS}'), "
                f"got {type(entry).__name__}."
            )
        return entry

    def _require_action_key(
        self, entry: dict[str, Any], key: str, index: int
    ) -> np.ndarray:
        if key not in entry:
            raise ValueError(
                f"actions[{index}]['action'] must contain the '{key}' key for this "
                f"action space; got keys {sorted(entry.keys())}."
            )
        value = cast(Any, entry)[key]
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    def _unpack_actions(self, actions: list[dict]) -> np.ndarray:
        space = self._env.single_action_space
        if isinstance(space, gym.spaces.Discrete):
            discrete_actions = [
                self._require_action_key(
                    self._action_dict(td, i), ACTION_KEY_DISCRETE, i
                )
                for i, td in enumerate(actions)
            ]
            raw = np.asarray(
                [np.asarray(a).reshape(-1)[0] for a in discrete_actions],
                dtype=np.int64,
            )
        elif isinstance(space, gym.spaces.MultiDiscrete):
            discrete_actions = [
                self._require_action_key(
                    self._action_dict(td, i), ACTION_KEY_DISCRETE, i
                )
                for i, td in enumerate(actions)
            ]
            raw = np.stack(
                [np.asarray(a).reshape(-1) for a in discrete_actions]
            ).astype(np.int64)
        else:
            continuous_actions = [
                self._require_action_key(
                    self._action_dict(td, i), ACTION_KEY_CONTINUOUS, i
                )
                for i, td in enumerate(actions)
            ]
            raw = np.stack(
                [np.asarray(a).reshape(-1) for a in continuous_actions]
            ).astype(np.float32)
            raw = raw.reshape((self.num_envs, *space.shape))
        return raw

    def _obs_for_index(self, obs: Any, i: int) -> dict[str, torch.Tensor]:
        """Build observation dict for env index ``i`` (may contain multiple keys).

        Dtypes come from the schema recorded at construction
        (:meth:`_build_obs_schema`), derived from the observation space rather than
        from channel/key names. Observations keep their native shape: image channels
        stay 2-D/3-D (e.g. ``(84, 84)`` for preprocessed Atari), continuous channels
        stay 1-D, and discrete channels stay scalar. No flattening is applied.
        """
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

        result: list[dict] = []
        for i in range(self.num_envs):
            entry: dict = {
                TIME_KEY: torch.tensor(
                    int(info["episode_time"][i]), dtype=torch.int64
                ),
                "observation": self._obs_for_index(obs, i),
                "reward": torch.tensor(float(reward_arr[i]), dtype=torch.float32),
                "done": torch.tensor(int(done_arr[i]), dtype=torch.int64),
                "episode_index": int(episode_index[i]),
                "reward_episodic": float(xformed_arr[i]),
            }
            if q_star is not None:
                entry["q_star"] = q_star[i]
            if ns_params is not None:
                entry["ns_params"] = self._ns_params_for_env(ns_params, i)
            result.append(entry)

        metrics = self._build_metrics(info, empty_episode_stats=is_reset)
        return result, metrics
