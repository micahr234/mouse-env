"""Public step API and rollout contract types for mouse-env ↔ mouse-core."""

from __future__ import annotations

from typing import Any, TypedDict

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

ACTION_KEY_DISCRETE = "discrete"
ACTION_KEY_CONTINUOUS = "continuous"

OBS_KEY_DISCRETE = "discrete"
OBS_KEY_CONTINUOUS = "continuous"
OBS_KEY_IMAGE = "image"

REWARD_KEY_STEP = "step"
REWARD_KEY_EPISODIC = "episodic"

TIME_KEY = "time"

DONE_RUNNING = 0
DONE_TERMINATED = 1
DONE_TRUNCATED = 2


class RewardDict(TypedDict):
    """Per-step reward payload (documentation only — runtime type is a nested dict of tensors)."""

    step: float
    episodic: float


class RolloutStepCore(TypedDict):
    """Logical fields for one env at one step (single-env view).

    At runtime each record is a ``TensorDict`` with ``batch_size=[]``.
    ``group_id`` and ``episode_index`` live in ``metadata``, not inside each TensorDict.
    """

    time: int
    observation: dict[str, Any]
    done: int
    reward: RewardDict


class RolloutMetrics(TypedDict):
    """Batch-level episode statistics."""

    episode_cum_reward: Any
    episode_length: Any


class RolloutMetadata(TypedDict, total=False):
    """Batch-level metadata returned alongside metrics."""

    group_ids: list[str]
    episode_index: Any
    q_star: Any
    ns_params: Any


class MouseVectorEnv:
    """Wraps a Gymnasium vector env and returns (list[TensorDict], metadata, metrics).

    Each TensorDict in the returned list corresponds to one parallel environment index
    (``batch_size=[]`` — a scalar TensorDict with no batch dimension).
    The field layout follows the public step API documented in docs/guide.md.

    Call ``step()`` only — there is no public ``reset()``. The first ``step()`` after
    construction performs an internal reset and returns initial observations with dummy
    ``reward`` (zeros) and ``done`` (``0``); actions passed on that call are ignored.
    Subsequent ``step()`` calls apply actions normally. Finished sub-envs are
    auto-reset by the inner ``SyncVectorEnv`` (``AutoresetMode.NEXT_STEP``) on the
    next step; that autoreset frame uses dummy ``reward`` and ``done`` (``0``), like
    the initial reset. Episode boundaries appear as non-zero ``done`` on the
    terminal transition.

    Every record contains the same keys:
        time (int64),
        observation (dict with any combination of "discrete", "continuous", and "image" tensors),
        reward — dict with "step" and "episodic" float32 tensors,
        done   — int64  (0=running, 1=terminated, 2=truncated)

    Actions are input to ``step()`` only; they are not echoed in ``data``.

    ``group_id`` and ``episode_index`` are NOT stored inside each TensorDict.
    They are returned in ``metadata`` as batch-level fields.

    ``metrics`` holds episode statistics:
        episode_cum_reward: float64[num_envs]   — NaN for running envs, filled on done != 0
        episode_length:     float64[num_envs]   — NaN for running envs

    ``metadata`` holds batch-level context:
        group_ids:     list[str]                 — one per env index (always present)
        episode_index: int64[num_envs]           — monotonic episode counter per stream (always present)
        q_star:        float64[num_envs, action_dim] (optional, when q_star_source set)
        ns_params:     any                      (optional, NS-Gym envs only)

    ``time`` is 0-based within the episode. Internal ``info["episode_time"]`` from
    StepCounterWrapper is 1-based after the first real step; ``MouseVectorEnv`` maps this
    at the public boundary. Initial reset records have ``time == 0``.
    """

    def __init__(self, env: gym.vector.VectorEnv, group_ids: list[str]):
        self._env = env
        self._group_ids = group_ids
        self._needs_initial_reset = True

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

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

    def sample_random_actions(self) -> list[TensorDict]:
        """Sample random actions and return them as ``list[TensorDict]``."""
        raw = self._env.sample_random_actions()
        space = self._env.single_action_space
        tds: list[TensorDict] = []
        for i in range(self.num_envs):
            arr = np.asarray(raw[i]).flatten()
            if isinstance(space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
                action = {"discrete": torch.tensor(arr, dtype=torch.int64)}
            else:
                action = {"continuous": torch.tensor(arr, dtype=torch.float32)}
            tds.append(TensorDict({"action": action}, batch_size=[]))
        return tds

    def step(self, actions: list[TensorDict]) -> tuple[list[TensorDict], dict, dict]:
        """Step all envs; return ``(data, metadata, metrics)``.

        On the first call after construction, performs an internal reset and returns
        initial observations (actions are ignored). Otherwise applies ``actions`` to
        all parallel envs.
        """
        if self._needs_initial_reset:
            self._needs_initial_reset = False
            obs, info = self._env.reset()
            return self._build_records(obs, info, is_reset=True)

        raw_actions = self._unpack_actions(actions)
        obs, reward, _terminated, _truncated, info = self._env.step(raw_actions)
        return self._build_records(obs, info, reward=reward, is_reset=False)

    def close(self) -> None:
        self._env.close()

    def _unpack_actions(self, actions: list[TensorDict]) -> np.ndarray:
        space = self._env.single_action_space
        if isinstance(space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
            raw = np.stack(
                [td["action"]["discrete"].numpy() for td in actions]
            ).squeeze(-1).astype(np.int64)
        else:
            raw = np.stack(
                [td["action"]["continuous"].numpy() for td in actions]
            ).astype(np.float32)
        return raw

    def _obs_for_index(self, obs: Any, i: int) -> dict[str, torch.Tensor]:
        """Build observation dict for env index ``i`` (may contain multiple keys)."""
        if isinstance(obs, dict):
            fields: dict[str, torch.Tensor] = {}
            for k, v in obs.items():
                arr = np.asarray(v[i]).flatten()
                dtype = torch.int64 if k == "discrete" else torch.float32
                fields[k] = torch.tensor(arr, dtype=dtype)
            return fields
        raw = np.asarray(obs[i]).flatten()
        if self.obs_key == "observation_discrete":
            return {"discrete": torch.tensor(raw, dtype=torch.int64)}
        if self.obs_key == "observation_image":
            return {"image": torch.tensor(raw, dtype=torch.float32)}
        return {"continuous": torch.tensor(raw, dtype=torch.float32)}

    def _build_metrics(self, info: dict, *, nan_episode_stats: bool = False) -> dict:
        nan_arr = np.full(self.num_envs, np.nan, dtype=np.float64)
        if nan_episode_stats:
            return {
                "episode_cum_reward": nan_arr.copy(),
                "episode_length": nan_arr.copy(),
            }
        return {
            "episode_cum_reward": (
                np.asarray(info["episode_cum_reward"], dtype=np.float64)
                if "episode_cum_reward" in info
                else nan_arr.copy()
            ),
            "episode_length": (
                np.asarray(info["episode_length"], dtype=np.float64)
                if "episode_length" in info
                else nan_arr.copy()
            ),
        }

    def _build_metadata(self, info: dict) -> dict:
        metadata: dict = {
            "group_ids": list(self._group_ids),
            "episode_index": np.asarray(info["episode_index"], dtype=np.int64),
        }
        if "metadata_q_star" in info:
            metadata["q_star"] = np.asarray(info["metadata_q_star"], dtype=np.float64)
        if "ns_params" in info:
            metadata["ns_params"] = info["ns_params"]
        return metadata

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
    ) -> tuple[list[TensorDict], dict, dict]:
        reset_mask = self._reset_frame_mask(info, is_reset=is_reset)

        if is_reset:
            reward_arr = np.zeros(self.num_envs, dtype=np.float32)
            xformed_arr = np.zeros(self.num_envs, dtype=np.float64)
            done_arr = np.zeros(self.num_envs, dtype=np.int64)
        else:
            reward_arr = np.asarray(reward, dtype=np.float32)
            xformed_arr = np.asarray(info["xformed_reward"], dtype=np.float64)
            done_arr = np.asarray(info["done"], dtype=np.int64)
            reward_arr[reset_mask] = 0.0
            xformed_arr[reset_mask] = 0.0
            done_arr[reset_mask] = DONE_RUNNING

        records: list[TensorDict] = []
        for i in range(self.num_envs):
            td = TensorDict(
                {
                    TIME_KEY: torch.tensor(
                        int(info["episode_time"][i]), dtype=torch.int64
                    ),
                    "observation": self._obs_for_index(obs, i),
                    "reward": {
                        REWARD_KEY_STEP: torch.tensor(
                            float(reward_arr[i]), dtype=torch.float32
                        ),
                        REWARD_KEY_EPISODIC: torch.tensor(
                            float(xformed_arr[i]), dtype=torch.float32
                        ),
                    },
                    "done": torch.tensor(int(done_arr[i]), dtype=torch.int64),
                },
                batch_size=[],
            )
            records.append(td)
        metrics = self._build_metrics(info, nan_episode_stats=is_reset)
        return records, self._build_metadata(info), metrics
