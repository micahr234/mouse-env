"""Expert policy adapters for injecting Q-values and optimal actions into rollout info.

Provides:
- ExpertPolicyAdapter: dataclass that wraps a loaded policy and exposes a uniform interface
  for deriving ``metadata_q_star`` from env info, observation queries, or action hints.
- build_q_star_source_adapter: factory that reads a ``q_star_source`` config dict and
  returns the appropriate adapter (SB3 policy, tabular Q-table, or env-info passthrough).
- action_star_to_one_hot_q_star: convert integer expert actions to one-hot Q-value rows.
- apply_q_star_source_env_kwargs: inject ``emit_q_star`` kwargs for first-party worlds.
"""

from __future__ import annotations

import contextlib
import io
import pickle
import sys
import warnings
import zipfile
from dataclasses import dataclass
from typing import Any, Iterator

import gymnasium as gym
import numpy as np
from huggingface_hub import hf_hub_download

from mouse.envs.env_ids import PROCEDURAL_FROZENLAKE_ENV_ID, SYNTHETIC_ENV_ID


def _to_batched_action_star(value: Any, num_envs: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim == 0:
        arr = np.full((num_envs,), int(arr), dtype=np.int64)
    if arr.ndim >= 1 and arr.shape[0] != num_envs:
        raise ValueError(
            f"expert policy produced action_star batch with shape {arr.shape}, "
            f"expected first dim {num_envs}."
        )
    return arr


def _to_batched_q_values(value: Any, num_envs: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        if num_envs != 1:
            raise ValueError(
                f"expert policy produced q-values with shape {arr.shape}, "
                f"expected first dim {num_envs}."
            )
        arr = arr[None, :]
    if arr.ndim < 2 or arr.shape[0] != num_envs:
        raise ValueError(
            f"expert policy produced q-values batch with shape {arr.shape}, "
            f"expected first dim {num_envs}."
        )
    return arr


def action_star_to_one_hot_q_star(actions: np.ndarray, num_actions: int) -> np.ndarray:
    """Convert integer expert actions to one-hot Q-value rows.

    Produces a ``[num_envs, num_actions]`` float64 array with 1.0 at the expert
    action index and 0.0 elsewhere, suitable for use as ``metadata_q_star`` when
    the environment only exposes a scalar expert action rather than full Q-values.
    """
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    one_hot = np.zeros((actions.shape[0], num_actions), dtype=np.float64)
    one_hot[np.arange(actions.shape[0]), actions] = 1.0
    return one_hot


@dataclass
class ExpertPolicyAdapter:
    """Adapter that extracts expert Q-values or optimal actions from various sources.

    Used internally by :class:`~mouse.envs.wrappers.QStarWrapper` to populate
    ``info["metadata_q_star"]`` on each step. Instantiated by
    :func:`build_q_star_source_adapter`; do not construct directly.

    Attributes:
        name (str): Normalised provider name (``"metadata_q_star"``, ``"sb3_rl_zoo"``,
            or ``"hf_q_table"``).
        require_metadata_action_star (bool): When ``True``, :meth:`action_star_from_infos`
            raises if ``"action_star"`` is missing from the env info dict.
        external_policy: Loaded SB3 policy or :class:`_TabularQPolicy` instance, or
            ``None`` when Q-values come directly from env info.
        deterministic (bool): Whether to use deterministic (argmax) action selection
            when querying ``external_policy.predict``.
        obs_key (str): Canonical observation key used when preparing observations for
            the external policy.
        external_frame_stack (int): Number of stacked frames expected by the policy
            (inferred from the policy's observation space shape). ``0`` or ``1`` means
            no stacking.
        external_obs_history (np.ndarray | None): Rolling frame buffer for policies that
            require frame stacking. Managed internally.
    """

    name: str
    require_metadata_action_star: bool = True
    external_policy: Any | None = None
    deterministic: bool = True
    obs_key: str = "observation"
    external_frame_stack: int = 0
    external_obs_history: np.ndarray | None = None

    def action_star_from_infos(
        self,
        infos: dict[str, Any],
        num_envs: int,
    ) -> np.ndarray | None:
        """Extract expert actions from ``info["action_star"]`` if present.

        Args:
            infos: Step info dict from the env.
            num_envs: Expected batch size for shape validation.

        Returns:
            ``int64[num_envs]`` array of expert actions, or ``None`` if the key is absent
            and ``require_metadata_action_star`` is ``False``.

        Raises:
            ValueError: If the key is absent and ``require_metadata_action_star`` is ``True``.
        """
        value = infos.get("action_star", None)
        if value is None:
            if self.require_metadata_action_star:
                raise ValueError(
                    f"q_star_source={self.name!r} requires env info key 'action_star'. "
                    "Make sure the env emits action_star metadata."
                )
            return None
        return _to_batched_action_star(value=value, num_envs=num_envs)

    def q_star_from_infos(
        self,
        infos: dict[str, Any],
        num_envs: int,
    ) -> np.ndarray | None:
        """Extract Q-values from ``info["q_star"]`` if present.

        Args:
            infos: Step info dict from the env.
            num_envs: Expected batch size for shape validation.

        Returns:
            ``float64[num_envs, action_dim]`` array, or ``None`` if the key is absent.
        """
        value = infos.get("q_star", None)
        if value is None:
            return None
        return _to_batched_q_values(value=value, num_envs=num_envs)

    def q_star_from_action_star_infos(
        self,
        infos: dict[str, Any],
        num_envs: int,
        num_actions: int,
    ) -> np.ndarray | None:
        """Derive one-hot Q-values from ``info["action_star"]``.

        Returns ``None`` if ``action_star`` is absent or ``num_actions`` is zero.

        Args:
            infos: Step info dict from the env.
            num_envs: Expected batch size for shape validation.
            num_actions: Number of actions in the action space (determines one-hot width).

        Returns:
            ``float64[num_envs, num_actions]`` one-hot array, or ``None``.
        """
        if num_actions <= 0:
            return None
        value = infos.get("action_star", None)
        if value is None:
            return None
        actions = _to_batched_action_star(value=value, num_envs=num_envs)
        return action_star_to_one_hot_q_star(actions=actions, num_actions=num_actions)

    def action_star_from_observation(
        self,
        obs: np.ndarray,
        done_mask: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Query ``external_policy.predict(obs)`` to get expert actions.

        Args:
            obs: Current observations, shape ``(num_envs, *obs_shape)``.
            done_mask: Boolean array of shape ``(num_envs,)`` indicating which envs
                just finished an episode (used to reset frame-stack history).

        Returns:
            ``int64[num_envs]`` expert actions, or ``None`` if no external policy is set.
        """
        if self.external_policy is None:
            return None
        obs_in = self._policy_obs(obs=obs, done_mask=done_mask)
        actions, _ = self.external_policy.predict(obs_in, deterministic=self.deterministic)
        return np.asarray(actions, dtype=np.int64)

    def q_star_from_observation(
        self,
        obs: np.ndarray,
        done_mask: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Query ``external_policy.predict_q(obs)`` to get Q-values directly.

        Falls back to ``None`` if the policy does not expose a ``predict_q`` method
        (e.g. SB3 policy-gradient algorithms).

        Args:
            obs: Current observations, shape ``(num_envs, *obs_shape)``.
            done_mask: Boolean array indicating episode boundaries for frame-stack reset.

        Returns:
            ``float64[num_envs, action_dim]`` Q-values, or ``None``.
        """
        if self.external_policy is None:
            return None
        predict_q = getattr(self.external_policy, "predict_q", None)
        if not callable(predict_q):
            return None
        obs_in = self._policy_obs(obs=obs, done_mask=done_mask)
        return _to_batched_q_values(value=predict_q(obs_in), num_envs=obs_in.shape[0])

    def _policy_obs(self, obs: np.ndarray, done_mask: np.ndarray | None) -> np.ndarray:
        if self.external_frame_stack <= 1:
            arr = np.asarray(obs)
            # SyncVectorEnv batches Discrete observations as (n, 1); SB3 expects (n,)
            if arr.ndim == 2 and arr.shape[1] == 1 and self.external_policy is not None:
                from gymnasium.spaces import Discrete
                pol_obs_space = getattr(self.external_policy, "observation_space", None)
                if isinstance(pol_obs_space, Discrete):
                    arr = arr.squeeze(axis=1)
            return arr
        arr = np.asarray(obs)
        if arr.ndim != 3:
            return arr
        if self.external_obs_history is None:
            self.external_obs_history = np.repeat(
                arr[:, None, :, :], self.external_frame_stack, axis=1
            )
        else:
            if done_mask is not None:
                done_bool = np.asarray(done_mask, dtype=bool)
                if np.any(done_bool):
                    self.external_obs_history[done_bool] = np.repeat(
                        arr[done_bool, None, :, :], self.external_frame_stack, axis=1
                    )
            self.external_obs_history[:, :-1, :, :] = self.external_obs_history[:, 1:, :, :]
            self.external_obs_history[:, -1, :, :] = arr
        return self.external_obs_history


def _const_schedule(value: float):
    return lambda _: value


def _ensure_gym_import_compatibility() -> None:
    if "gym" in sys.modules:
        return
    try:
        import gym as _gym  # type: ignore  # noqa: F401
        return
    except Exception:
        pass
    import gymnasium as gymnasium_mod
    sys.modules["gym"] = gymnasium_mod


@contextlib.contextmanager
def _sb3_zip_torch_load_compat() -> Iterator[None]:
    """Stabilize SB3 expert checkpoint loading with current PyTorch.

    ``stable_baselines3`` opens each ``.pth`` inside the SB3 zip via ``ZipFile.open`` and passes
    that stream to ``torch.load(..., weights_only=True)``. PyTorch's zip reader often fails on
    those non-seekable streams (``RuntimeError: PytorchStreamReader failed reading file version``
    / miniz). Buffer each shard and use ``weights_only=False`` (SB3 checkpoints are trusted local
    or HF hub files) only while loading the expert policy.
    """
    import torch

    real_load = torch.load

    def _compat_load(f: Any, *args: Any, **kwargs: Any) -> Any:
        kw = dict(kwargs)
        kw["weights_only"] = False
        if isinstance(f, zipfile.ZipExtFile):
            return real_load(io.BytesIO(f.read()), *args, **kw)
        return real_load(f, *args, **kw)

    torch.load = _compat_load  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.load = real_load  # type: ignore[method-assign]


def _to_gymnasium_space(space: Any) -> Any:
    from gymnasium import spaces

    if isinstance(
        space,
        (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary, spaces.Dict, spaces.Tuple),
    ):
        return space

    cls_name = type(space).__name__
    if cls_name == "Box":
        low = np.asarray(space.low)
        high = np.asarray(space.high)
        return spaces.Box(low=low, high=high, shape=space.shape, dtype=space.dtype)
    if cls_name == "Discrete":
        start = int(getattr(space, "start", 0))
        return spaces.Discrete(int(space.n), start=start)
    if cls_name == "MultiDiscrete":
        return spaces.MultiDiscrete(np.asarray(space.nvec, dtype=np.int64))
    if cls_name == "MultiBinary":
        return spaces.MultiBinary(space.n)
    if cls_name == "Dict":
        return spaces.Dict({k: _to_gymnasium_space(v) for k, v in dict(space.spaces).items()})
    if cls_name == "Tuple":
        return spaces.Tuple(tuple(_to_gymnasium_space(v) for v in tuple(space.spaces)))
    return space


def _load_space_custom_objects(
    model_path: str,
    custom_objects: dict[str, Any],
) -> dict[str, Any]:
    from gymnasium import spaces
    from stable_baselines3.common.save_util import load_from_zip_file

    try:
        data, _, _ = load_from_zip_file(
            model_path, device="cpu", custom_objects=custom_objects
        )
    except ModuleNotFoundError:
        return {}
    if data is None:
        return {}
    out: dict[str, Any] = {}
    obs_space = data.get("observation_space")
    if obs_space is not None and not isinstance(
        obs_space,
        (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary, spaces.Dict, spaces.Tuple),
    ):
        out["observation_space"] = _to_gymnasium_space(obs_space)
    action_space = data.get("action_space")
    if action_space is not None and not isinstance(
        action_space,
        (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary, spaces.Dict, spaces.Tuple),
    ):
        out["action_space"] = _to_gymnasium_space(action_space)
    return out


def _obs_space_dims_from_space(space: Any) -> tuple[int, ...] | None:
    if isinstance(space, gym.spaces.Discrete):
        return (int(space.n),)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return tuple(int(x) for x in np.asarray(space.nvec, dtype=np.int64).ravel())
    if isinstance(space, gym.spaces.Tuple):
        dims: list[int] = []
        for s in space.spaces:
            if not isinstance(s, gym.spaces.Discrete):
                return None
            dims.append(int(s.n))
        return tuple(dims)
    return None


class _TabularQPolicy:
    """Simple tabular policy with an SB3-like ``predict`` interface."""

    def __init__(self, qtable: np.ndarray, obs_space_dims: tuple[int, ...] | None = None):
        qt = np.asarray(qtable)
        if qt.ndim != 2:
            raise ValueError(f"qtable must have shape [num_states, num_actions], got {qt.shape}.")
        self._qtable = qt
        self._obs_space_dims = tuple(int(x) for x in obs_space_dims) if obs_space_dims else None

    def _to_state_index(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs)
        if arr.ndim == 0:
            arr = arr.reshape(1)

        if self._obs_space_dims is not None:
            dims = self._obs_space_dims
            if arr.ndim == 2 and int(arr.shape[0]) == len(dims) and int(arr.shape[-1]) != len(dims):
                arr = np.swapaxes(arr, 0, 1)
            if arr.ndim == 1 and arr.shape[0] == len(dims):
                arr = arr.reshape(1, -1)
            if arr.ndim != 2 or int(arr.shape[-1]) != len(dims):
                raise ValueError(
                    f"Expected batched observations with shape [N, {len(dims)}], got {arr.shape}."
                )
            obs_i = arr.astype(np.int64, copy=False)
            for d, size in enumerate(dims):
                obs_i[:, d] = np.clip(obs_i[:, d], 0, size - 1)
            state_idx = np.ravel_multi_index(obs_i.T, dims=dims)
            return np.atleast_1d(state_idx).astype(np.int64, copy=False)

        state_idx = np.asarray(arr, dtype=np.int64).reshape(-1)
        state_idx = np.clip(state_idx, 0, self._qtable.shape[0] - 1)
        return state_idx

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        del deterministic
        idx = self._to_state_index(obs)
        qvals = self._qtable[idx]
        actions = np.asarray(np.argmax(qvals, axis=-1), dtype=np.int64)
        return actions, None

    def predict_q(self, obs: np.ndarray) -> np.ndarray:
        idx = self._to_state_index(obs)
        return np.asarray(self._qtable[idx], dtype=np.float64)


def _infer_external_frame_stack(*, policy: Any, obs_key: str) -> int:
    obs_space = getattr(policy, "observation_space", None)
    shape = getattr(obs_space, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 3:
        return 0
    if obs_key != "observation_image":
        return 0
    channels = int(shape[0])
    return channels if channels > 1 else 0


def normalize_q_star_source_name(q_star_source: dict[str, Any] | None) -> str | None:
    if q_star_source is None:
        return None
    raw_name = str(q_star_source.get("name", "")).strip().lower()
    raw_provider = str(q_star_source.get("provider", "")).strip().lower()
    raw = raw_name or raw_provider
    if raw in ("", "none", "off", "disabled"):
        return None
    aliases = {
        "metadata_q_star": "metadata_q_star",
        "info_q_star": "metadata_q_star",
        "env_q_star": "metadata_q_star",
        "sb3_rl_zoo": "sb3_rl_zoo",
        "hf_q_table": "hf_q_table",
        "q_table": "hf_q_table",
        "qlearning": "hf_q_table",
    }
    if raw not in aliases:
        raise ValueError(
            f"Unsupported q_star_source.name={raw!r}. "
            "Supported: 'metadata_q_star' (or 'env_q_star' / 'info_q_star'), "
            "'sb3_rl_zoo', or 'hf_q_table'."
        )
    return aliases[raw]


def build_q_star_source_adapter(
    env_id: str,
    q_star_source: dict[str, Any] | None,
    obs_key: str = "observation",
    single_observation_space: Any | None = None,
) -> ExpertPolicyAdapter | None:
    """Build an :class:`ExpertPolicyAdapter` from a ``q_star_source`` config dict.

    Resolves the provider name and loads the appropriate expert policy:

    - ``"metadata_q_star"`` (aliases: ``"env_q_star"``, ``"info_q_star"``) — reads
      Q-values or action hints directly from env info keys; no external model loaded.
    - ``"sb3_rl_zoo"`` — downloads and loads an SB3 policy checkpoint from the
      Hugging Face Hub (or a local ``path``).
    - ``"hf_q_table"`` (alias: ``"q_table"``) — downloads and loads a tabular Q-table
      pickle from the Hub (or a local ``path``).

    Args:
        env_id: Environment id; used when selecting the correct SB3 algo or table.
        q_star_source: Config dict with at minimum a ``"provider"`` (or ``"name"``) key.
            ``None`` or an empty/disabled config returns ``None``.
        obs_key: Canonical observation key forwarded to the adapter (used for
            frame-stack inference and observation pre-processing).
        single_observation_space: The env's ``single_observation_space``; used to
            infer ``obs_space_dims`` for multi-dimensional tabular Q-tables.

    Returns:
        An :class:`ExpertPolicyAdapter` instance, or ``None`` if ``q_star_source`` is
        disabled/absent.

    Raises:
        ValueError: If the provider name is unrecognised, required keys are missing,
            or the loaded checkpoint is malformed.
        ImportError: If ``sb3_rl_zoo`` is requested but ``stable-baselines3`` is not
            installed, or ``hf_q_table`` requires ``huggingface_hub`` which is missing.
    """
    policy_name = normalize_q_star_source_name(q_star_source)
    if policy_name is None:
        return None
    if policy_name == "metadata_q_star":
        return ExpertPolicyAdapter(name=policy_name, require_metadata_action_star=False)

    cfg = dict(q_star_source or {})
    deterministic = bool(cfg.get("deterministic", True))

    if policy_name == "hf_q_table":
        model_path = cfg.get("path", None)
        if model_path is None:
            repo_id = str(cfg.get("repo_id", "")).strip()
            filename = str(cfg.get("filename", "")).strip() or "q-learning.pkl"
            if not repo_id:
                raise ValueError(
                    "q_star_source.provider=hf_q_table requires either 'path' or 'repo_id'."
                )
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )
        loaded: Any | None = None
        load_errors: list[str] = []
        for encoding in (None, "latin1", "bytes"):
            try:
                kw: dict[str, Any] = {} if encoding is None else {"encoding": encoding}
                with open(model_path, "rb") as f:
                    loaded = pickle.load(f, **kw)
                break
            except Exception as e:
                load_errors.append(str(e))
        if loaded is None:
            raise ValueError(
                "Could not load tabular policy pickle from q_star_source. "
                f"Tried standard encodings and got: {load_errors!r}"
            )
        qtable = loaded.get("qtable", None) if isinstance(loaded, dict) else loaded
        if qtable is None:
            raise ValueError(
                "Tabular expert policy file must be a qtable array or dict containing 'qtable'."
            )
        obs_space_dims: tuple[int, ...] | None = None
        if isinstance(loaded, dict) and loaded.get("obs_space_dims", None) is not None:
            obs_space_dims = tuple(int(x) for x in loaded["obs_space_dims"])
        if obs_space_dims is None and single_observation_space is not None:
            inferred = _obs_space_dims_from_space(single_observation_space)
            if inferred is not None:
                qtable_arr = np.asarray(qtable)
                num_states = int(qtable_arr.shape[0]) if qtable_arr.ndim >= 1 else 0
                inferred_states = int(np.prod(inferred))
                if inferred_states == num_states:
                    obs_space_dims = inferred
        policy = _TabularQPolicy(qtable=np.asarray(qtable), obs_space_dims=obs_space_dims)
        return ExpertPolicyAdapter(
            name=policy_name,
            require_metadata_action_star=False,
            external_policy=policy,
            deterministic=deterministic,
            obs_key=obs_key,
            external_frame_stack=0,
        )

    if policy_name != "sb3_rl_zoo":
        raise ValueError(f"Unsupported expert policy adapter kind: {policy_name!r}.")

    _ensure_gym_import_compatibility()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
            from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
    except Exception as e:
        raise ImportError(
            "q_star_source.provider=sb3_rl_zoo requires stable-baselines3."
        ) from e

    algo = str(cfg.get("algo", "")).strip().lower()
    algo_to_cls = {
        "a2c": A2C,
        "ddpg": DDPG,
        "dqn": DQN,
        "ppo": PPO,
        "sac": SAC,
        "td3": TD3,
    }
    if algo == "qrdqn":
        try:
            from sb3_contrib import QRDQN  # type: ignore
        except Exception as e:
            raise ImportError(
                "q_star_source.algo='qrdqn' requires sb3-contrib."
            ) from e
        algo_to_cls["qrdqn"] = QRDQN
    model_cls = algo_to_cls.get(algo)
    if model_cls is None:
        raise ValueError(
            f"Unsupported q_star_source.algo={algo!r}. Supported: {sorted(algo_to_cls)}."
        )

    model_path = cfg.get("path", None)
    if model_path is None:
        repo_id = str(cfg.get("repo_id", "")).strip()
        filename = str(cfg.get("filename", "")).strip()
        if not repo_id or not filename:
            raise ValueError(
                "q_star_source requires either 'path' or both 'repo_id' and 'filename'."
            )
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": _const_schedule(0.0),
        "clip_range": _const_schedule(0.0),
        "clip_range_vf": _const_schedule(0.0),
        "exploration_schedule": _const_schedule(0.0),
    }
    sb3_device = str(cfg.get("device", "cpu")).strip().lower() or "cpu"
    if sb3_device not in ("cpu", "cuda", "auto"):
        raise ValueError(
            f"Unsupported q_star_source.device={sb3_device!r}. Supported: 'cpu', 'cuda', or 'auto'."
        )
    with _sb3_zip_torch_load_compat():
        custom_objects.update(
            _load_space_custom_objects(model_path=str(model_path), custom_objects=custom_objects)
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="You are probably loading a DQN model saved with SB3 < 2.4.0"
            )
            policy = model_cls.load(
                model_path,
                device=sb3_device,
                custom_objects=custom_objects,
            )
    return ExpertPolicyAdapter(
        name=policy_name,
        require_metadata_action_star=False,
        external_policy=policy,
        deterministic=deterministic,
        obs_key=obs_key,
        external_frame_stack=_infer_external_frame_stack(policy=policy, obs_key=obs_key),
    )


def apply_q_star_source_env_kwargs(
    env_id: str,
    env_kwargs: dict[str, Any],
    q_star_source: dict[str, Any] | None,
) -> dict[str, Any]:
    """Inject any env-construction kwargs required by the ``q_star_source`` config.

    For first-party worlds (Procedural Frozen Lake / Synthetic Environment) with ``provider="metadata_q_star"``,
    this sets ``emit_q_star=True`` in ``env_kwargs`` so the env solves the tabular MDP
    and emits Q-values into ``info["q_star"]`` each step.

    Args:
        env_id: Gymnasium env id used to determine whether the env supports
            ``emit_q_star``.
        env_kwargs: Current env kwargs dict (not mutated; a copy is returned).
        q_star_source: Expert source config; see :class:`~mouse.envs.config.EnvConfig`.

    Returns:
        A new ``dict`` with any required keys added. Returns ``env_kwargs`` unchanged
        when ``q_star_source`` is ``None`` or the env does not support ``emit_q_star``.
    """
    policy_name = normalize_q_star_source_name(q_star_source)
    if policy_name is None:
        return env_kwargs

    out = dict(env_kwargs)
    emit_q_star = bool((q_star_source or {}).get("emit_q_star", False))
    if policy_name == "metadata_q_star" and env_id in (PROCEDURAL_FROZENLAKE_ENV_ID, SYNTHETIC_ENV_ID):
        emit_q_star = True
    if emit_q_star and env_id in (PROCEDURAL_FROZENLAKE_ENV_ID, SYNTHETIC_ENV_ID):
        out["emit_q_star"] = True

    if env_id == PROCEDURAL_FROZENLAKE_ENV_ID and policy_name == "metadata_q_star":
        random_map_wrapper = out.get("random_map_wrapper", None)
        if isinstance(random_map_wrapper, dict):
            wrapped = dict(random_map_wrapper)
            if emit_q_star:
                wrapped["emit_q_star"] = True
            out["random_map_wrapper"] = wrapped
    return out
