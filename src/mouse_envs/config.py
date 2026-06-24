"""Environment configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

OBSERVATION_KINDS = ("continuous", "discrete", "image")


@dataclass
class EnvConfig:
    """Configuration for building an environment via :func:`mouse_envs.make_env`.

    Attributes:
        id: Gymnasium env ID (e.g. ``"CartPole-v1"``). Used as the base name when
            ``name`` is not set.
        seed: RNG seed applied to the env on construction.
        num_envs: Number of parallel env instances (slots).
        episodes_per_task: Number of episodes before the task terminates.
        name: Optional display name; overrides ``id`` for slot naming.
        kwargs: Extra keyword arguments forwarded to ``gymnasium.make``.
        render: Enable render mode (``"human"``).
        q_star_source: Optional dict that attaches expert Q-values to every step
            output as ``outputs[i]["info_env_q_star"]``. Must contain a
            ``"provider"`` key. Three providers are supported:

            ``"env_q_star"`` — env-computed Q* (no extra fields):

            .. code-block:: python

                {"provider": "env_q_star"}

            Only works with ``SyntheticEnv-v1`` and ``Procedural-FrozenLake-v1``.

            ``"hf_q_table"`` — tabular Q-table from a pickle file:

            - ``"path"`` *(str)* — local path to a ``.pkl`` file, **or**
            - ``"repo_id"`` *(str)* + ``"filename"`` *(str, default* ``"q-learning.pkl"``*)* — download from HF Hub
            - ``"deterministic"`` *(bool, default* ``True``*)* — argmax action selection

            The pickle must be ``{"qtable": ndarray[states, actions]}`` or a bare ``ndarray``.

            ``"sb3_rl_zoo"`` — Stable-Baselines3 checkpoint (requires ``stable-baselines3``):

            - ``"algo"`` *(str, required)* — ``"a2c"``, ``"ddpg"``, ``"dqn"``, ``"ppo"``, ``"sac"``, ``"td3"``, ``"qrdqn"``
            - ``"path"`` *(str)* — local path to an SB3 ``.zip`` file, **or**
            - ``"repo_id"`` *(str)* + ``"filename"`` *(str)* — download from HF Hub
            - ``"device"`` *(str, default* ``"cpu"``*)* — ``"cpu"``, ``"cuda"``, or ``"auto"``
            - ``"deterministic"`` *(bool, default* ``True``*)* — deterministic action selection

        env_fn: Zero-arg factory that returns a freshly built Gymnasium env. When
            set, ``id`` is used only for naming; the factory is called once per slot.
        observation_kind: Force the observation channel (``"continuous"``,
            ``"discrete"``, or ``"image"``). Defaults to auto-detection; required
            for image envs.
        observation_indices: Mask dimensions on continuous-vector observations.
        reward_scale: Multiply the raw reward before it appears in outputs.
        reward_shift: Add to the (already scaled) reward.
        reset_reward: Reward value injected into the reset frame (default ``0.0``).
    """

    id: str
    seed: int
    num_envs: int
    episodes_per_task: int
    name: str | None = None
    kwargs: dict | None = None
    render: bool = False
    q_star_source: dict[str, Any] | None = None
    env_fn: Callable[[], Any] | None = None
    observation_kind: str | None = None
    observation_indices: list[int] | None = None
    reward_scale: float = 1.0
    reward_shift: float = 0.0
    reset_reward: float = 0.0
