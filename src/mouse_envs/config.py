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
        seed: RNG seed used for mouse-env's internal Gymnasium reset stream when
            ``reset_seed`` is not set.
        reset_seed: Optional seed for mouse-env's internal Gymnasium reset stream.
        episodes_per_task: Number of episodes before the task terminates. Defaults to
            ``0`` (unlimited) — the task boundary (done codes 3/4) never fires
            automatically.
        name: Optional display name; overrides ``id`` for env instance naming.
        kwargs: Extra keyword arguments forwarded to ``gymnasium.make``.
        episode_reset_options: Extra options forwarded to every internal
            ``env.reset(options=...)``.
        task_reset_options: Extra options overlaid on top of ``episode_reset_options``
            when an internal reset starts a new task.
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
            set, ``id`` is used only for naming.
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
    reset_seed: int | None = None
    episodes_per_task: int = 0
    name: str | None = None
    kwargs: dict | None = None
    episode_reset_options: dict | None = None
    task_reset_options: dict | None = None
    render: bool = False
    q_star_source: dict[str, Any] | None = None
    env_fn: Callable[[], Any] | None = None
    observation_kind: str | None = None
    observation_indices: list[int] | None = None
    reward_scale: float = 1.0
    reward_shift: float = 0.0
    reset_reward: float = 0.0
