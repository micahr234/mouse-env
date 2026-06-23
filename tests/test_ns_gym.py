"""Non-stationary env tests — built through the general ``env_fn`` factory.

mouse-env has no NS-Gym integration code. A non-stationary env is built like any other
custom env: construct the NS-Gym wrapper in an ``env_fn`` and apply a small adapter that
flattens NS-Gym's dict observation and exposes ``info["ns_params"]`` (which mouse-env
surfaces as ``results[i]["ns_params"]``).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from mouse_envs import EnvConfig, make_vector_env

pytest.importorskip("ns_gym")
from ns_gym.schedulers import ContinuousScheduler  # noqa: E402
from ns_gym.update_functions import OscillatingUpdate  # noqa: E402
from ns_gym.wrappers import NSClassicControlWrapper  # noqa: E402


class NSAdapter(gym.Wrapper):
    """Flatten NS-Gym dict obs to its ``state`` and expose ground-truth changes as ns_params."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        space = env.observation_space
        if isinstance(space, gym.spaces.Dict) and "state" in space.spaces:
            self.observation_space = space["state"]

    @staticmethod
    def _adapt(obs: Any, info: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        state = obs["state"] if isinstance(obs, dict) and "state" in obs else obs
        env_change = info.get("Ground Truth Env Change", {}) or {}
        delta = info.get("Ground Truth Delta Change", {}) or {}
        ns_params: dict[str, Any] = {}
        for key, flag in env_change.items():
            if not key.startswith("_"):
                ns_params[key] = np.asarray(delta.get(key, 0))
                ns_params[f"{key}_flag"] = np.asarray(flag)
        return state, {"ns_params": ns_params}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._adapt(obs, info)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state, info = self._adapt(obs, info)
        return state, reward, terminated, truncated, info


def _make_ns_cartpole() -> gym.Env:
    update_fns = {"length": OscillatingUpdate(ContinuousScheduler(), delta=0.01)}
    base = gym.make("CartPole-v1", max_episode_steps=500)
    ns_env = NSClassicControlWrapper(
        base, update_fns, change_notification=True, delta_change_notification=True
    )
    return NSAdapter(ns_env)


def test_non_stationary_cartpole() -> None:
    cfg = EnvConfig(
        id="CartPole-ns",
        seed=0,
        num_envs=2,
        max_episode_steps=500,
        env_fn=_make_ns_cartpole,
    )
    env = make_vector_env(cfg)
    try:
        result, _metrics = env.step(env.sample_random_actions())
        for _ in range(3):
            result, _metrics = env.step(env.sample_random_actions())
        assert "ns_params" in result[0]
        assert "length" in result[0]["ns_params"]
        assert "continuous" in result[0]["observation"]
    finally:
        env.close()
