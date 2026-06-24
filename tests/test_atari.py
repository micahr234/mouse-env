"""Atari (ALE) tests — built through the general ``env_fn`` + ``observation_kind`` API.

mouse-env has no Atari-specific code. Atari is configured like any other env: build it in
an ``env_fn`` factory (register ``ale_py``, ``gym.make``, then ``AtariPreprocessing``) and
route the image observation with ``observation_kind="image"``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from mouse_envs import EnvConfig, make_env

ale_py = pytest.importorskip("ale_py")
pytest.importorskip("gymnasium.wrappers.atari_preprocessing")
from gymnasium.wrappers import AtariPreprocessing  # noqa: E402

gym.register_envs(ale_py)


def _pong_factory(max_episode_steps: int, *, preprocess_kwargs: dict | None):
    def make() -> gym.Env:
        env = gym.make("ALE/Pong-v5", frameskip=1, max_episode_steps=max_episode_steps)
        if preprocess_kwargs is not None:
            env = AtariPreprocessing(env, **preprocess_kwargs)
        return env

    return make


def test_atari_vector_preprocessing() -> None:
    cfg = EnvConfig(
        id="ALE/Pong-v5",
        seed=0,
        num_envs=2,
        episodes_per_task=5,
        observation_kind="image",
        env_fn=_pong_factory(
            500, preprocess_kwargs={"frame_skip": 4, "screen_size": 84, "noop_max": 0}
        ),
    )
    env = make_env(cfg)
    try:
        assert env._slots[0].obs_key == "observation_image"
        outputs = env.step(env.sample_random_inputs())
        assert len(outputs) == 2

        batch = np.stack([r["observation"].numpy() for r in outputs])
        assert batch.shape == (2, 84, 84)
        assert batch.dtype == np.float32

        assert env.output_specs[0].observation.shape == (84, 84)
        assert env.output_specs[0].observation.dtype == torch.float32

        for r in outputs:
            assert r["time"].item() == 0
            assert r["reward"].item() == 0.0
            assert r["done"].item() == 0
    finally:
        env.close()


def test_atari_multi_step_rollout() -> None:
    cfg = EnvConfig(
        id="ALE/Pong-v5",
        seed=1,
        num_envs=1,
        episodes_per_task=5,
        observation_kind="image",
        env_fn=_pong_factory(500, preprocess_kwargs={"noop_max": 0}),
    )
    env = make_env(cfg)
    try:
        env.step(env.sample_random_inputs())
        outputs = env.step(env.sample_random_inputs())
        assert outputs[0]["time"].item() >= 1
        assert "observation" in outputs[0]
        assert outputs[0]["observation"].dtype == torch.float32
    finally:
        env.close()


def test_atari_discrete_action_sampling() -> None:
    cfg = EnvConfig(
        id="ALE/Pong-v5",
        seed=0,
        num_envs=1,
        episodes_per_task=5,
        observation_kind="image",
        env_fn=_pong_factory(100, preprocess_kwargs={"noop_max": 0}),
    )
    env = make_env(cfg)
    try:
        inputs = env.sample_random_inputs()
        assert len(inputs) == 1
        assert "action" in inputs[0]
        assert inputs[0]["action"].dtype == torch.int64
        assert env.input_specs[0].action.dtype == torch.int64
    finally:
        env.close()


def test_atari_without_preprocessing() -> None:
    cfg = EnvConfig(
        id="ALE/Pong-v5",
        seed=0,
        num_envs=1,
        episodes_per_task=5,
        observation_kind="image",
        env_fn=_pong_factory(100, preprocess_kwargs=None),
    )
    env = make_env(cfg)
    try:
        outputs = env.step(env.sample_random_inputs())
        # Raw ALE frames are RGB; observation keeps its native (210, 160, 3) shape.
        img = outputs[0]["observation"]
        assert tuple(img.shape) == (210, 160, 3)
        assert env.output_specs[0].observation.shape == (210, 160, 3)
    finally:
        env.close()
