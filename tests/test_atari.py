"""Atari (ALE) tests — built through the general ``env_fn`` API.

mouse-env has no Atari-specific code. Atari is configured like any other env: build it in
an ``env_fn`` factory (register ``ale_py``, ``gym.make``, then ``AtariPreprocessing``).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from mouse_envs import EnvConfig, make_env, make_group_env

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
    env = make_group_env(
        [
            EnvConfig(
                id="ALE/Pong-v5",
                reset_seed=i,
                episodes_per_task=5,
                env_fn=_pong_factory(
                    500,
                    preprocess_kwargs={"frame_skip": 4, "screen_size": 84, "noop_max": 0},
                ),
            )
            for i in range(2)
        ]
    )
    try:
        outputs = env.step(env.sample_random_input())
        assert len(outputs) == 2

        batch = np.stack([r["observation"].numpy() for r in outputs])
        assert batch.shape == (2, 84, 84)
        assert batch.dtype == np.uint8

        assert env.output_specs[0].observation.shape == (84, 84)
        assert env.output_specs[0].observation.dtype == torch.uint8

        for r in outputs:
            assert r["time"].item() == 0
            assert r["reward"].item() == 0.0
            assert r["done"].item() == 0
    finally:
        env.close()


def test_atari_multi_step_rollout() -> None:
    cfg = EnvConfig(
        id="ALE/Pong-v5",
        reset_seed=1,
        episodes_per_task=5,
        env_fn=_pong_factory(500, preprocess_kwargs={"noop_max": 0}),
    )
    env = make_env(cfg)
    try:
        env.step(env.sample_random_input())
        output = env.step(env.sample_random_input())
        assert output["time"].item() >= 1
        assert "observation" in output
        assert output["observation"].dtype == torch.uint8
    finally:
        env.close()


def test_atari_discrete_action_sampling() -> None:
    cfg = EnvConfig(
        id="ALE/Pong-v5",
        reset_seed=0,
        episodes_per_task=5,
        env_fn=_pong_factory(100, preprocess_kwargs={"noop_max": 0}),
    )
    env = make_env(cfg)
    try:
        inp = env.sample_random_input()
        assert "action" in inp
        assert inp["action"].dtype == torch.int64
        assert env.input_spec.action.dtype == torch.int64
    finally:
        env.close()


def test_atari_without_preprocessing() -> None:
    cfg = EnvConfig(
        id="ALE/Pong-v5",
        reset_seed=0,
        episodes_per_task=5,
        env_fn=_pong_factory(100, preprocess_kwargs=None),
    )
    env = make_env(cfg)
    try:
        output = env.step(env.sample_random_input())
        # Raw ALE frames are RGB; observation keeps its native (210, 160, 3) shape.
        img = output["observation"]
        assert tuple(img.shape) == (210, 160, 3)
        assert env.output_spec.observation.shape == (210, 160, 3)
    finally:
        env.close()
