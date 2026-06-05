"""Atari (ALE) tests — built through the general ``env_fn`` + ``observation_kind`` API.

mouse-env has no Atari-specific code. Atari is configured like any other env: build it in
an ``env_fn`` factory (register ``ale_py``, ``gym.make``, then ``AtariPreprocessing``) and
route the image observation with ``observation_kind="image"``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from mouse_envs import EnvConfig, make_vector_env

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
        group_id="ALE/Pong-v5",
        seed=0,
        num_envs=2,
        max_episode_steps=500,
        observation_kind="image",
        env_fn=_pong_factory(
            500, preprocess_kwargs={"frame_skip": 4, "screen_size": 84, "noop_max": 0}
        ),
    )
    env = make_vector_env(cfg)
    try:
        assert env.obs_key == "observation_image"
        result, metrics = env.step(env.sample_random_actions())
        assert len(result) == 2
        assert len(metrics) == 2
        assert "group_id" in result[0]

        batch = np.stack([r["observation"]["image"].numpy() for r in result])
        assert batch.shape == (2, 84, 84)
        assert batch.dtype == np.float32

        for r in result:
            assert r["time"].item() == 0
            assert r["reward"].item() == 0.0
            assert r["done"].item() == 0
    finally:
        env.close()


def test_atari_multi_step_rollout() -> None:
    cfg = EnvConfig(
        group_id="ALE/Pong-v5",
        seed=1,
        num_envs=1,
        max_episode_steps=500,
        observation_kind="image",
        env_fn=_pong_factory(500, preprocess_kwargs={"noop_max": 0}),
    )
    env = make_vector_env(cfg)
    try:
        env.step(env.sample_random_actions())
        result, _metrics = env.step(env.sample_random_actions())
        assert result[0]["time"].item() >= 1
        assert "discrete" not in result[0]["observation"]
        assert "image" in result[0]["observation"]
    finally:
        env.close()


def test_atari_discrete_action_sampling() -> None:
    cfg = EnvConfig(
        group_id="ALE/Pong-v5",
        seed=0,
        num_envs=1,
        max_episode_steps=100,
        observation_kind="image",
        env_fn=_pong_factory(100, preprocess_kwargs={"noop_max": 0}),
    )
    env = make_vector_env(cfg)
    try:
        actions = env.sample_random_actions()
        assert len(actions) == 1
        assert "discrete" in actions[0]["action"]
    finally:
        env.close()


def test_atari_without_preprocessing() -> None:
    cfg = EnvConfig(
        group_id="ALE/Pong-v5",
        seed=0,
        num_envs=1,
        max_episode_steps=100,
        observation_kind="image",
        env_fn=_pong_factory(100, preprocess_kwargs=None),
    )
    env = make_vector_env(cfg)
    try:
        result, _metrics = env.step(env.sample_random_actions())
        # Raw ALE frames are RGB; observation.image keeps its native (210, 160, 3) shape.
        img = result[0]["observation"]["image"]
        assert tuple(img.shape) == (210, 160, 3)
    finally:
        env.close()
