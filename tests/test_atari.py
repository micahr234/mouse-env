"""Atari (ALE) integration tests — offline when ``ale_py`` ROMs are installed."""

from __future__ import annotations

import numpy as np
import pytest

from mouse.envs import EnvConfig, make_vector_env
from mouse.envs.integrations.atari import ensure_ale_registered, is_ale_env

ale_py = pytest.importorskip("ale_py")


def test_is_ale_env() -> None:
    assert is_ale_env("ALE/Pong-v5")
    assert not is_ale_env("CartPole-v1")


def test_ensure_ale_registered() -> None:
    ensure_ale_registered()


def test_atari_vector_preprocessing() -> None:
    cfg = EnvConfig.atari(
        "ALE/Pong-v5",
        seed=0,
        num_envs=2,
        max_episode_steps=500,
        frame_skip=4,
        screen_size=84,
        noop_max=0,
    )
    env = make_vector_env(cfg)
    try:
        assert env.obs_key == "observation_image"
        data, metadata, metrics = env.step(env.sample_random_actions())
        assert len(data) == 2
        assert len(metadata) == 2
        assert len(metrics) == 2
        assert "group_id" in metadata[0]

        batch = np.stack([td["observation"]["image"].numpy() for td in data])
        assert batch.shape == (2, 84 * 84)
        assert batch.dtype == np.float32

        for td in data:
            assert td["time"].item() == 0
            assert td["reward"].item() == 0.0
            assert td["done"].item() == 0
    finally:
        env.close()


def test_atari_multi_step_rollout() -> None:
    cfg = EnvConfig.atari(
        "ALE/Pong-v5",
        seed=1,
        num_envs=1,
        max_episode_steps=500,
        noop_max=0,
    )
    env = make_vector_env(cfg)
    try:
        env.step(env.sample_random_actions())
        data, _metadata, _metrics = env.step(env.sample_random_actions())
        assert data[0]["time"].item() >= 1
        assert "discrete" not in data[0]["observation"]
        assert "image" in data[0]["observation"]
    finally:
        env.close()


def test_atari_discrete_action_sampling() -> None:
    cfg = EnvConfig.atari("ALE/Pong-v5", seed=0, num_envs=1, max_episode_steps=100, noop_max=0)
    env = make_vector_env(cfg)
    try:
        actions = env.sample_random_actions()
        assert len(actions) == 1
        assert "discrete" in actions[0]["action"]
    finally:
        env.close()


def test_atari_rejects_observation_indices() -> None:
    cfg = EnvConfig.atari(
        "ALE/Pong-v5",
        seed=0,
        num_envs=1,
        max_episode_steps=100,
        observation_indices=[0, 1, 2],
    )
    with pytest.raises(ValueError, match="observation_indices is not supported"):
        make_vector_env(cfg)


def test_atari_without_preprocessing() -> None:
    cfg = EnvConfig(
        group_id="ALE/Pong-v5",
        seed=0,
        num_envs=1,
        max_episode_steps=100,
        atari_preprocessing=False,
        kwargs={"frameskip": 1},
    )
    env = make_vector_env(cfg)
    try:
        data, _metadata, _metrics = env.step(env.sample_random_actions())
        # Raw ALE frames are RGB; vector stack still flattens to observation.image.
        img = data[0]["observation"]["image"]
        assert img.numel() == 210 * 160 * 3
    finally:
        env.close()
