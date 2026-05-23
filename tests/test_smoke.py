"""Smoke tests for mouse-env — offline, no Hugging Face downloads."""

from __future__ import annotations

import numpy as np
import pytest

from mouse.envs import EnvConfig, make_vector_env

NS_PARAMS = {
    "length": {
        "scheduler": "continuous",
        "update_function": "oscillating",
        "update_kwargs": {"delta": 0.01},
    }
}


def _rollout(env, steps: int = 5) -> tuple[list, list, list]:
    data, metadata, metrics = env.step(env.sample_random_actions())
    for _ in range(steps - 1):
        data, metadata, metrics = env.step(env.sample_random_actions())
    return data, metadata, metrics


def test_cartpole_step_contract() -> None:
    cfg = EnvConfig.cartpole(
        seed=0,
        num_envs=2,
        max_episode_steps=50,
        q_star_source=None,
    )
    env = make_vector_env(cfg)
    try:
        data, metadata, metrics = _rollout(env)
        assert len(data) == 2
        assert len(metadata) == 2
        assert len(metrics) == 2
        assert metadata[0]["group_id"].endswith("#0")
        assert metadata[1]["group_id"].endswith("#1")
        for i, td in enumerate(data):
            assert set(td.keys()) >= {"time", "observation", "reward", "done"}
            assert "continuous" in td["observation"]
            assert metrics[i]["episode_cum_reward"] == [] or isinstance(
                metrics[i]["episode_cum_reward"][0], float
            )
    finally:
        env.close()


def test_procedural_frozenlake_vector() -> None:
    cfg = EnvConfig.procedural_frozenlake(seed=0, num_envs=2, max_episode_steps=50)
    env = make_vector_env(cfg)
    try:
        data, metadata, metrics = _rollout(env)
        assert len(data) == 2
        assert len(metadata) == 2
        assert "q_star" in metadata[0]
        assert metadata[0]["q_star"].shape == (4,)
        assert metadata[1]["q_star"].shape == (4,)
        for td in data:
            assert "discrete" in td["observation"]
    finally:
        env.close()


def test_synthetic_vector() -> None:
    cfg = EnvConfig.synthetic(seed=0, num_envs=2, max_episode_steps=50)
    env = make_vector_env(cfg)
    try:
        data, metadata, _metrics = _rollout(env)
        assert len(data) == 2
        assert "q_star" in metadata[0]
        for td in data:
            assert "discrete" in td["observation"]
    finally:
        env.close()


def test_ns_cartpole() -> None:
    cfg = EnvConfig.ns_cartpole(
        seed=0,
        num_envs=1,
        max_episode_steps=50,
        non_stationary_params=NS_PARAMS,
    )
    env = make_vector_env(cfg)
    try:
        _data, metadata, _metrics = _rollout(env, steps=3)
        assert "ns_params" in metadata[0]
        assert "length" in metadata[0]["ns_params"]
    finally:
        env.close()


def test_reward_shaping() -> None:
    cfg = EnvConfig.cartpole(
        seed=0,
        num_envs=1,
        max_episode_steps=50,
        reward_scale=0.5,
        reward_shift=1.0,
        q_star_source=None,
    )
    env = make_vector_env(cfg)
    try:
        _data, metadata, _metrics = _rollout(env, steps=3)
        assert isinstance(metadata[0]["reward_episodic"], float)
    finally:
        env.close()


def test_partial_observability() -> None:
    cfg = EnvConfig.cartpole(
        seed=0,
        num_envs=1,
        max_episode_steps=50,
        observation_indices=[0, 2],
        q_star_source=None,
    )
    env = make_vector_env(cfg)
    try:
        data, _metadata, _metrics = _rollout(env, steps=2)
        obs = data[0]["observation"]["continuous"]
        assert obs.shape == (2,)
    finally:
        env.close()


def test_first_step_is_reset_frame() -> None:
    cfg = EnvConfig.cartpole(
        seed=0,
        num_envs=1,
        max_episode_steps=50,
        q_star_source=None,
    )
    env = make_vector_env(cfg)
    try:
        data, metadata, metrics = env.step(env.sample_random_actions())
        assert data[0]["time"].item() == 0
        assert data[0]["reward"].item() == 0.0
        assert data[0]["done"].item() == 0
        assert metadata[0]["reward_episodic"] == 0.0
        assert metrics[0]["episode_cum_reward"] == []
    finally:
        env.close()


def _roll_until_autoreset(env, *, max_steps: int = 500) -> tuple[list, list, list, int]:
    data, metadata, metrics = env.step(env.sample_random_actions())
    for step in range(1, max_steps):
        prev_time = data[0]["time"].item()
        data, metadata, metrics = env.step(env.sample_random_actions())
        if data[0]["time"].item() == 0 and prev_time > 0:
            return data, metadata, metrics, step
    raise AssertionError(f"no autoreset frame within {max_steps} steps")


def test_autoreset_frame_zeros_reward_with_shift() -> None:
    cfg = EnvConfig.cartpole(
        seed=0,
        num_envs=1,
        max_episode_steps=50,
        reward_scale=0.5,
        reward_shift=1.0,
        q_star_source=None,
    )
    env = make_vector_env(cfg)
    try:
        data, metadata, metrics, _step = _roll_until_autoreset(env)
        assert data[0]["time"].item() == 0
        assert data[0]["reward"].item() == 0.0
        assert data[0]["done"].item() == 0
        assert metadata[0]["reward_episodic"] == 0.0
        assert metrics[0]["episode_cum_reward"] == []
    finally:
        env.close()


def test_to_json_str_roundtrip() -> None:
    from mouse.envs.utils import to_json_str
    import json

    payload = {"board": ["SFFF", "FFFF"], "rewards": {"3": 1.0}}
    s = to_json_str(payload)
    assert json.loads(s) == payload


def test_make_vector_env_requires_max_steps() -> None:
    with pytest.raises(ValueError, match="max_episode_steps"):
        make_vector_env(
            EnvConfig(
                group_id="CartPole-v1",
                seed=0,
                num_envs=1,
                max_episode_steps=None,
            )
        )
