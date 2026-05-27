"""Offline tests for expert Q* adapters and vector-env metadata injection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest

from mouse_envs import EnvConfig, make_vector_env
from mouse_envs.experts.action_star import (
    action_star_to_one_hot_q_star,
    build_q_star_source_adapter,
    normalize_q_star_source_name,
)


def _rollout(env, steps: int = 3):
    result, metrics = env.step(env.sample_random_actions())
    for _ in range(steps - 1):
        result, metrics = env.step(env.sample_random_actions())
    return result, metrics


def test_normalize_q_star_source_aliases() -> None:
    assert normalize_q_star_source_name({"provider": "env_q_star"}) == "metadata_q_star"
    assert normalize_q_star_source_name({"provider": "q_table"}) == "hf_q_table"
    assert normalize_q_star_source_name(None) is None
    assert normalize_q_star_source_name({"provider": "none"}) is None


def test_action_star_to_one_hot_q_star() -> None:
    actions = np.array([0, 2], dtype=np.int64)
    one_hot = action_star_to_one_hot_q_star(actions, num_actions=4)
    assert one_hot.shape == (2, 4)
    assert one_hot[0, 0] == 1.0
    assert one_hot[1, 2] == 1.0
    assert one_hot.sum(axis=1).tolist() == [1.0, 1.0]


def test_metadata_q_star_procedural_frozenlake_is_exact() -> None:
    cfg = EnvConfig.procedural_frozenlake(seed=3, num_envs=1, max_episode_steps=50)
    env = make_vector_env(cfg)
    try:
        result, _metrics = _rollout(env, steps=2)
        q_star = result[0]["q_star"]
        assert q_star.shape == (4,)
        assert np.all(np.isfinite(q_star))
        # Optimal Q* should have a unique argmax per state.
        assert q_star.max() >= q_star.min()
    finally:
        env.close()


def test_metadata_q_star_synthetic_matches_action_dim() -> None:
    cfg = EnvConfig.synthetic(
        seed=1,
        num_envs=2,
        max_episode_steps=50,
        env_kwargs={"obs_size": 8, "action_size": 3},
    )
    env = make_vector_env(cfg)
    try:
        result, _metrics = _rollout(env, steps=2)
        assert result[0]["q_star"].shape == (3,)
        assert result[1]["q_star"].shape == (3,)
    finally:
        env.close()


def test_sb3_local_path_injects_q_star_without_hf(
    cartpole_ppo_zip_path: Path,
) -> None:
    cfg = EnvConfig.cartpole(
        seed=0,
        num_envs=1,
        max_episode_steps=50,
        q_star_source={
            "provider": "sb3_rl_zoo",
            "algo": "ppo",
            "path": str(cartpole_ppo_zip_path),
            "device": "cpu",
        },
    )

    def _fail_hf_download(*_args, **_kwargs):
        raise AssertionError("hf_hub_download must not be called when path is set")

    with patch("mouse_envs.experts.action_star.hf_hub_download", side_effect=_fail_hf_download):
        env = make_vector_env(cfg)
    try:
        result, _metrics = _rollout(env, steps=2)
        q_star = result[0]["q_star"]
        assert q_star.shape == (2,)
        # PPO has no predict_q — wrapper falls back to one-hot expert actions.
        assert np.isclose(q_star.sum(), 1.0)
        assert np.all((q_star == 0.0) | (q_star == 1.0))
    finally:
        env.close()


def test_hf_q_table_local_path_returns_full_q_values(
    tabular_qtable_pickle_path: Path,
) -> None:
    obs_space = gym.spaces.Discrete(16)
    adapter = build_q_star_source_adapter(
        env_id="SyntheticEnv-v1",
        q_star_source={
            "provider": "hf_q_table",
            "path": str(tabular_qtable_pickle_path),
        },
        obs_key="observation_discrete",
        single_observation_space=obs_space,
    )
    assert adapter is not None
    q = adapter.q_star_from_observation(
        obs=np.array([[5]], dtype=np.int64),
        done_mask=None,
    )
    assert q is not None
    assert q.shape == (1, 4)
    assert q[0, 1] == 2.0
    assert q[0, 3] == 1.0


def test_hf_q_table_vector_env_integration(
    tabular_qtable_pickle_path: Path,
) -> None:
    cfg = EnvConfig.synthetic(
        seed=0,
        num_envs=1,
        max_episode_steps=50,
        env_kwargs={"obs_size": 16, "action_size": 4},
        q_star_source={
            "provider": "hf_q_table",
            "path": str(tabular_qtable_pickle_path),
        },
    )

    def _fail_hf_download(*_args, **_kwargs):
        raise AssertionError("hf_hub_download must not be called when path is set")

    with patch("mouse_envs.experts.action_star.hf_hub_download", side_effect=_fail_hf_download):
        env = make_vector_env(cfg)
    try:
        result, _metrics = _rollout(env, steps=2)
        q_star = result[0]["q_star"]
        assert q_star.shape == (4,)
        assert np.all(np.isfinite(q_star))
        assert not np.isclose(q_star.sum(), 1.0)
    finally:
        env.close()


def test_q_star_absent_when_disabled() -> None:
    cfg = EnvConfig.cartpole(
        seed=0,
        num_envs=1,
        max_episode_steps=50,
        q_star_source=None,
    )
    env = make_vector_env(cfg)
    try:
        result, _metrics = _rollout(env, steps=2)
        assert "q_star" not in result[0]
    finally:
        env.close()


def test_build_adapter_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported q_star_source"):
        build_q_star_source_adapter(
            env_id="CartPole-v1",
            q_star_source={"provider": "not_a_real_provider"},
            obs_key="observation",
        )
