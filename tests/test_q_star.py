"""Offline tests for expert Q* adapters and env Q* injection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest

from mouse_envs import EnvConfig, make_env
from mouse_envs.experts.action_star import (
    action_star_to_continuous_q_star,
    action_star_to_one_hot_q_star,
    build_q_star_source_adapter,
    normalize_q_star_source_name,
)


def _rollout(env, steps: int = 3) -> list:
    outputs = env.step(env.sample_random_inputs())
    for _ in range(steps - 1):
        outputs = env.step(env.sample_random_inputs())
    return outputs


def test_normalize_q_star_source_canonical_providers() -> None:
    assert normalize_q_star_source_name({"provider": "env_q_star"}) == "env_q_star"
    assert normalize_q_star_source_name({"provider": "hf_q_table"}) == "hf_q_table"
    assert normalize_q_star_source_name({"provider": "sb3_rl_zoo"}) == "sb3_rl_zoo"
    assert normalize_q_star_source_name(None) is None
    with pytest.raises(ValueError, match="Unsupported q_star_source.provider"):
        normalize_q_star_source_name({"provider": "q_table"})


def test_action_star_to_one_hot_q_star() -> None:
    actions = np.array([0, 2], dtype=np.int64)
    one_hot = action_star_to_one_hot_q_star(actions, num_actions=4)
    assert one_hot.shape == (2, 4)
    assert one_hot[0, 0] == 1.0
    assert one_hot[1, 2] == 1.0
    assert one_hot.sum(axis=1).tolist() == [1.0, 1.0]


def test_action_star_to_continuous_q_star() -> None:
    actions = np.array([[0.5], [-1.0]], dtype=np.float32)
    q_star = action_star_to_continuous_q_star(actions, num_envs=2, action_dim=1)
    assert q_star.shape == (2, 1)
    assert q_star.dtype == np.float64
    assert q_star[0, 0] == 0.5
    assert q_star[1, 0] == -1.0
    with pytest.raises(ValueError, match="action_dim"):
        action_star_to_continuous_q_star(actions, num_envs=2, action_dim=3)


def test_sb3_continuous_expert_injects_action_vector_q_star(
    pendulum_ppo_zip_path: Path,
) -> None:
    cfgs = [
        EnvConfig(
            id="Pendulum-v1",
            reset_seed=i,
            episodes_per_task=5,
            q_star_source={
                "provider": "sb3_rl_zoo",
                "algo": "ppo",
                "path": str(pendulum_ppo_zip_path),
                "device": "cpu",
            },
        )
        for i in range(2)
    ]

    def _fail_hf_download(*_args, **_kwargs):
        raise AssertionError("hf_hub_download must not be called when path is set")

    with patch("mouse_envs.experts.action_star.hf_hub_download", side_effect=_fail_hf_download):
        env = make_env(cfgs)
    try:
        assert env.input_specs[0].action.shape == (1,)
        result = _rollout(env, steps=2)
        q_star = result[0]["info_env_q_star"]
        # Continuous expert: q_star carries the expert action vector, not one-hot.
        assert q_star.shape == (1,)
        assert np.all(np.isfinite(q_star))
        assert -2.0 <= float(q_star[0]) <= 2.0
    finally:
        env.close()


def test_env_q_star_procedural_frozenlake_is_exact() -> None:
    cfg = EnvConfig(
        id="Procedural-FrozenLake-v1",
        reset_seed=3,
        episodes_per_task=5,
        q_star_source={"provider": "env_q_star"},
    )
    env = make_env(cfg)
    try:
        result = _rollout(env, steps=2)
        q_star = result[0]["info_env_q_star"]
        assert q_star.shape == (4,)
        assert np.all(np.isfinite(q_star))
        # Optimal Q* should have a unique argmax per state.
        assert q_star.max() >= q_star.min()
    finally:
        env.close()


def test_env_q_star_synthetic_matches_action_dim() -> None:
    cfgs = [
        EnvConfig(
            id="SyntheticEnv-v1",
            reset_seed=i,
            episodes_per_task=5,
            kwargs={"obs_size": 8, "action_size": 3},
            q_star_source={"provider": "env_q_star"},
        )
        for i in range(2)
    ]
    env = make_env(cfgs)
    try:
        result = _rollout(env, steps=2)
        assert result[0]["info_env_q_star"].shape == (3,)
        assert result[1]["info_env_q_star"].shape == (3,)
    finally:
        env.close()


def test_sb3_local_path_injects_q_star_without_hf(
    cartpole_ppo_zip_path: Path,
) -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
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
        env = make_env(cfg)
    try:
        result = _rollout(env, steps=2)
        q_star = result[0]["info_env_q_star"]
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
    cfg = EnvConfig(
        id="SyntheticEnv-v1",
        reset_seed=0,
        episodes_per_task=5,
        kwargs={"obs_size": 16, "action_size": 4},
        q_star_source={
            "provider": "hf_q_table",
            "path": str(tabular_qtable_pickle_path),
        },
    )

    def _fail_hf_download(*_args, **_kwargs):
        raise AssertionError("hf_hub_download must not be called when path is set")

    with patch("mouse_envs.experts.action_star.hf_hub_download", side_effect=_fail_hf_download):
        env = make_env(cfg)
    try:
        result = _rollout(env, steps=2)
        q_star = result[0]["info_env_q_star"]
        assert q_star.shape == (4,)
        assert np.all(np.isfinite(q_star))
        assert not np.isclose(q_star.sum(), 1.0)
    finally:
        env.close()


def test_q_star_absent_when_disabled() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
    )
    env = make_env(cfg)
    try:
        result = _rollout(env, steps=2)
        assert "info_env_q_star" not in result[0]
    finally:
        env.close()


def test_build_adapter_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported q_star_source"):
        build_q_star_source_adapter(
            env_id="CartPole-v1",
            q_star_source={"provider": "not_a_real_provider"},
        )
