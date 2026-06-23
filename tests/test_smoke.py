"""Smoke tests for mouse-env — offline, no Hugging Face downloads."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from mouse_envs import EnvConfig, make_vector_env


def _rollout(env, steps: int = 5) -> tuple[list, list]:
    result, metrics = env.step(env.sample_random_actions())
    for _ in range(steps - 1):
        result, metrics = env.step(env.sample_random_actions())
    return result, metrics


def test_cartpole_step_contract() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        name="train-cartpole",
        seed=0,
        num_envs=3,
        max_episode_steps=50,
    )
    env = make_vector_env(cfg)
    try:
        result, metrics = _rollout(env)
        assert len(result) == 3
        assert len(metrics) == 3
        assert env.name == "train-cartpole#0"
        assert env.names == ("train-cartpole#0", "train-cartpole#1", "train-cartpole#2")
        sampled_action = env.sample_random_actions()[0]["action"]["discrete"]
        assert sampled_action.ndim == 0
        for i, r in enumerate(result):
            assert set(r.keys()) >= {
                "time",
                "observation",
                "reward",
                "done",
                "episode_index",
                "reward_episodic",
            }
            assert "id" not in r
            assert "name" not in r
            assert "action" not in r
            assert "continuous" in r["observation"]
            assert metrics[i]["episode_cum_reward"] == [] or isinstance(
                metrics[i]["episode_cum_reward"][0], float
            )
    finally:
        env.close()


def test_pendulum_continuous_step_contract() -> None:
    cfg = EnvConfig(
        id="Pendulum-v1",
        seed=0,
        num_envs=2,
        max_episode_steps=50,
    )
    env = make_vector_env(cfg)
    try:
        assert env.action_dim == 1
        sampled = env.sample_random_actions()
        action = sampled[0]["action"]
        assert "continuous" in action
        assert "discrete" not in action
        assert action["continuous"].dtype == torch.float32
        assert action["continuous"].ndim == 0
        result, metrics = _rollout(env)
        assert len(result) == 2
        for r in result:
            assert "continuous" in r["observation"]
            assert "action" not in r
    finally:
        env.close()


def test_action_input_contract_is_enforced() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        seed=0,
        num_envs=1,
        max_episode_steps=50,
    )
    env = make_vector_env(cfg)
    try:
        env.step(env.sample_random_actions())  # initial reset frame
        bare = [{"action": torch.tensor(0)} for _ in range(env.num_envs)]
        with pytest.raises(ValueError, match="must be a dict"):
            env.step(bare)
        wrong_key = [{"action": {"continuous": torch.tensor(0.0)}} for _ in range(env.num_envs)]
        with pytest.raises(ValueError, match="discrete"):
            env.step(wrong_key)
    finally:
        env.close()


def test_dict_obs_dtype_follows_space_not_key_name() -> None:
    class DictObsEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Dict(
                {
                    "pos": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                    "tile": gym.spaces.Box(low=0, high=9, shape=(1,), dtype=np.int32),
                }
            )
            self.action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {"pos": np.zeros(2, np.float32), "tile": np.array([3], np.int32)}, {}

        def step(self, action):
            return (
                {"pos": np.zeros(2, np.float32), "tile": np.array([3], np.int32)},
                0.0,
                False,
                False,
                {},
            )

    cfg = EnvConfig(
        id="DictObs",
        seed=0,
        num_envs=1,
        max_episode_steps=10,
        env_fn=lambda: DictObsEnv(),
    )
    env = make_vector_env(cfg)
    try:
        result, _metrics = _rollout(env, steps=2)
        obs = result[0]["observation"]
        # Float subspace -> float32; integer subspace -> int64, regardless of key name.
        assert obs["pos"].dtype == torch.float32
        assert obs["tile"].dtype == torch.int64
    finally:
        env.close()


def test_procedural_frozenlake_vector() -> None:
    cfg = EnvConfig(
        id="Procedural-FrozenLake-v1",
        seed=0,
        num_envs=2,
        max_episode_steps=50,
        q_star_source={"provider": "metadata_q_star"},
    )
    env = make_vector_env(cfg)
    try:
        result, metrics = _rollout(env)
        assert len(result) == 2
        assert "q_star" in result[0]
        assert result[0]["q_star"].shape == (4,)
        assert result[1]["q_star"].shape == (4,)
        for r in result:
            assert "discrete" in r["observation"]
    finally:
        env.close()


def test_synthetic_vector() -> None:
    cfg = EnvConfig(
        id="SyntheticEnv-v1",
        seed=0,
        num_envs=2,
        max_episode_steps=50,
        q_star_source={"provider": "metadata_q_star"},
    )
    env = make_vector_env(cfg)
    try:
        result, _metrics = _rollout(env)
        assert len(result) == 2
        assert "q_star" in result[0]
        for r in result:
            assert "discrete" in r["observation"]
    finally:
        env.close()


def test_partial_observability() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        seed=0,
        num_envs=1,
        max_episode_steps=50,
        observation_indices=[0, 2],
    )
    env = make_vector_env(cfg)
    try:
        result, _metrics = _rollout(env, steps=2)
        obs = result[0]["observation"]["continuous"]
        assert obs.shape == (2,)
    finally:
        env.close()


def test_reset_frame_contract() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        seed=0,
        num_envs=1,
        max_episode_steps=50,
        reset_reward=-1.0,
    )
    env = make_vector_env(cfg)
    try:
        result, metrics = env.step(env.sample_random_actions())
        assert result[0]["time"].item() == 0
        assert "action" not in result[0]
        assert result[0]["reward"].item() == -1.0
        assert result[0]["done"].item() == 0
        assert result[0]["reward_episodic"] == 0.0
        assert metrics[0]["episode_cum_reward"] == []
    finally:
        env.close()


def _roll_until_autoreset(env, *, max_steps: int = 500) -> tuple[list, list, int]:
    result, metrics = env.step(env.sample_random_actions())
    for step in range(1, max_steps):
        prev_time = result[0]["time"].item()
        result, metrics = env.step(env.sample_random_actions())
        if result[0]["time"].item() == 0 and prev_time > 0:
            return result, metrics, step
    raise AssertionError(f"no autoreset frame within {max_steps} steps")


def test_autoreset_frame_zeros_reward_with_shift() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        seed=0,
        num_envs=1,
        max_episode_steps=50,
        reward_scale=0.5,
        reward_shift=1.0,
    )
    env = make_vector_env(cfg)
    try:
        result, metrics, _step = _roll_until_autoreset(env)
        assert result[0]["time"].item() == 0
        assert result[0]["reward"].item() == 0.0
        assert result[0]["done"].item() == 0
        assert result[0]["reward_episodic"] == 0.0
        assert metrics[0]["episode_cum_reward"] == []
    finally:
        env.close()


def test_env_fn_factory() -> None:
    def make_cartpole() -> gym.Env:
        env = gym.make("CartPole-v1", max_episode_steps=50)
        return gym.wrappers.TransformObservation(
            env, lambda o: np.zeros_like(o), env.observation_space
        )

    cfg = EnvConfig(
        id="CartPole-custom",
        seed=0,
        num_envs=2,
        max_episode_steps=50,
        env_fn=make_cartpole,
    )
    env = make_vector_env(cfg)
    try:
        result, _metrics = _rollout(env, steps=2)
        assert len(result) == 2
        assert env.names == ("CartPole-custom#0", "CartPole-custom#1")
        obs = result[0]["observation"]["continuous"].numpy()
        assert np.all(obs == 0.0)
    finally:
        env.close()


def test_observation_kind_override() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        seed=0,
        num_envs=1,
        max_episode_steps=50,
        observation_kind="discrete",
    )
    env = make_vector_env(cfg)
    try:
        assert env.obs_key == "observation_discrete"
        result, _metrics = _rollout(env, steps=2)
        assert "discrete" in result[0]["observation"]
    finally:
        env.close()


def test_make_vector_env_requires_max_steps() -> None:
    with pytest.raises(ValueError, match="max_episode_steps"):
        make_vector_env(
            EnvConfig(
                id="CartPole-v1",
                seed=0,
                num_envs=1,
                max_episode_steps=None,
            )
        )
