"""Smoke tests for mouse-env — offline, no Hugging Face downloads."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from mouse_envs import EnvConfig, FieldSpec, InputSpec, OutputSpec, make_env


def _rollout(env, steps: int = 5) -> tuple[list, list]:
    [(outputs, metrics)] = env.step(env.sample_random_inputs())
    for _ in range(steps - 1):
        [(outputs, metrics)] = env.step(env.sample_random_inputs())
    return outputs, metrics


def test_cartpole_step_contract() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        name="train-cartpole",
        seed=0,
        num_envs=3,
        max_episode_steps=50,
    )
    env = make_env(cfg)
    try:
        outputs, metrics = _rollout(env)
        assert len(outputs) == 3
        assert len(metrics) == 3
        assert env.names[0] == "train-cartpole#0"
        assert env.names == ("train-cartpole#0", "train-cartpole#1", "train-cartpole#2")
        sampled = env.sample_random_inputs()
        assert "action" in sampled[0][0]
        assert sampled[0][0]["action"].ndim == 0
        for i, r in enumerate(outputs):
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
            assert metrics[i]["episode_cum_reward"] == [] or isinstance(
                metrics[i]["episode_cum_reward"][0], float
            )
    finally:
        env.close()


def test_output_spec_and_input_spec_cartpole() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        seed=0,
        num_envs=1,
        max_episode_steps=50,
    )
    env = make_env(cfg)
    try:
        ospec = env.output_specs[0]
        ispec = env.input_specs[0]

        assert isinstance(ospec, OutputSpec)
        assert isinstance(ispec, InputSpec)

        assert isinstance(ospec.observation, FieldSpec)
        assert ospec.observation.dtype == torch.float32
        assert ospec.observation.shape == (4,)

        assert ospec.time.dtype == torch.int64
        assert ospec.time.shape == ()
        assert ospec.reward.dtype == torch.float32
        assert ospec.done.dtype == torch.int64
        assert ospec.episode_index.dtype == int
        assert ospec.reward_episodic.dtype == float
        assert ospec.q_star is None
        assert ospec.ns_params is None

        assert isinstance(ispec.action, FieldSpec)
        assert ispec.action.dtype == torch.int64
        assert ispec.action.shape == ()
    finally:
        env.close()


def test_output_spec_frozenlake_with_q_star() -> None:
    cfg = EnvConfig(
        id="Procedural-FrozenLake-v1",
        seed=0,
        num_envs=1,
        max_episode_steps=50,
        q_star_source={"provider": "metadata_q_star"},
    )
    env = make_env(cfg)
    try:
        ospec = env.output_specs[0]
        assert isinstance(ospec.observation, FieldSpec)
        assert ospec.observation.dtype == torch.int64
        assert ospec.observation.shape == ()
        assert ospec.q_star is not None
        assert ospec.q_star.dtype == np.float64
        assert ospec.q_star.shape == (4,)
    finally:
        env.close()


def test_pendulum_continuous_step_contract() -> None:
    cfg = EnvConfig(
        id="Pendulum-v1",
        seed=0,
        num_envs=2,
        max_episode_steps=50,
    )
    env = make_env(cfg)
    try:
        assert env.input_specs[0].action.shape == (1,)
        sampled = env.sample_random_inputs()
        action = sampled[0][0]
        assert "action" in action
        assert action["action"].dtype == torch.float32
        assert action["action"].ndim == 0

        assert env.input_specs[0].action.dtype == torch.float32
        assert env.output_specs[0].observation.dtype == torch.float32

        outputs, metrics = _rollout(env)
        assert len(outputs) == 2
        for r in outputs:
            assert "observation" in r
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
    env = make_env(cfg)
    try:
        env.step(env.sample_random_inputs())  # initial reset frame
        not_a_dict = [[torch.tensor(0)]]
        with pytest.raises(ValueError, match="must be a dict"):
            env.step(not_a_dict)
        missing_key = [[{"wrong": torch.tensor(0)}]]
        with pytest.raises(ValueError, match="action"):
            env.step(missing_key)
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
    env = make_env(cfg)
    try:
        outputs, _metrics = _rollout(env, steps=2)
        # Float subspace -> float32; integer subspace -> int64, regardless of key name.
        assert outputs[0]["pos"].dtype == torch.float32
        assert outputs[0]["tile"].dtype == torch.int64

        # output_specs[0].observation is a dict of FieldSpecs for Dict obs spaces
        ospec = env.output_specs[0]
        assert isinstance(ospec.observation, dict)
        assert ospec.observation["pos"].dtype == torch.float32
        assert ospec.observation["tile"].dtype == torch.int64
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
    env = make_env(cfg)
    try:
        outputs, metrics = _rollout(env)
        assert len(outputs) == 2
        assert "q_star" in outputs[0]
        assert outputs[0]["q_star"].shape == (4,)
        assert outputs[1]["q_star"].shape == (4,)
        for r in outputs:
            assert "observation" in r
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
    env = make_env(cfg)
    try:
        outputs, _metrics = _rollout(env)
        assert len(outputs) == 2
        assert "q_star" in outputs[0]
        for r in outputs:
            assert "observation" in r
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
    env = make_env(cfg)
    try:
        outputs, _metrics = _rollout(env, steps=2)
        obs = outputs[0]["observation"]
        assert obs.shape == (2,)
        assert env.output_specs[0].observation.shape == (2,)
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
    env = make_env(cfg)
    try:
        [(outputs, metrics)] = env.step(env.sample_random_inputs())
        assert outputs[0]["time"].item() == 0
        assert "action" not in outputs[0]
        assert outputs[0]["reward"].item() == -1.0
        assert outputs[0]["done"].item() == 0
        assert outputs[0]["reward_episodic"] == 0.0
        assert metrics[0]["episode_cum_reward"] == []
    finally:
        env.close()


def _roll_until_autoreset(env, *, max_steps: int = 500) -> tuple[list, list, int]:
    [(outputs, metrics)] = env.step(env.sample_random_inputs())
    for step in range(1, max_steps):
        prev_time = outputs[0]["time"].item()
        [(outputs, metrics)] = env.step(env.sample_random_inputs())
        if outputs[0]["time"].item() == 0 and prev_time > 0:
            return outputs, metrics, step
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
    env = make_env(cfg)
    try:
        outputs, metrics, _step = _roll_until_autoreset(env)
        assert outputs[0]["time"].item() == 0
        assert outputs[0]["reward"].item() == 0.0
        assert outputs[0]["done"].item() == 0
        assert outputs[0]["reward_episodic"] == 0.0
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
    env = make_env(cfg)
    try:
        outputs, _metrics = _rollout(env, steps=2)
        assert len(outputs) == 2
        assert env.names == ("CartPole-custom#0", "CartPole-custom#1")
        obs = outputs[0]["observation"].numpy()
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
    env = make_env(cfg)
    try:
        assert env._inners[0].obs_key == "observation_discrete"
        outputs, _metrics = _rollout(env, steps=2)
        assert "observation" in outputs[0]
        assert env.output_specs[0].observation.dtype == torch.int64
    finally:
        env.close()


def test_make_env_requires_max_steps() -> None:
    with pytest.raises(ValueError, match="max_episode_steps"):
        make_env(
            EnvConfig(
                id="CartPole-v1",
                seed=0,
                num_envs=1,
                max_episode_steps=None,
            )
        )
