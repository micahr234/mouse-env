"""Smoke tests for mouse-env — offline, no Hugging Face downloads."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from mouse_envs import EnvConfig, FieldSpec, InputSpec, OutputSpec, make_env


def _rollout(env, steps: int = 5) -> list:
    outputs = env.step(env.sample_random_inputs())
    for _ in range(steps - 1):
        outputs = env.step(env.sample_random_inputs())
    return outputs


def test_cartpole_step_contract() -> None:
    cfgs = [
        EnvConfig(
            id="CartPole-v1",
            name=f"train-cartpole_{i}",
            reset_seed=i,
            episodes_per_task=5,
        )
        for i in range(3)
    ]
    env = make_env(cfgs)
    try:
        outputs = _rollout(env)
        assert len(outputs) == 3
        assert env.names[0] == "train-cartpole_0"
        assert env.names == ("train-cartpole_0", "train-cartpole_1", "train-cartpole_2")
        sampled = env.sample_random_inputs()
        assert "action" in sampled[0]
        assert sampled[0]["action"].ndim == 0
        for r in outputs:
            assert set(r.keys()) >= {
                "time",
                "observation",
                "reward",
                "done",
                "episode_index",
                "task_index",
            }
            assert "id" not in r
            assert "name" not in r
            assert "action" not in r
        for per_env in env.tracker.episode_cum_rewards:
            assert all(isinstance(v, float) for v in per_env)
    finally:
        env.close()


def test_mouse_env_exposes_gym_tuple_spaces() -> None:
    env = make_env(
        [
            EnvConfig(id="CartPole-v1", reset_seed=0, episodes_per_task=5),
            EnvConfig(id="CartPole-v1", reset_seed=1, episodes_per_task=5),
        ]
    )
    try:
        assert isinstance(env, gym.Env)
        assert isinstance(env.action_space, gym.spaces.Tuple)
        assert isinstance(env.observation_space, gym.spaces.Tuple)
        assert len(env.action_space.spaces) == 2
        assert len(env.observation_space.spaces) == 2
        assert isinstance(env.action_space.spaces[0], gym.spaces.Discrete)
        assert isinstance(env.observation_space.spaces[0], gym.spaces.Box)
        assert not hasattr(env, "action_spaces")
    finally:
        env.close()


def test_output_spec_and_input_spec_cartpole() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
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
        assert ospec.task_index.dtype == int
        assert not hasattr(ospec, "q_star")
        assert not hasattr(ospec, "ns_params")

        assert isinstance(ispec.action, FieldSpec)
        assert ispec.action.dtype == torch.int64
        assert ispec.action.shape == ()
    finally:
        env.close()


def test_output_spec_frozenlake_obs() -> None:
    cfg = EnvConfig(
        id="Procedural-FrozenLake-v1",
        reset_seed=0,
        episodes_per_task=5,
        q_star_source={"provider": "env_q_star"},
    )
    env = make_env(cfg)
    try:
        ospec = env.output_specs[0]
        assert isinstance(ospec.observation, FieldSpec)
        assert ospec.observation.dtype == torch.int64
        assert ospec.observation.shape == ()
        # Q-values appear as info_env_q_star in step outputs, not in OutputSpec.
        assert not hasattr(ospec, "q_star")
    finally:
        env.close()


def test_pendulum_continuous_step_contract() -> None:
    env = make_env(
        [
            EnvConfig(id="Pendulum-v1", reset_seed=0, episodes_per_task=5),
            EnvConfig(id="Pendulum-v1", reset_seed=1, episodes_per_task=5),
        ]
    )
    try:
        assert env.input_specs[0].action.shape == (1,)
        sampled = env.sample_random_inputs()
        action = sampled[0]
        assert "action" in action
        assert action["action"].dtype == torch.float32
        assert action["action"].ndim == 0

        assert env.input_specs[0].action.dtype == torch.float32
        assert env.output_specs[0].observation.dtype == torch.float32

        outputs = _rollout(env)
        assert len(outputs) == 2
        for r in outputs:
            assert "observation" in r
            assert "action" not in r
    finally:
        env.close()


def test_action_input_contract_is_enforced() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
    )
    env = make_env(cfg)
    try:
        env.step(env.sample_random_inputs())
        not_a_dict = [torch.tensor(0)]
        with pytest.raises(ValueError, match="must be a dict"):
            env.step(not_a_dict)
        missing_key = [{"wrong": torch.tensor(0)}]
        with pytest.raises(ValueError, match="action"):
            env.step(missing_key)
        with pytest.raises(ValueError, match="exactly 1 entries"):
            env.step([])
        with pytest.raises(ValueError, match="one dict per env instance"):
            env.step({"action": torch.tensor(0)})
    finally:
        env.close()


def test_mouse_env_reset_is_not_implemented() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
    )
    env = make_env(cfg)
    try:
        with pytest.raises(NotImplementedError, match="reset-free Mouse rollout protocol"):
            env.reset()
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
        reset_seed=0,
        episodes_per_task=5,
        env_fn=lambda: DictObsEnv(),
    )
    env = make_env(cfg)
    try:
        outputs = _rollout(env, steps=2)
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
    cfgs = [
        EnvConfig(
            id="Procedural-FrozenLake-v1",
            reset_seed=i,
            episodes_per_task=5,
            q_star_source={"provider": "env_q_star"},
        )
        for i in range(2)
    ]
    env = make_env(cfgs)
    try:
        outputs = _rollout(env)
        assert len(outputs) == 2
        assert "info_env_q_star" in outputs[0]
        assert outputs[0]["info_env_q_star"].shape == (4,)
        assert outputs[1]["info_env_q_star"].shape == (4,)
        for r in outputs:
            assert "observation" in r
    finally:
        env.close()


def test_synthetic_vector() -> None:
    cfgs = [
        EnvConfig(
            id="SyntheticEnv-v1",
            reset_seed=i,
            episodes_per_task=5,
            q_star_source={"provider": "env_q_star"},
        )
        for i in range(2)
    ]
    env = make_env(cfgs)
    try:
        outputs = _rollout(env)
        assert len(outputs) == 2
        assert "info_env_q_star" in outputs[0]
        for r in outputs:
            assert "observation" in r
    finally:
        env.close()


def test_procedural_frozenlake_can_regenerate_map_from_episode_reset_options() -> None:
    cfg = EnvConfig(
        id="Procedural-FrozenLake-v1",
        reset_seed=7,
        episodes_per_task=5,
        episode_reset_options={"regenerate_map": True},
        kwargs={
            "emit_map": True,
            "hole_prob": 0.0,
            "max_episode_steps": 1,
        },
    )
    env = make_env(cfg)
    try:
        first = env.step(env.sample_random_inputs())[0]
        terminal = env.step(env.sample_random_inputs())[0]
        assert terminal["done"].item() != 0
        reset = env.step(env.sample_random_inputs())[0]

        assert "info_map" in first
        assert "info_map" in reset
        assert first["info_map"] != reset["info_map"]
    finally:
        env.close()


def test_synthetic_can_regenerate_map_from_episode_reset_options() -> None:
    cfg = EnvConfig(
        id="SyntheticEnv-v1",
        reset_seed=7,
        episodes_per_task=5,
        episode_reset_options={"regenerate_map": True},
        kwargs={
            "obs_size": 8,
            "action_size": 3,
            "max_episode_steps": 1,
        },
    )
    env = make_env(cfg)
    try:
        first = env.step(env.sample_random_inputs())[0]
        terminal = env.step(env.sample_random_inputs())[0]
        assert terminal["done"].item() != 0
        reset = env.step(env.sample_random_inputs())[0]

        assert "info_map" in first
        assert "info_map" in reset
        assert first["info_map"] != reset["info_map"]
    finally:
        env.close()


def test_first_party_random_maps_are_lazy_until_reset() -> None:
    from mouse_envs.worlds.procedural_frozenlake import ProceduralFrozenLakeEnv
    from mouse_envs.worlds.synthetic import SyntheticEnv

    synthetic = SyntheticEnv(obs_size=8, action_size=3, map_seed=1)
    assert synthetic.map == {}
    assert not synthetic._map_initialized
    _, synthetic_info = synthetic.reset(seed=2)
    assert synthetic._map_initialized
    assert "map" in synthetic_info

    frozen = ProceduralFrozenLakeEnv(map_seed=1, emit_map=True)
    assert frozen._gridmap is None
    _, frozen_info = frozen.reset(seed=2)
    assert frozen._gridmap is not None
    assert "map" in frozen_info


def test_map_seed_is_independent_from_reset_seed() -> None:
    from mouse_envs.worlds.synthetic import SyntheticEnv

    env_a = SyntheticEnv(obs_size=8, action_size=3, map_seed=11)
    env_b = SyntheticEnv(obs_size=8, action_size=3, map_seed=11)
    try:
        _, info_a = env_a.reset(seed=21)
        _, info_b = env_b.reset(seed=22)
        assert info_a["map"] == info_b["map"]
    finally:
        env_a.close()
        env_b.close()


def test_make_env_stream_seeds_are_independent() -> None:
    def _first_map(*, map_seed: int, reset_seed: int) -> str:
        cfg = EnvConfig(
            id="SyntheticEnv-v1",
            reset_seed=reset_seed,
            kwargs={"obs_size": 8, "action_size": 7, "map_seed": map_seed},
        )
        env = make_env(cfg)
        try:
            return env.step(env.sample_random_inputs())[0]["info_map"]
        finally:
            env.close()

    assert _first_map(map_seed=3, reset_seed=4) == _first_map(map_seed=3, reset_seed=40)
    assert _first_map(map_seed=3, reset_seed=4) != _first_map(map_seed=30, reset_seed=4)


def test_action_space_can_be_seeded_for_random_inputs() -> None:
    def _sampled_actions(*, reset_seed: int, action_space_seed: int) -> list[int]:
        cfg = EnvConfig(
            id="SyntheticEnv-v1",
            reset_seed=reset_seed,
            kwargs={"obs_size": 8, "action_size": 7, "map_seed": 1},
        )
        env = make_env(cfg)
        try:
            assert len(env.action_space.spaces) == 1
            env.action_space.spaces[0].seed(action_space_seed)
            return [int(env.sample_random_inputs()[0]["action"].item()) for _ in range(12)]
        finally:
            env.close()

    assert _sampled_actions(reset_seed=10, action_space_seed=20) == _sampled_actions(
        reset_seed=11, action_space_seed=20
    )
    assert _sampled_actions(reset_seed=10, action_space_seed=20) != _sampled_actions(
        reset_seed=10, action_space_seed=21
    )


def test_procedural_frozenlake_can_regenerate_map_at_task_boundary_only() -> None:
    from mouse_envs.format import DONE_EPISODE_TRUNCATED, DONE_TASK_TRUNCATED

    cfg = EnvConfig(
        id="Procedural-FrozenLake-v1",
        reset_seed=7,
        episodes_per_task=2,
        task_reset_options={"regenerate_map": True},
        kwargs={
            "emit_map": True,
            "hole_prob": 0.0,
            "max_episode_steps": 1,
        },
    )
    env = make_env(cfg)
    try:
        initial = env.step(env.sample_random_inputs())[0]
        first_terminal = env.step(env.sample_random_inputs())[0]
        episode_reset = env.step(env.sample_random_inputs())[0]
        second_terminal = env.step(env.sample_random_inputs())[0]
        task_reset = env.step(env.sample_random_inputs())[0]

        assert first_terminal["done"].item() == DONE_EPISODE_TRUNCATED
        assert second_terminal["done"].item() == DONE_TASK_TRUNCATED
        assert "info_map" in initial
        assert "info_map" not in episode_reset
        assert "info_map" in task_reset
        assert initial["info_map"] != task_reset["info_map"]
    finally:
        env.close()


def test_procedural_frozenlake_observation_space_uses_max_map_size() -> None:
    cfg = EnvConfig(
        id="Procedural-FrozenLake-v1",
        reset_seed=7,
        episodes_per_task=1,
        task_reset_options={"regenerate_map": True},
        kwargs={
            "emit_map": True,
            "min_width": 3,
            "max_width": 5,
            "min_height": 3,
            "max_height": 6,
            "max_episode_steps": 1,
        },
    )
    env = make_env(cfg)
    try:
        space = env.observation_space.spaces[0]
        assert isinstance(space, gym.spaces.Discrete)
        assert space.n == 30

        env.step(env.sample_random_inputs())
        env.step(env.sample_random_inputs())
        env.step(env.sample_random_inputs())

        space = env.observation_space.spaces[0]
        assert isinstance(space, gym.spaces.Discrete)
        assert space.n == 30
    finally:
        env.close()


def _task_regenerated_frozenlake_maps(seed: int) -> list[str]:
    cfg = EnvConfig(
        id="Procedural-FrozenLake-v1",
        reset_seed=seed,
        episodes_per_task=1,
        task_reset_options={"regenerate_map": True},
        kwargs={
            "emit_map": True,
            "map_seed": seed,
            "max_episode_steps": 1,
        },
    )
    env = make_env(cfg)
    maps: list[str] = []
    try:
        for _ in range(3):
            reset_frame = env.step([{"action": torch.tensor(0)}])[0]
            maps.append(reset_frame["info_map"])
            terminal = env.step([{"action": torch.tensor(0)}])[0]
            assert terminal["done"].item() != 0
        return maps
    finally:
        env.close()


def test_task_regenerated_maps_are_seed_reproducible() -> None:
    assert _task_regenerated_frozenlake_maps(7) == _task_regenerated_frozenlake_maps(7)
    assert _task_regenerated_frozenlake_maps(7) != _task_regenerated_frozenlake_maps(8)


def test_partial_observability() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
        observation_indices=[0, 2],
    )
    env = make_env(cfg)
    try:
        outputs = _rollout(env, steps=2)
        obs = outputs[0]["observation"]
        assert obs.shape == (2,)
        assert env.output_specs[0].observation.shape == (2,)
    finally:
        env.close()


def test_reset_frame_contract() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
        reset_reward=-1.0,
    )
    env = make_env(cfg)
    try:
        outputs = env.step(env.sample_random_inputs())
        assert outputs[0]["time"].item() == 0
        assert "action" not in outputs[0]
        assert outputs[0]["reward"].item() == -1.0
        assert outputs[0]["done"].item() == 0
        assert outputs[0]["task_index"] == 0
        assert env.tracker.episode_cum_rewards[0] == []
    finally:
        env.close()


def _roll_until_autoreset(env, *, max_steps: int = 500) -> tuple[list, int]:
    outputs = env.step(env.sample_random_inputs())
    for step in range(1, max_steps):
        prev_time = outputs[0]["time"].item()
        outputs = env.step(env.sample_random_inputs())
        if outputs[0]["time"].item() == 0 and prev_time > 0:
            return outputs, step
    raise AssertionError(f"no autoreset frame within {max_steps} steps")


def test_autoreset_frame_zeros_reward_with_shift() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
        reward_scale=0.5,
        reward_shift=1.0,
    )
    env = make_env(cfg)
    try:
        outputs, _step = _roll_until_autoreset(env)
        assert outputs[0]["time"].item() == 0
        assert outputs[0]["reward"].item() == 0.0
        assert outputs[0]["done"].item() == 0
        assert len(env.tracker.episode_cum_rewards[0]) >= 1
    finally:
        env.close()


def test_env_fn_factory() -> None:
    def make_cartpole() -> gym.Env:
        env = gym.make("CartPole-v1", max_episode_steps=50)
        return gym.wrappers.TransformObservation(
            env, lambda o: np.zeros_like(o), env.observation_space
        )

    cfgs = [
        EnvConfig(
            id="CartPole-custom",
            name=f"CartPole-custom_{i}",
            reset_seed=i,
            episodes_per_task=5,
            env_fn=make_cartpole,
        )
        for i in range(2)
    ]
    env = make_env(cfgs)
    try:
        outputs = _rollout(env, steps=2)
        assert len(outputs) == 2
        assert env.names == ("CartPole-custom_0", "CartPole-custom_1")
        obs = outputs[0]["observation"].numpy()
        assert np.all(obs == 0.0)
    finally:
        env.close()


def test_observation_kind_override() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
        observation_kind="discrete",
    )
    env = make_env(cfg)
    try:
        assert env._env_instances[0].obs_key == "observation_discrete"
        outputs = _rollout(env, steps=2)
        assert "observation" in outputs[0]
        assert env.output_specs[0].observation.dtype == torch.int64
    finally:
        env.close()


def test_task_done_codes_fire_at_task_boundary() -> None:
    from mouse_envs.format import DONE_EPISODE_TERMINATED, DONE_EPISODE_TRUNCATED, DONE_TASK_TERMINATED, DONE_TASK_TRUNCATED

    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=2,
        kwargs={"max_episode_steps": 10},
    )
    env = make_env(cfg)
    try:
        episode_dones: list[int] = []
        task_dones: list[int] = []
        for _ in range(300):
            outputs = env.step(env.sample_random_inputs())
            done = int(outputs[0]["done"].item())
            if done in (DONE_EPISODE_TERMINATED, DONE_EPISODE_TRUNCATED):
                episode_dones.append(done)
            elif done in (DONE_TASK_TERMINATED, DONE_TASK_TRUNCATED):
                task_dones.append(done)
        # With 2 episodes per task, for every 2 episode-end signals there should be 1 task-end.
        assert len(task_dones) > 0, "expected some task-done steps within 300 steps"
        assert len(episode_dones) > 0, "expected some episode-done steps within 300 steps"
        # task_index increments each time a task boundary is crossed
        outputs = env.step(env.sample_random_inputs())
        assert outputs[0]["task_index"] >= 0
    finally:
        env.close()


def test_tracker_accumulates_and_clears() -> None:
    from mouse_envs import MetricsTracker

    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
        kwargs={"max_episode_steps": 10},
    )
    env = make_env(cfg)
    try:
        assert isinstance(env.tracker, MetricsTracker)
        assert env.tracker.episode_cum_rewards == [[]]
        assert env.tracker.episode_lengths == [[]]

        # Roll until at least one episode completes
        for _ in range(200):
            env.step(env.sample_random_inputs())
            if env.tracker.episode_cum_rewards[0]:
                break
        else:
            raise AssertionError("no episode completed within 200 steps")

        rewards = env.tracker.episode_cum_rewards[0]
        lengths = env.tracker.episode_lengths[0]
        assert len(rewards) >= 1
        assert len(lengths) == len(rewards)
        assert all(isinstance(r, float) for r in rewards)
        assert all(isinstance(l, float) for l in lengths)

        # clear() wipes accumulated data
        env.tracker.clear()
        assert env.tracker.episode_cum_rewards == [[]]
        assert env.tracker.episode_lengths == [[]]
    finally:
        env.close()
