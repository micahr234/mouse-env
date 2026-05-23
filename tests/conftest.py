"""Shared pytest fixtures for mouse-env tests."""

from __future__ import annotations

import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO


@pytest.fixture(scope="session")
def cartpole_ppo_zip_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny locally trained PPO checkpoint — avoids Hugging Face downloads."""
    root = tmp_path_factory.mktemp("sb3_fixtures")
    path = root / "cartpole_ppo"
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=64, verbose=0)
    model.learn(total_timesteps=256)
    model.save(str(path))
    env.close()
    zip_path = path.with_suffix(".zip")
    assert zip_path.is_file()
    return zip_path


@pytest.fixture(scope="session")
def tabular_qtable_pickle_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tabular Q-table pickle for ``hf_q_table`` offline tests."""
    root = tmp_path_factory.mktemp("qtable_fixtures")
    path = root / "qtable.pkl"
    qtable = np.zeros((16, 4), dtype=np.float64)
    qtable[:, 1] = 2.0
    qtable[:, 3] = 1.0
    with path.open("wb") as f:
        pickle.dump({"qtable": qtable}, f)
    return path
