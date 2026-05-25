"""Atari (ALE) integration helpers — registration and Gymnasium ``AtariPreprocessing`` wiring."""

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing


def is_ale_env(env_id: str) -> bool:
    """True if ``env_id`` is an Atari (ALE) env."""
    return env_id.startswith("ALE/")


def ensure_ale_registered() -> None:
    """Import and register ``ale_py`` with Gymnasium."""
    try:
        import ale_py  # noqa: F401

        gym.register_envs(ale_py)
    except ImportError as e:
        raise ImportError(
            "ALE (Atari) envs require the ale_py package. Install with: pip install 'gymnasium[atari]'"
        ) from e


def wrap_atari_preprocessing(
    env: gym.Env,
    *,
    enabled: bool,
    preprocessing_kwargs: dict | None,
) -> gym.Env:
    """Optionally wrap an ALE env with ``AtariPreprocessing``."""
    if not enabled:
        return env
    return AtariPreprocessing(env, **(preprocessing_kwargs or {}))
