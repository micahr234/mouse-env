"""Env id routing helpers (no backend imports)."""

from typing import Any


def is_ns_gym_env(
    env_id: str,
    non_stationary_params: dict[str, Any] | None,
    env_type: str | None,
) -> bool:
    """Return ``True`` when the env should use the NS-Gym backend."""
    _ = non_stationary_params
    _ = env_type
    return env_id.startswith("NS-")


def normalize_env_id(env_id: str) -> str:
    """Strip the ``NS-`` routing prefix before ``gym.make``."""
    if env_id.startswith("NS-"):
        return env_id[3:]
    return env_id
