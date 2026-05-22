"""Environment factory (backward-compatible module path).

Prefer ``from mouse.envs import EnvConfig, make_vector_env`` for new code.
"""

from mouse.envs.backends.plain import PlainVectorEnv
from mouse.envs.config import (
    EnvBuildConfig,
    EnvConfig,
    RolloutConfig,
    resolve_q_star_source_for_env,
)
from mouse.envs.factory import make_vector_env
from mouse.envs.backends.ns import NSVectorEnv
from mouse.envs.routing import is_ns_gym_env, normalize_env_id

__all__ = [
    "EnvConfig",
    "EnvBuildConfig",
    "RolloutConfig",
    "make_vector_env",
    "PlainVectorEnv",
    "NSVectorEnv",
    "is_ns_gym_env",
    "normalize_env_id",
    "resolve_q_star_source_for_env",
]
