"""Environment backends that construct underlying ``gym.Env`` instances."""

from mouse.envs.backends.base import ConstructionSeedWrapper, VectorEnvBuilder
from mouse.envs.backends.plain import PlainVectorEnv, PlainVectorEnvBuilder

__all__ = [
    "ConstructionSeedWrapper",
    "VectorEnvBuilder",
    "PlainVectorEnv",
    "PlainVectorEnvBuilder",
]


def __getattr__(name: str):
    if name in ("NSVectorEnv", "NSVectorEnvBuilder"):
        from mouse.envs.backends import ns as _ns

        return getattr(_ns, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ConstructionSeedWrapper",
    "VectorEnvBuilder",
    "PlainVectorEnv",
    "PlainVectorEnvBuilder",
    "NSVectorEnv",
    "NSVectorEnvBuilder",
]
