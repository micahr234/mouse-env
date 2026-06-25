"""MOUSE environments — environments and rollout formatting for mouse-core."""

from importlib.metadata import version

from mouse_envs.build import make_env
from mouse_envs.config import EnvConfig
from mouse_envs.format import (
    FieldSpec,
    InputSpec,
    MetricsTracker,
    MouseEnv,
    OutputSpec,
    StepOutput,
)

__version__ = version("mouse-env")

__all__ = [
    "__version__",
    "EnvConfig",
    "FieldSpec",
    "InputSpec",
    "make_env",
    "MetricsTracker",
    "MouseEnv",
    "OutputSpec",
    "StepOutput",
]
