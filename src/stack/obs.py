"""Observation helpers for vector env stacks."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObservationWrapper


class ObservationSliceWrapper(ObservationWrapper):
    """Slice a ``Box`` observation vector to a subset of indices.

    Useful for making envs partially observable. For example, on CartPole
    pass ``indices=[0, 2]`` to keep only cart position and pole angle,
    dropping the two velocity components.

    Args:
        env: The env to wrap. Its observation space must be a ``Box``.
        indices: Non-empty list of integer indices into the flattened observation
            vector to retain.

    Raises:
        ValueError: If ``indices`` is empty, the observation space is not a ``Box``,
            or any index is out of range for the observation shape.
    """

    def __init__(self, env: gym.Env, indices: list[int]):
        if not indices:
            raise ValueError("observation_indices must be non-empty.")
        super().__init__(env)
        self._indices = np.array(indices, dtype=np.intp)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box):
            raise ValueError(
                f"ObservationSliceWrapper requires Box observation space, got {type(space).__name__}."
            )
        low = np.asarray(space.low).flatten()
        high = np.asarray(space.high).flatten()
        if len(low) != len(high) or max(self._indices) >= len(low):
            raise ValueError(
                f"observation_indices {indices} out of range for space shape {low.shape}."
            )
        self.observation_space = gym.spaces.Box(
            low=low[self._indices],
            high=high[self._indices],
            dtype=getattr(space, "dtype", np.float32),
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        flat = np.asarray(observation).flatten()
        return flat[self._indices].astype(self.observation_space.dtype)


def _is_discrete_like(space: gym.Space) -> bool:
    """True iff the space contains only discrete / integer-valued observations."""
    if isinstance(space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)):
        return True
    if isinstance(space, gym.spaces.Tuple):
        return all(_is_discrete_like(s) for s in space.spaces)
    if isinstance(space, gym.spaces.Dict):
        return all(_is_discrete_like(s) for s in space.spaces.values())
    if isinstance(space, gym.spaces.Box):
        obs_dtype = np.dtype(space.dtype)
        return bool(np.issubdtype(obs_dtype, np.integer) or np.issubdtype(obs_dtype, np.bool_))
    return False


def resolve_obs_key(env: gym.vector.VectorEnv, requested: str = "observation") -> str:
    """Return the canonical observation-dict key for this env's observation space.

    The three possible keys map to:

    - ``"observation_image"`` — returned as-is when explicitly requested (Atari envs).
    - ``"observation_discrete"`` — returned when the single observation space is purely
      integer/discrete (``Discrete``, ``MultiDiscrete``, ``MultiBinary``, or integer ``Box``).
    - ``requested`` — returned in all other cases (continuous ``Box`` envs).

    Args:
        env: A vector env whose ``single_observation_space`` is inspected.
        requested: The caller's preferred key; overridden only by the discrete detection
            logic unless the value is ``"observation_image"``.

    Returns:
        One of ``"observation_image"``, ``"observation_discrete"``, or ``requested``.
    """
    if requested == "observation_image":
        return requested
    if _is_discrete_like(env.single_observation_space):
        return "observation_discrete"
    return requested
