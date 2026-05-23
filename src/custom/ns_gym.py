"""NS-Gym integration: non-stationary Gymnasium environments.

Provides:
- NS-gym wrappers, update functions, and scheduler config
- NSGymInterfaceWrapper: adapts NS-Gym dict obs + info to flat obs + ns_params
- make_ns_env: create a single non-stationary env
- NSVectorEnv: vector runner for non-stationary envs
- is_ns_gym_env / normalize_env_id: routing helpers
"""

from typing import Any, cast

import gymnasium as gym
import numpy as np
from ns_gym.schedulers import (
    ContinuousScheduler,
    DiscreteScheduler,
    MemorylessScheduler,
    PeriodicScheduler,
    RandomScheduler,
)
from ns_gym.update_functions import (
    DeterministicTrend,
    DistributionCyclicUpdate,
    DistributionDecrementUpdate,
    DistributionIncrementUpdate,
    DistributionNoUpdate,
    DistributionStepWiseUpdate,
    ExponentialDecay,
    GeometricProgression,
    IncrementUpdate,
    NoUpdate,
    OscillatingUpdate,
    RandomWalk,
    RandomWalkWithDrift,
    RandomWalkWithDriftAndTrend,
    StepWiseUpdate,
)
from ns_gym.wrappers import NSClassicControlWrapper, NSCliffWalkingWrapper, NSFrozenLakeWrapper

from mouse.envs.routing import is_ns_gym_env, normalize_env_id

# -----------------------------------------------------------------------------
# NS-Gym: wrappers, update functions, single env, vector env
# -----------------------------------------------------------------------------

_NS_INFO_KEYS = ("Ground Truth Env Change", "Ground Truth Delta Change")

_NS_WRAPPER = {
    "CartPole-v1": NSClassicControlWrapper,
    "Acrobot-v1": NSClassicControlWrapper,
    "MountainCar-v0": NSClassicControlWrapper,
    "MountainCarContinuous-v0": NSClassicControlWrapper,
    "Pendulum-v1": NSClassicControlWrapper,
    "FrozenLake-v1": NSFrozenLakeWrapper,
    "CliffWalking-v1": NSCliffWalkingWrapper,
}

# Envs whose tunable params are non-scalar (e.g. transition-prob lists for grid worlds).
# ns_gym's base._get_delta_change does `updated_param - param`, which fails for lists,
# so delta_change_notification must be disabled for these.
_NS_NO_DELTA_CHANGE = {"FrozenLake-v1", "CliffWalking-v1"}

# Concrete scheduler types we instantiate; used for runtime check before passing to update functions
_scheduler_types = (
    ContinuousScheduler,
    DiscreteScheduler,
    MemorylessScheduler,
    PeriodicScheduler,
    RandomScheduler,
)


def _filter_ns_info(info: dict[str, Any]) -> dict[str, Any]:
    """Return only ground-truth env change and delta change from env info."""
    return {k: info.get(k, {}) for k in _NS_INFO_KEYS}


def extract_ns_params(infos: dict[str, Any]) -> dict[str, Any]:
    """Extract non-stationary parameter values from NS-Gym ground-truth info.

    Reads ``"Ground Truth Env Change"`` and ``"Ground Truth Delta Change"`` from the
    raw NS-Gym info dict and converts them to a flat dict of the form:

    - ``"{param}"`` — ``float`` or ``ndarray``: the delta change value for the parameter.
    - ``"{param}_flag"`` — ``bool`` or ``ndarray``: whether the parameter changed this step.

    Private keys (starting with ``"_"``) are skipped.

    Args:
        infos: Raw info dict from an NS-Gym wrapper (before :class:`NSGymInterfaceWrapper`
            processes it).

    Returns:
        Flat dict with one ``"{param}"`` and one ``"{param}_flag"`` entry per non-private
        parameter found in ``"Ground Truth Env Change"``.
    """
    ns_params = {}
    env_change = infos.get("Ground Truth Env Change", {})
    delta_change = infos.get("Ground Truth Delta Change", {})
    for k, flag in env_change.items():
        if not k.startswith("_"):
            flag = np.asarray(flag)
            quantity = np.asarray(delta_change.get(k, 0))
            ns_params[f"{k}_flag"] = flag
            ns_params[k] = quantity
    return ns_params


class NSGymInterfaceWrapper(gym.Wrapper):
    """Adapt an NS-Gym env to the standard Gymnasium single-env API.

    NS-Gym wrappers return a ``dict`` observation (with a ``"state"`` key) and inject
    ``"Ground Truth Env Change"`` / ``"Ground Truth Delta Change"`` keys into info.
    This wrapper strips both of those non-standard aspects so that the env behaves like
    a normal Gymnasium env:

    - **Observation**: only ``obs["state"]`` is returned; other dict keys are discarded.
    - **Info**: raw ground-truth keys are replaced with the output of
      :func:`extract_ns_params` (keys like ``"{param}"`` and ``"{param}_flag"``).

    The :class:`~mouse.envs.base.ObservationSliceWrapper` and the full wrapper stack can
    then be applied on top without any NS-Gym-specific logic.

    Args:
        env: An NS-Gym wrapped environment (e.g. the output of
            ``NSClassicControlWrapper(base_env, ...)``) whose observation space is a
            ``Dict`` containing a ``"state"`` key.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        inner_obs_space = env.observation_space
        if isinstance(inner_obs_space, gym.spaces.Dict) and "state" in inner_obs_space.spaces:
            self.observation_space = inner_obs_space["state"]
        else:
            self.observation_space = inner_obs_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        state = obs["state"] if isinstance(obs, dict) and "state" in obs else obs
        info = extract_ns_params(_filter_ns_info(info))
        return state, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = obs["state"] if isinstance(obs, dict) and "state" in obs else obs
        info = extract_ns_params(_filter_ns_info(info))
        return state, reward, terminated, truncated, info


def create_ns_gym_update_functions(ns_gym_config: dict[str, Any]) -> dict[str, Any]:
    """Build NS-Gym parameter update functions from a plain config dict.

    Instantiates a ``(Scheduler, UpdateFunction)`` pair for each physics parameter
    listed in ``ns_gym_config``. The resulting dict is passed directly to the
    appropriate NS-Gym wrapper (e.g. ``NSClassicControlWrapper``).

    Config format::

        {
            "<param_name>": {
                "scheduler": "continuous" | "periodic" | "random" | "discrete" | "memoryless",
                "update_function": "no_update" | "increment" | "random_walk" | "oscillating" | ...,
                "scheduler_kwargs": {...},   # optional
                "update_kwargs": {...},      # optional
            },
            ...
        }

    Supported schedulers: ``continuous``, ``periodic``, ``random``, ``discrete``,
    ``memoryless``.

    Supported update functions: ``no_update``, ``increment``, ``random_walk``,
    ``random_walk_with_drift``, ``random_walk_with_drift_and_trend``,
    ``deterministic_trend``, ``exponential_decay``, ``geometric_progression``,
    ``oscillating``, ``step_wise``, ``distribution_no_update``,
    ``distribution_increment``, ``distribution_decrement``,
    ``distribution_step_wise``, ``distribution_cyclic``.

    Args:
        ns_gym_config: Mapping from parameter name to a scheduler + update-function
            specification. Must be a plain ``dict``; nested dicts must not contain
            OmegaConf or other structured-config objects.

    Returns:
        Dict mapping each parameter name to an instantiated NS-Gym update function.

    Raises:
        TypeError: If ``ns_gym_config`` or any nested value is not a plain ``dict``.
        ValueError: If an unknown scheduler or update-function name is specified.
    """
    if not isinstance(ns_gym_config, dict):
        raise TypeError(
            f"ns_gym_config must be a dict, got {type(ns_gym_config).__name__}."
        )
    param_update_functions = {}
    for param_name, update_config in ns_gym_config.items():
        if not isinstance(update_config, dict):
            raise TypeError(
                f"ns_gym_config[{param_name!r}] must be a dict, got {type(update_config).__name__}."
            )
        scheduler_type = update_config["scheduler"]
        update_func_type = update_config["update_function"]
        scheduler_kwargs_raw = update_config.get("scheduler_kwargs", {}) or {}
        update_kwargs_raw = update_config.get("update_kwargs", {}) or {}
        if not isinstance(scheduler_kwargs_raw, dict):
            raise TypeError(
                f"ns_gym_config[{param_name!r}].scheduler_kwargs must be a plain dict, got {type(scheduler_kwargs_raw).__name__}."
            )
        if not isinstance(update_kwargs_raw, dict):
            raise TypeError(
                f"ns_gym_config[{param_name!r}].update_kwargs must be a plain dict, got {type(update_kwargs_raw).__name__}."
            )
        scheduler_kwargs = scheduler_kwargs_raw
        update_kwargs = update_kwargs_raw
        if scheduler_type == "continuous":
            scheduler = ContinuousScheduler(**scheduler_kwargs)
        elif scheduler_type == "periodic":
            scheduler = PeriodicScheduler(**scheduler_kwargs)
        elif scheduler_type == "random":
            scheduler = RandomScheduler(**scheduler_kwargs)
        elif scheduler_type == "discrete":
            scheduler_kwargs = dict(scheduler_kwargs)
            event_list = scheduler_kwargs.pop("event_list", [])
            scheduler = DiscreteScheduler(set(event_list), **scheduler_kwargs)
        elif scheduler_type == "memoryless":
            scheduler = MemorylessScheduler(**scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        if not isinstance(scheduler, _scheduler_types):
            raise TypeError(
                f"Expected a scheduler instance (one of {[t.__name__ for t in _scheduler_types]}), "
                f"got {type(scheduler).__name__}"
            )
        sched: Any = cast(Any, scheduler)
        if update_func_type == "increment":
            update_func = IncrementUpdate(sched, **update_kwargs)
        elif update_func_type == "random_walk":
            update_func = RandomWalk(sched, **update_kwargs)
        elif update_func_type == "no_update":
            update_func = NoUpdate(sched)
        elif update_func_type == "deterministic_trend":
            update_func = DeterministicTrend(sched, **update_kwargs)
        elif update_func_type == "exponential_decay":
            update_func = ExponentialDecay(sched, **update_kwargs)
        elif update_func_type == "geometric_progression":
            update_func = GeometricProgression(sched, **update_kwargs)
        elif update_func_type == "oscillating":
            update_func = OscillatingUpdate(sched, **update_kwargs)
        elif update_func_type == "random_walk_with_drift":
            update_func = RandomWalkWithDrift(sched, **update_kwargs)
        elif update_func_type == "random_walk_with_drift_and_trend":
            update_func = RandomWalkWithDriftAndTrend(sched, **update_kwargs)
        elif update_func_type == "step_wise":
            update_func = StepWiseUpdate(sched, **update_kwargs)
        elif update_func_type == "distribution_decrement":
            update_func = DistributionDecrementUpdate(sched, **update_kwargs)
        elif update_func_type == "distribution_increment":
            update_func = DistributionIncrementUpdate(sched, **update_kwargs)
        elif update_func_type == "distribution_step_wise":
            update_func = DistributionStepWiseUpdate(sched, **update_kwargs)
        elif update_func_type == "distribution_cyclic":
            update_func = DistributionCyclicUpdate(sched, **update_kwargs)
        elif update_func_type == "distribution_no_update":
            update_func = DistributionNoUpdate(sched)
        else:
            raise ValueError(f"Unknown update function type: {update_func_type}")
        param_update_functions[param_name] = update_func
    return param_update_functions


def make_ns_env(
    env_id: str,
    non_stationary_params: dict[str, Any],
    max_steps_per_episode: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
    render: bool = False,
) -> gym.Env:
    """Create a single non-stationary Gymnasium env wrapped with NS-Gym.

    Instantiates the base env, wraps it with the appropriate NS-Gym wrapper
    (selected from ``_NS_WRAPPER`` by ``env_id``), and applies
    :class:`NSGymInterfaceWrapper` so the result looks like a standard ``gym.Env``.
    The update functions are persistent — each env instance owns its own RNG state
    and parameter trajectory.

    Args:
        env_id: Plain env id (without the ``NS-`` prefix), e.g. ``"CartPole-v1"``.
        non_stationary_params: Scheduler + update-function config per parameter; see
            :func:`create_ns_gym_update_functions` for the format.
        max_steps_per_episode: Passed to ``gym.make(max_episode_steps=...)``.
        env_kwargs: Extra keyword arguments forwarded to ``gym.make``.
        render: Set ``render_mode="human"`` on the underlying env.

    Returns:
        A :class:`NSGymInterfaceWrapper`-wrapped non-stationary env.

    Raises:
        ValueError: If ``env_id`` has no registered NS-Gym wrapper.
    """
    env_kwargs = env_kwargs or {}
    if render and "render_mode" not in env_kwargs:
        env_kwargs = {**env_kwargs, "render_mode": "human"}

    param_update_functions = create_ns_gym_update_functions(non_stationary_params)
    base_env = gym.make(
        env_id,
        max_episode_steps=max_steps_per_episode,
        **env_kwargs,
    )
    wrapper_cls = _NS_WRAPPER.get(env_id)
    if wrapper_cls is None:
        raise ValueError(f"No NS-Gym wrapper registered for env_id={env_id!r}. Known: {list(_NS_WRAPPER)}")
    return NSGymInterfaceWrapper(wrapper_cls(  # type: ignore
        base_env,
        param_update_functions,
        change_notification=True,
        delta_change_notification=env_id not in _NS_NO_DELTA_CHANGE,
    ))


def __getattr__(name: str):
    if name == "NSVectorEnv":
        from mouse.envs.backends.ns import NSVectorEnv as _NSVectorEnv

        return _NSVectorEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


