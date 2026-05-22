# Environments

mouse-envs supports four categories of environments, all accessed through the unified `make_vector_env` factory. Each returns a wrapped `gym.vector.VectorEnv` with a consistent info dict (see [Wrapper Stack](wrappers.md)).

---

## Plain Gymnasium

Standard discrete-action Gymnasium environments — CartPole, MountainCar, LunarLander, Acrobot, and similar.

```python
from mouse.envs import EnvConfig, make_vector_env

cfg = EnvConfig(
    env_id="CartPole-v1",
    seed=0,
    num_envs=8,
    max_episode_steps=500,
    kwargs=None,
    render=False,
    non_stationary_params=None,
    num_steps=None,
    action_source_loop_prob_schedule=None,
    action_source_episode_prob_schedule=None,
    q_star_source=None,
    action_source="random",
    action_source_temperature=1.0,
    split="train",
    env_type="plain",
    atari_preprocessing=None,
    atari_preprocessing_kwargs=None,
    observation_indices=None,
)
env = make_vector_env(cfg)
```

### Partial observability

Use `observation_indices` to slice the observation vector. For CartPole this keeps only cart position and pole angle (removing velocities):

```python
cfg = EnvConfig(
    env_id="CartPole-v1",
    observation_indices=[0, 2],
    ...
)
```

### Reward shaping

Use `reward_scale` and `reward_shift` to scale/shift rewards linearly after each step. `episode_cum_reward` in info always reflects the raw (unscaled) return.

---

## Atari

Atari Learning Environment (ALE) environments with optional preprocessing.

```python
cfg = EnvConfig(
    env_id="ALE/Pong-v5",
    atari_preprocessing=True,
    atari_preprocessing_kwargs={"frame_skip": 4, "screen_size": 84},
    ...
)
```

When `atari_preprocessing=True` the env is wrapped with `gymnasium.wrappers.AtariPreprocessing` (grayscale conversion, frame skipping, no-op starts, screen resize). The obs key is automatically resolved to `observation_image`.

---

## NS-Gym (Non-Stationary)

Non-stationary variants of classic control environments where physics parameters shift over time according to configurable schedules. NS-Gym env IDs start with `NS-`:

```python
cfg = EnvConfig(
    env_id="NS-CartPole-v1",
    non_stationary_params={
        "masscart": {
            "scheduler": "ContinuousScheduler",
            "update_fn": "NoUpdate",
            "init_value": 1.0,
        },
        "length": {
            "scheduler": "ContinuousScheduler",
            "update_fn": "OscillatingUpdate",
            "init_value": 0.5,
        },
    },
    env_type="ns_gym",
    ...
)
```

### Supported schedulers

| Scheduler | Description |
|-----------|-------------|
| `ContinuousScheduler` | Updates parameter every step |
| `DiscreteScheduler` | Updates parameter every episode |
| `MemorylessScheduler` | Samples independently each step |
| `PeriodicScheduler` | Updates at fixed intervals |
| `RandomScheduler` | Updates randomly |

### Supported update functions

`NoUpdate`, `IncrementUpdate`, `StepWiseUpdate`, `OscillatingUpdate`, `RandomWalk`, `RandomWalkWithDrift`, `RandomWalkWithDriftAndTrend`, `DeterministicTrend`, `ExponentialDecay`, `GeometricProgression`, `DistributionNoUpdate`, `DistributionIncrementUpdate`, `DistributionDecrementUpdate`, `DistributionStepWiseUpdate`, `DistributionCyclicUpdate`.

---

## Custom FrozenLake

A procedurally generated FrozenLake environment registered as `Custom-FrozenLake-v1`. Supports optional value-iteration Q* expert for supervised pre-training data.

```python
cfg = EnvConfig(
    env_id="Custom-FrozenLake-v1",
    kwargs={"map_size": 8, "seed": 42, "is_slippery": True},
    q_star_source={"type": "env"},
    ...
)
```

---

## Custom SyntheticEnv

A random tabular MDP registered as `Custom-SyntheticEnv-v1`. Useful for controlled experiments over known reward structures.

```python
cfg = EnvConfig(
    env_id="Custom-SyntheticEnv-v1",
    kwargs={"num_states": 16, "num_actions": 4, "seed": 0},
    ...
)
```
