# Examples

Runnable scripts live in [`examples/`](../examples/). Each script is self-contained and can be run directly:

```bash
python examples/01_random_rollout.py
```

| Script | Description |
|--------|-------------|
| [01_random_rollout.py](../examples/01_random_rollout.py) | Basic vector rollout with CartPole |
| [02_q_star_expert.py](../examples/02_q_star_expert.py) | FrozenLake with Q* expert policy |
| [03_ns_gym_oscillating.py](../examples/03_ns_gym_oscillating.py) | Non-stationary CartPole with oscillating pole |
| [04_atari_preprocessing.py](../examples/04_atari_preprocessing.py) | Atari Pong with frame preprocessing |
| [05_partial_observability.py](../examples/05_partial_observability.py) | CartPole with partial observations |
| [06_reward_shaping.py](../examples/06_reward_shaping.py) | MountainCar with reward scale/shift |

---

## Collecting rollouts

<!-- see examples/01_random_rollout.py -->

```python
from mouse.envs import EnvConfig, make_vector_env

cfg = EnvConfig.cartpole(seed=0, num_envs=4, max_episode_steps=500)
env = make_vector_env(cfg)
obs, info = env.reset()

for step in range(1000):
    actions = env.sample_random_actions()
    obs, reward, terminated, truncated, info = env.step(actions)

    # Access rich metadata from info
    episode_step = info["episode_step"]   # int64[4]
    done_code    = info["done"]           # {0, 1, 2}[4]
    xformed_r    = info["xformed_reward"] # float64[4]
```

---

## NS-Gym with oscillating pole length

<!-- see examples/03_ns_gym_oscillating.py -->

```python
from mouse.envs import EnvConfig, make_vector_env

cfg = EnvConfig.ns_cartpole(
    seed=42,
    num_envs=2,
    non_stationary_params={
        "length": {
            "scheduler": "ContinuousScheduler",
            "update_fn": "OscillatingUpdate",
            "init_value": 0.5,
        },
    },
)

env = make_vector_env(cfg)
obs, info = env.reset()

for _ in range(500):
    obs, reward, terminated, truncated, info = env.step(env.sample_random_actions())
    ns_params = info.get("ns_params")  # current non-stationary parameter values
```

---

## Custom FrozenLake with Q* expert

<!-- see examples/02_q_star_expert.py -->

```python
import numpy as np
from mouse.envs import EnvConfig, make_vector_env

cfg = EnvConfig.frozenlake(seed=7, num_envs=1)
env = make_vector_env(cfg)
obs, info = env.reset()

for _ in range(200):
    # metadata_q_star is injected automatically from the value-iteration solution
    q_star = info["metadata_q_star"]  # float64[1, 4]
    action = q_star.argmax(axis=-1)
    obs, reward, terminated, truncated, info = env.step(action)
```

---

## Atari with preprocessing

<!-- see examples/04_atari_preprocessing.py -->

```python
from mouse.envs import EnvConfig, make_vector_env

cfg = EnvConfig(
    env_id="ALE/Pong-v5",
    seed=0,
    num_envs=4,
    max_episode_steps=10000,
    env_type="plain",
    atari_preprocessing=True,
    atari_preprocessing_kwargs={"frame_skip": 4, "screen_size": 84, "noop_max": 30},
    # ... other fields omitted for brevity
)

env = make_vector_env(cfg)
obs, info = env.reset()
# obs shape: (4, 84, 84) — grayscale frames
# env.obs_key == "observation_image"
```

---

## Partial observability

<!-- see examples/05_partial_observability.py -->

```python
from mouse.envs import EnvConfig, make_vector_env

# CartPole obs: 0=cart_pos, 1=cart_vel, 2=pole_angle, 3=pole_vel
cfg = EnvConfig.cartpole(seed=0, num_envs=2, observation_indices=[0, 2])
env = make_vector_env(cfg)
obs, info = env.reset()
print(obs.shape)  # (2, 2) — only cart position and pole angle
```

---

## Reward shaping

<!-- see examples/06_reward_shaping.py -->

Use `reward_scale` and `reward_shift` to transform rewards without affecting `episode_cum_reward` (which always reflects raw returns):

```python
from mouse.envs import EnvConfig, make_vector_env

cfg = EnvConfig(
    env_id="MountainCar-v0",
    seed=0,
    num_envs=1,
    max_episode_steps=200,
    env_type="plain",
    reward_scale=0.1,   # raw reward × 0.1
    reward_shift=1.0,   # then + 1.0
    # ... other fields omitted for brevity
)

env = make_vector_env(cfg)
obs, info = env.reset()

obs, reward, terminated, truncated, info = env.step(env.sample_random_actions())
raw_r     = reward[0]
xformed_r = info["xformed_reward"][0]  # raw * scale + shift
```
