# Examples

## Collecting rollouts

```python
from mouse.envs.envs import EnvConfig, make_vector_env

cfg = EnvConfig(
    env_id="CartPole-v1",
    seed=0,
    num_envs=4,
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

```python
from mouse.envs.envs import EnvConfig, make_vector_env

cfg = EnvConfig(
    env_id="NS-CartPole-v1",
    seed=42,
    num_envs=2,
    max_episode_steps=500,
    kwargs=None,
    render=False,
    non_stationary_params={
        "length": {
            "scheduler": "ContinuousScheduler",
            "update_fn": "OscillatingUpdate",
            "init_value": 0.5,
        },
    },
    num_steps=None,
    action_source_loop_prob_schedule=None,
    action_source_episode_prob_schedule=None,
    q_star_source=None,
    action_source="random",
    action_source_temperature=1.0,
    split="train",
    env_type="ns_gym",
    atari_preprocessing=None,
    atari_preprocessing_kwargs=None,
    observation_indices=None,
)

env = make_vector_env(cfg)
obs, info = env.reset()

for _ in range(500):
    obs, reward, terminated, truncated, info = env.step(env.sample_random_actions())
    ns_params = info.get("ns_params")  # current non-stationary parameter values
```

---

## Custom FrozenLake with Q* expert

```python
from mouse.envs.envs import EnvConfig, make_vector_env

cfg = EnvConfig(
    env_id="Custom-FrozenLake-v1",
    seed=7,
    num_envs=1,
    max_episode_steps=200,
    kwargs={"map_size": 8, "is_slippery": True},
    render=False,
    non_stationary_params=None,
    num_steps=None,
    action_source_loop_prob_schedule=None,
    action_source_episode_prob_schedule=None,
    q_star_source={"type": "env"},
    action_source="q_star",
    action_source_temperature=0.0,
    split="train",
    env_type="plain",
    atari_preprocessing=None,
    atari_preprocessing_kwargs=None,
    observation_indices=None,
)

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

```python
from mouse.envs.envs import EnvConfig, make_vector_env

cfg = EnvConfig(
    env_id="ALE/Pong-v5",
    seed=0,
    num_envs=4,
    max_episode_steps=10000,
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
    atari_preprocessing=True,
    atari_preprocessing_kwargs={"frame_skip": 4, "screen_size": 84, "noop_max": 30},
    observation_indices=None,
)

env = make_vector_env(cfg)
obs, info = env.reset()
# obs shape: (4, 84, 84) — grayscale frames
# env.obs_key == "observation_image"
```

---

## Reward shaping

Use `reward_scale` and `reward_shift` to transform rewards without affecting `episode_cum_reward` (which always reflects raw returns):

```python
cfg = EnvConfig(
    env_id="MountainCar-v0",
    reward_scale=0.1,   # shrink rewards
    reward_shift=1.0,   # shift up
    ...
)
```
