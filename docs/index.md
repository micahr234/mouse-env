# Meta-Optimization Using Sequential Experiences — Environments

<p align="center"><img src="mouse.png" alt="MOUSE" width="400"/></p>

**mouse-envs** is the environment package for [MOUSE](https://micahr234.github.io/mouse-core/), a modular PyTorch library for in-context reinforcement learning. It provides Gymnasium vector environments, NS-Gym non-stationary wrappers, custom tabular MDPs (FrozenLake, SyntheticEnv), and rich rollout metadata (episode stats, expert Q-values) used to collect training data for MOUSE agents.

---

## What mouse-envs provides

### Environments

- **Plain Gymnasium** — standard discrete-action envs (CartPole, MountainCar, LunarLander, etc.) wrapped in a vector stack with rich info injection.
- **Atari** — ALE environments with optional `AtariPreprocessing` (grayscale, frame skip, screen resize).
- **NS-Gym** — non-stationary variants of classic control envs where physics parameters shift over time according to configurable schedules.
- **Custom FrozenLake** — procedurally generated maps with optional value-iteration Q* expert.
- **Custom SyntheticEnv** — random tabular MDPs for controlled experiments.

### Wrapper stack

Every environment is wrapped in a standard stack that injects into `info`:

| Key | Description |
|-----|-------------|
| `episode_length` | Per-env episode length at termination (NaN otherwise) |
| `episode_cum_reward` | Raw cumulative reward at termination (NaN otherwise) |
| `episode_step` | Step count within current episode |
| `global_step` | Monotonically increasing step count |
| `xformed_reward` | Normalised reward used for model training |
| `done` | 0 = running, 1 = terminated, 2 = truncated |
| `env_name` | Environment name string |
| `env_idx` | Per-env integer index |
| `metadata_q_star` | Expert Q-values (when a `q_star_source` is configured) |

---

## Install

```bash
pip install mouse-envs
```

---

## Quick start

```python
from mouse.envs.envs import EnvConfig, make_vector_env

cfg = EnvConfig(
    env_id="CartPole-v1",
    seed=42,
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

for _ in range(100):
    actions = env.sample_random_actions()
    obs, reward, terminated, truncated, info = env.step(actions)
```

---

## Guides

- [Environments](environments.md) — supported environment types and configuration
- [Wrapper Stack](wrappers.md) — the vector env wrapper stack in detail
- [Examples](examples.md) — collecting rollouts, NS-Gym, expert Q-values

---

The full API reference is available [here](api/envs.md).
