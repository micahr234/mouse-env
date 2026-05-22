# Guide

mouse-env builds **vector Gymnasium environments** for [mouse-core](https://github.com/micahr234/mouse-core) with a shared wrapper stack and rollout metadata.

## Quick start

```python
from mouse.envs import EnvConfig, make_vector_env

env = make_vector_env(EnvConfig.cartpole(num_envs=4, seed=0))
obs, info = env.reset()

for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.sample_random_actions())
```

Keyword form also works:

```python
env = make_vector_env("CartPole-v1", 0, max_steps_per_episode=500, num_envs=4)
```

## Package layout

| Path | Purpose |
|------|---------|
| `config.py` | `EnvConfig`, presets, Q* defaults |
| `factory.py` | `make_vector_env` |
| `backends/` | Plain, NS-Gym, Atari env construction |
| `stack/` | Vector wrappers (`build_vector_env_stack`) |
| `action_star.py` | Expert policies → `metadata_q_star` |
| `frozenlake.py` / `synthetic.py` | Custom registered envs |

API details live in **docstrings** (`EnvConfig`, `make_vector_env`, wrappers).

## Environment types

See [environments.md](environments.md) for CartPole, Atari, NS-Gym, FrozenLake, and SyntheticEnv configuration.

## Rollout data (mouse-core)

See [rollout_contract.md](rollout_contract.md) for the v1 step record (`env_id`, `episode_index`, `step_index`, action/observation/reward dicts, `done`).

Legacy keys still emitted by the stack (`env_name`, `episode_step`, `xformed_reward`, …) are documented in [wrappers.md](wrappers.md).

## Examples

[examples.md](examples.md) — NS-Gym, Atari, expert Q*, partial observability.

## mouse-core integration

[mouse_core_alignment.md](mouse_core_alignment.md) — how datasets and `DatasetStore` should map to the contract.
