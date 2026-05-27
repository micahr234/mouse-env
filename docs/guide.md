# Guide

<p align="center"><img src="mouse-env.png" width="400" alt="MOUSE environments"/></p>

**mouse-env** builds vector Gymnasium environments and formats their output for [mouse-core](https://github.com/micahr234/mouse-core) training. You configure an environment, call `step()`, and receive structured TensorDict records — no Gymnasium `info` dicts to parse.

> MOUSE is in early development. APIs may change without notice.

## Install

```bash
pip install mouse-env
```

For local development, clone the repo and run `source scripts/install.sh`.

## Create an environment

```python
from mouse_envs import EnvConfig, make_vector_env

cfg = EnvConfig(
    group_id="CartPole-v1",
    seed=0,
    num_envs=4,
    max_episode_steps=500,
)
env = make_vector_env(cfg)
```

`make_vector_env` returns a `MouseVectorEnv`. `EnvConfig` also has preset helpers (`cartpole()`, `procedural_frozenlake()`, `synthetic()`, `atari()`, and others) — see the `EnvConfig` docstrings for options.

Required fields:

| Field | Purpose |
|-------|---------|
| `group_id` | Gymnasium env id or a custom id registered by this package |
| `seed` | Base seed for parallel streams |
| `num_envs` | Number of environments stepped in parallel |
| `max_episode_steps` | Episode length budget (also used for reward normalisation) |

Everything else on `EnvConfig` is optional (reward shaping, partial observations, Atari preprocessing, non-stationary physics, expert Q-values, reset-frame defaults, and so on). Check the docstrings when you need them.

## Run a rollout

There is **no public `reset()`**. Call `step()` only. The first call performs an internal reset and returns initial observations; actions on that call are ignored.

mouse-env does this so the rollout stream has one shape from the first token onward. A reset frame still contains the same fields as a normal transition: `observation`, `reward`, `done`, `time`, and the rest of the training metadata. Training code does not need a separate method for the first environment interaction, and sequence models do not need to handle a shorter reset-only record.

```python
for _ in range(1000):
    actions = env.sample_random_actions()
    results, metrics = env.step(actions)
```

Every `step()` returns the same two-part shape:

```python
results, metrics = env.step(actions)
# actions[i]["action"]: dict — discrete or continuous (input to step)
# results[i]:  all per-step fields — observation, reward, done, time, group_id, episode_index, reward_episodic, optional q_star/ns_params
# metrics[i]:  evaluation stats — cum reward, length
```

**`results`** is the rollout stream. Each `results[i]` is a dict containing both the sequence-model inputs (observation, reward, done, time) and training/analysis context (group_id, episode_index, reward_episodic, and optionally q_star and ns_params). **`metrics`** sits alongside it at the same env index and summarizes episode outcomes for evaluation and logging.

When a sub-environment finishes, it auto-resets on the next step. That autoreset frame looks like the initial reset frame: it uses the configured `reset_reward` and always has `done == 0`. The actual episode boundary is the step where `done` is non-zero.

Configure `reset_reward` on `EnvConfig` when the initial/reset token should carry a value other than zero:

```python
cfg = EnvConfig.cartpole(
    seed=0,
    num_envs=4,
    max_episode_steps=500,
    reset_reward=0.0,
)
```

---

## Input: actions

Pass a `list[TensorDict]` of length `num_envs`. Like `observation` in `data`, each **`action` is a dict** — use `"discrete"` or `"continuous"` to match the environment's action space:

```python
from tensordict import TensorDict
import torch

# Discrete env (e.g. Procedural Frozen Lake, Atari):
actions = [
    TensorDict({"action": {"discrete": torch.tensor([2])}}, batch_size=[])
    for _ in range(env.num_envs)
]

# Continuous env (e.g. CartPole with a Box action space would use "continuous"):
# TensorDict({"action": {"continuous": torch.tensor([...])}}, batch_size=[])

results, metrics = env.step(actions)
```

For continuous action spaces, use `"continuous"` instead of `"discrete"`.

`env.sample_random_actions()` generates a valid action list with the same dict layout. On the first `step()` after construction, actions are ignored.

---

## Output: `results`

`results` is a list of length `num_envs`. Each `results[i]` is a plain dict with all per-step fields:

```python
{
    "time": torch.tensor(int, dtype=torch.int64),
    "observation": {
        "discrete":   torch.tensor([...], dtype=torch.int64),    # optional
        "continuous": torch.tensor([...], dtype=torch.float32),  # optional
        "image":      torch.tensor([...], dtype=torch.float32),  # optional
    },
    "reward": torch.tensor(float, dtype=torch.float32),
    "done": torch.tensor(int, dtype=torch.int64),
    "group_id": str,
    "episode_index": int,
    "reward_episodic": float,
    # optional:
    "q_star": np.ndarray,   # float64[action_dim], when configured
    "ns_params": dict,      # NS-Gym envs only
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `time` | int64 tensor | Step index within the current episode (0-based). `0` on reset frames. |
| `observation` | dict of tensors | Any combination of `discrete`, `continuous`, and/or `image` keys. |
| `reward` | float32 tensor | Raw environment reward. Uses `reset_reward` on reset frames. |
| `done` | int64 tensor | `0` running · `1` terminated · `2` truncated. Reset frames always use `0`. |
| `group_id` | str | Env identity string (e.g. `"CartPole-v1#0"`). |
| `episode_index` | int | Episode counter for this parallel env. |
| `reward_episodic` | float | Normalised training signal; `0.0` on reset frames. |
| `q_star` | float64 array | Expert Q-values when configured (optional). |
| `ns_params` | dict | Current non-stationary parameters; NS-Gym envs only (optional). |

Image observations (e.g. preprocessed Atari) are flattened vectors in `observation["image"]`.

---

## Output: `metrics`

**Not model input — evaluation.** Episode statistics for the current step, aligned with `results[i]`. Use these to measure returns and episode lengths without parsing the rollout stream. For env `i`, read `metrics[i]`:

| Field | Description |
|-------|-------------|
| `episode_cum_reward` | Cumulative raw return for each episode env `i` finished on this step |
| `episode_length` | Length in steps for each episode env `i` finished on this step |

Each field is a (possibly empty) list of floats:

- **`[]`** — env `i` did not finish on this step (including reset/autoreset frames).
- **`[value]`** — env `i` finished once; one entry per finish on this step.
- **`[v1, v2, …]`** — env `i` finished multiple times on this step (unusual, but supported by the shape).

Note: `metrics[i]["episode_cum_reward"]` always reflects the **raw** (unscaled) return, even when reward shaping is enabled. The shaped training signal is in `results[i]["reward_episodic"]`.

---

## Non-stationary environments (NS-Gym)

[NS-Gym](https://github.com/scope-lab-vu/ns_gym) is an external open-source framework for non-stationary MDPs — mouse-env **uses** it; we did not create it. See the [NS-Gym docs](https://nsgym.io/) for the underlying schedulers, update functions, and wrappers.

What mouse-env adds on top:

- **Plain dict configs** — pass `non_stationary_params` on `EnvConfig` (e.g. `EnvConfig.ns_cartpole(...)`) instead of wiring NS-Gym scheduler/update classes by hand
- **Standard observations** — `NSGymInterfaceWrapper` strips NS-Gym’s dict observations down to flat state vectors compatible with the rest of the stack
- **`results[i]["ns_params"]`** — current parameter values and change flags each step, for logging or auxiliary training signals
- **Same vector `step()` API** — non-stationary envs run through `make_vector_env` like CartPole or Atari

Example: [examples/03_ns_gym_oscillating.ipynb](../examples/03_ns_gym_oscillating.ipynb).

---

## Atari (ALE integration)

[Atari Learning Environment (ALE)](https://github.com/Farama-Foundation/Arcade-Learning-Environment) envs are provided by Gymnasium — mouse-env **uses** them as-is; we did not modify the games. See the [Gymnasium Atari docs](https://gymnasium.farama.org/environments/atari/).

What mouse-env adds:

- **`EnvConfig.atari()`** — bundles common `AtariPreprocessing` defaults (frame skip, grayscale 84×84, noop warm-up)
- **Flattened `observation.image`** — preprocessed frames in the usual `data[i]` layout
- **Same vector `step()` API** — parallel Atari streams like any other env

Requires `gymnasium[atari]` / `ale_py`. Example: [examples/04_atari_preprocessing.ipynb](../examples/04_atari_preprocessing.ipynb).

## Examples

Jupyter notebooks in [`examples/`](../examples/) walk through specific setups (random rollout, expert Q*, Atari preprocessing, partial observability, reward shaping, non-stationary physics).
