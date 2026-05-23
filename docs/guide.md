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
from mouse.envs import EnvConfig, make_vector_env

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

Everything else on `EnvConfig` is optional (reward shaping, partial observations, Atari preprocessing, non-stationary physics, expert Q-values, and so on). Check the docstrings when you need them.

## Run a rollout

There is **no public `reset()`**. Call `step()` only. The first call performs an internal reset and returns initial observations; actions on that call are ignored.

```python
for _ in range(1000):
    actions = env.sample_random_actions()
    data, metadata, metrics = env.step(actions)
```

Every `step()` returns the same three-part shape:

```python
data, metadata, metrics = env.step(actions)
# actions[i]["action"]: dict — discrete or continuous (input to step)
# data[i]:     sequence-model tokens — observation (dict), reward, done, time
# metadata[i]: training & analysis — group_id, episode_index, reward_episodic, optional q_star
# metrics[i]:  evaluation stats — cum reward, length
```

**`data`** is the rollout stream you generally pass to an LLM or other sequence model. **`metadata`** and **`metrics`** sit alongside it at the same env index: they are not part of the model input by default, but **`metadata`** is often used to help training (e.g. Q* supervision), analyze performance, or inspect rollouts; **`metrics`** summarizes episode outcomes for evaluation and logging.

When a sub-environment finishes, it auto-resets on the next step. That autoreset frame looks like the initial reset frame: dummy reward and `done == 0`. The actual episode boundary is the step where `done` is non-zero.

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

data, metadata, metrics = env.step(actions)
```

For continuous action spaces, use `"continuous"` instead of `"discrete"`.

`env.sample_random_actions()` generates a valid action list with the same dict layout. On the first `step()` after construction, actions are ignored.

---

## Output: `data`

**Model input.** `data` is what you generally feed to an LLM or sequence model — the per-step rollout stream. It is a list of length `num_envs`. Each element is a scalar TensorDict (`batch_size=[]`) with the same keys:

```python
TensorDict({
    "time": torch.tensor(int, dtype=torch.int64),
    "observation": {
        "discrete":   torch.tensor([...], dtype=torch.int64),    # optional
        "continuous": torch.tensor([...], dtype=torch.float32),  # optional
        "image":      torch.tensor([...], dtype=torch.float32),  # optional
    },
    "reward": torch.tensor(float, dtype=torch.float32),
    "done": torch.tensor(int, dtype=torch.int64),
}, batch_size=[])
```

### Fields

| Field | Description |
|-------|-------------|
| `time` | Step index within the current episode (0-based). `0` on reset frames. |
| `observation` | **Dict** of tensors — any combination of `discrete`, `continuous`, and/or `image` (whichever keys the environment provides). Not a single flat vector. |
| `reward` | Raw environment reward. `0.0` on reset frames (initial or autoreset). |
| `done` | `0` running · `1` terminated · `2` truncated. `0` on reset frames. |

Image observations (e.g. preprocessed Atari) are flattened vectors in `observation["image"]`.

Actions are **not** echoed in `data` — they are input to `step()` only.

---

## Output: `metadata`

**Not model input — training & analysis.** Per-env context aligned with `data[i]`. You typically do not tokenize or embed this into the sequence model, but it is often used to support training (expert Q-values, auxiliary losses), analyze performance, track env identity across parallel streams, or inspect non-stationary dynamics. For env `i`, read `metadata[i]`:

| Field | Always | Access |
|-------|--------|--------|
| `group_id` | yes | `metadata[i]["group_id"]` (e.g. `"CartPole-v1#0"`) |
| `episode_index` | yes | `metadata[i]["episode_index"]` |
| `reward_episodic` | yes | `metadata[i]["reward_episodic"]` — normalised training signal; `0.0` on reset frames |
| `q_star` | no | `metadata[i]["q_star"]` — expert Q-values when configured |
| `ns_params` | no | `metadata[i]["ns_params"]` — non-stationary parameters (NS-Gym envs only) |

---

## Output: `metrics`

**Not model input — evaluation.** Episode statistics for the current step, aligned with `data[i]`. Use these to measure returns and episode lengths without parsing the rollout stream. For env `i`, read `metrics[i]`:

| Field | Description |
|-------|-------------|
| `episode_cum_reward` | Cumulative raw return for each episode env `i` finished on this step |
| `episode_length` | Length in steps for each episode env `i` finished on this step |

Each field is a (possibly empty) list of floats:

- **`[]`** — env `i` did not finish on this step (including reset/autoreset frames).
- **`[value]`** — env `i` finished once; one entry per finish on this step.
- **`[v1, v2, …]`** — env `i` finished multiple times on this step (unusual, but supported by the shape).

Note: `metrics[i]["episode_cum_reward"]` always reflects the **raw** (unscaled) return, even when reward shaping is enabled. The shaped training signal is in `metadata[i]["reward_episodic"]`.

---

## Non-stationary environments (NS-Gym)

[NS-Gym](https://github.com/scope-lab-vu/ns_gym) is an external open-source framework for non-stationary MDPs — mouse-env **uses** it; we did not create it. See the [NS-Gym docs](https://nsgym.io/) for the underlying schedulers, update functions, and wrappers.

What mouse-env adds on top:

- **Plain dict configs** — pass `non_stationary_params` on `EnvConfig` (e.g. `EnvConfig.ns_cartpole(...)`) instead of wiring NS-Gym scheduler/update classes by hand
- **Standard observations** — `NSGymInterfaceWrapper` strips NS-Gym’s dict observations down to flat state vectors compatible with the rest of the stack
- **`metadata[i]["ns_params"]`** — current parameter values and change flags each step, for logging or auxiliary training signals
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
