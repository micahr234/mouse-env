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

`make_vector_env` returns a `MouseVectorEnv`. `EnvConfig` also has preset helpers (`cartpole()`, `atari()`, `frozenlake()`, and others) — see the `EnvConfig` docstrings for options.

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
# data:     list[TensorDict]  — one record per parallel env (batch_size=[])
# metadata: dict              — batch-level context
# metrics:  dict              — batch-level episode statistics
```

When a sub-environment finishes, it auto-resets on the next step. That autoreset frame looks like the initial reset frame: dummy reward and `done == 0`. The actual episode boundary is the step where `done` is non-zero.

---

## Input: actions

Pass a `list[TensorDict]` of length `num_envs`. Each element wraps an `"action"` dict with `"discrete"` or `"continuous"`, matching the environment's action space:

```python
from tensordict import TensorDict
import torch

actions = [
    TensorDict({"action": {"discrete": torch.tensor([2])}}, batch_size=[])
    for _ in range(env.num_envs)
]
data, metadata, metrics = env.step(actions)
```

For continuous action spaces, use `"continuous"` instead of `"discrete"`.

`env.sample_random_actions()` generates a valid action list. On the first `step()` after construction, actions are ignored.

---

## Output: `data`

`data` is a list of length `num_envs`. Each element is a scalar TensorDict (`batch_size=[]`) with the same keys:

```python
TensorDict({
    "time": torch.tensor(int, dtype=torch.int64),
    "observation": {
        "discrete":   torch.tensor([...], dtype=torch.int64),    # optional
        "continuous": torch.tensor([...], dtype=torch.float32),  # optional
        "image":      torch.tensor([...], dtype=torch.float32),  # optional
    },
    "reward": {
        "step":     torch.tensor(float, dtype=torch.float32),
        "episodic": torch.tensor(float, dtype=torch.float32),
    },
    "done": torch.tensor(int, dtype=torch.int64),
}, batch_size=[])
```

### Fields

| Field | Description |
|-------|-------------|
| `time` | Step index within the current episode (0-based). `0` on reset frames. |
| `observation` | Any combination of `discrete`, `continuous`, and/or `image`. Include whichever keys the environment provides. |
| `reward.step` | Raw environment reward. `0.0` on reset frames (initial or autoreset). |
| `reward.episodic` | Normalised training signal. `0.0` on reset frames. |
| `done` | `0` running · `1` terminated · `2` truncated. `0` on reset frames. |

Image observations (e.g. preprocessed Atari) are flattened vectors in `observation["image"]`.

Actions are **not** echoed in `data` — they are input to `step()` only.

---

## Output: `metadata`

Batch-level context. Not stored inside individual TensorDicts.

| Key | Always | Description |
|-----|--------|-------------|
| `group_ids` | yes | `list[str]` — one id per env index (e.g. `"CartPole-v1#0"`) |
| `episode_index` | yes | `int64[num_envs]` — monotonic episode counter per stream |
| `q_star` | no | `float64[num_envs, action_dim]` — expert Q-values when configured |
| `ns_params` | no | Current non-stationary parameter values (NS-Gym envs) |

Use `group_ids[i]` to identify which environment stream a record came from.

---

## Output: `metrics`

Episode statistics at batch level. Values are `float64[num_envs]`.

| Key | Description |
|-----|-------------|
| `episode_cum_reward` | Cumulative raw return for the episode that just ended |
| `episode_length` | Length in steps of the episode that just ended |

Both are `NaN` while an episode is running. They are filled when `done != 0` on that env index.

Note: `metrics["episode_cum_reward"]` always reflects the **raw** (unscaled) return, even when reward shaping is enabled. Transformed rewards appear in `data[i]["reward"]["episodic"]`.

## Examples

Jupyter notebooks in [`examples/`](../examples/) walk through specific setups (random rollout, expert Q*, Atari preprocessing, partial observability, reward shaping, non-stationary physics).
