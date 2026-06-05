# Guide

<p align="center"><img src="mouse-env.png" width="400" alt="MOUSE environments"/></p>

**mouse-env** builds vector Gymnasium environments and formats their output for [mouse-core](https://github.com/micahr234/mouse-core) training. You configure an environment, call `step()`, and receive plain dict records with tensor fields — no Gymnasium `info` dicts to parse.

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

`make_vector_env` returns a `MouseVectorEnv`. Use `group_id` for any Gymnasium environment id, including custom ids registered by this package such as `Procedural-FrozenLake-v1` and `SyntheticEnv-v1`.

To build the env yourself instead of by id, pass `env_fn` — a zero-arg factory returning a fresh Gymnasium env (mouse-env calls it once per parallel env, so it must return a new instance each time, not a shared one). `group_id` then acts purely as the identity label and `max_episode_steps` is still required for reward normalisation; `kwargs`, `render`, and the time limit become your factory's responsibility. This is the place to do construction and wrapping outside mouse-env (Atari preprocessing, non-stationary NS-Gym envs, and so on).

```python
def make_env():
    env = gym.make("CartPole-v1", max_episode_steps=500)
    return MyWrapper(env)

cfg = EnvConfig(group_id="my-cartpole", seed=0, num_envs=4, max_episode_steps=500, env_fn=make_env)
```

Required fields:

| Field | Purpose |
|-------|---------|
| `group_id` | Gymnasium env id or a custom id registered by this package |
| `seed` | Base seed for parallel streams |
| `num_envs` | Number of environments stepped in parallel |
| `max_episode_steps` | Episode length budget (also used for reward normalisation) |

Everything else on `EnvConfig` is optional (reward shaping, partial observations, custom env wrappers, observation-channel routing, non-stationary physics, expert Q-values, reset-frame defaults, and so on). Check the docstrings when you need them.

Expert Q-values are opt-in. Pass `q_star_source` directly when a rollout should include `results[i]["q_star"]`:

```python
cfg = EnvConfig(
    group_id="CartPole-v1",
    seed=0,
    num_envs=4,
    max_episode_steps=500,
    q_star_source={
        "provider": "sb3_rl_zoo",
        "algo": "ppo",
        "repo_id": "sb3/ppo-CartPole-v1",
        "filename": "ppo-CartPole-v1.zip",
        "deterministic": True,
    },
)
```

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
cfg = EnvConfig(
    group_id="CartPole-v1",
    seed=0,
    num_envs=4,
    max_episode_steps=500,
    reset_reward=0.0,
)
```

---

## Input: actions

Pass a `list[TensorDict]` of length `num_envs`. Each **`action` is a dict** — use `"discrete"` or `"continuous"` to match the environment's action space:

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
    "ns_params": dict,      # when an env wrapper sets info["ns_params"]
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
| `ns_params` | dict | Surfaced when an env wrapper sets `info["ns_params"]` (e.g. non-stationary envs); optional. |

Observations keep their native shape. Image observations (e.g. preprocessed Atari) stay 2-D/3-D in `observation["image"]` — for example an 84×84 `AtariPreprocessing` frame arrives as an `(84, 84)` tensor, not a flat vector. Continuous channels stay 1-D and discrete channels stay scalar.

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

## Custom envs and observation routing

mouse-env has no per-environment integration code. Two general `EnvConfig` knobs cover environment-specific needs:

- **`env_fn`** — build (and wrap) the env yourself in a zero-arg factory, instead of by `group_id`. mouse-env calls it once per parallel env, so return a fresh instance each time. This is where you apply any Gymnasium wrapper (preprocessing, observation transforms, time limits, and so on). It's also how you use envs such as Atari (`gymnasium.wrappers.AtariPreprocessing`) or non-stationary NS-Gym envs — construct and wrap them in the factory; see the examples below.
- **`observation_kind`** — force the observation channel: `"continuous"`, `"discrete"`, or `"image"`. When `None` (default), mouse-env auto-detects from the observation space. Auto-detection cannot recognise image spaces (an image is a `uint8` `Box`, which otherwise looks discrete), so image envs must set `observation_kind="image"`.

Envs that need an extra package install it via an optional extra: `pip install 'mouse-env[atari]'` (`ale_py` + `opencv-python`) or `pip install 'mouse-env[non-stationary]'` (`ns_gym`). To pull every optional env in one go, use `pip install 'mouse-env[all]'`.

---

## Examples

Runnable Jupyter notebooks in [`examples/`](../examples/):

- [01_random_rollout.ipynb](../examples/01_random_rollout.ipynb) — minimal end-to-end rollout and the `results`/`metrics` shape.
- [02_q_star_expert.ipynb](../examples/02_q_star_expert.ipynb) — expert Q-values via `q_star_source` (Procedural Frozen Lake).
- [03_ns_gym_oscillating.ipynb](../examples/03_ns_gym_oscillating.ipynb) — non-stationary CartPole (NS-Gym) built with an `env_fn` factory.
- [04_atari_preprocessing.ipynb](../examples/04_atari_preprocessing.ipynb) — Atari Pong with `AtariPreprocessing` via `env_fn` + `observation_kind="image"`.
- [05_partial_observability.ipynb](../examples/05_partial_observability.ipynb) — masking observation dimensions with `observation_indices`.
- [06_reward_shaping.ipynb](../examples/06_reward_shaping.ipynb) — `reward_scale` / `reward_shift` and `reward_episodic`.
- [07_synthetic_env.ipynb](../examples/07_synthetic_env.ipynb) — random finite discrete MDP for tabular experiments.
