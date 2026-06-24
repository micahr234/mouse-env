# Guide

<p align="center"><img src="mouse-env.png" width="400" alt="MOUSE environments"/></p>

**mouse-env** builds Gymnasium environments and formats their output for [mouse-core](https://github.com/micahr234/mouse-core) training. You configure an environment, call `step()`, and receive plain dict records with tensor fields — no Gymnasium `info` dicts to parse.

> MOUSE is in early development. APIs may change without notice.

## Install

```bash
pip install mouse-env
```

For local development, clone the repo and run `source scripts/install.sh`.

## Create an environment

```python
from mouse_envs import EnvConfig, make_env

cfg = EnvConfig(
    id="CartPole-v1",
    seed=0,
    num_envs=4,
    max_episode_steps=500,
)
env = make_env(cfg)
```

`make_env` accepts a single `EnvConfig` or a `list[EnvConfig]` and returns a `MouseEnv`. Use `id` for any Gymnasium environment id, including custom ids registered by this package such as `Procedural-FrozenLake-v1` and `SyntheticEnv-v1`. The constructed env exposes indexed names as `env.names`, for example `("CartPole-v1#0", "CartPole-v1#1")`; `env.names[0]` returns the first name for single-env use. Pass optional `name` when the public environment name should differ from the Gymnasium id:

```python
cfg = EnvConfig(id="CartPole-v1", name="train-cartpole", seed=0, num_envs=2, max_episode_steps=500)
env = make_env(cfg)
env.names  # ("train-cartpole#0", "train-cartpole#1")
```

To build the env yourself instead of by id, pass `env_fn` — a zero-arg factory returning a fresh Gymnasium env (mouse-env calls it once per parallel env, so it must return a new instance each time, not a shared one). `name` if set, otherwise `id`, acts as the base for `env.names`, and `max_episode_steps` is still required for reward normalisation; `kwargs`, `render`, and the time limit become your factory's responsibility. This is the place to do construction and wrapping outside mouse-env (Atari preprocessing, non-stationary NS-Gym envs, and so on).

```python
def make_cartpole():
    env = gym.make("CartPole-v1", max_episode_steps=500)
    return MyWrapper(env)

cfg = EnvConfig(id="my-cartpole", seed=0, num_envs=4, max_episode_steps=500, env_fn=make_cartpole)
```

Required fields:

| Field | Purpose |
|-------|---------|
| `id` | Gymnasium env id or a custom id registered by this package |
| `seed` | Base seed for parallel streams |
| `num_envs` | Number of environments stepped in parallel |
| `max_episode_steps` | Episode length budget (also used for reward normalisation) |

Everything else on `EnvConfig` is optional (`name` overrides the base for `env.names`; other knobs cover reward shaping, partial observations, custom env wrappers, observation-channel routing, non-stationary physics, expert Q-values, reset-frame defaults, and so on). Check the docstrings when you need them.

Expert Q-values are opt-in. Pass `q_star_source` directly when a rollout should include `outputs[i]["q_star"]`:

```python
cfg = EnvConfig(
    id="CartPole-v1",
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

### Multiple heterogeneous environments

Pass a `list[EnvConfig]` to combine different environment types in one `MouseEnv`. Each config becomes an independent inner env with its own parallel slots. Envs are stepped sequentially; outputs are never concatenated across configs:

```python
env = make_env([
    EnvConfig(id="CartPole-v1", seed=0, num_envs=4, max_episode_steps=500),
    EnvConfig(id="MountainCar-v0", seed=1, num_envs=2, max_episode_steps=200),
])
```

## Run a rollout

There is **no public `reset()`**. Call `step()` only. The first call performs an internal reset and returns initial observations; inputs on that call are ignored.

mouse-env does this so the rollout stream has one shape from the first token onward. A reset frame still contains the same fields as a normal transition: `observation`, `reward`, `done`, `time`, and the rest of the training metadata. Training code does not need a separate method for the first environment interaction, and sequence models do not need to handle a shorter reset-only record.

`step` always uses a nested structure indexed by env. `inputs_per_env[i]` is the input `list[dict]` for the i-th inner env; `step` returns `[(outputs, metrics), ...]`, one tuple per inner env.

```python
for _ in range(1000):
    inputs_per_env = env.sample_random_inputs()
    [(outputs, metrics)] = env.step(inputs_per_env)  # single-config case
```

For multiple configs:

```python
for _ in range(1000):
    results = env.step(env.sample_random_inputs())
    for outputs, metrics in results:
        ...
```

Every `step()` returns the same two-part shape per inner env:

```python
[(outputs, metrics)] = env.step(inputs_per_env)
# inputs_per_env[0][i]["action"]: tensor (input to step for slot i of env 0)
# env.names[i]: environment name for slot i
# outputs[i]:  all per-step fields — observation, reward, done, time, episode_index, reward_episodic, optional q_star/ns_params
# metrics[i]:  evaluation stats — cum reward, length
```

**`env.names`** holds all environment slot names flattened across inner envs. These start with `EnvConfig.name` when provided, otherwise `EnvConfig.id`, and append `#0`, `#1`, and so on. **`outputs`** is the rollout stream for one inner env. Each `outputs[i]` is a dict containing both the sequence-model inputs (observation, reward, done, time) and training/analysis context (episode_index, reward_episodic, and optionally q_star and ns_params). **`metrics`** sits alongside it at the same slot index.

When a sub-environment finishes, it auto-resets on the next step. That autoreset frame looks like the initial reset frame: it uses the configured `reset_reward` and always has `done == 0`. The actual episode boundary is the step where `done` is non-zero.

Configure `reset_reward` on `EnvConfig` when the initial/reset token should carry a value other than zero:

```python
cfg = EnvConfig(
    id="CartPole-v1",
    seed=0,
    num_envs=4,
    max_episode_steps=500,
    reset_reward=0.0,
)
```

---

## Input: inputs

Pass a `list[list[dict]]` — the outer list is indexed by inner env, the inner list by parallel slot. Each input dict has a single **`"action"`** key holding a tensor — no type suffix, no nested dict. `env.sample_random_inputs()` generates valid inputs automatically:

```python
import torch

# Single-config env — one inner env, num_envs=4 parallel slots:
inputs_per_env = [
    [{"action": torch.tensor(2, dtype=torch.int64)} for _ in range(env.num_envs)]
]
[(outputs, metrics)] = env.step(inputs_per_env)
```

Use `env.input_specs[i]` to find the expected dtype and shape for inner env `i`:

```python
spec = env.input_specs[0]
# spec.action.dtype  — torch.int64 for discrete spaces; torch.float32 for continuous
# spec.action.shape  — () for scalar actions; (n,) for multi-dimensional
```

`env.sample_random_inputs()` generates a valid nested input list automatically. On the first `step()` after construction, inputs are ignored.

---

## Output: `outputs`

Each inner env's `outputs` is a list indexed by parallel slot. Each `outputs[i]` is a plain dict with all per-step fields:

```python
{
    "time":            torch.tensor(int,   dtype=torch.int64),
    "observation":     torch.tensor([...], dtype=...),   # tensor; dtype and shape from env.output_specs[0].observation
    "reward":          torch.tensor(float, dtype=torch.float32),
    "done":            torch.tensor(int,   dtype=torch.int64),
    "episode_index":   int,
    "reward_episodic": float,
    # optional:
    "q_star":   np.ndarray,  # float64[action_dim], when configured;
                             # one-hot/Q-values for discrete spaces,
                             # the expert action vector for continuous spaces
    "ns_params": dict,       # when an env wrapper sets info["ns_params"]
}
```

For `gym.spaces.Dict` observation spaces, the subspace keys appear **directly** on the output dict instead of under `"observation"` (e.g. `outputs[0]["pos"]`, `outputs[0]["tile"]`).

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `time` | int64 tensor | Step index within the current episode (0-based). `0` on reset frames. |
| `observation` | tensor | The observation tensor. dtype and shape are described by `env.output_specs[i].observation`. |
| `reward` | float32 tensor | Raw environment reward. Uses `reset_reward` on reset frames. |
| `done` | int64 tensor | `0` running · `1` terminated · `2` truncated. Reset frames always use `0`. |
| `episode_index` | int | Episode counter for this parallel slot. |
| `reward_episodic` | float | Normalised training signal; `0.0` on reset frames. |
| `q_star` | float64 array | Expert Q-values when configured (optional). |
| `ns_params` | dict | Surfaced when an env wrapper sets `info["ns_params"]` (e.g. non-stationary envs); optional. |

Observations keep their native shape. Image observations (e.g. preprocessed Atari) stay 2-D/3-D — for example an 84×84 `AtariPreprocessing` frame arrives as an `(84, 84)` tensor, not a flat vector. Continuous channels stay 1-D and discrete channels stay scalar.

### Introspecting the contract: `output_specs` and `input_specs`

`env.output_specs[i]` and `env.input_specs[i]` are dataclasses with one `FieldSpec(dtype, shape)` attribute per key in the output / input dict for inner env `i`. They describe the full contract at construction time, before any steps:

```python
from mouse_envs import FieldSpec, OutputSpec, InputSpec

ospec = env.output_specs[0]   # OutputSpec for first inner env
ispec = env.input_specs[0]    # InputSpec for first inner env

ospec.observation.dtype   # torch.float32 (continuous/image) or torch.int64 (discrete)
ospec.observation.shape   # (4,) for CartPole; () for FrozenLake; (84, 84) for Atari
ospec.time.dtype          # torch.int64
ospec.reward.dtype        # torch.float32
ospec.q_star              # FieldSpec(np.float64, (action_dim,)) or None when not configured

ispec.action.dtype        # torch.int64 (discrete) or torch.float32 (continuous)
ispec.action.shape        # () for scalar actions; (n,) for multi-dimensional

# Dict observation space: observation is a dict of FieldSpecs
ospec.observation         # {"pos": FieldSpec(torch.float32, (2,)), "tile": FieldSpec(torch.int64, (1,))}
```

---

## Output: `metrics`

**Not model input — evaluation.** Episode statistics for the current step, aligned with `outputs[i]`. Use these to measure returns and episode lengths without parsing the rollout stream. For slot `i`, read `metrics[i]`:

| Field | Description |
|-------|-------------|
| `episode_cum_reward` | Cumulative raw return for each episode slot `i` finished on this step |
| `episode_length` | Length in steps for each episode slot `i` finished on this step |

Each field is a (possibly empty) list of floats:

- **`[]`** — slot `i` did not finish on this step (including reset/autoreset frames).
- **`[value]`** — slot `i` finished once; one entry per finish on this step.
- **`[v1, v2, …]`** — slot `i` finished multiple times on this step (unusual, but supported by the shape).

Note: `metrics[i]["episode_cum_reward"]` always reflects the **raw** (unscaled) return, even when reward shaping is enabled. The shaped training signal is in `outputs[i]["reward_episodic"]`.

---

## Custom envs and observation routing

mouse-env has no per-environment integration code. Two general `EnvConfig` knobs cover environment-specific needs:

- **`env_fn`** — build (and wrap) the env yourself in a zero-arg factory, instead of by `id`. mouse-env calls it once per parallel env, so return a fresh instance each time. This is where you apply any Gymnasium wrapper (preprocessing, observation transforms, time limits, and so on). It's also how you use envs such as Atari (`gymnasium.wrappers.AtariPreprocessing`) or non-stationary NS-Gym envs — construct and wrap them in the factory; see the examples below.
- **`observation_kind`** — force the observation channel: `"continuous"`, `"discrete"`, or `"image"`. When `None` (default), mouse-env auto-detects from the observation space. Auto-detection cannot recognise image spaces (an image is a `uint8` `Box`, which otherwise looks discrete), so image envs must set `observation_kind="image"`.

Envs that need an extra package install it via an optional extra: `pip install 'mouse-env[atari]'` (`ale_py` + `opencv-python`) or `pip install 'mouse-env[non-stationary]'` (`ns_gym`). To pull every optional env in one go, use `pip install 'mouse-env[all]'`.

---

## Examples

Runnable Jupyter notebooks in [`examples/`](../examples/):

- [01_random_rollout.ipynb](../examples/01_random_rollout.ipynb) — minimal end-to-end rollout and the `outputs`/`metrics` shape.
- [02_q_star_expert.ipynb](../examples/02_q_star_expert.ipynb) — expert Q-values via `q_star_source`: compute Q* offline with value iteration on standard `FrozenLake-v1` and inject it with `hf_q_table`.
- [03_ns_gym_oscillating.ipynb](../examples/03_ns_gym_oscillating.ipynb) — non-stationary CartPole (NS-Gym) built with an `env_fn` factory.
- [04_atari_preprocessing.ipynb](../examples/04_atari_preprocessing.ipynb) — Atari Pong with `AtariPreprocessing` via `env_fn` + `observation_kind="image"`.
- [05_partial_observability.ipynb](../examples/05_partial_observability.ipynb) — masking observation dimensions with `observation_indices`.
- [06_reward_shaping.ipynb](../examples/06_reward_shaping.ipynb) — `reward_scale` / `reward_shift` and `reward_episodic`.
- [07_synthetic_env.ipynb](../examples/07_synthetic_env.ipynb) — random finite discrete MDP for tabular experiments.
- [08_multi_env.ipynb](../examples/08_multi_env.ipynb) — combining multiple heterogeneous environments in one `MouseEnv` via a `list[EnvConfig]`.
- [09_procedural_frozenlake.ipynb](../examples/09_procedural_frozenlake.ipynb) — `Procedural-FrozenLake-v1`: fresh random map every episode with per-map Q* via `metadata_q_star`.
