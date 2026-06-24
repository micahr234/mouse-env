# MOUSE Environments 🐭

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-env/main/mouse-env.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for production use. APIs may change without notice.

**mouse-env** turns episodic reinforcement learning environments into <u>continuing environments</u>. Instead of asking user code to alternate between `step()` and `reset()`, mouse-env handles resets internally so a rollout can continue through one uninterrupted `step()` loop.

Most RL benchmarks are episodic: an agent acts until termination or truncation, the caller calls `reset()`, and a new trial begins. That is a good interface when each episode is an independent sample. It is less natural when the experiment studies behavior **across multiple episodes**, where what the agent observes or discovers in one episode can affect what it does in a later one.

You can stitch episodes together on top of Gymnasium yourself, but the result is usually ad hoc. Important choices become arbitrary: whether reset observations are kept, how episode boundaries are marked, and when an RL algorithm should bootstrap. **mouse-env** makes the episode-to-continuing conversion explicit and consistent in three ways:

* **Reset-free rollout.** Users keep calling `step(inputs)`. When an episode ends, mouse-env resets the underlying environment internally and returns the next observation without requiring a public `reset()` call.
* **Visible episode structure.** Terminations, truncations, and reset frames stay in the data returned by the environment, so agents and analysis code can see where one episode ended and the next began.
* **Task-level boundaries.** In episodic RL, credit is cut off at the reset boundary. mouse-env introduces a task level — a group of N consecutive episodes — and signals task boundaries with distinct `done` codes. The RL algorithm bootstraps at task end, not at each episode reset, so value can propagate freely across the episodes within a task.

The result is a continuing interface for episodic RL: ordinary episodic Gymnasium environments generate reset-free trajectories, with visible episode boundaries inside each task and explicit task boundaries that tell the algorithm when to cut credit.

---

## Install 📦

```bash
pip install mouse-env
```

For development:

```bash
git clone https://github.com/micahr234/mouse-env.git
cd mouse-env
source scripts/install.sh
```

---

## Quick start 🚀

Build an env, sample inputs, and keep stepping:

```python
from mouse_envs import EnvConfig, make_env

cfg = EnvConfig(
    id="CartPole-v1",
    seed=0,
    num_envs=4,
    episodes_per_task=5,
)
env = make_env(cfg)

for _ in range(1000):
    inputs = env.sample_random_inputs()
    outputs = env.step(inputs)

# Episode stats accumulate in env.tracker automatically
print(env.tracker.episode_cum_rewards)  # list[list[float]] — per slot
env.close()
```

Runnable notebooks in [`examples/`](examples/) cover every feature with worked code and explanations:

| Notebook | What it covers |
|----------|----------------|
| [01 — Random rollout](examples/01_random_rollout.ipynb) | End-to-end loop; output fields; `done` codes; reset frames; `EnvConfig`; `input_specs`/`output_specs`; tracker |
| [02 — Expert Q-values](examples/02_q_star_expert.ipynb) | `q_star_source`; `hf_q_table` provider; value iteration; greedy expert rollout |
| [03 — Non-stationary env](examples/03_ns_gym_oscillating.ipynb) | `env_fn` factory pattern; NS-Gym adapter; `ns_params` in outputs |
| [04 — Atari preprocessing](examples/04_atari_preprocessing.ipynb) | `env_fn` + `AtariPreprocessing`; `observation_kind="image"` |
| [05 — Partial observability](examples/05_partial_observability.ipynb) | `observation_indices`; masking observation dimensions |
| [06 — Reward shaping](examples/06_reward_shaping.ipynb) | `reward_scale`/`reward_shift`; effect on the raw `reward` field |
| [07 — Synthetic env](examples/07_synthetic_env.ipynb) | `SyntheticEnv-v1`; `env_q_star`; tabular experiments |
| [08 — Multiple envs](examples/08_multi_env.ipynb) | `list[EnvConfig]`; heterogeneous specs; env slot names |
| [09 — Procedural FrozenLake](examples/09_procedural_frozenlake.ipynb) | `Procedural-FrozenLake-v1`; per-map Q*; continual training |

---

## Core API ⚙️

There is no public rollout-time `reset()` call. The first `step()` quietly performs an internal reset and returns the initial observation using the same record shape as every other step. Inputs passed on that first call are ignored.

After an episode terminates or truncates, the next call to `step()` emits the reset observation for the next episode before normal stepping resumes.

`step()` returns a single flat `list[dict]` of **outputs** — one entry per slot. Each output dict contains model-visible training data: an `observation` tensor, rewards, done flags, time, episode metadata, optional `q_star` expert action-values, and environment-specific fields.

`inputs` is a flat `list[dict]` — one dict per slot, each with a single `"action"` tensor key. Use `env.input_specs[i]` to discover the expected dtype and shape for slot `i`; use `env.output_specs[i]` for the full output contract.

**Episode statistics** are kept separate from the per-step stream and are accumulated automatically in `env.tracker` (a `MetricsTracker`):

```python
env.tracker.episode_cum_rewards   # list[list[float]] — per-slot raw cumulative returns
env.tracker.episode_lengths       # list[list[float]] — per-slot episode step counts
env.tracker.clear()               # wipe accumulated data between evaluation runs
```

Boundaries are represented by integer-coded `done` values:

* `0` = running (normal step or reset frame)
* `1` = episode terminated naturally
* `2` = episode truncated by time limit
* `3` = episode terminated naturally, and this was the last episode in the task
* `4` = episode truncated, and this was the last episode in the task

Codes 1 and 2 indicate how an episode ended. Codes 3 and 4 carry the same episode-end meaning but additionally mark a task boundary. The RL algorithm bootstraps at codes 3 or 4 and treats codes 1 and 2 as interior dynamics — value keeps propagating forward through those episode resets. `episodes_per_task` in `EnvConfig` sets how many episodes make up one task.

Reset frames are ordinary `outputs` records with:

* the first observation of the new episode (or new task)
* `time=0`
* the configured `reset_reward`, which is `0` by default
* `done=0`

This keeps the rollout stream uniform while still making both episode and task structure explicit.

---

## Gymnasium environments 🌎

Pass any Gymnasium environment id as `id`. mouse-env builds the underlying Gymnasium env, steps it internally, and exposes the concatenated non-episodic stream through the same API.

Each constructed env exposes indexed names in `env.names`, formed from optional `EnvConfig.name` when provided, otherwise `EnvConfig.id`, plus `_0`, `_1`, and so on. `env.name` returns the first name for single-env use. Step outputs do not repeat this name on every record.

mouse-env also includes a couple of custom environments. Other envs that need their own package — Atari (`gymnasium[atari]`) or non-stationary NS-Gym (`ns_gym`) — have no special code here; you build them in an `env_fn` factory (see [Bring your own env](#bring-your-own-env-env_fn) and the [examples](examples/)).

### Procedural Frozen Lake

* **ID:** `Procedural-FrozenLake-v1`
* Random valid grid generation: size, holes, start/goal, and optional per-goal rewards.
* Example: [examples/09_procedural_frozenlake.ipynb](examples/09_procedural_frozenlake.ipynb)

### Synthetic Environment

* **ID:** `SyntheticEnv-v1`
* Random finite discrete MDP for controlled tabular experiments.
* Example: [examples/07_synthetic_env.ipynb](examples/07_synthetic_env.ipynb)

---

## Environment Tools 🛠️

mouse-env also includes a few knobs for augmenting and modifying environments.

### Expert Q-values (`q_star_source`)

Set `q_star_source` on `EnvConfig` to attach expert Q-values to every step output as `outputs[i]["info_env_q_star"]`. Useful for imitation learning, diagnostics, or guided exploration.

`q_star_source` is a plain `dict` with a required `"provider"` key plus provider-specific fields:

#### `"env_q_star"` — env-computed Q*

The env runs value iteration internally and injects Q-values into every step. Requires `SyntheticEnv-v1` or `Procedural-FrozenLake-v1`; no extra fields needed.

```python
q_star_source={"provider": "env_q_star"}
```

#### `"hf_q_table"` — precomputed tabular Q-table

Loads a Q-table from a local pickle or the Hugging Face Hub. The pickle must be `{"qtable": ndarray[states, actions]}` or a bare `ndarray`.

| Field | Required | Description |
|---|---|---|
| `"path"` | if no `repo_id` | Local path to a `.pkl` file |
| `"repo_id"` | if no `path` | HF Hub repo ID (e.g. `"user/my-qtable"`) |
| `"filename"` | no | File in the Hub repo (default: `"q-learning.pkl"`) |
| `"deterministic"` | no | Argmax action selection (default: `True`) |

#### `"sb3_rl_zoo"` — Stable-Baselines3 checkpoint

Loads an SB3 policy from a local `.zip` or the Hugging Face Hub. Requires `stable-baselines3`; `"qrdqn"` additionally requires `sb3-contrib`.

| Field | Required | Description |
|---|---|---|
| `"algo"` | yes | `"a2c"`, `"ddpg"`, `"dqn"`, `"ppo"`, `"sac"`, `"td3"`, `"qrdqn"` |
| `"path"` | if no `repo_id` | Local path to an SB3 `.zip` checkpoint |
| `"repo_id"` | if no `path` | HF Hub repo ID |
| `"filename"` | if no `path` | File in the Hub repo |
| `"device"` | no | `"cpu"`, `"cuda"`, or `"auto"` (default: `"cpu"`) |
| `"deterministic"` | no | Deterministic action selection (default: `True`) |

Example: [examples/02_q_star_expert.ipynb](examples/02_q_star_expert.ipynb)

### Bring your own env (`env_fn`)

Instead of using `id` to build a Gymnasium env, pass `env_fn` — a zero-arg factory that returns a freshly built (and already-wrapped, if you like) Gymnasium env. mouse-env calls it once per parallel env, so it must return a **new** env each time (not a shared instance). `name` if set, otherwise `id`, is used as the base for `env.names`. Time-limit truncation and any other wrappers are left entirely to your factory.

```python
def make_cartpole():
    env = gym.make("CartPole-v1", max_episode_steps=500)
    return MyWrapper(env)  # apply any Gymnasium wrappers here

cfg = EnvConfig(id="my-cartpole", seed=0, num_envs=4, episodes_per_task=5, env_fn=make_cartpole)
```

This is also how you apply custom Gymnasium wrappers (preprocessing, observation transforms, etc.): wrap inside your factory.

### Observation routing (`observation_kind`)

Force the observation channel with `observation_kind` (`"continuous"`, `"discrete"`, or `"image"`). Defaults to auto-detection from the observation space; required (`"image"`) for image envs, which auto-detection cannot recognise.

### Partial observability

Use `observation_indices` to mask dimensions on continuous-vector observation spaces.

Example: [examples/05_partial_observability.ipynb](examples/05_partial_observability.ipynb)

### Reward shaping

Use `reward_scale` and `reward_shift` to scale and shift the raw per-step reward before it appears in `outputs[i]["reward"]`. The formula is `reward = raw × scale + shift`.

Example: [examples/06_reward_shaping.ipynb](examples/06_reward_shaping.ipynb)

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a record of notable changes.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
