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


## News 📰

- **2026-06-26 — `SingleEnv` / `GroupEnv`** `make_env` returns one standalone env; `make_group_env` handles parallel rollouts — a cleaner fit for online training loops that step envs individually or in batches.

See [CHANGELOG.md](CHANGELOG.md) for the full release history.


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


## Quick start 🚀

Build an env, sample inputs, and keep stepping:

```python
from mouse_envs import EnvConfig, make_env

cfg = EnvConfig(
    id="CartPole-v1",
    reset_seed=0,
    episodes_per_task=5,
)
env = make_env(cfg)

for _ in range(1000):
    output = env.step(env.sample_random_input())

# Episode stats accumulate in env.tracker automatically
print(env.tracker.episode_cum_rewards)  # list[float]
env.close()
```

For multiple envs, use `make_group_env`:

```python
from mouse_envs import EnvConfig, make_group_env

env = make_group_env([
    EnvConfig(id="CartPole-v1", reset_seed=0, name="cp-0", episodes_per_task=5),
    EnvConfig(id="CartPole-v1", reset_seed=1, name="cp-1", episodes_per_task=5),
])
for _ in range(1000):
    outputs = env.step(env.sample_random_input())  # list[dict]
env.close()
```

Runnable notebooks in [`examples/`](examples/) cover every feature with worked code and explanations:

| Notebook | What it covers |
|----------|----------------|
| [01 — Random rollout](examples/01_random_rollout.ipynb) | End-to-end loop; output fields; `done` codes; reset frames; `EnvConfig`; `input_spec`/`output_spec` |
| [02 — Expert Q-values](examples/02_q_star_expert.ipynb) | `q_star_source`; `hf_q_table` provider; value iteration; greedy expert rollout |
| [03 — Non-stationary env](examples/03_ns_gym_oscillating.ipynb) | `env_fn` factory pattern; NS-Gym adapter; `ns_params` in outputs |
| [04 — Atari preprocessing](examples/04_atari_preprocessing.ipynb) | `env_fn` + `AtariPreprocessing`; preprocessed frame passthrough |
| [05 — Partial observability](examples/05_partial_observability.ipynb) | `observation_indices`; masking observation dimensions |
| [06 — Reward shaping](examples/06_reward_shaping.ipynb) | `reward_scale`/`reward_shift`; effect on the raw `reward` field |
| [07 — Synthetic env](examples/07_synthetic_env.ipynb) | `SyntheticEnv-v1`; `q_star`; tabular experiments |
| [08 — Multiple envs](examples/08_multi_env.ipynb) | `make_group_env`; heterogeneous specs; env instance names |
| [09 — Procedural FrozenLake](examples/09_procedural_frozenlake.ipynb) | `Procedural-FrozenLake-v1`; per-map Q*; continual training |
| [10 — RNG seeding control](examples/10_rng_seeding_control.ipynb) | `map_seed`; `reset_seed`; reproducible generated maps and resets |
| [11 — Play Procedural FrozenLake](examples/11_play_procedural_frozenlake.ipynb) | D-pad controls and rendered output for manually playing generated lakes |
| [12 — Tracker](examples/12_metrics_tracker.ipynb) | `env.tracker`; raw vs shaped returns; `clear()` between eval runs; multi-env aggregation |


## Core API ⚙️

`SingleEnv` and `GroupEnv` use the Mouse reset-free rollout protocol instead of the standard Gymnasium `reset()`/`step(action)` loop. Public `reset()` raises `NotImplementedError`; the first `step()` quietly performs an internal reset and returns the initial observation. Inputs passed on that first call are ignored.

After an episode terminates or truncates, the next call to `step()` emits the reset observation for the next episode before normal stepping resumes.

**`make_env(EnvConfig)`** returns a `SingleEnv`. **`make_group_env(list[EnvConfig])`** returns a `GroupEnv`. Both share the same method names; return types differ:

| Method | `SingleEnv` | `GroupEnv` |
|---|---|---|
| `step(...)` | `(dict) -> dict` | `(list[dict]) -> list[dict]` |
| `sample_random_input()` | `-> dict` | `-> list[dict]` |
| `tracker` | `Tracker` | `GroupTracker` |

Each output dict contains model-visible training data: an `observation` value, rewards, done flags, time, episode metadata, optional expert Q-values as `info_q_star`, and environment-specific `info_*` fields.

On a `SingleEnv`, use `env.input_spec` and `env.output_spec`. On a `GroupEnv`, use `env.input_specs[i]` and `env.output_specs[i]`. Actions and observations preserve the underlying Gymnasium spaces' native dtypes wherever possible.

`GroupEnv` exposes tuple `action_space` and `observation_space` for Gymnasium compatibility. On a `SingleEnv`, `action_space` and `observation_space` are the underlying gym spaces directly.

**Episode statistics** are kept separate from the per-step stream and accumulate automatically in `env.tracker`:

```python
# SingleEnv — flat lists (Tracker)
env.tracker.episode_cum_rewards   # list[float]
env.tracker.episode_lengths       # list[float]

# GroupEnv — per-env lists (GroupTracker, read-through, no own storage)
env.tracker.episode_cum_rewards   # list[list[float]]
env.tracker.episode_lengths       # list[list[float]]
env.tracker.clear()               # wipe accumulated data between evaluation runs
```

Boundaries are represented by integer-coded `done` values:

* `0` = running (normal step or reset frame)
* `1` = episode terminated naturally
* `2` = episode truncated by time limit
* `3` = episode terminated naturally, and this was the last episode in the task
* `4` = episode truncated, and this was the last episode in the task

Codes 1 and 2 indicate how an episode ended. Codes 3 and 4 carry the same episode-end meaning but additionally mark a task boundary. The RL algorithm bootstraps at codes 3 or 4 and treats codes 1 and 2 as interior dynamics — value keeps propagating forward through those episode resets. `episodes_per_task` in `EnvConfig` sets how many episodes make up one task. Defaults to `0` (unlimited) — the task boundary never fires automatically.

Reset frames are ordinary `outputs` records with:

* the first observation of the new episode (or new task)
* `time=0`
* the configured `reset_reward`, which is `0` by default
* `done=0`

This keeps the rollout stream uniform while still making both episode and task structure explicit.


## Gymnasium environments 🌎

Pass any Gymnasium environment id as `id`. mouse-env builds the underlying Gymnasium env, steps it internally, and exposes the concatenated non-episodic stream through the same API.

Each constructed env exposes a name via `env.name` on `SingleEnv` or `env.names` on `GroupEnv`, formed from optional `EnvConfig.name` when provided, otherwise `EnvConfig.id`. Step outputs do not repeat this name on every record.

mouse-env also includes a couple of custom environments. Other envs that need their own package — Atari (`gymnasium[atari]`) or non-stationary NS-Gym (`ns_gym`) — have no special code here; you build them in an `env_fn` factory (see [Bring your own env](#bring-your-own-env-env_fn) and the [examples](examples/)).

### Procedural Frozen Lake

* **ID:** `Procedural-FrozenLake-v1`
* Random valid grid generation: size, holes, start/goal, and optional per-goal rewards.
* Random maps are generated lazily on the first reset, not during construction. By default, each env instance keeps one generated map across resets. Pass `episode_reset_options={"regenerate_map": True}` to generate a fresh map on every episode reset, or `task_reset_options={"regenerate_map": True}` to regenerate only when a new task starts.
* Variable-size random maps expose a stable observation space sized to the largest possible map (`max_width * max_height`), so output specs do not change after regeneration.
* Example: [examples/09_procedural_frozenlake.ipynb](examples/09_procedural_frozenlake.ipynb)

### Synthetic Environment

* **ID:** `SyntheticEnv-v1`
* Random finite discrete MDP for controlled tabular experiments.
* Random MDP maps are generated lazily on the first reset, not during construction. By default, each env instance keeps one generated MDP across resets. Pass `episode_reset_options={"regenerate_map": True}` to sample a fresh MDP on every episode reset, or `task_reset_options={"regenerate_map": True}` to regenerate only when a new task starts.
* Example: [examples/07_synthetic_env.ipynb](examples/07_synthetic_env.ipynb)


## Environment Tools 🛠️

mouse-env also includes a few knobs for augmenting and modifying environments.

### Expert Q-values (`q_star_source`)

Set `q_star_source` on `EnvConfig` to attach expert Q-values to every step output as `outputs[i]["info_q_star"]`. Useful for imitation learning, diagnostics, or guided exploration.

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

Instead of using `id` to build a Gymnasium env, pass `env_fn` — a zero-arg factory that returns a freshly built (and already-wrapped, if you like) Gymnasium env. `name` if set, otherwise `id`, is used as the env name. Time-limit truncation and any other wrappers are left entirely to your factory.

```python
def make_cartpole():
    env = gym.make("CartPole-v1", max_episode_steps=500)
    return MyWrapper(env)  # apply any Gymnasium wrappers here

cfg = EnvConfig(id="my-cartpole", reset_seed=0, episodes_per_task=5, env_fn=make_cartpole)
```

This is also how you apply custom Gymnasium wrappers (preprocessing, observation transforms, etc.): wrap inside your factory.

### Reset options and seeding

Use `episode_reset_options` to pass a dict to every internal `env.reset(options=...)`. Use `task_reset_options` for options that apply only when the reset starts a new task; these are overlaid on top of `episode_reset_options`.

`EnvConfig.reset_seed` controls mouse-env's internal `env.reset(seed=...)` stream. To build multiple env instances, pass a list to `make_group_env` and choose reset seeds explicitly for each config:

* `kwargs={"map_seed": ...}` controls first-party procedural map/MDP generation (`SyntheticEnv-v1` and `Procedural-FrozenLake-v1`). It is an env-specific constructor argument, not a base `EnvConfig` field.
* In Gymnasium, reset seeding normally controls the random number generator used for reset-time randomness: initial state sampling, randomized reset observations, and other randomness that belongs to starting a new episode.

Use these seeds when you want to hold one source of randomness fixed while varying another. For random action sampling, use the normal Gymnasium action-space API through `env.action_space.spaces[i]`. See [examples/10_rng_seeding_control.ipynb](examples/10_rng_seeding_control.ipynb) for a runnable walkthrough.

### Partial observability

Use `observation_indices` to mask dimensions on continuous-vector observation spaces.

Example: [examples/05_partial_observability.ipynb](examples/05_partial_observability.ipynb)

### Reward shaping

Use `reward_scale` and `reward_shift` to scale and shift the raw per-step reward before it appears in `outputs[i]["reward"]`. The formula is `reward = raw × scale + shift`.

Example: [examples/06_reward_shaping.ipynb](examples/06_reward_shaping.ipynb)


## Contributing 🔧

See [CONTRIBUTING.md](CONTRIBUTING.md).


## License 🔑

GNU General Public License v3.0 — see [LICENSE](LICENSE).
