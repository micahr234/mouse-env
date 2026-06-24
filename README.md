# MOUSE Environments 🐭

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-env/main/docs/mouse-env.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for production use. APIs may change without notice.

**mouse-env** turns episodic reinforcement learning environments into <u>continuing environments</u>. Instead of asking user code to alternate between `step()` and `reset()`, mouse-env handles resets internally so a rollout can continue through one uninterrupted `step()` loop.

Most RL benchmarks are episodic: an agent acts until termination or truncation, the caller calls `reset()`, and a new trial begins. That is a good interface when each episode is an independent sample. It is less natural when the experiment studies behavior **across multiple episodes**, where what the agent observes or discovers in one episode can affect what it does in a later one.

You can stitch episodes together on top of Gymnasium yourself, but the result is usually ad hoc. Important choices become arbitrary: whether reset observations are kept, how episode boundaries are marked, and how rewards behave at the boundary. **mouse-env** makes the episode-to-continuing conversion explicit and consistent in three ways:

* **Reset-free rollout.** Users keep calling `step(inputs)`. When an episode ends, mouse-env resets the underlying environment internally and returns the next observation without requiring a public `reset()` call.
* **Visible episode structure.** Terminations, truncations, and reset frames stay in the data returned by the environment, so agents and analysis code can see where one episode ended and the next began.
* **Cross-episode friendly rewards.** In episodic RL, credit is cut off at the reset boundary. A reward in the next episode does not encourage useful behavior in the previous one. mouse-env keeps raw environment rewards available, and also exposes a transformed reward signal that allows credit to pass across resets.

The result is a continuing interface for episodic RL: ordinary episodic Gymnasium environments can generate reset-free trajectories for multi-episode problems, with visible episode boundaries and rewards that allow value to propagate across trials.

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
    max_episode_steps=500,
)
env = make_env(cfg)

for _ in range(1000):
    inputs_per_env = env.sample_random_inputs()
    [(outputs, metrics)] = env.step(inputs_per_env)

env.close()
```

See **[docs/guide.md](docs/guide.md)** for full field-level documentation, plus runnable notebooks in [`examples/`](examples/).

---

## Core API ⚙️

There is no public rollout-time `reset()` call. The first `step()` quietly performs an internal reset and returns the initial observation using the same record shape as every other step. Inputs passed on that first call are ignored.

After an episode terminates or truncates, the next call to `step()` emits the reset observation for the next episode before normal stepping resumes.

Each call returns two objects:

* **`outputs`** — model-visible training data, including an `observation` tensor, rewards, done flags, time, episode metadata, optional `q_star` expert action-values, and environment-specific fields
* **`metrics`** — logging data, such as true episodic return and episode length, emitted when episodes end

`inputs` are plain dictionaries with a single `"action"` tensor key. Use `env.input_spec` to discover the expected dtype and shape for the current env. Use `env.output_spec` to discover the dtype and shape of every field in the output dict.

Episode boundaries are represented by integer-coded `done` values:

* `0` = running
* `1` = terminated
* `2` = truncated

Reset frames are ordinary `outputs` records with:

* the first observation of the new episode
* `time=0`
* the configured `reset_reward`, which is `0` by default
* `done=0`

This keeps the rollout stream uniform while still making episode structure explicit.

---

## Gymnasium environments 🌎

Pass any Gymnasium environment id as `id`. mouse-env builds the underlying Gymnasium env, steps it internally, and exposes the concatenated non-episodic stream through the same API.

Each constructed env exposes indexed names in `env.names`, formed from optional `EnvConfig.name` when provided, otherwise `EnvConfig.id`, plus `#0`, `#1`, and so on. `env.name` returns the first name for single-env use. Step outputs do not repeat this name on every record.

mouse-env also includes a couple of custom environments. Other envs that need their own package — Atari (`gymnasium[atari]`) or non-stationary NS-Gym (`ns_gym`) — have no special code here; you build them in an `env_fn` factory (see [Bring your own env](#bring-your-own-env-env_fn) and the [examples](examples/)).

### Procedural Frozen Lake

* **ID:** `Procedural-FrozenLake-v1`
* Random valid grid generation: size, holes, start/goal, and optional per-goal rewards.
* Example: [examples/02_q_star_expert.ipynb](examples/02_q_star_expert.ipynb)

### Synthetic Environment

* **ID:** `SyntheticEnv-v1`
* Random finite discrete MDP for controlled tabular experiments.
* Example: [examples/07_synthetic_env.ipynb](examples/07_synthetic_env.ipynb)

---

## Environment Tools 🛠️

mouse-env also includes a few knobs for augmenting and modifying environments.

### Expert Q-values (`q_star_source`)

Expert Q-values are exposed as `outputs[i]["q_star"]`. They are useful for supervision, diagnostics, or comparing learned behavior against an expert or exact tabular solution.

Example: [examples/02_q_star_expert.ipynb](examples/02_q_star_expert.ipynb)

### Bring your own env (`env_fn`)

Instead of using `id` to build a Gymnasium env, pass `env_fn` — a zero-arg factory that returns a freshly built (and already-wrapped, if you like) Gymnasium env. mouse-env calls it once per parallel env, so it must return a **new** env each time (not a shared instance). `name` if set, otherwise `id`, is used as the base for `env.names`, and `max_episode_steps` is still required (for reward normalisation); `kwargs`, `render`, and the internal `max_episode_steps` time limit are left to your factory.

```python
def make_cartpole():
    env = gym.make("CartPole-v1", max_episode_steps=500)
    return MyWrapper(env)  # apply any Gymnasium wrappers here

cfg = EnvConfig(id="my-cartpole", seed=0, num_envs=4, max_episode_steps=500, env_fn=make_cartpole)
```

This is also how you apply custom Gymnasium wrappers (preprocessing, observation transforms, etc.): wrap inside your factory.

### Observation routing (`observation_kind`)

Force the observation channel with `observation_kind` (`"continuous"`, `"discrete"`, or `"image"`). Defaults to auto-detection from the observation space; required (`"image"`) for image envs, which auto-detection cannot recognise.

### Partial observability

Use `observation_indices` to mask dimensions on continuous-vector observation spaces.

Example: [examples/05_partial_observability.ipynb](examples/05_partial_observability.ipynb)

### Reward shaping

Use `reward_scale` and `reward_shift`; the normalized training signal appears in `outputs[i]["reward_episodic"]`.

Example: [examples/06_reward_shaping.ipynb](examples/06_reward_shaping.ipynb)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
