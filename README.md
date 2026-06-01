# MOUSE Environments 🐭

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-env/main/docs/mouse-env.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for production use. APIs may change without notice.

**mouse-env** converts episodic reinforcement learning environments into continuing, non-episodic streams by concatenating episodes together.

Callers keep stepping through one long stream of experience. Episode boundaries remain visible in the returned data, but they do not require public `reset()` calls or interrupt the control flow.

---

## Why mouse-env exists 🧠

Most reinforcement learning environments are episodic:

1. call `reset()` to start an episode
2. call `step(action)` until the episode ends
3. call `reset()` again
4. repeat steps 2 and 3

mouse-env wraps that reset-driven pattern and presents it as a **step-only API**. The underlying environment still terminates, truncates, and resets episodes; mouse-env emits those transitions as records in a single continuous stream.

This is useful when training sequence models or other agents that learn from recent history across episode boundaries. The model can see where episodes end, while the caller keeps using the same `step(actions)` loop.

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

Build an env, sample actions, and keep stepping:

```python
from mouse_envs import EnvConfig, make_vector_env

cfg = EnvConfig(
    group_id="CartPole-v1",
    seed=0,
    num_envs=4,
    max_episode_steps=500,
)
env = make_vector_env(cfg)

for _ in range(1000):
    actions = env.sample_random_actions()
    results, metrics = env.step(actions)

env.close()
```

See **[docs/guide.md](docs/guide.md)** for full field-level documentation, plus runnable notebooks in [`examples/`](examples/).

---

## Core API ⚙️

There is no public rollout-time `reset()` call. The first `step()` quietly performs an internal reset and returns the initial observation using the same record shape as every other step. Actions passed on that first call are ignored.

After an episode terminates or truncates, the next call to `step()` emits the reset observation for the next episode before normal stepping resumes.

Each call returns two objects:

* **`results`** — model-visible training data, including observations (`discrete`, `continuous`, and/or `image` tensor channels), rewards, done flags, time, episode metadata, optional `q_star` target expert action-values, and environment-specific fields
* **`metrics`** — logging data, such as true episodic return and episode length, emitted when episodes end

`actions` follow the same typed-dictionary structure as observations.

Episode boundaries are represented by integer-coded `done` values:

* `0` = running
* `1` = terminated
* `2` = truncated

Reset frames are ordinary `results` records with:

* the first observation of the new episode
* `time=0`
* the configured `reset_reward`, which is `0` by default
* `done=0`

This keeps the rollout stream uniform while still making episode structure explicit.

---

## Gymnasium environments and integrations 🌎

Pass any Gymnasium environment id as `group_id`. mouse-env builds the underlying Gymnasium env, steps it internally, and exposes the concatenated non-episodic stream through the same API.

mouse-env also includes a few custom environments and optional integrations.

### Procedural Frozen Lake

* **ID:** `Procedural-FrozenLake-v1`
* Random valid grid generation: size, holes, start/goal, and optional per-goal rewards.
* Example: [examples/02_q_star_expert.ipynb](examples/02_q_star_expert.ipynb)

### Synthetic Environment

* **ID:** `SyntheticEnv-v1`
* Random finite discrete MDP for controlled tabular experiments.
* Example: [examples/07_synthetic_env.ipynb](examples/07_synthetic_env.ipynb)

### NS-Gym integration

mouse-env integrates [NS-Gym](https://github.com/scope-lab-vu/ns_gym) to support non-stationary dynamics through the same API.

Example: [examples/03_ns_gym_oscillating.ipynb](examples/03_ns_gym_oscillating.ipynb)
NS-Gym docs: [nsgym.io](https://nsgym.io/)

### Atari integration

mouse-env keeps ALE/Gymnasium Atari semantics intact and exposes them through the same API.

Requirement: `gymnasium[atari]` (`ale_py`).
Example: [examples/04_atari_preprocessing.ipynb](examples/04_atari_preprocessing.ipynb)

---

## Environment Tools 🛠️

mouse-env also includes a few knobs for augmenting and modifying environments.

### Expert Q-values (`q_star_source`)

Expert Q-values are exposed as `results[i]["q_star"]`. They are useful for supervision, diagnostics, or comparing learned behavior against an expert or exact tabular solution.

Example: [examples/02_q_star_expert.ipynb](examples/02_q_star_expert.ipynb)

### Partial observability

Use `observation_indices` to mask dimensions on continuous-vector observation spaces.

Example: [examples/05_partial_observability.ipynb](examples/05_partial_observability.ipynb)

### Reward shaping

Use `reward_scale` and `reward_shift`; the normalized training signal appears in `results[i]["reward_episodic"]`.

Example: [examples/06_reward_shaping.ipynb](examples/06_reward_shaping.ipynb)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
