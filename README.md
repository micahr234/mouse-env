# MOUSE Environments 🐭

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-env/main/docs/mouse-env.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for production use. APIs may change without notice.

**mouse-env** is the environment layer for [MOUSE](https://github.com/micahr234/mouse-core), a modular PyTorch library for in-context reinforcement learning.

It gives tabular environments, Gymnasium control tasks, Atari, and non-stationary setups one compact API built for long streams of experience.

---

## Why mouse-env exists 🧠

Most reinforcement learning environment APIs are built around explicit episode resets:

1. call `reset()` to start an episode
2. call `step(action)` until the episode ends
3. call `reset()` again
4. repeat steps 2 and 3

That pattern works well for traditional RL, where each episode is usually treated as its own little run.

**In-context reinforcement learning (ICRL) is different.** The policy is usually a sequence model that acts from recent history, not just the current observation. It may fail on one episode, observe the rewards, and use that experience to do better on the next episode.

For that setting, episode boundaries should stay visible in the data, but they should not force the caller into a reset-driven control flow.

mouse-env therefore uses a **step-only API** that naturally fits ICRL: callers keep calling `step(actions)`, while episode boundaries travel along in the returned data instead of forcing explicit reset calls.

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

cfg = EnvConfig.cartpole(seed=0, num_envs=4, max_episode_steps=500)
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

## Included environments and integrations 🌎

mouse-env includes ready-to-use environments and integrations so you can start experimenting without wiring every backend by hand.

### Procedural Frozen Lake

* **ID:** `Procedural-FrozenLake-v1`
* **Config helper:** `EnvConfig.procedural_frozenlake()`
* Random valid grid generation: size, holes, start/goal, and optional per-goal rewards.
* Distinct from Gymnasium's fixed `FrozenLake-v1` benchmark.

### Synthetic Environment

* **ID:** `SyntheticEnv-v1`
* **Config helper:** `EnvConfig.synthetic()`
* Random finite discrete MDP for controlled tabular experiments.

### NS-Gym integration

mouse-env integrates [NS-Gym](https://github.com/scope-lab-vu/ns_gym) to support non-stationary dynamics through the same API.

* `EnvConfig.ns_cartpole(non_stationary_params={...})` for scheduler and update configuration
* `NSGymInterfaceWrapper` to normalize observations and expose `results[i]["ns_params"]`
* Vectorized non-stationary streams with standard `(results, metrics)` outputs

Example: [examples/03_ns_gym_oscillating.ipynb](examples/03_ns_gym_oscillating.ipynb)
NS-Gym docs: [nsgym.io](https://nsgym.io/)

### Atari integration

mouse-env keeps ALE/Gymnasium Atari semantics intact and exposes them through the same API.

* `EnvConfig.atari()` preset:

  * `AtariPreprocessing`
  * frame skip 4
  * grayscale 84×84
  * no-op warm-up
* Frames are surfaced in `data[i]["observation"]["image"]` as flattened tensors

Requirement: `gymnasium[atari]` (`ale_py`).
Example: [examples/04_atari_preprocessing.ipynb](examples/04_atari_preprocessing.ipynb)

---

## Environment Tools 🛠️

mouse-env also includes a few knobs for augmenting and modifying environments.

### Expert Q-values (`q_star_source`)

Expert Q-values are exposed as `results[i]["q_star"]`. They are useful for supervision, diagnostics, or comparing learned behavior against an expert or exact tabular solution.

Available backends:

* **`sb3_rl_zoo`** — pretrained Stable-Baselines3 policies from Hugging Face Hub. The CartPole preset uses this by default.
* **`metadata_q_star`** — exact Q* solved from tabular MDP dynamics. Used by Procedural Frozen Lake and Synthetic Environment.

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
