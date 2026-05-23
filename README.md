# MOUSE Environments

<p align="center"><img src="docs/mouse-env.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for use. APIs will change without notice.

**mouse-env** is the environment package for [MOUSE](https://github.com/micahr234/mouse-core), a modular PyTorch library for in-context reinforcement learning. It builds Gymnasium vector environments and reformats step output into TensorDict records for mouse-core training.

## Install

```bash
pip install mouse-env
```

Development:

```bash
git clone https://github.com/micahr234/mouse-env.git
cd mouse-env
source scripts/install.sh
```

## Quick start

Configure a vector env, call `step()` in a loop, and read TensorDict records — no Gymnasium `info` dicts to parse.

```python
from mouse.envs import EnvConfig, make_vector_env

cfg = EnvConfig.cartpole(seed=0, num_envs=4, max_episode_steps=500)
env = make_vector_env(cfg)

for _ in range(1000):
    actions = env.sample_random_actions()
    data, metadata, metrics = env.step(actions)

env.close()
```

Compared to Gymnasium:

- **No `reset()`** — call `step()` only. The first call resets internally; actions on that call are ignored.
- **Dict observations and actions** — `data[i]["observation"]` and `actions[i]["action"]` are both dicts of tensors. Observations use `discrete`, `continuous`, and/or `image`; actions use `discrete` or `continuous`, matching the env's spaces.
- **Different return value** — `(data, metadata, metrics)` instead of `(obs, reward, terminated, truncated, info)`.
  - **`data[i]`** — the sequence-model payload: `time`, `observation`, `reward`, and `done`.
  - **`metadata[i]`** — per-env training and analysis context: `group_id`, `episode_index`, `reward_episodic`, optional `q_star`. Not fed directly to the sequence model, but commonly used to support training (shaped returns, expert targets, auxiliary losses), analyze performance, and debug rollouts.
  - **`metrics[i]`** — episode finish stats for this step (empty lists if none). Used for evaluation and logging, not model input.

See **[docs/guide.md](docs/guide.md)** for the full step API. Example Jupyter notebooks are in [`examples/`](examples/).

## Environments & features

Beyond plain Gymnasium envs, mouse-env ships custom worlds, non-stationary dynamics, and expert-policy metadata for in-context RL experiments.

**Procedural Frozen Lake** (`Procedural-FrozenLake-v1`, `EnvConfig.procedural_frozenlake()`) — not Gymnasium's fixed 4×4 `FrozenLake-v1`. Each env instance draws a new valid grid (random size, holes, start/goal placement) with optional per-goal rewards.

**Synthetic Environment** (`SyntheticEnv-v1`, `EnvConfig.synthetic()`) — a finite discrete MDP with randomly sampled transitions and rewards, for controlled tabular experiments without hand-designing a grid.

**NS-Gym integration** — [NS-Gym](https://github.com/scope-lab-vu/ns_gym) is an external framework for non-stationary MDPs; we did not build it, but mouse-env integrates it so you can use time-varying physics (e.g. oscillating CartPole pole length) through the same `step()` API as everything else. Our layer adds:

- **`EnvConfig.ns_cartpole(non_stationary_params={...})`** — plain dict configs for NS-Gym schedulers and update functions (no manual wrapper wiring)
- **`NSGymInterfaceWrapper`** — adapts NS-Gym’s dict observations and ground-truth info into flat observations and per-env `metadata[i]["ns_params"]`
- **Vector env support** — parallel non-stationary streams with the usual `(data, metadata, metrics)` return shape

See [examples/03_ns_gym_oscillating.ipynb](examples/03_ns_gym_oscillating.ipynb). NS-Gym docs: [nsgym.io](https://nsgym.io/).

**Atari integration** — [Gymnasium Atari (ALE)](https://gymnasium.farama.org/environments/atari/) envs are unchanged under the hood; mouse-env does not modify the games themselves. We bundle the usual training presets so you can run them through the same `step()` API:

- **`EnvConfig.atari()`** — common defaults: Gymnasium's `AtariPreprocessing` with frame skip 4, grayscale 84×84 resize, and noop warm-up (all overridable via `atari_preprocessing_kwargs`)
- **Observation layout** — preprocessed frames surface as flattened `observation.image` in `data`

Requires the `gymnasium[atari]` extra (`ale_py`). See [examples/04_atari_preprocessing.ipynb](examples/04_atari_preprocessing.ipynb).

**Expert Q-values (Q\*)** — attach optimal or near-optimal action values to supported envs via `q_star_source`; values appear in `metadata[i]["q_star"]` each step:

- **Standard Gymnasium envs** — load a pretrained Stable-Baselines3 policy from the Hugging Face Hub (`provider: sb3_rl_zoo`). The CartPole preset includes this by default.
- **Tabular envs** (Procedural Frozen Lake, Synthetic Environment) — exact Q* is computed by solving the MDP (`provider: metadata_q_star`); no external Q-table download required.

**Partial observability** — mask observation dimensions with `observation_indices` on any env that exposes a continuous observation vector. See [examples/05_partial_observability.ipynb](examples/05_partial_observability.ipynb).

**Reward shaping** — scale and shift per-step rewards with `reward_scale` / `reward_shift`; the normalised training signal appears in `metadata[i]["reward_episodic"]`. See [examples/06_reward_shaping.ipynb](examples/06_reward_shaping.ipynb).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
