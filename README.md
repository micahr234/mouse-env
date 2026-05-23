# MOUSE Environments

<p align="center"><img src="docs/mouse-env.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for use. APIs will change without notice.

**mouse-env** is the environment layer for [MOUSE](https://github.com/micahr234/mouse-core), a modular PyTorch library for in-context reinforcement learning.

In standard RL, episodes are independent. The algorithm resets the environment, collects one episode, updates its weights, and repeats — what happened in episode 1 has no bearing on how the agent behaves in episode 2. In-context RL works differently: the agent is a sequence model whose context window spans *multiple episodes*. It reads its own history and adapts its strategy within a single forward pass, without a gradient update. Episode N is not a clean slate — the model uses what happened in episode N-1 (and earlier) to play better in episode N. Learning happens across episodes, not just within them.

mouse-env is built around this model. Rather than treating each episode as an isolated run, it presents experience as a continuous stream where episode boundaries are just marked transitions — the environment resets automatically and the agent keeps running with its accumulated context. Every `step()` call returns the same three aligned lists regardless of env type:

- **`data[i]`** — what the sequence model reads: `observation`, `reward`, `done`, and `time`. One record per env per step, ready to concatenate into a context window.
- **`metadata[i]`** — what the training loop needs but the model doesn't see: expert Q-values (`q_star`) for imitation targets, normalised episodic return (`reward_episodic`) for the training signal, `episode_index` to track how many episodes the agent has seen, and `group_id` to identify which env variant this stream came from.
- **`metrics[i]`** — what you log: true cumulative reward and episode length, emitted once at episode end (empty lists on every other step).

Episode tracking, autoreset, expert Q-value annotation, and reward normalisation are all built in. The same API covers tabular MDPs, Atari, non-stationary envs, and any Gymnasium env — no boilerplate required.

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

One config, one call to `make_vector_env`, then `step()` in a loop:

```python
from mouse.envs import EnvConfig, make_vector_env

cfg = EnvConfig.cartpole(seed=0, num_envs=4, max_episode_steps=500)
env = make_vector_env(cfg)

for _ in range(1000):
    actions = env.sample_random_actions()
    data, metadata, metrics = env.step(actions)

env.close()
```

**A few things to know before writing code:**

- **Call `step()` only** — the first call resets internally and returns the initial observation with `reward=0`, `done=0`, and `time=0`; the actions on that call are ignored. Every call after applies actions normally. The same zeroed record appears whenever a sub-env finishes and resets the starting observation with `reward=0`, `done=0`, and `time=0` are given before normal stepping resumes.
- **Observations and actions are dicts of tensors** — `data[i]["observation"]` uses keys `discrete`, `continuous`, and/or `image`. Actions follow the same shape: `actions[i]["action"]["discrete"]` or `["continuous"]`.
- **`done` is an int** — `0` = running, `1` = terminated, `2` = truncated.

See **[docs/guide.md](docs/guide.md)** for the full field reference. Example Jupyter notebooks are in [`examples/`](examples/).

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
