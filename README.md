# MOUSE Environments

<p align="center"><img src="docs/mouse-env.png" width="400"/></p>

<p><strong><span style="color:#d00000;">Warning:</span></strong> MOUSE is in early development and is not yet ready for production use. APIs may change without notice.</p>

**mouse-env** is the environment layer for [MOUSE](https://github.com/micahr234/mouse-core), a modular PyTorch library for in-context reinforcement learning (ICRL).

## Why mouse-env exists

In standard RL, episodes are usually treated as independent rollouts:

1. Reset environment.
2. Collect one episode.
3. Update model weights.
4. Repeat.

In **in-context RL**, the policy is a sequence model whose context window spans **multiple episodes**. The model can adapt behavior from recent history in a single forward pass, without gradient updates between episodes.

mouse-env is built for this regime: experience is emitted as a continuous stream where episode boundaries are events in the stream, not control-flow branches in user code.

## Core API

The main contract is simple:

- Build one vectorized environment with `make_vector_env(...)`
- Call `step(actions)` in a loop
- Receive a fixed output shape every step:
  - `data`
  - `metadata`
  - `metrics`

For each sub-environment index `i`:

- **`data[i]`** (model-visible)
  - `observation`
  - `reward`
  - `done`
  - `time`
- **`metadata[i]`** (training-only)
  - `q_star`
  - `reward_episodic`
  - `episode_index`
  - `group_id`
  - plus optional env-specific metadata (for example `ns_params`)
- **`metrics[i]`** (logging)
  - true episodic return and episode length, emitted only on episode end

This API is shared across tabular envs, Gymnasium control tasks, Atari, and non-stationary setups.

## Install

```bash
pip install mouse-env
```

Development setup:

```bash
git clone https://github.com/micahr234/mouse-env.git
cd mouse-env
source scripts/install.sh
```

## Quick start

```python
from mouse.envs import EnvConfig, make_vector_env

cfg = EnvConfig.cartpole(seed=0, num_envs=4, max_episode_steps=500)
env = make_vector_env(cfg)

for _ in range(1000):
    actions = env.sample_random_actions()
    data, metadata, metrics = env.step(actions)

env.close()
```

### Important stepping semantics

- **Use `step()` only.**
  - The first `step()` performs an internal reset and returns initial observations with `reward=0`, `done=0`, `time=0`.
  - Actions passed on that first call are ignored.
- **Autoreset is built in.**
  - After termination/truncation, the next emitted transition includes the reset observation (`reward=0`, `done=0`, `time=0`) before normal stepping resumes.
- **Observations/actions are typed dicts of tensors.**
  - Observation channels can include `discrete`, `continuous`, and/or `image`.
  - Actions follow the same keyed structure.
- **`done` is integer-coded.**
  - `0` = running, `1` = terminated, `2` = truncated.

See **[docs/guide.md](docs/guide.md)** for full field-level documentation, plus runnable notebooks in [`examples/`](examples/).

## Included environments and integrations

### 1) Procedural Frozen Lake

- **ID:** `Procedural-FrozenLake-v1`
- **Config helper:** `EnvConfig.procedural_frozenlake()`
- Random valid grid generation (size, holes, start/goal), optional per-goal rewards.
- Distinct from Gymnasium's fixed `FrozenLake-v1` benchmark.

### 2) Synthetic Environment

- **ID:** `SyntheticEnv-v1`
- **Config helper:** `EnvConfig.synthetic()`
- Random finite discrete MDP for controlled tabular experiments.

### 3) NS-Gym integration (external framework)

mouse-env integrates [NS-Gym](https://github.com/scope-lab-vu/ns_gym) to support non-stationary dynamics through the same API.

- `EnvConfig.ns_cartpole(non_stationary_params={...})` for scheduler/update config
- `NSGymInterfaceWrapper` to normalize observations + expose `metadata[i]["ns_params"]`
- Vectorized non-stationary streams with standard `(data, metadata, metrics)` outputs

Example: [examples/03_ns_gym_oscillating.ipynb](examples/03_ns_gym_oscillating.ipynb)  
NS-Gym docs: [nsgym.io](https://nsgym.io/)

### 4) Atari integration

mouse-env keeps ALE/Gymnasium Atari semantics intact and exposes them through the same API.

- `EnvConfig.atari()` preset:
  - `AtariPreprocessing`
  - frame skip 4
  - grayscale 84×84
  - noop warm-up
- Frames are surfaced in `data[i]["observation"]["image"]` (flattened)

Requirement: `gymnasium[atari]` (`ale_py`).  
Example: [examples/04_atari_preprocessing.ipynb](examples/04_atari_preprocessing.ipynb)

## Training-oriented features

### Expert Q-values (`q_star_source`)

- Exposed as `metadata[i]["q_star"]`.
- Backends:
  - **`sb3_rl_zoo`**: pretrained Stable-Baselines3 policies from Hugging Face Hub (CartPole preset uses this by default).
  - **`metadata_q_star`**: exact Q* solved from tabular MDP dynamics (Procedural Frozen Lake + Synthetic Environment).

### Partial observability

Use `observation_indices` to mask dimensions on continuous-vector observation spaces.  
Example: [examples/05_partial_observability.ipynb](examples/05_partial_observability.ipynb)

### Reward shaping

Use `reward_scale` and `reward_shift`; normalized training signal appears in `metadata[i]["reward_episodic"]`.  
Example: [examples/06_reward_shaping.ipynb](examples/06_reward_shaping.ipynb)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
