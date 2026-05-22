# Meta-Optimization Using Sequential Experiences — Environments

<p align="center"><img src="docs/mouse-env.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for use. APIs will change without notice.

**mouse-env** is the environment package for [MOUSE](https://github.com/micahr234/mouse-core), a modular PyTorch library for in-context reinforcement learning. It provides Gymnasium vector environments, NS-Gym non-stationary wrappers, custom tabular MDPs, and rollout metadata for training data collection.

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

```python
from mouse.envs import EnvConfig, make_vector_env

env = make_vector_env(EnvConfig.cartpole(num_envs=4, seed=0))
obs, info = env.reset()

for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.sample_random_actions())
```

## Examples

Runnable scripts live in [`examples/`](examples/). See [docs/examples.md](docs/examples.md) for descriptions and inline code.

## Documentation

All docs are Markdown in [`docs/`](docs/) (read on GitHub or in the repo):

| Doc | Description |
|-----|-------------|
| [guide.md](docs/guide.md) | Overview, layout, quick start |
| [rollout_contract.md](docs/rollout_contract.md) | **mouse-env ↔ mouse-core** step schema (`env_id`, dicts, rewards) |
| [environments.md](docs/environments.md) | Env types and `EnvConfig` options |
| [examples.md](docs/examples.md) | Runnable scripts in [`examples/`](examples/) — NS-Gym, Atari, Q*, partial observability, reward shaping |
| [wrappers.md](docs/wrappers.md) | Current wrapper stack and `info` keys |
| [mouse_core_alignment.md](docs/mouse_core_alignment.md) | Updating mouse-core for the contract |

API reference: Python docstrings in `src/` (e.g. `EnvConfig`, `make_vector_env`).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
