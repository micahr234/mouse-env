# Meta-Optimization Using Sequential Experiences — Environments

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

```python
from mouse.envs import EnvConfig, make_vector_env

env = make_vector_env(EnvConfig(
    group_id="CartPole-v1",
    seed=0,
    num_envs=4,
    max_episode_steps=500,
))

for _ in range(1000):
    data, metadata, metrics = env.step(env.sample_random_actions())
```

See **[docs/guide.md](docs/guide.md)** for the full step API — input actions, output records, reset behaviour, and field reference.

Interactive examples (Jupyter notebooks) live in [`examples/`](examples/) — see the [index](examples/README.md). After `source scripts/install.sh`, open them with Jupyter Lab or VS Code’s notebook UI:

```bash
.venv/bin/jupyter lab examples/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
