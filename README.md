# Meta-Optimization Using Sequential Experiences — Environments

<p align="center"><img src="docs/mouse.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for use. APIs will change without notice.

**mouse-envs** is the environment package for [MOUSE](https://github.com/micahr234/mouse-core), a modular PyTorch library for in-context reinforcement learning. It provides Gymnasium vector environments, NS-Gym non-stationary wrappers, custom tabular MDPs, and rich rollout metadata (episode stats, expert Q-values) used to collect training data for MOUSE agents.

## Install

```bash
pip install mouse-envs
```

For the latest development version:

```bash
pip install "git+https://github.com/micahr234/mouse-env.git"
```

## Documentation

📖 **[micahr234.github.io/mouse-env](https://micahr234.github.io/mouse-env/)**

The **[rollout contract](docs/rollout_contract.md)** defines the data mouse-envs exposes to [mouse-core](https://github.com/micahr234/mouse-core) (`env_id`, episode/step indices, action/observation/reward dicts).

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
