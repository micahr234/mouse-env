# Examples

Jupyter notebooks that demonstrate `mouse-env` setups. Open in [JupyterLab](https://jupyter.org/) or VS Code after `source scripts/install.sh`.

| Notebook | Topic |
|----------|--------|
| [01_random_rollout.ipynb](01_random_rollout.ipynb) | CartPole vector env, random actions, step output |
| [02_q_star_expert.ipynb](02_q_star_expert.ipynb) | FrozenLake with greedy Q* from metadata |
| [03_ns_gym_oscillating.ipynb](03_ns_gym_oscillating.ipynb) | Non-stationary CartPole (oscillating pole length) |
| [04_atari_preprocessing.ipynb](04_atari_preprocessing.ipynb) | Atari Pong with grayscale 84×84 frames |
| [05_partial_observability.ipynb](05_partial_observability.ipynb) | CartPole with masked observation indices |
| [06_reward_shaping.ipynb](06_reward_shaping.ipynb) | MountainCar with reward scale/shift |

```bash
.venv/bin/jupyter lab examples/
```
