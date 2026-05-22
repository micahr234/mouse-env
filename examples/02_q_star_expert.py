"""FrozenLake rollout driven by the Q* (value-iteration) expert policy."""

import numpy as np
from mouse.envs import EnvConfig, make_vector_env

if __name__ == "__main__":
    cfg = EnvConfig.frozenlake(seed=7, num_envs=1)
    env = make_vector_env(cfg)
    obs, info = env.reset()

    episodes = 0
    for step in range(200):
        # metadata_q_star is injected by the wrapper stack from the value-iteration solution
        q_star = info["metadata_q_star"]  # float64[1, num_actions]
        action = np.argmax(q_star, axis=-1)

        obs, reward, terminated, truncated, info = env.step(action)

        done_code = info["done"]  # 1=terminated (goal/hole), 2=truncated (time limit)
        if done_code[0] != 0:
            episodes += 1
            outcome = "terminated" if terminated[0] else "truncated"
            print(f"step={step:3d}  episode={episodes}  outcome={outcome}  reward={reward[0]:.1f}")

    env.close()
