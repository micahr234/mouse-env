"""Basic CartPole vector rollout using random actions."""

from mouse.envs import EnvConfig, make_vector_env

if __name__ == "__main__":
    cfg = EnvConfig.cartpole(seed=0, num_envs=4, max_episode_steps=500)
    env = make_vector_env(cfg)
    obs, info = env.reset()

    for step in range(1000):
        actions = env.sample_random_actions()
        obs, reward, terminated, truncated, info = env.step(actions)

        episode_step = info["episode_step"]    # int64[4] — step within current episode
        done_code    = info["done"]            # 0=running, 1=terminated, 2=truncated
        xformed_r    = info["xformed_reward"]  # reward after scale/shift transforms

        if step % 100 == 0:
            print(
                f"step={step:4d}  "
                f"episode_step={episode_step}  "
                f"done_code={done_code}  "
                f"xformed_reward={xformed_r}"
            )

    env.close()
