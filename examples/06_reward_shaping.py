"""MountainCar with reward scale/shift: xformed_reward differs from raw reward."""

from mouse.envs import EnvConfig, make_vector_env

if __name__ == "__main__":
    cfg = EnvConfig(
        env_id="MountainCar-v0",
        seed=0,
        num_envs=1,
        max_episode_steps=200,
        env_type="plain",
        reward_scale=0.1,   # raw reward × 0.1
        reward_shift=1.0,   # then + 1.0
        # Required fields with defaults
        kwargs=None,
        render=False,
        non_stationary_params=None,
        num_steps=None,
        action_source_loop_prob_schedule=None,
        action_source_episode_prob_schedule=None,
        q_star_source=None,
        action_source="random",
        action_source_temperature=1.0,
        split="train",
        atari_preprocessing=None,
        atari_preprocessing_kwargs=None,
        observation_indices=None,
    )

    env = make_vector_env(cfg)
    obs, info = env.reset()

    for step in range(20):
        obs, reward, terminated, truncated, info = env.step(env.sample_random_actions())

        raw_r     = reward[0]
        xformed_r = info["xformed_reward"][0]  # raw * scale + shift
        print(f"step={step:2d}  raw_reward={raw_r:.3f}  xformed_reward={xformed_r:.3f}")

    env.close()
