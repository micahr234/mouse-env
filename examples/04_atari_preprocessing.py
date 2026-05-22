"""Atari Pong with grayscale frame preprocessing (84×84, frame-skip 4)."""

from mouse.envs import EnvConfig, make_vector_env

if __name__ == "__main__":
    cfg = EnvConfig(
        env_id="ALE/Pong-v5",
        seed=0,
        num_envs=4,
        max_episode_steps=10000,
        env_type="plain",
        atari_preprocessing=True,
        atari_preprocessing_kwargs={"frame_skip": 4, "screen_size": 84, "noop_max": 30},
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
        observation_indices=None,
    )

    env = make_vector_env(cfg)
    obs, info = env.reset()

    # After preprocessing: obs shape is (num_envs, 84, 84) — grayscale frames
    print(f"obs shape: {obs.shape}")   # expected: (4, 84, 84)
    print(f"obs key:   {env.obs_key}") # "observation_image"

    for step in range(50):
        obs, reward, terminated, truncated, info = env.step(env.sample_random_actions())

    env.close()
