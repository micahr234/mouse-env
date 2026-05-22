"""CartPole with partial observations: only cart position and pole angle."""

from mouse.envs import EnvConfig, make_vector_env

# CartPole obs indices: 0=cart_pos, 1=cart_vel, 2=pole_angle, 3=pole_vel
# We expose only cart position and pole angle — hiding velocity information.
VISIBLE_INDICES = [0, 2]

if __name__ == "__main__":
    cfg = EnvConfig.cartpole(seed=0, num_envs=2, observation_indices=VISIBLE_INDICES)
    env = make_vector_env(cfg)
    obs, info = env.reset()

    # Full CartPole obs has 4 features; partial obs has only 2.
    print(f"obs shape: {obs.shape}")  # expected: (2, 2)

    for step in range(200):
        obs, reward, terminated, truncated, info = env.step(env.sample_random_actions())

    print(f"final obs shape: {obs.shape}")  # still (2, 2) throughout
    env.close()
