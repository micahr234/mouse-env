"""Non-stationary CartPole with an oscillating pole length."""

from mouse.envs import EnvConfig, make_vector_env

NS_PARAMS = {
    "length": {
        "scheduler": "ContinuousScheduler",
        "update_fn": "OscillatingUpdate",
        "init_value": 0.5,
    }
}

if __name__ == "__main__":
    cfg = EnvConfig.ns_cartpole(seed=42, num_envs=2, non_stationary_params=NS_PARAMS)
    env = make_vector_env(cfg)
    obs, info = env.reset()

    for step in range(500):
        obs, reward, terminated, truncated, info = env.step(env.sample_random_actions())

        # Log current non-stationary parameter values every 50 steps
        if step % 50 == 0:
            ns_params = info.get("ns_params")
            print(f"step={step:3d}  ns_params={ns_params}")

    env.close()
