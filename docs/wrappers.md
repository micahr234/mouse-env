# Wrapper stack

> **Note:** The target rollout shape for mouse-core is [rollout_contract.md](rollout_contract.md). This page describes **what the code emits today** (legacy `info` keys).

Every environment returned by `make_vector_env` passes through a standard vector wrapper stack built by `build_vector_env_stack`. The wrappers are composed from innermost to outermost as follows:

```
SyncVectorEnv
  → EpisodeStatisticsWrapper   raw-reward episode stats
  → _RewardTransformWrapper    reward * scale + shift
  → StepCounterWrapper         episode_step, global_step in info
  → XformedRewardWrapper       xformed_reward in info
  → DoneEncodingWrapper        done as {0, 1, 2} in info
  → EnvIdentityWrapper         env_name, env_idx, seed, obs_key, action_dim
  → QStarWrapper               metadata_q_star (only when q_star_source is set)
```

---

## EpisodeStatisticsWrapper

Tracks per-episode length and cumulative raw reward. Injects into `info` at episode boundaries (NaN on non-terminal steps).

| Info key | Type | Description |
|----------|------|-------------|
| `episode_length` | `float[N]` | Episode length; NaN unless done |
| `episode_cum_reward` | `float[N]` | Cumulative raw reward; NaN unless done |

Placed innermost so it always sees the raw (unscaled) rewards, even when `reward_scale` or `reward_shift` are non-trivial.

---

## _RewardTransformWrapper

Scales and shifts rewards: `r_out = r * scale + shift`. Controlled by `reward_scale` and `reward_shift` in `EnvConfig`.

---

## StepCounterWrapper

Injects per-env step counters into `info`.

| Info key | Type | Description |
|----------|------|-------------|
| `episode_step` | `int64[N]` | Step count within current episode (resets on new episode) |
| `global_step` | `int64[N]` | Monotonically increasing step count across all episodes |

---

## XformedRewardWrapper

Computes a normalised reward useful as a training signal:

```
xformed_reward = (episode_reward_sum + (episode_step - 1) * r) / max_steps
```

Reads `episode_step` from `StepCounterWrapper` (which must be inside this wrapper).

| Info key | Type | Description |
|----------|------|-------------|
| `xformed_reward` | `float64[N]` | Normalised cumulative reward |

---

## DoneEncodingWrapper

Encodes episode termination status as an integer.

| Value | Meaning |
|-------|---------|
| `0` | Running |
| `1` | Terminated (natural end) |
| `2` | Truncated (time limit) |

When both `terminated` and `truncated` fire simultaneously, `1` takes priority.

---

## EnvIdentityWrapper

Injects environment identity metadata and exposes convenience attributes on the wrapper object.

| Info key | Type | Description |
|----------|------|-------------|
| `env_name` | `str[N]` | Environment name string |
| `env_idx` | `int64[N]` | Per-env integer index |

**Wrapper attributes:**

| Attribute | Description |
|-----------|-------------|
| `env_seed` | Seed used for the initial reset |
| `obs_key` | Canonical observation key (`observation`, `observation_discrete`, or `observation_image`) |
| `action_dim` | Number of discrete actions |
| `sample_random_actions()` | Sample a random action batch as `int64` array |

`reset()` seeds automatically from `env_seed` when no explicit seed is passed.

---

## QStarWrapper

Injects expert Q-values into `info["metadata_q_star"]` when a `q_star_source` is configured in `EnvConfig`. Tries sources in order:

1. Env infos (custom envs that emit Q-values directly)
2. Observation-based Q prediction (e.g. SB3 policy)
3. Action-star infos (expert action → one-hot Q-values)
4. Observation-based action-star prediction (one-hot encoded)

| Info key | Type | Description |
|----------|------|-------------|
| `metadata_q_star` | `float64[N, A]` | Expert Q-values; shape `(num_envs, action_dim)` |
