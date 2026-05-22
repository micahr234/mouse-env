# MOUSE rollout contract (v1 — core)

This document defines the **minimum data** a vector environment built by **mouse-envs** should expose so **mouse-core** can collect rollouts without env-specific branches. It is the explicit version of what the wrapper stack implements today, refined toward a single ordered record per env per step.

---

## Design principles

1. **Ordered fields** — same logical sequence every time (identity → time → action → observation → termination → reward).
2. **Dictionaries for variable-shape payloads** — action and observation use keyed dicts so discrete, continuous, and image modalities do not require separate top-level columns.
3. **Required vs optional** — core fields are always present; extensions (expert Q*, episode-end stats, NS params) are optional and namespaced.
4. **Per parallel env** — in a `VectorEnv` with `num_envs = N`, each field is batched (length `N`) or a length-`N` list of dicts.

---

## Core record (required every step)

For each parallel env index `i` at each `step`, the rollout row MUST provide:

| Order | Field | Type (per env `i`) | Description |
|------:|-------|-------------------|-------------|
| 1 | `env_name` | `str` | Which environment this row belongs to (e.g. `"CartPole-v1"`). |
| 2 | `episode_index` | `int` | Episode counter for this env since the last collector `reset()` (0 on first episode, +1 each time `done != 0`). |
| 3 | `step_index` | `int` | Step within the current episode, **0-based** (0 = first step after `reset()`). |
| 4 | `action` | `dict[str, Any]` | Action taken **on this step**. See [Action dict](#action-dict). |
| 5 | `observation` | `dict[str, Any]` | Observation **after** this step (post-transition). See [Observation dict](#observation-dict). |
| 6 | `done` | `int` | `0` = still running, `1` = terminated (natural end), `2` = truncated (e.g. time limit). |
| 7 | `reward` | `dict[str, float]` | Reward signals for this step. See [Reward dict](#reward-dict). |

### Action dict

Exactly one primary key should be set for v1 discrete-only builds:

| Key | When used | Value |
|-----|-----------|--------|
| `discrete` | Discrete `Discrete` / `MultiDiscrete` action spaces | `int` action index |
| `continuous` | *(reserved)* | vector or scalar |
| `…` | Future modalities | TBD |

mouse-core v1 may assume `action["discrete"]` exists when using current envs.

### Observation dict

Use one or more keys depending on modality (multiple keys allowed if the env truly exposes both, but typical rows use one):

| Key | When used | Value |
|-----|-----------|--------|
| `discrete` | Integer / discrete obs | scalar or vector |
| `continuous` | Real-valued vector obs | `float` array |
| `image` | Pixel / frame obs | `H×W` or `H×W×C` array |

The env builder SHOULD document which keys appear for a given `env_name`. A convenience `obs_key` on the env object (e.g. `"observation_discrete"`) may remain for backward compatibility but the **canonical** payload is the dict.

### Reward dict

| Key | Required | Description |
|-----|----------|-------------|
| `step` | yes | Raw environment reward for this transition (before optional scale/shift used for training). |
| `episodic` | yes | Transformed episodic signal used for MOUSE training (today: normalised running average over the episode budget — `xformed_reward` in code). |

Optional future keys (e.g. `shaped`, `raw_unscaled`) belong here, not as duplicate top-level fields.

### `done` semantics

| Value | Meaning |
|------:|---------|
| `0` | Episode continues |
| `1` | Terminal transition (task ended) |
| `2` | Truncation (e.g. `TimeLimit`); treat bootstrap / value targets accordingly |

If Gymnasium reports both `terminated` and `truncated`, **`1` wins** (same rule as today).

---

## What is intentionally *not* in core

| Excluded | Reason |
|----------|--------|
| `global_step` | Redundant if `episode_index` + `step_index` + `env_name` identify the row; schedulers can use loop step from the runner. |
| Expert Q*, maps, NS params | Optional extension block (below). |
| `episode_length` / return at episode end only | Optional logging block; not needed every step for core training rows. |

---

## Optional extension block (`metadata` or `extensions`)

Optional keys SHOULD live under a single namespace to avoid polluting core:

```python
info["extensions"] = {
    "metadata_q_star": ...,      # float vector length action_dim
    "episode_return": ...,     # only meaningful when done != 0
    "episode_length": ...,
    "ns_params": ...,            # NS-Gym only
}
```

mouse-core SHOULD feature-detect: if `"extensions" not in info`, skip optional behaviour.

---

## Vector env delivery (today vs target)

| Core field | Target location | Current mouse-envs |
|------------|-----------------|----------------------|
| `env_name` | `info["env_name"]` | yes |
| `episode_index` | `info["episode_index"]` | **not yet** — use runner-side counter or derive from resets |
| `step_index` | `info["step_index"]` (0-based) | `info["episode_step"]` is **1-based** |
| `action` | `info["action"]` dict | **not in info** — runner stores action from `step(a)` call |
| `observation` | `info["observation"]` dict | obs is **return value** of `step`, not dict |
| `done` | `info["done"]` | yes |
| `reward.step` | Gymnasium `reward` return | yes (return value) |
| `reward.episodic` | `info["reward"]["episodic"]` | `info["xformed_reward"]` (top-level) |

Adopting this contract is a **small, staged migration**: add new keys alongside old ones, switch mouse-core, deprecate old keys.

---

## Minimal Python shape (reference)

```python
# One env index i — conceptual TypedDict
class RolloutStepCore(TypedDict):
    env_name: str
    episode_index: int
    step_index: int
    action: dict[str, Any]
    observation: dict[str, Any]
    done: int
    reward: dict[str, float]
```

For batched vector steps, each value is an array of length `num_envs` (or `list` of dicts per env for nested dict fields).

---

## Agreement summary

The core components you listed map cleanly to this contract:

1. **Environment name** — required identifier.  
2. **Episode number** — `episode_index`.  
3. **Step number** — `step_index`, 0-based within the episode.  
4. **Action** — dict (`discrete` today).  
5. **Observation** — dict (`discrete` / `continuous` / `image`).  
6. **Done** — integer 0/1/2.  
7. **Reward** — dict with at least `step` and `episodic`.

That is sufficient for mouse-core to function without knowing whether the backend is CartPole, Atari, or a custom tabular MDP.
