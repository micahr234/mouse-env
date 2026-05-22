# MOUSE rollout contract

This document defines what a **MouseEnvironment** must produce so **mouse-core** can train and evaluate without knowing whether the underlying world is CartPole, Atari, or a custom tabular MDP.

The contract is intentionally small: a fixed **core** record every step, plus an optional **extensions** block. Everything else (wrapper mechanics, Gymnasium ids, NS-Gym physics) stays inside mouse-env.

---

## Why this exists

Gymnasium gives you `reset` / `step`, observations, rewards, and termination flags. MOUSE needs more than that:

- The same **column layout** in every rollout dataset.
- **Episode-aware** indexing and training rewards, not only raw step rewards.
- **Modalities** (discrete, continuous, image) without separate pipelines per env type.
- A stable **environment identifier** on every row so multi-env and multi-dataset runs do not rely on implicit order or display names.

mouse-env builds envs that satisfy this contract; mouse-core reads datasets (or live buffers) that follow it. If both sides agree on the spec below, either repository can evolve independently behind a versioned interface.

---

## Terminology: `env_id` not “name”

| Term | Meaning |
|------|---------|
| **`env_id`** | Stable, unique string for the environment **instance type** in a rollout (e.g. `"CartPole-v1"`, `"NS-CartPole-v1"`, `"Custom-FrozenLake-v1"`). This is the canonical identifier on each step. |
| **Display / logging name** | Optional human-readable label; **not** part of the core contract. If you need one, put it in `extensions` — do not duplicate `env_id`. |

Use **`env_id`** everywhere in the contract, dataset schemas, and mouse-core configs. “Name” is ambiguous (pretty title vs Gym id vs Hub dataset name); **id** means “this is the key you join and filter on.”

---

## Core record (required every step)

Think of one **row** per environment index per `step`. Fields are listed in **reading order** — the order collectors and readers should expect.

### 1. `env_id` (string)

Which environment produced this row.

- Example: `"CartPole-v1"`.
- In a vector env with `num_envs > 1`, each index has its own `env_id` (often the same string repeated, or distinct ids if you multiplex different env types).

### 2. `episode_index` (integer)

Which episode this is for **this env stream** since the collector last called `reset()`.

- First episode after `reset()` → `0`.
- When `done != 0`, the **next** step starts a new episode with `episode_index` incremented by 1.
- Resets on `env.reset()` zero the episode counter for that stream.

This is the “episode number” you care about for in-context windows — not a global training step.

### 3. `step_index` (integer)

Position inside the current episode, **0-based**.

- First transition after `reset()` → `step_index == 0`.
- Increments by 1 each `step` until `done != 0`.

We intentionally **do not** require `global_step` in the core contract: `env_id` + `episode_index` + `step_index` uniquely locates a row. Loop-level schedules in mouse-core can use the runner’s own iteration counter.

### 4. `action` (dictionary)

What was executed on this step. Keys denote modality; values hold the payload.

| Key | v1 usage | Value |
|-----|----------|--------|
| `discrete` | Required for current discrete-only envs | `int` action index |
| `continuous` | Reserved | vector or scalar |
| *(future)* | e.g. `multi_discrete` | TBD |

Example: `{"discrete": 2}`.

The action belongs in the **record** (typically `info["action"]` after `step`) so stored trajectories are self-contained, not only in the caller that invoked `step(action)`.

### 5. `observation` (dictionary)

Observation **after** the transition (post-`step` state). Same key scheme as actions:

| Key | When | Value |
|-----|------|--------|
| `discrete` | Discrete / integer state | scalar or vector |
| `continuous` | Vector state | float array |
| `image` | Pixels / frames | array, `H×W` or `H×W×C` |

Example: `{"continuous": [x, x_dot, theta, theta_dot]}`.

Usually **one** key is set per row. Multiple keys are allowed only when the env truly exposes multiple modalities at once.

### 6. `done` (integer)

| Value | Meaning |
|------:|---------|
| `0` | Episode still running |
| `1` | Natural termination (terminal state) |
| `2` | Truncation (e.g. time limit) |

If Gymnasium sets both `terminated` and `truncated`, treat as **`1`** (terminated wins).

### 7. `reward` (dictionary)

At least two floats per step:

| Key | Required | Meaning |
|-----|----------|---------|
| `step` | yes | Raw environment reward for this transition (unaltered env signal; scale/shift for logging can be documented separately). |
| `episodic` | yes | MOUSE training signal derived from the episode so far (running episodic / normalised quantity — today implemented as `xformed_reward` in mouse-env). |

Example: `{"step": 1.0, "episodic": 0.04}`.

Additional shaping keys (if ever needed) stay **inside** `reward`, not as extra top-level columns.

---

## Optional: `extensions`

Non-core data MUST live under one namespace so mouse-core can ignore what it does not need:

```python
info["extensions"] = {
    "metadata_q_star": np.ndarray,   # shape (action_dim,) per env
    "episode_return": float,         # often only when done != 0
    "episode_length": int,
    "ns_params": {...},              # NS-Gym only
}
```

Rules:

- **Feature-detect** — if `extensions` or a key is missing, skip that behaviour.
- **Never required** for basic RL / ICRL training on `reward["episodic"]` and `observation`.

---

## Vector environments

For `num_envs = N`, each core field is batched:

- Scalars: arrays of shape `(N,)` (`env_id` may be `N` strings or one repeated id).
- Dicts: per-env dicts, e.g. `N` observation dicts, or a documented batched layout.

The **logical** row is still the seven fields above; batching is transport.

---

## Relationship to mouse-core

mouse-core today consumes **HuggingFace dataset columns** (see [mouse-core `docs/data.md`](https://github.com/micahr234/mouse-core/blob/main/docs/data.md)), not raw Gym `info` directly. The contract is the **target shape** for:

1. **Live collection** — wrappers populate `info` (and/or append dicts) in this form.
2. **Dataset export** — flatten dicts into columns for `DatasetStore.append` / Parquet.

### Suggested column mapping (v1 → mouse-core)

| Contract field | Suggested dataset column | Current mouse-core column |
|----------------|-------------------------|-------------------------|
| `env_id` | `env_id` | *(not standard today — add)* |
| `episode_index` | `episode_index` | *(not standard today — add)* |
| `step_index` | `step_index` | `episode_step` (1-based; migrate to 0-based `step_index`) |
| `action["discrete"]` | `action` | `action` |
| `observation["continuous"]` | `observation` | `observation` |
| `observation["discrete"]` | `observation_discrete` | `observation_discrete` |
| `observation["image"]` | `observation_image` | `observation_image` |
| `done` | `done` | `done` |
| `reward["step"]` | `reward` | `reward` |
| `reward["episodic"]` | `reward_episodic` or keep `xformed_reward` | `xformed_reward` |
| `extensions["metadata_q_star"]` | `metadata_q_star` | `metadata_q_star` → `q_star` in TensorDict |

mouse-core changes (when you adopt v1):

- Teach `DatasetStore` to accept `env_id`, `episode_index`, `step_index` (and map `step_index` → `time` in TensorDict if you keep `time` as the internal name).
- Prefer reading `reward_episodic` with fallback to `xformed_reward` during migration.
- Document that `env_id` is the join key across multi-env datasets, not a free-form name.

---

## mouse-env implementation status

| Contract | Status |
|----------|--------|
| `env_id` | Partial — today `info["env_name"]` (same string as Gym id); rename/add `env_id` planned |
| `episode_index` | Not yet emitted |
| `step_index` (0-based) | Partial — today `info["episode_step"]` is 1-based |
| `action` dict | Not yet in `info` |
| `observation` dict | Not yet in `info` (obs is return value only) |
| `done` | Implemented |
| `reward` dict | Partial — `reward` return + top-level `xformed_reward` |

Migration strategy: **emit v1 alongside legacy keys**, update mouse-core, deprecate legacy keys in a later release.

---

## Reference types (Python)

See `mouse.envs.contract`:

- `RolloutStepCore` — core TypedDict with `env_id`
- `RewardDict`, action/observation key constants
- `RolloutExtensions` — optional block

---

## Versioning

This document describes **contract v1 (core)**. When the shape changes, bump the version in this file and in `contract.py` so mouse-env and mouse-core can gate behaviour (`contract_version=1` in dataset metadata, for example).
