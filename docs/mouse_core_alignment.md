# Aligning mouse-core with the rollout contract

This page is for **mouse-core** maintainers. The authoritative spec is [rollout_contract.md](rollout_contract.md). mouse-env implements env construction; mouse-core implements **dataset ingestion and training**. Both should converge on the same row shape.

---

## Goal

A developer should be able to:

1. Read [rollout_contract.md](rollout_contract.md) and understand the seven core fields.
2. Point mouse-core’s data pipeline at datasets collected from mouse-env without env-specific code paths.
3. Use **`env_id`** as the stable key for filtering and multi-env experiments (not informal “names”).

---

## What mouse-core does today

`DatasetStore` ([`mouse.data.dataset_store`](https://github.com/micahr234/mouse-core/blob/main/src/data/dataset_store.py)) maps HuggingFace **columns** to a `TensorDict` batch. Relevant mappings (from mouse-core `docs/data.md`):

| Dataset column | TensorDict key | Notes |
|----------------|----------------|--------|
| `action` | `action` | discrete index |
| `reward` | `reward` | step reward |
| `xformed_reward` | `xformed_reward` | episodic training signal |
| `done` | `done` | 0 / 1 / 2 |
| `episode_step` | `time` | episode step index |
| `observation` | `obs_continuous` | |
| `observation_discrete` | `obs_discrete` | |
| `observation_image` | `obs_image` | |
| `metadata_q_star` | `q_star` | optional |

There is **no standard `env_id` column** yet; episode boundaries are implied by `done` and step counters, not by an explicit `episode_index`.

---

## Recommended changes in mouse-core

### 1. Document the contract

- Link to mouse-env’s `rollout_contract.md` from mouse-core `docs/data.md` and `docs/index.md`.
- State that v1 is the target schema for new datasets.

### 2. Extend `DatasetStore` column support

| Priority | Column | Maps from contract |
|----------|--------|-------------------|
| P0 | `env_id` | `env_id` |
| P0 | `step_index` | `step_index` (0-based); keep accepting `episode_step` as alias → `time` |
| P1 | `episode_index` | `episode_index` |
| P1 | `reward_episodic` | `reward["episodic"]`; fallback `xformed_reward` |
| P2 | nested `action` / `observation` | flatten on append: already flat columns are fine |

### 3. `append()` API

Accept either flat columns (today) or contract-shaped dicts:

```python
store.append({
    "env_id": "CartPole-v1",
    "episode_index": 0,
    "step_index": 3,
    "action": {"discrete": 1},
    "observation": {"continuous": [0.1, 0.2, 0.3, 0.4]},
    "done": 0,
    "reward": {"step": 1.0, "episodic": 0.02},
})
```

Implement a small normalizer: `contract_row_to_store_row(row) -> dict` shared by tests in both repos.

### 4. TensorDict / losses

- Keep internal name `time` if desired, but document that it is **`step_index`** in the contract.
- Losses using `use_xformed_reward` should accept `reward_episodic` or `xformed_reward` during migration.

### 5. No change required to model code immediately

Backbone and heads consume `TensorDict` keys; as long as `DatasetStore` produces the same keys (or aliases), training code can stay stable while columns evolve.

---

## Suggested PR split

1. **mouse-env**: docs + emit `env_id` and v1 keys alongside legacy `info` keys.
2. **mouse-core**: docs link + `DatasetStore` accepts `env_id`, `episode_index`, `step_index`, `reward_episodic`.
3. **mouse-env**: switch collectors to v1-only.
4. **mouse-core**: deprecate `episode_step` / `xformed_reward` column names in docs.

---

## Contract version metadata

When pushing datasets to the Hub, set dataset info metadata, e.g. `mouse_rollout_contract_version: 1`, so loaders can validate columns before training.
