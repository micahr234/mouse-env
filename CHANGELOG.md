# Changelog

All notable changes to mouse-env are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.5.0] - 2026-06-25

### Added
- `EnvConfig.reset_seed` now controls mouse-env's internal Gymnasium reset seeding.
- `MouseEnv` now subclasses `gymnasium.Env` and exposes dynamic tuple `action_space` and `observation_space` attributes for underlying env instances.
- Added an RNG seeding control notebook showing how to reproduce or vary map generation and reset behavior independently.
- `EnvConfig.episode_reset_options` forwards options to every internal `env.reset(options=...)`; `EnvConfig.task_reset_options` overlays options only when a reset starts a new task. `Procedural-FrozenLake-v1` and `SyntheticEnv-v1` support `{"regenerate_map": True}` to sample a fresh map/MDP at either cadence.
- `episodes_per_task` now defaults to `0` (unlimited) — the task boundary (done codes 3/4) never fires automatically. Passing any positive integer restores the previous fixed-count behaviour.
- `MetricsTracker` class attached to `MouseEnv` as `env.tracker`; accumulates per-env `episode_cum_rewards` and `episode_lengths` automatically on every `step()` call and can be cleared with `env.tracker.clear()`.
- All Gymnasium `info` dict keys are now forwarded verbatim as `info_<key>` in every step output. For example, `info["env_q_star"]` appears as `outputs[i]["info_env_q_star"]`, `info["map"]` as `outputs[i]["info_map"]`, and `info["ns_params"]` as `outputs[i]["info_ns_params"]`. No env-specific filtering is applied.

### Changed
- Per-instance action-space access now uses the standard Gymnasium tuple space API (`env.action_space.spaces[i]`) instead of the removed `env.action_spaces` helper.
- `SyntheticEnv-v1` and `Procedural-FrozenLake-v1` now use `map_seed` instead of constructor `seed` for generated maps/MDPs, and random maps are generated lazily on first reset rather than during construction.
- Renamed flattened-env terminology to "env instance" / "env index"; the flat `outputs[i]`, `inputs[i]`, `env.names`, `env.input_specs`, and `env.output_specs` API shape is unchanged.
- `OutputSpec` no longer has `q_star`, `ns_params`, or `map` fields; info keys are dynamic and discovered from step outputs.
- `Procedural-FrozenLake-v1` now exposes a stable maximum discrete observation space for variable-size generated maps, so the spec remains valid across map regeneration.

### Fixed
- `ProceduralFrozenLakeEnv`: Q-table is now computed once per map initialization (guarded by the `_map_dirty` flag) instead of on every `reset()` call. For fixed maps this eliminates redundant value-iteration sweeps every episode.
- All example notebooks updated to use `episodes_per_task` (required field) instead of the removed `max_episode_steps` on `EnvConfig`. Episode time limits moved to `kwargs` where needed.
- Notebook 06 removed stale `reward_episodic` output field references.

### Changed
- `MouseEnv.step()` now returns `list[dict]` (outputs only) instead of `tuple[list[dict], list[dict]]` (outputs, metrics). Episode statistics are no longer returned inline; read them from `env.tracker` instead.
- Renamed `q_star_source` provider `"metadata_q_star"` to `"env_q_star"`; the output key `info_metadata_q_star` is now `info_env_q_star`.

### Removed
- `MouseEnv.action_spaces`; use `env.action_space.spaces[i]` or drill down into the underlying Gymnasium env instance instead.
- `first_visit_bonus` removed from `Procedural-FrozenLake-v1`; Q* outputs now reflect the solved map directly without undocumented novelty shaping.
- `RolloutMetrics` TypedDict removed from the public API (`mouse_envs` no longer exports it).

## [0.4.1] - 2026-06-24
