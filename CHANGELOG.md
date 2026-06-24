# Changelog

All notable changes to mouse-env are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `episodes_per_task` now defaults to `0` (unlimited) — the task boundary (done codes 3/4) never fires automatically. Passing any positive integer restores the previous fixed-count behaviour.
- `MetricsTracker` class attached to `MouseEnv` as `env.tracker`; accumulates per-slot `episode_cum_rewards` and `episode_lengths` automatically on every `step()` call and can be cleared with `env.tracker.clear()`.
- All Gymnasium `info` dict keys are now forwarded verbatim as `info_<key>` in every step output. For example, `info["env_q_star"]` appears as `outputs[i]["info_env_q_star"]`, `info["map"]` as `outputs[i]["info_map"]`, and `info["ns_params"]` as `outputs[i]["info_ns_params"]`. No env-specific filtering is applied.

### Changed
- `OutputSpec` no longer has `q_star`, `ns_params`, or `map` fields; info keys are dynamic and discovered from step outputs.

### Fixed
- `ProceduralFrozenLakeEnv`: Q-table is now computed once per map initialization (guarded by the `_map_dirty` flag) instead of on every `reset()` call. For fixed maps this eliminates redundant value-iteration sweeps every episode.
- All example notebooks updated to use `episodes_per_task` (required field) instead of the removed `max_episode_steps` on `EnvConfig`. Episode time limits moved to `kwargs` where needed.
- Notebook 06 removed stale `reward_episodic` output field references.

### Changed
- `MouseEnv.step()` now returns `list[dict]` (outputs only) instead of `tuple[list[dict], list[dict]]` (outputs, metrics). Episode statistics are no longer returned inline; read them from `env.tracker` instead.
- Renamed `q_star_source` provider `"metadata_q_star"` to `"env_q_star"`; the output key `info_metadata_q_star` is now `info_env_q_star`.

### Removed
- `RolloutMetrics` TypedDict removed from the public API (`mouse_envs` no longer exports it).

## [0.4.1] - 2026-06-24
