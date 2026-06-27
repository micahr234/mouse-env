# Changelog

All notable changes to mouse-env are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `SingleEnv`: standalone env class wrapping one gym env with its own `Tracker`. Constructed via `make_env(EnvConfig(...))`.
- `GroupEnv`: pure-reference container that delegates `step`, `close`, `render`, and tracker reads to a list of `SingleEnv` instances. Constructed via `make_group_env([cfg1, cfg2, ...])` or directly as `GroupEnv([env_a, env_b])`. Multiple `GroupEnv` objects may point to overlapping sets of `SingleEnv` instances without data conflicts.
- `make_group_env(list[EnvConfig])`: explicit factory for `GroupEnv`.
- `GroupTracker`: live read-through view over each constituent `SingleEnv`'s `Tracker`. Stores no episode data; `episode_cum_rewards[i]` and `episode_lengths[i]` delegate directly to `envs[i].tracker`.
- `SingleEnv`, `GroupEnv`, `Tracker`, and `GroupTracker` exported from `mouse_envs`.
- Added a tracker example notebook (`12_metrics_tracker.ipynb`) covering raw vs shaped returns, `clear()` between eval runs, and multi-env aggregation.
- Added a playable Procedural FrozenLake notebook with D-pad controls and rendered output.
- `mouse_envs.__version__` now exposes the installed `mouse-env` package version from package metadata.

### Changed
- `make_env(EnvConfig)` returns a `SingleEnv` (was `MouseEnv`). Passing a list to `make_env` is no longer supported — use `make_group_env`.
- `SingleEnv.step(input: dict) -> dict` and `GroupEnv.step(inputs: list[dict]) -> list[dict]`.
- Both env types expose `sample_random_input()` — returns `dict` on `SingleEnv`, `list[dict]` on `GroupEnv` (GroupEnv no longer has `sample_random_inputs()`).
- `SingleEnv.tracker` is a flat `Tracker` with `episode_cum_rewards: list[float]` and `episode_lengths: list[float]`.
- `GroupEnv.tracker` is a `GroupTracker` with per-env `list[list[float]]` views.
- `SingleEnv.name`, `SingleEnv.output_spec`, `SingleEnv.input_spec` replace indexed `names`, `output_specs[0]`, `input_specs[0]` from old single-env `MouseEnv` usage.
- `SingleEnv.action_space` and `SingleEnv.observation_space` expose the underlying gym space directly (not `gym.spaces.Tuple`).
- `EnvConfig.reset_seed` is now the sole config field for mouse-env's internal Gymnasium reset seeding.
- Observations and actions now preserve the underlying Gymnasium space dtype where possible instead of being routed through semantic observation kinds or canonical float/int dtypes.
- `QStarWrapper` now publishes normalized expert values as `info["q_star"]`, so step outputs emit `info_q_star` instead of `info_env_q_star`.

### Removed
- `MouseEnv` removed; replaced by `SingleEnv` (one env) and `GroupEnv` (multiple envs).
- `MetricsTracker` renamed to `Tracker`.
- `EnvConfig.observation_kind`; observations are always emitted under `observation`.
- `EnvConfig.seed`; use `EnvConfig.reset_seed` for reset-time seeding and `kwargs={"map_seed": ...}` for first-party procedural map/MDP generation.

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
- Repeated env instances are now created by passing an explicit `list[EnvConfig]` to `make_env`; each `EnvConfig` builds exactly one env instance.
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
- `EnvConfig.num_envs`; use one `EnvConfig` per env instance so per-env seeds and constructor kwargs such as `map_seed` are explicit.
- `MouseEnv.action_spaces`; use `env.action_space.spaces[i]` or drill down into the underlying Gymnasium env instance instead.
- `first_visit_bonus` removed from `Procedural-FrozenLake-v1`; Q* outputs now reflect the solved map directly without undocumented novelty shaping.
- `RolloutMetrics` TypedDict removed from the public API (`mouse_envs` no longer exports it).

## [0.4.1] - 2026-06-24
