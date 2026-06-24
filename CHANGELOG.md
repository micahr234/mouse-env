# Changelog

All notable changes to mouse-env are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `MetricsTracker` class attached to `MouseEnv` as `env.tracker`; accumulates per-slot `episode_cum_rewards` and `episode_lengths` automatically on every `step()` call and can be cleared with `env.tracker.clear()`.

### Changed
- `MouseEnv.step()` now returns `list[dict]` (outputs only) instead of `tuple[list[dict], list[dict]]` (outputs, metrics). Episode statistics are no longer returned inline; read them from `env.tracker` instead.

### Removed
- `RolloutMetrics` TypedDict removed from the public API (`mouse_envs` no longer exports it).

## [0.4.1] - 2026-06-24
