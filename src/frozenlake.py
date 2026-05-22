"""Custom FrozenLake environment with generated map and optional q_star labels."""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium.envs.registration import register, registry
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

from mouse.envs.tabular.value_iteration import value_iteration_gymnasium_p
from mouse.utils import map_payload_to_json_str

from mouse.envs.env_ids import CUSTOM_FROZENLAKE_ENV_ID


class CustomFrozenLakeEnv(FrozenLakeEnv):
    """FrozenLake variant with generated valid map and optional q_star info.

    ``goal_reward_low`` / ``goal_reward_high`` (default ``1.0`` each, matching Gymnasium's goal
    reward) sample one reward **per goal tile** when the map is generated (not on each ``reset``),
    stored in :attr:`_goal_rewards_by_state`; equal bounds yield a constant reward per goal.
    For a **fixed** map, pass ``fixed_map`` as a dict with
    ``board`` and ``rewards`` (same shape as emitted metadata). When ``emit_q_star`` is True,
    :meth:`compute_q_table` applies those terminal rewards in value iteration.

    When ``emit_map`` is True, ``info["map"]`` is a **JSON string** encoding
    ``{"board": [...], "rewards": {...}}`` (see :func:`utils.map_payload_to_json_str`).

    Random maps place ``S`` / ``G`` and holes per ``hole_prob`` (see :meth:`_generate_map`).

    Map validity uses :meth:`_find_path_to_goal` so the shortest path from each ``S`` meets
    ``min_hops``. When ``emit_q_star`` is True, labels use
    **tabular value iteration** on Gymnasium's ``P`` matrix (same scheme as
    ``CustomSyntheticEnv.compute_q_table``). The Q-table is rebuilt on every
    :meth:`reset`.

    ``step_penalty`` is added to the scalar reward on **every** :meth:`step` (including
    terminal transitions). Value iteration includes the same offset so ``q_star`` matches rollout
    rewards when ``emit_q_star`` is True.

    When ``first_visit_bonus`` is non-zero and ``emit_q_star`` is True, each finite ``info["q_star"][a]``
    is increased by that amount for every action ``a`` that, under :attr:`P`, has positive probability
    of transitioning from the current state to a state not yet visited on this env instance (the
    visited set is cleared only at construction; supervision only; :meth:`compute_q_table` is unchanged).
    """

    def __init__(
        self,
        render_mode: str | None = None,
        is_slippery: bool = False,
        min_hops: int = 3,
        min_width: int = 3,
        max_width: int = 8,
        min_height: int = 3,
        max_height: int = 8,
        hole_prob: float = 0.2,
        start_pos: int | list[int] | None = None,
        start_pos_prob: float | None = None,
        goal_pos: int | list[int] | None = None,
        goal_pos_prob: float | None = None,
        max_tries: int = 10_000,
        fixed_map: list[str] | tuple[str, ...] | Mapping[str, Any] | None = None,
        emit_q_star: bool = False,
        emit_map: bool = False,
        goal_reward_low: float = 1.0,
        goal_reward_high: float = 1.0,
        step_penalty: float = 0.0,
        seed: int | None = None,
        first_visit_bonus: float = 0.0,
    ):
        """
        Args:
            render_mode: Gymnasium render mode (e.g. ``"human"``). ``None`` disables rendering.
            is_slippery: When ``True``, movement succeeds with probability 1/3 and may slide
                sideways — the standard stochastic FrozenLake setting.
            min_hops: Minimum BFS distance from each start tile to the nearest goal tile.
                Maps that don't satisfy this are rejected and regenerated.
            min_width: Minimum map width (columns) for random maps.
            max_width: Maximum map width (columns) for random maps.
            min_height: Minimum map height (rows) for random maps.
            max_height: Maximum map height (rows) for random maps.
            hole_prob: Probability that any non-start, non-goal tile becomes a hole ``"H"``.
            start_pos: Fixed start position(s) as a tile index or list of indices.
                ``None`` uses ``start_pos_prob`` or places one start randomly.
            start_pos_prob: Probability that each non-goal tile becomes a start tile.
                Ignored when ``start_pos`` is set.
            goal_pos: Fixed goal position(s). ``None`` uses ``goal_pos_prob`` or one random goal.
            goal_pos_prob: Probability that each available tile becomes a goal tile.
            max_tries: Number of map generation attempts before raising a ``RuntimeError``.
            fixed_map: Override the random map with a fixed layout. Accepted forms:

                - ``list[str]`` / ``tuple[str, ...]`` — rows of ``S``, ``F``, ``H``, ``G`` chars.
                - ``dict`` with ``"board"`` key (list of row strings) and optional ``"rewards"``
                  key mapping goal state indices to reward values.
            emit_q_star: When ``True``, run value iteration on every ``reset()`` and inject
                the resulting Q-table into ``info["q_star"]`` each step.
            emit_map: When ``True``, inject the map layout as a JSON string into
                ``info["map"]`` once per episode (on the first step/reset).
            goal_reward_low: Lower bound for per-goal reward sampling (inclusive).
            goal_reward_high: Upper bound for per-goal reward sampling. Equal bounds yield
                a fixed reward.
            step_penalty: Scalar added to the reward on every step (e.g. ``-0.01`` for a
                step cost). Also applied inside value iteration so ``q_star`` matches.
            seed: Random seed for map generation and Q-table RNG. ``None`` = non-deterministic.
            first_visit_bonus: Bonus added to ``q_star[a]`` for any action that has positive
                probability of reaching an unvisited state. Applied at supervision time only;
                does not affect ``compute_q_table``.
        """
        lo, hi = float(goal_reward_low), float(goal_reward_high)
        if lo > hi:
            lo, hi = hi, lo
        self._goal_reward_low = lo
        self._goal_reward_high = hi
        self._step_penalty = float(step_penalty)
        self._first_visit_bonus = float(first_visit_bonus)
        self._visited_states: set[int] = set()
        self.emit_q_star = bool(emit_q_star)
        self.emit_map = bool(emit_map)
        self.gamma = 1.0
        self._map_rng = random.Random(seed)
        self._q_table_rng = np.random.default_rng(seed)
        fixed_rewards: Mapping[int | str, float] | None = None
        if fixed_map is not None:
            self._gridmap, fixed_rewards = self._parse_fixed_map_spec(fixed_map)
        else:
            self._gridmap = self._generate_valid_map(
                self._map_rng,
                min_hops=int(min_hops),
                max_tries=int(max_tries),
                min_width=int(min_width),
                max_width=int(max_width),
                min_height=int(min_height),
                max_height=int(max_height),
                hole_prob=float(hole_prob),
                start_pos=start_pos,
                start_pos_prob=start_pos_prob,
                goal_pos=goal_pos,
                goal_pos_prob=goal_pos_prob,
            )
        self._goal_rewards_by_state = self._compute_goal_rewards_for_map(
            gridmap=self._gridmap,
            rng=self._map_rng,
            overrides=fixed_rewards,
        )
        super().__init__(
            render_mode=render_mode,
            desc=self._gridmap,
            is_slippery=bool(is_slippery),
        )
        self._map_metadata = map_payload_to_json_str(self._make_map_metadata_dict())
        self._map_dirty = True
        self._q_table: np.ndarray | None = None

    def _make_map_metadata_dict(self) -> dict[str, Any]:
        """Structured map payload; serialized to JSON for ``info["map"]`` when ``emit_map`` is True."""
        return {
            "board": list(self._gridmap),
            "rewards": {
                str(k): float(v) for k, v in sorted(self._goal_rewards_by_state.items())
            },
        }

    @staticmethod
    def goal_states_from_gridmap(gridmap: list[str]) -> list[int]:
        """Linear state indices of every ``G`` tile (row-major)."""
        cols = len(gridmap[0])
        out: list[int] = []
        for r, row in enumerate(gridmap):
            for c, ch in enumerate(row):
                if ch == "G":
                    out.append(r * cols + c)
        return out

    @classmethod
    def _parse_fixed_map_spec(
        cls,
        fixed_map: list[str] | tuple[str, ...] | Mapping[str, Any],
    ) -> tuple[list[str], Mapping[int | str, float] | None]:
        """Return ``(gridmap, rewards)`` for a fixed map.

        * **List/tuple of rows** — same as before; ``rewards`` is ``None`` (use
          ``goal_reward_low`` / ``goal_reward_high``, defaulting to 1.0 / 1.0 per goal).
        * **Dict** — must have ``board`` (``list`` | ``tuple`` of row strings). Optional
          ``rewards``: mapping goal state index → reward (must cover every ``G`` if present).
        """
        if isinstance(fixed_map, Mapping):
            if "board" not in fixed_map:
                raise ValueError(
                    "fixed_map dict must include 'board' (list of row strings), "
                    "and optionally 'rewards' (goal state index → reward)."
                )
            board_raw = fixed_map["board"]
            if not isinstance(board_raw, (list, tuple)):
                raise ValueError("fixed_map['board'] must be a list or tuple of row strings.")
            gridmap = cls._normalize_and_validate_fixed_map(fixed_map=board_raw)
            rewards_raw = fixed_map.get("rewards")
            if rewards_raw is None:
                return gridmap, None
            if not isinstance(rewards_raw, Mapping):
                raise ValueError(
                    "fixed_map['rewards'] must be a mapping from goal state index to reward."
                )
            return gridmap, rewards_raw
        return cls._normalize_and_validate_fixed_map(fixed_map=fixed_map), None

    def _compute_goal_rewards_for_map(
        self,
        gridmap: list[str],
        rng: random.Random,
        overrides: Mapping[int | str, float] | None,
    ) -> dict[int, float]:
        """One reward per goal state, fixed for the lifetime of this map."""
        goal_states = self.goal_states_from_gridmap(gridmap)
        if not goal_states:
            raise ValueError("Map must contain at least one 'G' goal tile.")
        goal_set = set(goal_states)
        if overrides is not None:
            norm = {int(k): float(v) for k, v in overrides.items()}
            if set(norm.keys()) != goal_set:
                raise ValueError(
                    "fixed_map['rewards'] must contain exactly one entry per goal state. "
                    f"Expected states {sorted(goal_set)}, got {sorted(norm.keys())}."
                )
            return norm
        lo, hi = self._goal_reward_low, self._goal_reward_high
        return {s: float(rng.uniform(lo, hi)) for s in goal_states}

    def _landed_on_goal(self, obs: int) -> bool:
        """True if ``obs`` indexes a ``G`` cell on the current ``desc``."""
        state = int(obs)
        row = state // int(self.ncol)
        col = state % int(self.ncol)
        cell = self.desc[row, col]
        if isinstance(cell, np.ndarray):
            cell = cell.item()
        if isinstance(cell, bytes):
            return cell == b"G"
        if isinstance(cell, str):
            return cell == "G"
        return bytes(cell) == b"G"

    @staticmethod
    def _find_path_to_goal(gridmap: list[str], state: int) -> tuple[list[tuple[int, int]], list[int]] | None:
        """Shortest path to any ``G`` via BFS (used for ``min_hops`` validation only)."""
        rows = len(gridmap)
        cols = len(gridmap[0])
        board = [list(row) for row in gridmap]
        start_r = state // cols
        start_c = state % cols
        start_pos = (start_r, start_c)
        goals = [(i, j) for i in range(rows) for j in range(cols) if board[i][j] == "G"]
        if not goals:
            return None
        directions = [
            (0, -1),   # left 0
            (1, 0),    # down 1
            (0, 1),    # right 2
            (-1, 0),   # up 3
        ]
        queue = deque()
        queue.append((start_pos, [], []))
        visited = set()
        while queue:
            (r, c), path, actions = queue.popleft()
            if (r, c) in goals:
                return [start_pos] + path, actions
            if (r, c) in visited:
                continue
            visited.add((r, c))
            for action, (dr, dc) in enumerate(directions):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != "H":
                    queue.append(((nr, nc), path + [(nr, nc)], actions + [action]))
        return None

    @classmethod
    def _map_is_valid(cls, gridmap: list[str], min_hops: int) -> bool:
        rows = len(gridmap)
        cols = len(gridmap[0])
        board = [list(row) for row in gridmap]
        found_start = False
        for r in range(rows):
            for c in range(cols):
                if board[r][c] != "S":
                    continue
                found_start = True
                state = r * cols + c
                result = cls._find_path_to_goal(gridmap=gridmap, state=state)
                if result is None:
                    return False
                _, actions = result
                if len(actions) < min_hops:
                    return False
        return found_start

    @classmethod
    def _normalize_and_validate_fixed_map(
        cls,
        fixed_map: list[str] | tuple[str, ...],
    ) -> list[str]:
        if not fixed_map:
            raise ValueError("fixed_map cannot be empty.")
        gridmap = [str(row) for row in fixed_map]
        row_width = len(gridmap[0])
        if row_width == 0:
            raise ValueError("fixed_map rows cannot be empty.")
        if any(len(row) != row_width for row in gridmap):
            raise ValueError("fixed_map rows must all have the same width.")

        allowed = {"S", "F", "H", "G"}
        invalid_chars = sorted({ch for row in gridmap for ch in row if ch not in allowed})
        if invalid_chars:
            raise ValueError(
                f"fixed_map contains unsupported characters {invalid_chars}. Allowed: {sorted(allowed)}."
            )

        num_starts = sum(row.count("S") for row in gridmap)
        num_goals = sum(row.count("G") for row in gridmap)
        if num_starts < 1:
            raise ValueError("fixed_map must contain at least one 'S' start tile.")
        if num_goals < 1:
            raise ValueError("fixed_map must contain at least one 'G' goal tile.")
        return gridmap

    @staticmethod
    def _generate_map(
        rng: random.Random,
        min_width: int,
        max_width: int,
        min_height: int,
        max_height: int,
        hole_prob: float,
        start_pos: int | list[int] | None,
        start_pos_prob: float | None,
        goal_pos: int | list[int] | None,
        goal_pos_prob: float | None,
    ) -> list[str]:
        width = rng.randint(min_width, max_width)
        height = rng.randint(min_height, max_height)
        map_index = ["F"] * (width * height)
        available_index = list(range(width * height))

        if isinstance(start_pos, int):
            start_positions: list[int] | None = [start_pos]
        elif start_pos is None:
            start_positions = None
        else:
            start_positions = list(start_pos)
        if start_positions is None and start_pos_prob is None:
            start_positions = [rng.choice(available_index)]
        elif start_positions is None and start_pos_prob is not None:
            start_positions = [i for i in available_index if rng.random() < start_pos_prob]
        for p in (start_positions or []):
            if p in available_index:
                map_index[p] = "S"
                available_index.remove(p)

        if isinstance(goal_pos, int):
            goal_positions: list[int] | None = [goal_pos]
        elif goal_pos is None:
            goal_positions = None
        else:
            goal_positions = list(goal_pos)
        if goal_positions is None and goal_pos_prob is None:
            goal_positions = [rng.choice(available_index)]
        elif goal_positions is None and goal_pos_prob is not None:
            goal_positions = [i for i in available_index if rng.random() < goal_pos_prob]
        for p in (goal_positions or []):
            if p in available_index:
                map_index[p] = "G"
                available_index.remove(p)

        for i in available_index:
            if rng.random() < hole_prob:
                map_index[i] = "H"

        return ["".join(map_index[i * width:(i + 1) * width]) for i in range(height)]

    @classmethod
    def _generate_valid_map(
        cls,
        rng: random.Random,
        min_hops: int,
        max_tries: int,
        **kwargs: Any,
    ) -> list[str]:
        for _ in range(max_tries):
            gridmap = cls._generate_map(rng=rng, **kwargs)
            if cls._map_is_valid(gridmap, min_hops=min_hops):
                return gridmap
        raise RuntimeError(
            "Could not generate a valid FrozenLake map. "
            "Try lower hole_prob, lower min_hops, or larger dimensions."
        )

    def compute_q_table(
        self,
        max_iter: int = 10_000,
        tolerance: float = 1e-10,
    ) -> np.ndarray:
        """Compute the optimal Q-table via value iteration on Gymnasium's ``P`` matrix.

        Uses the same Bellman update scheme as :meth:`CustomSyntheticEnv.compute_q_table
        <mouse.envs.synthetic.CustomSyntheticEnv.compute_q_table>`. Goal-tile rewards are
        overridden with the per-tile values from ``_goal_rewards_by_state``, and
        ``step_penalty`` is added to every non-terminal transition.

        Args:
            max_iter: Maximum number of value-iteration sweeps.
            tolerance: Convergence threshold on the max absolute change in Q-values.

        Returns:
            ``float64`` array of shape ``(num_states, 4)`` — optimal Q-values for
            every (state, action) pair under the current map.
        """
        n_s = int(self.nrow * self.ncol)
        return value_iteration_gymnasium_p(
            self.P,
            n_states=n_s,
            n_actions=4,
            gamma=float(self.gamma),
            step_penalty=float(self._step_penalty),
            goal_rewards_by_state=self._goal_rewards_by_state,
            max_iter=max_iter,
            tolerance=tolerance,
        )

    def _optimal_action_for_obs(self, obs: int) -> int:
        if self._q_table is None:
            return 0
        state = int(obs)
        n_s, _ = self._q_table.shape
        if state < 0 or state >= n_s:
            return 0
        return int(np.argmax(self._q_table[state]))

    def _q_star_for_obs(self, obs: int) -> np.ndarray:
        action_dim = int(getattr(self.action_space, "n", 0))
        if self._q_table is None:
            fallback = np.full((action_dim,), np.nan, dtype=np.float64)
            fallback[self._optimal_action_for_obs(obs)] = 0.0
            return fallback
        state = int(obs)
        n_s, _ = self._q_table.shape
        if state < 0 or state >= n_s:
            fallback = np.full((action_dim,), np.nan, dtype=np.float64)
            fallback[self._optimal_action_for_obs(obs)] = 0.0
            return fallback
        return np.asarray(self._q_table[state], dtype=np.float64).copy()

    def _q_star_for_obs_with_first_visit(self, obs: int) -> np.ndarray:
        """Return ``q_star`` for ``obs``, adding :attr:`_first_visit_bonus` to each action that can
        reach an unvisited successor (see class docstring)."""
        q = self._q_star_for_obs(obs)
        state = int(obs)
        self._visited_states.add(state)
        b = self._first_visit_bonus
        if b == 0.0:
            return q
        out = q.copy()
        P = self.P
        n_a = int(getattr(self.action_space, "n", 0))
        visited = self._visited_states
        for a in range(n_a):
            leads_to_new = False
            for trans in P[state][a]:
                prob, next_s, _, _ = trans
                if float(prob) <= 0.0:
                    continue
                if int(next_s) not in visited:
                    leads_to_new = True
                    break
            if leads_to_new:
                out[a] = out[a] + b
        return out

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        if self.emit_q_star:
            self._q_table = self.compute_q_table()
        info = dict[str, Any](info)
        if self._map_dirty:
            if self.emit_map:
                info["map"] = self._map_metadata
            self._map_dirty = False
        if self.emit_q_star:
            info["q_star"] = self._q_star_for_obs_with_first_visit(int(obs))
        return obs, info

    def step(self, a: Any):
        obs, reward, terminated, truncated, info = super().step(a)
        if terminated and self._landed_on_goal(int(obs)):
            reward = float(self._goal_rewards_by_state[int(obs)])
        reward = float(reward) + self._step_penalty
        info = dict[str, Any](info)
        if self._map_dirty:
            if self.emit_map:
                info["map"] = self._map_metadata
            self._map_dirty = False
        if self.emit_q_star:
            info["q_star"] = self._q_star_for_obs_with_first_visit(int(obs))
        return obs, reward, terminated, truncated, info


def ensure_custom_frozenlake_registered() -> None:
    """Register ``Custom-FrozenLake-v1`` with Gymnasium exactly once.

    Safe to call multiple times; subsequent calls are no-ops. Called automatically
    by :func:`~mouse.envs.envs.PlainVectorEnv` when ``env_id`` matches
    :data:`CUSTOM_FROZENLAKE_ENV_ID`.
    """
    if CUSTOM_FROZENLAKE_ENV_ID in registry:
        return
    register(
        id=CUSTOM_FROZENLAKE_ENV_ID,
        entry_point="mouse.envs.frozenlake:CustomFrozenLakeEnv",
    )
