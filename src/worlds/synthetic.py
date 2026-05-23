"""Synthetic Environment — random discrete tabular MDP with optional q_star labels."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry

from mouse.envs.experts.value_iteration import solve_tabular_mdp
from mouse.envs.utils import to_json_str

from mouse.envs.env_ids import SYNTHETIC_ENV_ID


def _uniform_n(
    rng: np.random.Generator,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    """Half-open uniform on ``[low, high)`` (``numpy.random.Generator.uniform``); constant if ``low == high``."""
    lo, hi = float(low), float(high)
    if lo > hi:
        raise ValueError(f"require low <= high, got low={low!r} high={high!r}.")
    if size <= 0:
        return np.zeros((0,), dtype=np.float64)
    if lo == hi:
        return np.full((size,), lo, dtype=np.float64)
    return rng.uniform(lo, hi, size=size)


class SyntheticEnv(gym.Env[int, int]):
    """Finite MDP with random tabular transitions, per-(state, action) rewards and goal flags.

    Dynamics are fully specified by ``map['transition']`` (next-state indices, including
    self-loops). ``transition_prob`` is **only** used when sampling a new random map (constructor
    ``map`` is ``None``): it controls how often each sampled edge is kept vs replaced by a
    self-loop at ``(s, a)`` (with ``goal`` cleared there). When ``map`` is provided, it is ignored;
    bake slip / no-op behavior into ``map['transition']`` instead.

    Random maps assign one **escape** action per state (index ``s % action_size``) to a non-self
    next state before ``transition_prob`` baking; that action is never turned into a self-loop.

    When sampling a random map (``map`` is ``None``), per-(state, action) rewards are drawn
    uniformly (half-open intervals): non-goal pairs use ``[reward_low, reward_high)``; goal pairs
    use ``[goal_reward_low, goal_reward_high)``. Equal lower and upper bounds yield a constant.

    ``step_penalty`` is added to the reward on every :meth:`step` (including terminal steps).
    :meth:`compute_q_table` applies the same offset so ``q_star`` matches rollout rewards when
    ``emit_q_star`` is True (via :func:`~mouse.envs.experts.solve_tabular_mdp`).

    :meth:`step` sets ``truncated=False`` like Gymnasium ``FrozenLakeEnv`` / Procedural Frozen Lake
    wrapper; step-limit truncation is applied by ``TimeLimit`` when using
    ``gym.make(..., max_episode_steps=...)``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        obs_size: int = 16,
        action_size: int = 4,
        reward_low: float = 0.0,
        reward_high: float = 0.0,
        goal_reward_low: float = -1.0,
        goal_reward_high: float = 1.0,
        goal_prob: float = 0.05,
        start_prob: float = 1.0,
        min_distance: int = 0,
        max_tries: int = 10000,
        map: dict[str, Any] | None = None,
        emit_q_star: bool = False,
        emit_map: bool = True,
        step_penalty: float = 0.0,
        transition_prob: float = 0.5,
        seed: int | None = None,
    ):
        """
        Args:
            obs_size: Number of discrete states (``observation_space = Discrete(obs_size)``).
                Ignored when ``map`` is provided (inferred from ``map["transition"]``).
            action_size: Number of discrete actions. Ignored when ``map`` is provided.
            reward_low: Lower bound (inclusive) for non-goal per-(state, action) rewards.
            reward_high: Upper bound (exclusive) for non-goal rewards. Equal to ``reward_low``
                gives constant zero rewards on non-goal transitions.
            goal_reward_low: Lower bound for goal per-(state, action) rewards.
            goal_reward_high: Upper bound for goal rewards.
            goal_prob: Probability that any ``(state, action)`` pair is designated as a
                goal (terminal) transition when sampling a random MDP.
            start_prob: Probability that each valid (non-goal) state is included as a
                start state in the sampled MDP.
            min_distance: Minimum BFS distance from any start state to the nearest goal
                transition. ``0`` allows starting adjacent to a goal.
            max_tries: Number of MDP sampling attempts before raising a ``ValueError``.
            map: Fixed map dict with keys ``"transition"``, ``"goal"``, ``"reward"``,
                and ``"start"`` (all as ``numpy`` arrays). When provided, all random-sampling
                parameters above are ignored.
            emit_q_star: When ``True``, run value iteration at construction and inject the
                Q-table into ``info["q_star"]`` on each step/reset.
            emit_map: When ``True``, inject the MDP arrays as a JSON string into
                ``info["map"]`` once per episode (on the first step/reset after construction).
            step_penalty: Scalar added to every step reward (including terminal steps).
                Included in value iteration so ``q_star`` matches rollout rewards.
            transition_prob: When sampling a random MDP, probability that a ``(state, action)``
                pair leads to a non-self-loop next state. The escape action
                ``a = state % action_size`` is always a non-self-loop. Ignored when ``map``
                is provided.
            seed: Random seed for MDP sampling. ``None`` = non-deterministic.

        Raises:
            ValueError: For invalid parameter combinations (e.g. ``reward_low > reward_high``,
                ``obs_size < 2``, or a fixed ``map`` that fails validation).
        """
        super().__init__()
        rl, rh = float(reward_low), float(reward_high)
        if rl > rh:
            raise ValueError(f"reward_low must be <= reward_high, got {reward_low!r} and {reward_high!r}.")
        gl, gh = float(goal_reward_low), float(goal_reward_high)
        if gl > gh:
            gh, gl = gl, gh
        if not (0.0 <= float(goal_prob) <= 1.0):
            raise ValueError(f"goal_prob must be in [0, 1], got {goal_prob!r}.")
        if not (0.0 < float(start_prob) <= 1.0):
            raise ValueError(f"start_prob must be in (0, 1], got {start_prob!r}.")
        if int(min_distance) < 0:
            raise ValueError(f"min_distance must be >= 0, got {min_distance!r}.")
        if int(max_tries) < 1:
            raise ValueError(f"max_tries must be >= 1, got {max_tries!r}.")
        sp = float(step_penalty)
        if not math.isfinite(sp):
            raise ValueError(f"step_penalty must be finite, got {step_penalty!r}.")
        self._step_penalty = sp
        if map is None:
            if not (0.0 <= float(transition_prob) <= 1.0):
                raise ValueError(f"transition_prob must be in [0, 1], got {transition_prob!r}.")
            self.transition_prob = float(transition_prob)
        else:
            self.transition_prob = None
        if map is None:
            resolved_obs_size = int(obs_size)
            resolved_action_size = int(action_size)
            if resolved_obs_size < 2:
                raise ValueError(f"obs_size must be >= 2, got {obs_size!r}.")
            if resolved_action_size < 2:
                raise ValueError(f"action_size must be >= 2, got {action_size!r}.")
        else:
            if "transition" not in map:
                raise ValueError("map must include key 'transition'.")
            transition = np.asarray(map["transition"], dtype=np.int64)
            if transition.ndim != 2:
                raise ValueError("map['transition'] must be a 2D array [obs_size, action_size].")
            resolved_obs_size, resolved_action_size = int(transition.shape[0]), int(transition.shape[1])
            if resolved_obs_size < 2:
                raise ValueError("map['transition'] must have obs_size >= 2.")
            if resolved_action_size < 2:
                raise ValueError("map['transition'] must have action_size >= 2.")

        self.obs_size = resolved_obs_size
        self.action_size = resolved_action_size
        self.reward_low = rl
        self.reward_high = rh
        self.goal_reward_low = gl
        self.goal_reward_high = gh
        self.goal_prob = float(goal_prob)
        self.start_prob = float(start_prob)
        self.emit_q_star = bool(emit_q_star)
        self.emit_map = bool(emit_map)
        self.gamma = 1.0
        self.min_distance = int(min_distance)
        self.max_tries = int(max_tries)
        self._init_seed = seed
        self._q_table_rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Discrete(self.obs_size)
        self.action_space = gym.spaces.Discrete(self.action_size)

        self.map: dict[str, np.ndarray] = {}
        self._map_dirty = True
        self._q_table: np.ndarray | None = None
        self._state = 0
        if map is None:
            self._build_random_mdp()
        else:
            self._load_env_map(map)
        if self.emit_q_star:
            self._refresh_q_table()

    def _map_payload_for_info(self) -> dict[str, Any]:
        return {k: np.array(v, copy=True) for k, v in self.map.items()}

    def _assign_escape_actions(self, rng: np.random.Generator) -> None:
        """Set ``transition[s, s % action_size]`` to a random state other than ``s``."""
        t = self.map["transition"]
        for s in range(self.obs_size):
            a_esc = s % self.action_size
            nxt = int(rng.integers(0, self.obs_size))
            while nxt == s:
                nxt = int(rng.integers(0, self.obs_size))
            t[s, a_esc] = nxt

    def _load_env_map(self, map: dict[str, Any]) -> None:
        """Load a fixed map structure and validate constraints."""
        if not {"transition", "goal", "reward", "start"} <= map.keys():
            raise ValueError("map must include 'transition', 'goal', 'reward', and 'start'.")
        transition = np.asarray(map["transition"], dtype=np.int64)
        goal = np.asarray(map["goal"], dtype=np.bool_)
        reward = np.asarray(map["reward"], dtype=np.float64)
        start = np.asarray(map["start"], dtype=np.bool_)
        if transition.shape != (self.obs_size, self.action_size):
            raise ValueError(
                "map['transition'] shape does not match inferred [obs_size, action_size]."
            )
        if goal.shape != (self.obs_size, self.action_size):
            raise ValueError("map['goal'] must have shape [obs_size, action_size].")
        if reward.shape != (self.obs_size, self.action_size):
            raise ValueError("map['reward'] must have shape [obs_size, action_size].")
        if start.shape != (self.obs_size,):
            raise ValueError("map['start'] must have shape [obs_size].")
        if np.any(transition < 0) or np.any(transition >= self.obs_size):
            raise ValueError("map['transition'] contains out-of-range next-state indices.")
        state_has_goal = np.any(goal, axis=1)
        if not np.any(state_has_goal):
            raise ValueError("map must contain at least one state with a goal transition.")
        if np.all(state_has_goal):
            raise ValueError("map must contain at least one state without any goal transition.")
        if not np.all(np.isfinite(reward)):
            raise ValueError("map['reward'] contains non-finite values.")
        if not np.any(start):
            raise ValueError("map['start'] must contain at least one start state.")
        if np.any(start & state_has_goal):
            raise ValueError("map['start'] must not select states that already have a goal transition.")

        self.map["transition"] = np.array(transition, copy=True)
        self.map["goal"] = np.array(goal, copy=True)
        self.map["reward"] = np.array(reward, copy=True)
        self.map["start"] = np.array(start, copy=True)
        self._map_dirty = True

    def _apply_transition_prob_to_sampled_map(self, rng: np.random.Generator) -> None:
        """With probability ``1 - transition_prob``, replace ``(s, a)`` with a self-loop and clear ``goal``.

        The escape action ``a = s % action_size`` is never replaced (see :meth:`_assign_escape_actions`).
        """
        p = self.transition_prob
        assert p is not None
        if p >= 1.0:
            return
        t = self.map["transition"]
        g = self.map["goal"]
        for s in range(self.obs_size):
            a_esc = s % self.action_size
            for a in range(self.action_size):
                if a == a_esc:
                    continue
                if rng.random() >= p:
                    t[s, a] = s
                    g[s, a] = False

    def _build_random_mdp(self) -> None:
        rng = np.random.default_rng(self._init_seed)
        for _ in range(self.max_tries):
            self.map["transition"] = rng.integers(
                low=0,
                high=self.obs_size,
                size=(self.obs_size, self.action_size),
                dtype=np.int64,
            )
            self.map["goal"] = rng.random(size=(self.obs_size, self.action_size)) < self.goal_prob
            state_has_goal = np.any(self.map["goal"], axis=1)
            if not np.any(state_has_goal):
                # Ensure at least one (state, action) reaches a goal.
                s = int(rng.integers(0, self.obs_size))
                a = int(rng.integers(0, self.action_size))
                self.map["goal"][s, a] = True
                state_has_goal[s] = True
            if np.all(state_has_goal):
                # Keep at least one state with no goal transition so reset/rollout is well-defined.
                s = int(rng.integers(0, self.obs_size))
                self.map["goal"][s, :] = False
                state_has_goal[s] = False
            self._assign_escape_actions(rng)
            self._apply_transition_prob_to_sampled_map(rng)
            state_has_goal = np.any(self.map["goal"], axis=1)
            if not np.any(state_has_goal):
                continue
            if np.all(state_has_goal):
                continue
            candidates = np.asarray(
                [s for s in range(self.obs_size) if self._is_valid_start_state(s)],
                dtype=np.int64,
            )
            if candidates.size == 0:
                continue
            # Each candidate is included as a start state with probability start_prob.
            start = np.zeros((self.obs_size,), dtype=np.bool_)
            for s in candidates:
                if rng.random() < self.start_prob:
                    start[s] = True
            if not np.any(start):
                start[candidates[int(rng.integers(0, candidates.size))]] = True
            self.map["start"] = start
            n_sa = self.obs_size * self.action_size
            reward = _uniform_n(rng=rng, low=self.reward_low, high=self.reward_high, size=n_sa).reshape(
                self.obs_size, self.action_size
            )
            goal_mask = self.map["goal"]
            n_goal = int(goal_mask.sum())
            if n_goal > 0:
                reward[goal_mask] = _uniform_n(
                    rng=rng, low=self.goal_reward_low, high=self.goal_reward_high, size=n_goal
                )
            self.map["reward"] = reward.astype(np.float64)
            self._map_dirty = True
            return
        raise ValueError(
            "Could not sample a valid map (reachable goal from valid start states) "
            f"within max_tries={self.max_tries}, min_distance={self.min_distance}. "
            "Try lowering min_distance, increasing max_tries or obs_size, or changing seed."
        )

    def _goal_distance(self, start_state: int) -> int | None:
        """Shortest directed path length from start_state to any state with a goal transition; None if unreachable."""
        s0 = int(start_state)
        if np.any(self.map["goal"][s0]):
            return 0
        visited = np.zeros((self.obs_size,), dtype=np.bool_)
        q: deque[tuple[int, int]] = deque([(s0, 0)])
        visited[s0] = True
        while q:
            state, dist = q.popleft()
            for nxt in np.unique(self.map["transition"][state]):
                nxt_i = int(nxt)
                if visited[nxt_i]:
                    continue
                if np.any(self.map["goal"][nxt_i]):
                    return dist + 1
                visited[nxt_i] = True
                q.append((nxt_i, dist + 1))
        return None

    def _is_valid_start_state(self, state: int) -> bool:
        s = int(state)
        if np.any(self.map["goal"][s]):
            return False
        d = self._goal_distance(s)
        return d is not None and d >= self.min_distance

    def compute_q_table(
        self,
        max_iter: int = 10000,
        tolerance: float = 1e-10,
    ) -> np.ndarray:
        """Compute the optimal Q-table for the current map via :func:`solve_tabular_mdp`.

        Args:
            max_iter: Maximum number of value-iteration sweeps.
            tolerance: Convergence threshold (max absolute change in Q-values).

        Returns:
            ``float64`` array of shape ``(obs_size, action_size)`` — optimal Q-values.
        """
        return solve_tabular_mdp(
            reward=self.map["reward"],
            transition=self.map["transition"],
            goal=self.map["goal"],
            gamma=float(self.gamma),
            step_penalty=float(self._step_penalty),
            max_iter=max_iter,
            tolerance=tolerance,
        )

    def _refresh_q_table(self) -> None:
        if self._q_table is not None:
            return
        self._q_table = self.compute_q_table()

    def _optimal_action_for_obs(self, obs: int) -> int:
        if self._q_table is None:
            return 0
        state = int(obs)
        if state < 0 or state >= self.obs_size:
            return 0
        return int(np.argmax(self._q_table[state]))

    def _q_star_for_obs(self, obs: int) -> np.ndarray:
        if self._q_table is None:
            fallback = np.full((self.action_size,), np.nan, dtype=np.float64)
            fallback[self._optimal_action_for_obs(obs)] = 0.0
            return fallback
        state = int(obs)
        if state < 0 or state >= self.obs_size:
            fallback = np.full((self.action_size,), np.nan, dtype=np.float64)
            fallback[self._optimal_action_for_obs(obs)] = 0.0
            return fallback
        return np.asarray(self._q_table[state], dtype=np.float64).copy()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        start_states = np.where(self.map["start"])[0]
        self._state = int(start_states[self.np_random.integers(0, start_states.size)])
        info: dict[str, Any] = {}
        if self._map_dirty:
            if self.emit_map:
                info["map"] = to_json_str(self._map_payload_for_info())
            self._map_dirty = False
        if self.emit_q_star:
            info["q_star"] = self._q_star_for_obs(self._state)
        return int(self._state), info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        a = int(action)
        if a < 0 or a >= self.action_size:
            raise ValueError(f"action {a} is out of bounds for Discrete({self.action_size}).")
        s = int(self._state)
        reward = float(self.map["reward"][s, a]) + self._step_penalty
        terminated = bool(self.map["goal"][s, a])
        self._state = int(self.map["transition"][s, a])
        obs = int(self._state)
        truncated = False
        info: dict[str, Any] = {}
        if self._map_dirty:
            if self.emit_map:
                info["map"] = to_json_str(self._map_payload_for_info())
            self._map_dirty = False
        if self.emit_q_star:
            info["q_star"] = self._q_star_for_obs(self._state)
        return obs, reward, terminated, truncated, info


def ensure_synthetic_env_registered() -> None:
    """Register ``SyntheticEnv-v1`` with Gymnasium exactly once.

    Safe to call multiple times; subsequent calls are no-ops. Called automatically
    by :func:`~mouse.envs.make_vector_env` when ``group_id`` matches
    :data:`SYNTHETIC_ENV_ID`.
    """
    if SYNTHETIC_ENV_ID in registry:
        return
    register(
        id=SYNTHETIC_ENV_ID,
        entry_point="mouse.envs.worlds.synthetic:SyntheticEnv",
    )
