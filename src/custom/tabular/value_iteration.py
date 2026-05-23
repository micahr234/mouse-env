"""Shared tabular value iteration for custom envs."""

from __future__ import annotations

from typing import Any

import numpy as np


def value_iteration_tabular(
    reward: np.ndarray,
    transition: np.ndarray,
    goal: np.ndarray,
    *,
    gamma: float,
    step_penalty: float = 0.0,
    max_iter: int = 10_000,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Value iteration on a dense tabular MDP (synthetic env layout).

    Bellman update::

        Q(s, a) = reward(s, a) + step_penalty + gamma * (not_goal(s, a)) * V(transition(s, a))

    Args:
        reward: Per-(state, action) rewards, shape ``(S, A)``.
        transition: Next-state index per (state, action), shape ``(S, A)``.
        goal: Bool mask; ``False`` at terminal goal transitions, shape ``(S, A)``.
        gamma: Discount factor.
        step_penalty: Added to every transition reward.
        max_iter: Maximum sweeps.
        tolerance: Stop when max absolute Q change is below this.

    Returns:
        Optimal Q-table, shape ``(S, A)``, ``float64``.
    """
    g = float(gamma)
    not_goal = ~np.asarray(goal, dtype=bool)
    step = float(step_penalty)
    obs_size, action_size = reward.shape
    q = np.zeros((obs_size, action_size), dtype=np.float64)
    for _ in range(int(max_iter)):
        v = q.max(axis=1)
        q_new = np.asarray(reward, dtype=np.float64) + step + g * not_goal * v[transition]
        if np.max(np.abs(q_new - q)) <= float(tolerance):
            return q_new
        q = q_new
    return q


def value_iteration_gymnasium_p(
    P: Any,
    *,
    n_states: int,
    n_actions: int,
    gamma: float,
    step_penalty: float,
    goal_rewards_by_state: dict[int, float],
    max_iter: int = 10_000,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Value iteration on Gymnasium toy-text ``P[s][a]`` transition lists.

    Args:
        P: Gymnasium env dynamics, indexed by state and action.
        n_states: Number of states.
        n_actions: Number of actions.
        gamma: Discount factor.
        step_penalty: Added to each transition reward before the Bellman backup.
        goal_rewards_by_state: Override terminal rewards at goal states.
        max_iter: Maximum sweeps.
        tolerance: Convergence threshold.

    Returns:
        Optimal Q-table, shape ``(n_states, n_actions)``, ``float64``.
    """
    g = float(gamma)
    q = np.zeros((n_states, n_actions), dtype=np.float64)
    for _ in range(int(max_iter)):
        v = q.max(axis=1)
        q_new = np.zeros((n_states, n_actions), dtype=np.float64)
        for s in range(n_states):
            for a in range(n_actions):
                acc = 0.0
                for prob, next_s, r, done in P[s][a]:
                    p = float(prob)
                    ns = int(next_s)
                    rr = float(r)
                    if done and ns in goal_rewards_by_state:
                        rr = float(goal_rewards_by_state[ns])
                    rr += step_penalty
                    if done:
                        acc += p * rr
                    else:
                        acc += p * (rr + g * v[ns])
                q_new[s, a] = acc
        if np.max(np.abs(q_new - q)) <= float(tolerance):
            return q_new
        q = q_new
    return q
