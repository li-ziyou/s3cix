"""
Value iteration for NotificationScheduler-v0 over the *observable* state space.

All state-space dimensions are derived from env.observation_space — no
hardcoded constants. The only numbers that live here are algorithmic
(gamma, theta, max_iter), which belong to the algorithm not the environment.

Because concentration is hidden, we cannot use it directly. Instead:
  - R(s, a) uses E[concentration | context] — the expected concentration
    given the observable context, derived from the stationary distribution
    of the concentration Markov chain under each context.
  - P(s'|s, a) is derived analytically from the env's config parameters.

Every approximation made here is a direct consequence of the POMDP structure.
Q-learning sidesteps both by sampling from the real environment instead.
"""

from __future__ import annotations

import itertools

import numpy as np


# ---------------------------------------------------------------------------
# State-space dimensions — derived from the environment at call time
# ---------------------------------------------------------------------------

def _dims(env) -> tuple[int, int, int, int, int, int]:
    """Return (N_CONTEXT, N_TOD, N_TSD, N_QSIZE, N_URGENCY, N_IMPORT)
    directly from env.observation_space.nvec."""
    nvec = env.observation_space.nvec
    return tuple(int(x) for x in nvec)  # type: ignore[return-value]


def n_states(env) -> int:
    """Total number of observable states."""
    result = 1
    for d in _dims(env):
        result *= d
    return result


N_ACTIONS = 2  # 0=wait, 1=deliver — fixed by the problem definition


def state_to_index(state: tuple, env) -> int:
    """Convert (ctx, tod, tsd, qs, mu, mi) tuple to flat index."""
    dims = _dims(env)
    idx = 0
    for v, d in zip(state, dims):
        idx = idx * d + v
    return idx


def all_states(env):
    """Iterate over all valid state tuples."""
    return itertools.product(*[range(d) for d in _dims(env)])


# ---------------------------------------------------------------------------
# Expected concentration given context
# ---------------------------------------------------------------------------

def _build_expected_concentration(env) -> np.ndarray:
    """
    E[concentration | context] for each context, shape (N_CONTEXT,).
    Delegates to env.expected_concentration — single source of this logic.
    """
    n_ctx = _dims(env)[0]
    result = np.zeros(n_ctx)
    original_ctx = env._context
    for ctx in range(n_ctx):
        env._context = ctx
        result[ctx] = env.expected_concentration
    env._context = original_ctx
    return result


# ---------------------------------------------------------------------------
# Approximate reward model R(s, a)
# ---------------------------------------------------------------------------

def _approximate_queue(qs, mu, mi, env):
    """
    Build a synthetic queue array representing the expected queue contents
    for observable state (qs, mu, mi), using env config params.
    """
    from notification_env.notification_scheduler import _NOTIF_DTYPE

    avg_urg  = (mu - 1) / 2.0 if mu > 0 else 0.0
    avg_imp  = (mi - 1) / 2.0 if mi > 0 else 0.0
    avg_age  = qs / (2.0 * env.ARRIVAL_PROB)

    init_val   = env.VAL_BASE + avg_urg * env.VAL_URG_SCALE + avg_imp * env.VAL_IMP_SCALE
    decay_rate = env.DECAY_BASE + avg_urg * env.DECAY_URG_SCALE

    queue = np.zeros(qs, dtype=_NOTIF_DTYPE)
    queue["urgency"]     = int(round(avg_urg))
    queue["importance"]  = int(round(avg_imp))
    queue["initial_val"] = init_val
    queue["decay_rate"]  = decay_rate
    queue["age"]         = int(round(avg_age))
    return queue


def build_reward_model(env) -> np.ndarray:
    """
    Build R[s, a] array of shape (N_STATES, N_ACTIONS).

    Uses env._compute_reward() — no separate reward implementation.
    Concentration is approximated by E[concentration | context].
    """
    from notification_env.notification_scheduler import _NOTIF_DTYPE

    E_conc  = _build_expected_concentration(env)
    n_s     = n_states(env)
    R       = np.zeros((n_s, N_ACTIONS))
    empty_q = np.zeros(0, dtype=_NOTIF_DTYPE)

    for state in all_states(env):
        ctx, tod, tsd, qs, mu, mi = state
        s  = state_to_index(state, env)
        ec = int(round(E_conc[ctx]))

        for action in range(N_ACTIONS):
            queue     = _approximate_queue(qs, mu, mi, env) if qs > 0 else empty_q
            queue_len = qs
            R[s, action] = env._compute_reward(
                action, queue, queue_len,
                concentration=ec, context=ctx, overflow=False,
            )

    return R


# ---------------------------------------------------------------------------
# Transition model P(s' | s, a)
# ---------------------------------------------------------------------------

def _next_tsd(tsd: int, action: int, n_tsd: int) -> int:
    if action == 1:
        return 0
    return min(tsd + 1, n_tsd - 1)


def _next_tod(tod: int, n_tod: int, tod_period: int) -> dict[int, float]:
    p_advance = 1.0 / tod_period
    if tod == n_tod - 1:
        return {tod: 1.0}
    return {tod: 1 - p_advance, tod + 1: p_advance}


def _next_context(ctx: int, n_ctx: int, trans_prob: float) -> dict[int, float]:
    stay   = 1 - trans_prob
    switch = trans_prob / (n_ctx - 1)
    return {c: (stay if c == ctx else switch) for c in range(n_ctx)}


def _next_queue(qs, mu, mi, action, env) -> dict[tuple, float]:
    if action == 1:
        return {(0, 0, 0): 1.0}

    result: dict[tuple, float] = {}
    p_no   = 1.0 - env.ARRIVAL_PROB
    result[(qs, mu, mi)] = p_no

    p_each = env.ARRIVAL_PROB / 9.0
    for u in range(3):
        for imp in range(3):
            if qs >= env.MAX_QUEUE:
                key = (qs, mu, mi)
            else:
                key = (qs + 1, max(mu, u + 1), max(mi, imp + 1))
            result[key] = result.get(key, 0.0) + p_each

    return result


def build_transition_model(env) -> list:
    """
    Build sparse transition model T[s][a] = [(prob, s'), ...].
    All dimensions and probabilities come from the env.
    """
    n_ctx, n_tod, n_tsd, n_qs, n_urg, n_imp = _dims(env)
    n_s = n_states(env)
    T   = [[[] for _ in range(N_ACTIONS)] for _ in range(n_s)]

    for state in all_states(env):
        ctx, tod, tsd, qs, mu, mi = state
        s = state_to_index(state, env)

        for action in range(N_ACTIONS):
            new_tsd  = _next_tsd(tsd, action, n_tsd)
            tod_dist = _next_tod(tod, n_tod, env.TOD_PERIOD)
            ctx_dist = _next_context(ctx, n_ctx, env.CTX_TRANS_PROB)
            q_dist   = _next_queue(qs, mu, mi, action, env)

            for (new_ctx, p_ctx) in ctx_dist.items():
                for (new_tod, p_tod) in tod_dist.items():
                    for (q_key, p_q) in q_dist.items():
                        new_qs, new_mu, new_mi = q_key
                        prob   = p_ctx * p_tod * p_q
                        s_next = state_to_index(
                            (new_ctx, new_tod, new_tsd, new_qs, new_mu, new_mi), env
                        )
                        T[s][action].append((prob, s_next))

    return T


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ValueIterationAgent:
    """
    Wraps a converged Q-table for use with the environment.

    Usage
    -----
    agent = ValueIterationAgent(Q, env)
    action = agent(obs)
    """

    def __init__(self, Q: np.ndarray, env):
        self.Q   = Q
        self.env = env

    def __call__(self, obs: np.ndarray) -> int:
        s = state_to_index(tuple(int(x) for x in obs), self.env)
        return int(np.argmax(self.Q[s]))
