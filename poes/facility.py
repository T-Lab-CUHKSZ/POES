"""Facility-location subset selection.

Objective over a candidate pool of p items:

    G(S) = sum_{i in S} u_i + lam * (1/p) * sum_j max_{i in S} s(i, j)

`greedy` builds S from scratch; `swap` starts from the previous subset and applies a few
one-for-one exchanges. Both work on positions within the candidate pool.
"""

import numpy as np


def greedy(utility, sim, k, lam):
    p = len(utility)
    if k >= p:
        return list(range(p))

    best = np.zeros(p, dtype=np.float32)
    chosen = np.zeros(p, dtype=bool)
    order = []
    for _ in range(k):
        cov_gain = (np.maximum(best[None, :], sim).sum(axis=1) - best.sum()) / p
        gain = utility + lam * cov_gain
        gain[chosen] = -np.inf
        j = int(np.argmax(gain))
        chosen[j] = True
        best = np.maximum(best, sim[j])
        order.append(j)
    return order


def _coverage_state(sim, selected):
    """Best and second-best similarity of every pool item to the selected set."""
    rows = sim[selected]
    nearest = np.argmax(rows, axis=0)
    cols = np.arange(sim.shape[1])
    best = rows[nearest, cols]
    if len(selected) >= 2:
        second = np.partition(rows, len(selected) - 2, axis=0)[-2:].min(axis=0)
    else:
        second = np.zeros_like(best)
    return best, np.asarray(selected)[nearest], second


def swap(utility, sim, selected, lam, switch_cost, budget, n_candidates):
    """Warm-start update of `selected`.

    At most `budget` swaps are made. Each round tries every (out, in) pair with `in` drawn from
    the `n_candidates` unselected items of highest utility, and applies the best one if it raises
    G by more than `switch_cost`. Returns the new selection and the number of accepted swaps.
    """
    p = len(utility)
    mask = np.zeros(p, dtype=bool)
    mask[selected] = True
    accepted = 0

    for _ in range(max(0, int(budget))):
        current = np.flatnonzero(mask)
        free = np.flatnonzero(~mask)
        if free.size == 0:
            break
        if free.size > n_candidates:
            free = free[np.argsort(-utility[free])[:n_candidates]]

        best, owner, second = _coverage_state(sim, current)
        cov = float(best.mean())
        cand_sim = sim[free]

        top, pair = 0.0, None
        for out in current:
            base = np.where(owner == out, second, best)
            cov_new = np.maximum(base[None, :], cand_sim).mean(axis=1)
            delta = utility[free] - utility[out] + lam * (cov_new - cov) - switch_cost
            j = int(np.argmax(delta))
            if delta[j] > top:
                top, pair = float(delta[j]), (out, free[j])

        if pair is None:
            break
        mask[pair[0]] = False
        mask[pair[1]] = True
        accepted += 1

    return np.flatnonzero(mask).tolist(), accepted
