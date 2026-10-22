import numpy as np

from poes.facility import greedy, swap


def objective(utility, sim, selected, lam):
    return utility[selected].sum() + lam * sim[selected].max(axis=0).mean()


def random_instance(p=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(p, 6))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    sim = np.maximum(x @ x.T, 0.0).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    return rng.random(p).astype(np.float32), sim


def test_greedy_returns_k_distinct():
    u, s = random_instance()
    chosen = greedy(u, s, 10, lam=0.6)
    assert len(chosen) == 10 and len(set(chosen)) == 10


def test_greedy_without_coverage_is_top_k():
    u, s = random_instance()
    assert set(greedy(u, s, 7, lam=0.0)) == set(np.argsort(-u)[:7].tolist())


def test_greedy_small_pool():
    u, s = random_instance(p=5)
    assert sorted(greedy(u, s, 8, lam=0.6)) == list(range(5))


def test_swap_respects_budget_and_improves():
    u, s = random_instance(seed=3)
    start = list(range(10))
    for budget in (0, 1, 3):
        new, accepted = swap(u, s, start, lam=0.6, switch_cost=0.0, budget=budget, n_candidates=128)
        assert len(new) == 10 and accepted <= budget
        assert len(set(new) ^ set(start)) <= 2 * budget
        assert objective(u, s, new, 0.6) >= objective(u, s, start, 0.6) - 1e-6


def test_swap_switch_cost_blocks_small_gains():
    u, s = random_instance(seed=4)
    start = greedy(u, s, 10, lam=0.6)
    new, accepted = swap(u, s, start, lam=0.6, switch_cost=10.0, budget=5, n_candidates=128)
    assert accepted == 0 and sorted(new) == sorted(start)
