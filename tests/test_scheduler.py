import numpy as np

from poes import POES, fit_1pl


def make_scheduler(n=120, seed=0, **kw):
    texts = [f"question {i} about topic {i % 7} with value {i * 13 % 29}" for i in range(n)]
    return POES(texts, seed=seed, **kw)


def simulate(poes, n_prompts=12, k=20, seed=0):
    """Feed outcomes of synthetic prompts; returns the subsets used."""
    rng = np.random.default_rng(seed)
    b = rng.normal(0, 1, poes.n)
    subsets = []
    for p in range(n_prompts):
        idx = poes.select(k, progress=p / n_prompts)
        subsets.append(idx)
        prob = 1 / (1 + np.exp(-(p / 4 - b[idx])))
        poes.observe(f"prompt {p}", idx, (rng.random(k) < prob).astype(float))
    return subsets


def test_select_returns_valid_subset():
    poes = make_scheduler()
    for idx in simulate(poes):
        assert len(idx) == 20 and len(set(idx)) == 20
        assert min(idx) >= 0 and max(idx) < poes.n


def test_warmup_is_bounded():
    poes = make_scheduler(warmup_max=3)
    simulate(poes, n_prompts=5)
    assert poes.warm


def test_bounded_drift_after_warmup():
    poes = make_scheduler(budget_max=5)
    subsets = simulate(poes, n_prompts=15)
    for prev, cur in zip(subsets[4:], subsets[5:]):
        assert len(set(prev) ^ set(cur)) <= 2 * 5


def test_observe_ignores_repeated_pairs():
    poes = make_scheduler()
    poes.observe("p", [1, 2, 3], [1, 0, 1])
    poes.observe("p", [1, 2, 3], [0, 0, 0])
    assert poes.n_evals == 1
    assert poes.outcomes[("p", 1)] == 1.0


def test_fit_1pl_recovers_parameters():
    rng = np.random.default_rng(0)
    theta, b = rng.normal(0, 1, 30), rng.normal(0, 1, 80)
    obs = [
        (p, i, float(rng.random() < 1 / (1 + np.exp(-(theta[p] - b[i])))))
        for p in range(30)
        for i in rng.choice(80, 40, replace=False)
    ]
    est_theta, est_b = fit_1pl(obs, 30, 80)
    assert np.corrcoef(theta, est_theta)[0, 1] > 0.7
    assert np.corrcoef(b, est_b)[0, 1] > 0.7


def test_calibration_corrects_for_subset_difficulty():
    # A strong prompt scored on hard items and a weak prompt scored on easy items:
    # raw accuracy prefers the weak one, the calibrated score should not.
    poes = make_scheduler(n=100)
    rng = np.random.default_rng(1)
    b = np.linspace(-2.5, 2.5, 100)
    idx = np.arange(100)
    for p in range(10):
        poes.observe(f"ref {p}", idx, (rng.random(100) < 1 / (1 + np.exp(-(0.0 - b)))).astype(float))
    hard, easy = np.arange(70, 100), np.arange(0, 30)
    strong = (rng.random(30) < 1 / (1 + np.exp(-(1.5 - b[hard])))).astype(float)
    weak = (rng.random(30) < 1 / (1 + np.exp(-(-1.5 - b[easy])))).astype(float)
    poes.observe("strong", hard, strong)
    poes.observe("weak", easy, weak)

    assert weak.mean() > strong.mean()
    cal_strong, cal_weak = poes.calibrated_scores(["strong", "weak"])
    assert cal_strong > cal_weak
