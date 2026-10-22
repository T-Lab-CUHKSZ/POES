import numpy as np

from .facility import greedy, swap
from .irt import OnlineIRT, fit_1pl, sigmoid
from .similarity import tfidf_similarity


def _minmax(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    if np.isclose(lo, hi):
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _sym_kl(p, q, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    kl_pq = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    kl_qp = q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))
    return kl_pq + kl_qp


class POES:
    """Decides which k items of the training pool a candidate prompt is scored on.

    Usage inside an optimizer loop::

        poes = POES(train_inputs)
        idx = poes.select(k=20, progress=step / n_steps)
        outcomes = [score(prompt, i) for i in idx]      # 0/1 or any value in [0, 1]
        poes.observe(prompt, idx, outcomes)
        scores = poes.calibrated_scores(candidates)     # comparable across subsets

    Item choice combines an IRT discrimination utility (items the two strongest prompts are
    most likely to disagree on) with facility-location coverage of the pool. After a short
    random warmup the subset is updated by a few warm-start swaps per call, with the swap
    budget and switching cost set by a controller from optimization progress and the recent
    score trend.

    Because different prompts end up scored on different items, raw subset accuracies are not
    comparable. `calibrated_scores` refits the IRT model on all outcomes seen so far and returns
    each prompt's predicted accuracy on the same reference set (every item observed so far).

    Args:
        texts: inputs of the training pool; used for the TF-IDF similarity.
        similarity: optional precomputed (n, n) similarity matrix, replaces the TF-IDF one.
        pool_size: items with the highest utility considered for selection (all if n <= pool_size).
        n_candidates: incoming candidates tried per swap round.
        lam_start, lam_end: coverage weight, interpolated linearly over progress.
        budget_start, budget_end: swap budget as a fraction of k, interpolated over progress.
        budget_min, budget_max: bounds of the swap budget.
        switch_cost: base gain a swap must exceed.
        switch_cost_min, switch_cost_max: bounds of the switching cost.
        progress_gain, noise_gain: raise the switching cost with progress / with score noise.
        stagnation_factor, improving_factor: switching-cost multipliers when the score trend is
            flat / improving; a flat trend also adds one to the swap budget, an improving one
            removes one.
        trend_tol, trend_window: tolerance and window of the score-trend test.
        warmup_min, warmup_max: bounds on the number of random-subset evaluations; in between,
            warmup ends once the mean utility of the top-k items exceeds `warmup_ratio` times
            the pool mean.
        top_m: number of leading prompts whose pairwise disagreement defines the utility.
        irt_lr, irt_l2, irt_clip: online IRT update parameters.
        seed: random seed.
    """

    def __init__(
        self,
        texts,
        similarity=None,
        pool_size=512,
        n_candidates=128,
        lam_start=0.8,
        lam_end=0.4,
        budget_start=0.25,
        budget_end=0.10,
        budget_min=1,
        budget_max=5,
        switch_cost=0.05,
        switch_cost_min=0.01,
        switch_cost_max=0.30,
        progress_gain=0.5,
        noise_gain=0.5,
        stagnation_factor=0.8,
        improving_factor=1.1,
        trend_tol=0.01,
        trend_window=5,
        warmup_min=2,
        warmup_max=3,
        warmup_ratio=1.05,
        top_m=2,
        irt_lr=0.05,
        irt_l2=0.01,
        irt_clip=6.0,
        tie_noise=1e-6,
        seed=42,
    ):
        self.n = len(texts)
        self.sim = np.asarray(similarity, dtype=np.float32) if similarity is not None else tfidf_similarity(texts)
        if self.sim.shape != (self.n, self.n):
            raise ValueError(f"similarity must have shape ({self.n}, {self.n}), got {self.sim.shape}")

        self.pool_size = pool_size
        self.n_candidates = n_candidates
        self.lam_start, self.lam_end = lam_start, lam_end
        self.budget_start, self.budget_end = budget_start, budget_end
        self.budget_min, self.budget_max = budget_min, budget_max
        self.switch_cost = switch_cost
        self.switch_cost_min, self.switch_cost_max = switch_cost_min, switch_cost_max
        self.progress_gain, self.noise_gain = progress_gain, noise_gain
        self.stagnation_factor, self.improving_factor = stagnation_factor, improving_factor
        self.trend_tol, self.trend_window = trend_tol, trend_window
        self.warmup_min, self.warmup_max, self.warmup_ratio = warmup_min, warmup_max, warmup_ratio
        self.top_m = max(2, top_m)
        self.irt_params = dict(lr=irt_lr, l2=irt_l2, clip=irt_clip)
        self.tie_noise = tie_noise
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.irt = OnlineIRT(self.n, **self.irt_params)
        self.outcomes = {}  # (prompt, item) -> outcome
        self.history = []  # mean new outcome of each observe() call
        self.n_evals = 0
        self.warm = False
        self.selected = None
        self.last_swaps = 0

    # ------------------------------------------------------------------ selection

    def select(self, k, progress=0.0):
        """Indices of the k items to evaluate the next prompt(s) on.

        progress: fraction of the optimization budget used so far, in [0, 1].
        """
        k = min(int(k), self.n)
        progress = float(np.clip(progress, 0.0, 1.0))

        utility = None
        if not self.warm:
            in_warmup, utility = self._warmup(k)
            if in_warmup:
                return self.rng.choice(self.n, k, replace=False).tolist()
        if utility is None:
            utility = self.utility(noise=True)
            if utility is None:
                return self.rng.choice(self.n, k, replace=False).tolist()

        utility = _minmax(utility)
        lam = float(np.clip(
            self.lam_start + (self.lam_end - self.lam_start) * progress,
            min(self.lam_start, self.lam_end),
            max(self.lam_start, self.lam_end),
        ))
        pool = self._pool(utility, k)
        u = utility[pool]
        s = self.sim[np.ix_(pool, pool)]

        position = {int(i): j for j, i in enumerate(pool)}
        prev = [position[i] for i in (self.selected or []) if i in position]
        self.last_swaps = 0
        if self.selected is None or len(self.selected) != k or len(prev) < k:
            chosen = greedy(u, s, k, lam)
        else:
            cost, budget = self._controller(k, progress)
            chosen, self.last_swaps = swap(u, s, prev, lam, cost, budget, self.n_candidates)

        self.selected = [int(pool[j]) for j in chosen]
        return list(self.selected)

    def utility(self, noise=False):
        """Mean pairwise symmetric KL between the success probabilities of the leading prompts."""
        leaders = self.irt.top(self.top_m)
        if len(leaders) < 2:
            return None
        probs = [self.irt.success_prob(p) for p in leaders]
        total, pairs = np.zeros(self.n), 0
        for a in range(len(probs)):
            for c in range(a + 1, len(probs)):
                total += _sym_kl(probs[a], probs[c])
                pairs += 1
        u = total / pairs
        if noise and self.tie_noise > 0:
            u = u + self.rng.normal(0.0, self.tie_noise, size=u.shape)
        return u

    def _warmup(self, k):
        """Returns (still_in_warmup, utility computed while deciding to leave warmup)."""
        if self.n_evals >= self.warmup_max:
            self.warm = True
            return False, None
        if self.n_evals < self.warmup_min:
            return True, None
        u = self.utility(noise=False)
        if u is None:
            return True, None
        mean = float(np.mean(u))
        if not np.isfinite(mean) or mean <= 1e-6:
            return True, None
        ratio = float(np.mean(np.partition(u, -k)[-k:])) / (mean + 1e-8)
        if ratio >= self.warmup_ratio:
            self.warm = True
            return False, u
        return True, None

    def _pool(self, utility, k):
        if self.pool_size is None or self.pool_size >= self.n:
            return np.arange(self.n)
        size = max(self.pool_size, k)
        ranked = np.argsort(-utility)[:size]
        if self.selected is None:
            return ranked
        pool = sorted(set(self.selected))
        seen = set(pool)
        for i in ranked:
            if len(pool) >= size:
                break
            if int(i) not in seen:
                pool.append(int(i))
                seen.add(int(i))
        return np.asarray(pool)

    def _trend(self):
        h, w = self.history, self.trend_window
        if not h:
            return 0.0, 0.0
        delta = h[-1] - h[-w - 1] if len(h) > w else 0.0
        recent = h[-w:]
        spread = float(np.std(recent)) if len(recent) >= 2 else 0.0
        return delta, spread

    def _controller(self, k, progress):
        """Switching cost and swap budget for the next update."""
        delta, spread = self._trend()
        flat = abs(delta) < self.trend_tol
        improving = delta > self.trend_tol

        ratio = self.budget_start + (self.budget_end - self.budget_start) * progress
        budget = min(self.budget_max, max(self.budget_min, int(np.round(ratio * k))), k)
        cost = self.switch_cost * (1.0 + self.progress_gain * progress + self.noise_gain * spread)
        if flat:
            budget = min(self.budget_max, k, budget + 1)
            cost *= self.stagnation_factor
        elif improving:
            budget = max(self.budget_min, budget - 1)
            cost *= self.improving_factor
        cost = float(np.clip(cost, self.switch_cost_min, self.switch_cost_max))
        return cost, budget

    # ------------------------------------------------------------------ feedback

    def observe(self, prompt, items, outcomes):
        """Record per-item outcomes of `prompt`. Pairs seen before are ignored."""
        new_items, new_y = [], []
        for i, y in zip(items, outcomes):
            y = float(y)
            key = (prompt, int(i))
            if not np.isfinite(y) or key in self.outcomes:
                continue
            y = min(max(y, 0.0), 1.0)
            self.outcomes[key] = y
            new_items.append(int(i))
            new_y.append(y)
        if not new_items:
            return

        self.history.append(float(np.mean(new_y)))
        del self.history[: -10 * self.trend_window]
        self.n_evals += 1
        self.irt.update(prompt, new_items, new_y)

    def calibrated_scores(self, prompts):
        """Predicted accuracy of each prompt on every item observed so far.

        Fits a 1PL model on all recorded outcomes (all prompts), so scores of prompts evaluated
        on different subsets are on one scale. Uses no extra model calls.
        """
        prompts = list(prompts)
        if not self.outcomes:
            return [float("nan")] * len(prompts)

        index = {p: j for j, p in enumerate(prompts)}
        obs = []
        for (p, i), y in self.outcomes.items():
            if p not in index:
                index[p] = len(index)
            obs.append((index[p], i, y))
        theta, b = fit_1pl(obs, len(index), self.n)

        reference = sorted({i for _, i in self.outcomes})
        b_ref = b[reference]
        return [float(np.mean(sigmoid(theta[j] - b_ref))) for j in range(len(prompts))]
