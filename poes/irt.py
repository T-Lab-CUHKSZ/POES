"""One-parameter IRT model: P(prompt p solves item i) = sigmoid(theta_p - b_i)."""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class OnlineIRT:
    """MAP estimate updated with one gradient step per observation."""

    def __init__(self, n_items, lr=0.05, l2=0.01, clip=6.0):
        self.lr = lr
        self.l2 = l2
        self.clip = clip
        self.b = np.zeros(n_items)
        self.theta = {}

    def update(self, prompt, items, outcomes):
        theta = self.theta.get(prompt, 0.0)
        for i, y in zip(items, outcomes):
            b = float(np.clip(self.b[i], -self.clip, self.clip))
            err = y - float(sigmoid(theta - b))
            theta = float(np.clip(theta + self.lr * (err - self.l2 * theta), -self.clip, self.clip))
            self.b[i] = np.clip(b + self.lr * (-err - self.l2 * b), -self.clip, self.clip)
        self.theta[prompt] = theta

    def top(self, m):
        """The m prompts with the highest ability estimate."""
        ranked = sorted(self.theta.items(), key=lambda kv: kv[1], reverse=True)
        return [p for p, _ in ranked[:m]]

    def success_prob(self, prompt):
        return sigmoid(self.theta[prompt] - self.b)


def fit_1pl(obs, n_prompts, n_items, l2=0.5, iters=300, lr=0.5):
    """Batch MAP fit from all observations.

    obs: iterable of (prompt_index, item_index, outcome in [0, 1]).
    Returns (theta, b). Each parameter's gradient is scaled by 1 / (1 + its number of observations).
    """
    obs = np.asarray(list(obs), dtype=np.float64)
    theta = np.zeros(n_prompts)
    b = np.zeros(n_items)
    if obs.size == 0:
        return theta, b

    p_idx = obs[:, 0].astype(int)
    i_idx = obs[:, 1].astype(int)
    y = obs[:, 2]
    p_count = np.bincount(p_idx, minlength=n_prompts) + 1.0
    i_count = np.bincount(i_idx, minlength=n_items) + 1.0

    for _ in range(iters):
        err = y - sigmoid(theta[p_idx] - b[i_idx])
        theta += lr * (np.bincount(p_idx, weights=err, minlength=n_prompts) - l2 * theta) / p_count
        b += lr * (-np.bincount(i_idx, weights=err, minlength=n_items) - l2 * b) / i_count
    return theta, b
