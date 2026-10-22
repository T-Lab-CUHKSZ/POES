"""POES on a synthetic benchmark, no LLM needed.

Every item has a difficulty b_i, every prompt an ability a_p, and prompt p solves item i with
probability sigmoid(a_p - b_i + e_pi), where e_pi is a fixed prompt-item effect. A small
hill-climbing optimizer proposes prompts, scores each one on k items chosen by POES, and
finally returns the prompt with the best score. The script reports the pool accuracy of the
returned prompt when scores are raw subset accuracies vs. calibrated scores.

    python examples/simulate.py --runs 20
"""

import argparse

import numpy as np

from poes import POES


class World:
    def __init__(self, n_items, seed):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.b = rng.normal(0.0, 1.2, n_items)
        clusters = rng.integers(0, 8, n_items)
        self.sim = np.where(clusters[:, None] == clusters[None, :], 0.6, 0.1).astype(np.float32)
        np.fill_diagonal(self.sim, 1.0)
        self.abilities = []
        self.outcomes = []

    def new_prompt(self, ability):
        n = len(self.b)
        effect = self.rng.normal(0.0, 0.7, n)
        p = 1.0 / (1.0 + np.exp(-(ability - self.b + effect)))
        self.abilities.append(ability)
        self.outcomes.append((self.rng.random(n) < p).astype(float))
        return len(self.outcomes) - 1

    def accuracy(self, prompt):
        return float(self.outcomes[prompt].mean())


def run(world, calibrate, n_steps, per_step, k, seed):
    poes = POES([""] * len(world.b), similarity=world.sim, seed=seed)
    rng = np.random.default_rng(seed + 1)
    scores = {}

    def evaluate(prompts, progress):
        idx = poes.select(k, progress=progress)
        for p in prompts:
            poes.observe(p, idx, world.outcomes[p][idx])
        if calibrate:
            # rescore everything so old and new prompts share one scale
            names = list(scores) + [p for p in prompts if p not in scores]
            scores.update(zip(names, poes.calibrated_scores(names)))
        else:
            for p in prompts:
                scores[p] = float(world.outcomes[p][idx].mean())

    start = world.new_prompt(0.0)
    evaluate([start], 0.0)
    for step in range(n_steps):
        parent = max(scores, key=scores.get)
        for _ in range(per_step):
            child = world.new_prompt(world.abilities[parent] + rng.normal(0.0, 0.4))
            evaluate([child], step / n_steps)
    best = max(scores, key=scores.get)
    return world.accuracy(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--per-step", type=int, default=8)
    ap.add_argument("-k", type=int, default=20)
    args = ap.parse_args()

    results = {"raw": [], "calibrated": []}
    for r in range(args.runs):
        for name, calibrate in (("raw", False), ("calibrated", True)):
            world = World(args.items, seed=r)
            results[name].append(run(world, calibrate, args.steps, args.per_step, args.k, seed=r))

    for name, accs in results.items():
        print(f"{name:>10}: accuracy of returned prompt {np.mean(accs):.3f} +- {np.std(accs) / np.sqrt(len(accs)):.3f}")


if __name__ == "__main__":
    main()
