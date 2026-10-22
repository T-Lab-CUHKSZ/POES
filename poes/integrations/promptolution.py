"""Use POES inside promptolution (tested with promptolution 2.3).

    task = POESTask(ClassificationTask(train_df, ...), n_steps=10, k=20)
    optimizer = OPRO(predictor=predictor, task=task, meta_llm=llm, callbacks=[task.callback()])
    prompts = optimizer.optimize(n_steps=10)

Works with optimizers that score prompts through `task.evaluate(prompts, predictor)`
(OPRO, EvoPromptGA, EvoPromptDE). Calls that pass `eval_strategy` or `block_idx`
explicitly, e.g. a final `evaluate(..., eval_strategy="full")`, go to the wrapped task unchanged.
"""

import dataclasses

import numpy as np
from promptolution.utils.callbacks import BaseCallback
from promptolution.utils.prompt import Prompt, sort_prompts_by_scores

from ..scheduler import POES


class POESTask:
    """Wraps a promptolution task so optimization-time evaluations use POES subsets.

    Args:
        task: a promptolution task (ClassificationTask, JudgeTask, ...).
        n_steps: number of optimizer steps, used to compute progress.
        k: subset size per evaluation.
        calibrate: report IRT-calibrated scores instead of raw subset accuracy.
        **poes_kwargs: passed to `POES`.
    """

    def __init__(self, task, n_steps, k=20, calibrate=True, **poes_kwargs):
        self.task = task
        self.n_steps = n_steps
        self.k = k
        self.calibrate = calibrate
        self.scheduler = POES(task.xs, **poes_kwargs)
        self.steps_done = 0

    def __getattr__(self, name):
        return getattr(self.task, name)

    @property
    def progress(self):
        return min(1.0, self.steps_done / self.n_steps) if self.n_steps else 0.0

    def evaluate(self, prompts, predictor, system_prompts=None, eval_strategy=None, block_idx=None):
        if eval_strategy is not None or block_idx is not None:
            return self.task.evaluate(
                prompts, predictor, system_prompts=system_prompts, eval_strategy=eval_strategy, block_idx=block_idx
            )

        prompts = [prompts] if isinstance(prompts, Prompt) else list(prompts)
        idx = self.scheduler.select(self.k, progress=self.progress)
        xs = [self.task.xs[i] for i in idx]
        ys = [self.task.ys[i] for i in idx]

        # promptolution's evaluate() gets its data from subsample(); point it at the chosen items.
        self.task.subsample = lambda eval_strategy=None, block_idx=None: (xs, ys)
        try:
            result = self.task.evaluate(prompts, predictor, system_prompts=system_prompts, eval_strategy="subsample")
        finally:
            del self.task.subsample

        for prompt, row in zip(prompts, result.scores):
            self.scheduler.observe(str(prompt), idx, row)

        if self.calibrate:
            scores = self.scheduler.calibrated_scores([str(p) for p in prompts])
            result = dataclasses.replace(result, agg_scores=np.asarray(scores, dtype=float))
        return result

    def callback(self):
        """Callback to pass to the optimizer; advances progress and refreshes kept scores."""
        return POESCallback(self)


class POESCallback(BaseCallback):
    """At the end of every step: advance progress and, with calibration on, rescore all kept
    prompts under the current IRT fit so earlier and newer scores stay on the same scale."""

    def __init__(self, poes_task):
        super().__init__()
        self.poes_task = poes_task

    def on_step_end(self, optimizer):
        task = self.poes_task
        task.steps_done += 1
        if not task.calibrate or not optimizer.prompts:
            return True

        scores = task.scheduler.calibrated_scores([str(p) for p in optimizer.prompts])
        prompts, scores = sort_prompts_by_scores(optimizer.prompts, scores)
        optimizer.prompts = prompts
        optimizer.scores = np.asarray(scores) if isinstance(optimizer.scores, np.ndarray) else scores

        # OPRO builds its meta-prompt from the score list at the end of each step.
        if hasattr(optimizer, "meta_prompt_template") and hasattr(optimizer, "_format_instructions"):
            optimizer.meta_prompt = optimizer.meta_prompt_template.replace(
                "<instructions>", optimizer._format_instructions()
            ).replace("<examples>", optimizer._sample_examples())
        return True

    def on_train_end(self, optimizer):
        return True
