import zlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("promptolution")

from promptolution.llms.base_llm import BaseLLM  # noqa: E402
from promptolution.optimizers import OPRO, EvoPromptDE, EvoPromptGA  # noqa: E402
from promptolution.predictors import MarkerBasedPredictor  # noqa: E402
from promptolution.tasks import ClassificationTask  # noqa: E402
from promptolution.utils.prompt import Prompt  # noqa: E402

from poes.integrations.promptolution import POESTask  # noqa: E402


class FakeLLM(BaseLLM):
    """Answers optimizer requests with new instructions and task inputs with a label that depends
    on a hash of (instruction, input), so different prompts get different accuracies."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def _get_response(self, prompts, system_prompts):
        out = []
        for text in prompts:
            if "<prompt>" in text:
                self.count += 1
                out.append(f"<prompt>Classify the input carefully, variant {self.count}.</prompt>")
            else:
                label = "yes" if zlib.crc32(text.encode()) % 3 else "no"
                out.append(f"<final_answer>{label}</final_answer>")
        return out


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    rows = [{"x": f"Is {i} a lucky number in case {rng.integers(100)}?", "y": "yes" if i % 3 else "no"} for i in range(120)]
    return pd.DataFrame(rows)


def build(data, calibrate=True):
    llm = FakeLLM()
    predictor = MarkerBasedPredictor(llm, classes=["yes", "no"])
    task = POESTask(ClassificationTask(data, x_column="x", y_column="y"), n_steps=3, k=10, calibrate=calibrate)
    return llm, predictor, task


@pytest.mark.parametrize("calibrate", [True, False])
def test_opro_runs(data, calibrate):
    llm, predictor, task = build(data, calibrate)
    opt = OPRO(
        predictor=predictor,
        task=task,
        meta_llm=llm,
        initial_prompts=["Answer yes or no."],
        num_instructions_per_step=3,
        callbacks=[task.callback()],
    )
    prompts = opt.optimize(n_steps=3)
    assert task.steps_done == 3
    assert task.scheduler.n_evals > 1
    assert all(np.isfinite(opt.scores))
    full = task.evaluate(prompts[0], predictor, eval_strategy="full")
    assert full.scores.shape == (1, len(task.xs))


@pytest.mark.parametrize("optimizer", [EvoPromptGA, EvoPromptDE])
def test_evoprompt_runs(data, optimizer):
    llm, predictor, task = build(data)
    opt = optimizer(
        predictor=predictor,
        task=task,
        meta_llm=llm,
        initial_prompts=[f"Answer yes or no, style {i}." for i in range(5)],
        callbacks=[task.callback()],
    )
    opt.optimize(n_steps=2)
    assert task.steps_done == 2
    assert all(np.isfinite(opt.scores))


def test_evaluate_uses_scheduled_items_and_restores_task(data):
    llm, predictor, task = build(data)
    task.evaluate([Prompt("Answer yes or no.")], predictor)
    assert len({i for (_, i) in task.scheduler.outcomes}) == 10
    assert "subsample" not in task.task.__dict__


@pytest.mark.parametrize("name", ["opro", "evoprompt_ga", "evoprompt_de"])
def test_run_builds_every_optimizer(data, name):
    import run

    llm, predictor, task = build(data)
    opt = run.build_optimizer(name, predictor, task, llm, run.DEFAULT_PROMPTS)
    opt.optimize(n_steps=2)
    assert task.steps_done == 2
    assert all(np.isfinite(opt.scores))


def test_run_load_data_lowercases_labels(tmp_path):
    import argparse

    import run

    path = tmp_path / "d.csv"
    pd.DataFrame({"x": [f"q{i}" for i in range(50)], "y": ["Yes", "No"] * 25}).to_csv(path, index=False)
    args = argparse.Namespace(data=str(path), test=None, train_frac=0.8, seed=0, max_train=None, max_test=5)
    train, test = run.load_data(args)
    assert len(train) == 40 and len(test) == 5
    assert set(train["y"]) == {"yes", "no"}
