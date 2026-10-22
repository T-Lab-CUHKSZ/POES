"""Prompt optimization with POES on a CSV dataset.

    export OPENAI_API_KEY=...
    python run.py --data data/navigate.csv --optimizer opro \
        --worker-model <model-id> --worker-url https://.../v1

The CSV needs an input column `x` and a label column `y` (other columns are ignored).
Without --test, 20% of the rows are held out for the final test score.
"""

import argparse
import json
import os

import pandas as pd
from promptolution.llms import APILLM
from promptolution.optimizers import OPRO, EvoPromptDE, EvoPromptGA
from promptolution.predictors import MarkerBasedPredictor
from promptolution.tasks import ClassificationTask

from poes.integrations.promptolution import POESTask

OPTIMIZERS = {"opro": OPRO, "evoprompt_ga": EvoPromptGA, "evoprompt_de": EvoPromptDE}

# Starting population for EvoPrompt; OPRO only uses the first one.
DEFAULT_PROMPTS = [
    "Give me your response within <final_answer> tags.",
    "Read the input carefully and answer within <final_answer> tags.",
    "Think step by step, then put the final answer within <final_answer> tags.",
    "Solve the task and write only the answer between <final_answer> and </final_answer>.",
    "Explain your reasoning briefly. End with the answer inside <final_answer> tags.",
    "You are an expert in this task. Answer within <final_answer> tags.",
]


def load_data(args):
    df = pd.read_csv(args.data)
    if args.test:
        train, test = df, pd.read_csv(args.test)
    else:
        train = df.sample(frac=args.train_frac, random_state=args.seed)
        test = df.drop(train.index)
    if args.max_train:
        train = train.sample(n=min(args.max_train, len(train)), random_state=args.seed)
    if args.max_test:
        test = test.sample(n=min(args.max_test, len(test)), random_state=args.seed)
    for part in (train, test):
        part["y"] = part["y"].astype(str).str.lower()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_llm(model, url, key_env, kwargs_json, max_tokens, max_concurrency):
    key = os.environ.get(key_env) or os.environ.get("OPENAI_API_KEY")
    call_kwargs = json.loads(kwargs_json) if kwargs_json else None
    return APILLM(
        api_url=url,
        model_id=model,
        api_key=key,
        max_tokens=max_tokens,
        max_concurrent_calls=max_concurrency,
        call_kwargs=call_kwargs,
    )


def build_optimizer(name, predictor, task, meta_llm, initial_prompts):
    cls = OPTIMIZERS[name]
    prompts = initial_prompts[:1] if name == "opro" else initial_prompts
    return cls(predictor=predictor, task=task, meta_llm=meta_llm, initial_prompts=prompts, callbacks=[task.callback()])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV with columns x, y")
    ap.add_argument("--test", help="separate test CSV; default: hold out part of --data")
    ap.add_argument("--optimizer", choices=sorted(OPTIMIZERS), default="opro")
    ap.add_argument("--worker-model", required=True)
    ap.add_argument("--worker-url", required=True)
    ap.add_argument("--meta-model", help="model that writes new prompts (default: worker model)")
    ap.add_argument("--meta-url", help="default: --worker-url")
    ap.add_argument("--worker-kwargs", help="JSON passed to every worker request, e.g. extra_body")
    ap.add_argument("--meta-kwargs", help="JSON passed to every meta request")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-concurrency", type=int, default=16, help="parallel requests per model")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("-k", type=int, default=20, help="items per evaluation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--max-train", type=int)
    ap.add_argument("--max-test", type=int)
    ap.add_argument("--task-description", default="")
    ap.add_argument("--initial-prompts", help="text file, one prompt per line")
    ap.add_argument("--no-calibration", action="store_true", help="feed raw subset accuracy back instead")
    ap.add_argument("--out", help="write the final prompts and scores to this JSON file")
    args = ap.parse_args()

    train, test = load_data(args)
    worker = build_llm(
        args.worker_model, args.worker_url, "WORKER_API_KEY", args.worker_kwargs, args.max_tokens, args.max_concurrency
    )
    if args.meta_model or args.meta_url or args.meta_kwargs:
        meta = build_llm(args.meta_model or args.worker_model, args.meta_url or args.worker_url,
                         "META_API_KEY", args.meta_kwargs, args.max_tokens, args.max_concurrency)
    else:
        meta = worker

    classes = sorted(set(train["y"]) | set(test["y"]))
    # MarkerBasedPredictor only accepts word labels; otherwise the raw answer is compared
    predictor = MarkerBasedPredictor(worker, classes=classes if all(c.islower() for c in classes) else None)

    task = POESTask(
        ClassificationTask(train, task_description=args.task_description, x_column="x", y_column="y"),
        n_steps=args.steps,
        k=args.k,
        calibrate=not args.no_calibration,
        seed=args.seed,
    )
    if args.initial_prompts:
        with open(args.initial_prompts) as f:
            initial = [line.strip() for line in f if line.strip()]
    else:
        initial = DEFAULT_PROMPTS
    optimizer = build_optimizer(args.optimizer, predictor, task, meta, initial)
    prompts = optimizer.optimize(n_steps=args.steps)
    best = prompts[0]

    test_task = ClassificationTask(test, task_description=args.task_description, x_column="x", y_column="y")
    test_score = float(test_task.evaluate(best, predictor, eval_strategy="full").agg_scores[0])
    print("best prompt:", best)
    print(f"test accuracy: {test_score:.4f} ({len(test)} examples)")

    if args.out:
        record = {
            "args": vars(args),
            "best_prompt": str(best),
            "test_accuracy": test_score,
            "prompts": [str(p) for p in prompts],
            "train_scores": [float(s) for s in optimizer.scores],
        }
        with open(args.out, "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
