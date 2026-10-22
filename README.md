<h2 align="center">
  <a href="https://arxiv.org/abs/2604.11328">Select Smarter, Not More? Prompt-Aware Evaluation Scheduling with Submodular Guarantees</a>
</h2>

<h5 align="center">
  If our project helps you, please give us a star ⭐ and cite our <a href="#citation">paper</a>!
</h5>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2604.11328-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2604.11328)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-blue.svg)](https://arxiv.org/abs/2604.11328)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</h5>

## News

- 🎉 Our paper is accepted to NeurIPS 2026!
- **[2026.10]** Code released.

## Why Do We Need POES?

Automatic prompt optimizers such as OPRO and EvoPrompt propose many candidate prompts, and every candidate has to be scored before the optimizer can decide what to keep. Scoring each candidate on the full training set is expensive, so in practice each one is scored on a small evaluation subset, often 20 examples.

Which 20 examples matters a lot. A fixed random subset never adapts to the candidates: once the leading prompts all solve the easy items, those items stop telling them apart. Re-drawing the subset every time makes scores jumpy, and a prompt can look strong simply because it drew easy questions. In both cases the optimizer receives noisy feedback and may keep the wrong prompt.

**POES (Prompt-Aware Online Evaluation Scheduling)** chooses the subset online. It looks at the prompts the optimizer currently has, picks the examples that best separate the leading candidates while still covering the training pool, changes the subset only a little at a time, and puts the scores of prompts evaluated on different subsets onto one scale.

## How POES Works

### Prompt-Aware Discrimination

POES keeps a 1PL item response theory (IRT) model of every observed outcome: each prompt has an ability, each example a difficulty. An example is useful when the two strongest prompts are predicted to disagree on it, measured by the symmetric KL divergence between their predicted success probabilities. This is the classical adaptive-testing criterion (Fisher information at the ability midpoint) applied to the two current leaders.

### Coverage of the Training Pool

Picking only the most discriminative examples tends to concentrate the subset on a narrow region of the data. POES adds a facility-location coverage term over TF-IDF similarity, so the subset stays representative of the whole pool. Discrimination plus coverage is a monotone submodular objective, so building the first subset greedily carries the usual (1 - 1/e) guarantee.

### Warm-Start Swap Updates

Instead of re-solving the subset from scratch, POES starts from the previous subset and makes a few one-for-one swaps, accepting a swap only if it improves the objective by more than a switching cost. Between two consecutive calls the subset changes by at most 2B items, where B is the swap budget.

### Adaptive Controller and Auto Warmup

The first two or three evaluations use random subsets while the IRT estimates are still unreliable. After that, a controller sets the coverage weight, swap budget, and switching cost from optimization progress and the recent score trend: more exploration early, more stability late.

### Calibrated Scoring

Because the subset evolves, prompts end up scored on different examples, and raw subset accuracies are not directly comparable. POES refits the IRT model on all outcomes observed so far and reports each prompt's predicted accuracy on the same reference set (every example evaluated so far). All retained prompts are rescored after every optimizer step. This reuses outcomes that were already collected and needs no extra LLM calls.

## Features

- Works with OPRO, EvoPrompt-GA, and EvoPrompt-DE through [promptolution](https://github.com/gepromptet/promptolution), without modifying the optimizers.
- The scheduler itself has no LLM dependency; `select()` / `observe()` / `calibrated_scores()` can be plugged into any optimization loop.
- No extra model calls: selection and calibration only reuse outcomes the optimizer already paid for.
- Any OpenAI-compatible endpoint can be used as worker or meta model, including local vLLM servers.

## Supported Optimizers

| Optimizer | `--optimizer` | Note |
| --- | --- | --- |
| OPRO | `opro` | starts from one initial prompt |
| EvoPrompt (genetic algorithm) | `evoprompt_ga` | starts from a population of prompts |
| EvoPrompt (differential evolution) | `evoprompt_de` | starts from a population of prompts |

## Directory Specification

```text
POES/
├── poes/
│   ├── scheduler.py            # POES: selection, controller, warmup, calibrated scores
│   ├── irt.py                  # online 1PL IRT and the batch refit used for calibration
│   ├── facility.py             # facility-location greedy and warm-start swaps
│   ├── similarity.py           # TF-IDF similarity between training inputs
│   └── integrations/
│       └── promptolution.py    # POESTask wrapper and step callback
├── run.py                      # main entry: OPRO / EvoPrompt with POES on a CSV dataset
├── examples/
│   └── simulate.py             # synthetic benchmark, no LLM needed
└── tests/
```

## Environment Setup

### Requirements

- Python 3.10, 3.11, or 3.12 (promptolution 2.3 does not support 3.13 yet)

Install POES together with the promptolution integration:

```bash
git clone https://github.com/T-Lab-CUHKSZ/POES.git
cd POES
pip install -e ".[promptolution]"
```

or install the dependencies directly:

```bash
pip install -r requirements.txt
```

If you only need the scheduler in your own loop, `pip install -e .` installs it with numpy, scipy, and scikit-learn only (Python 3.9+).

## API Configuration

POES uses two LLM roles:

| Role | Purpose | Arguments |
| --- | --- | --- |
| worker | answers the task inputs with the candidate prompt | `--worker-model`, `--worker-url` |
| meta | writes new candidate prompts | `--meta-model`, `--meta-url` (default: same as worker) |

API keys are read from environment variables:

```bash
export OPENAI_API_KEY="sk-..."      # used for both roles
# or separately
export WORKER_API_KEY="..."
export META_API_KEY="..."
```

### Local LLM Support

Any OpenAI-compatible server works. For example, with vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct --port 8000

python run.py --data data/navigate.csv \
  --worker-model Qwen/Qwen2.5-7B-Instruct --worker-url http://localhost:8000/v1
```

For reasoning models that put their answer into a separate reasoning field, turn thinking off through `--worker-kwargs`, e.g.

```bash
--worker-kwargs '{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'
```

## Data Preparation

`run.py` reads a CSV file with an input column `x` and a label column `y`:

```text
x,y
"If you follow these instructions, do you return to the starting point? Take 3 steps. ...",yes
...
```

Labels are lowercased. Word labels (`yes`/`no`, `positive`/`negative`, option letters) are matched against the text between `<final_answer>` tags; for other labels, such as numbers, describe the label set with `--task-description`. Without `--test`, 80% of the rows form the optimization pool and 20% are held out (`--train-frac`, `--max-train`, `--max-test` change this).

## Usage

### Quick Test

Run the tests and the synthetic benchmark, neither needs an API key:

```bash
pytest tests
python examples/simulate.py --runs 20
```

`simulate.py` builds a toy world of prompts and items, runs a small hill-climbing optimizer with POES, and reports the true accuracy of the returned prompt with raw subset scores and with calibrated scores.

### Running Experiments

```bash
python run.py --data <CSV> --worker-model <MODEL> --worker-url <URL> [OPTIONS]
```

Required arguments:

| Argument | Description |
| --- | --- |
| `--data` | CSV with columns `x`, `y` |
| `--worker-model` | model id of the worker LLM |
| `--worker-url` | base URL of the worker endpoint |

Optional arguments:

| Argument | Description |
| --- | --- |
| `--optimizer` | `opro` (default), `evoprompt_ga`, or `evoprompt_de` |
| `--steps N` | number of optimizer steps (default 10) |
| `-k N` | examples per evaluation (default 20) |
| `--seed N` | random seed for the data split and the scheduler (default 42) |
| `--test CSV` | separate test file instead of holding out part of `--data` |
| `--meta-model`, `--meta-url` | a different model for writing prompts |
| `--worker-kwargs`, `--meta-kwargs` | JSON added to every request, e.g. `extra_body` |
| `--max-concurrency N` | parallel requests per model (default 16); lower it if the endpoint rate-limits |
| `--task-description TEXT` | short description shown to the optimizer |
| `--initial-prompts FILE` | starting prompts, one per line |
| `--no-calibration` | feed raw subset accuracy back to the optimizer |
| `--out FILE` | save the final prompts and scores as JSON |

### Examples

```bash
# OPRO with POES
python run.py --data data/navigate.csv --worker-model <MODEL> --worker-url <URL>

# EvoPrompt-GA with POES, 10 examples per evaluation
python run.py --data data/navigate.csv --optimizer evoprompt_ga -k 10 \
  --worker-model <MODEL> --worker-url <URL>

# Separate worker and meta models
python run.py --data data/navigate.csv \
  --worker-model Qwen/Qwen2.5-7B-Instruct --worker-url http://localhost:8000/v1 \
  --meta-model gpt-4o-mini --meta-url https://api.openai.com/v1

# Raw subset accuracy instead of calibrated scores
python run.py --data data/navigate.csv --no-calibration --worker-model <MODEL> --worker-url <URL>
```

### Using POES in Your Own Code

With promptolution:

```python
from poes.integrations.promptolution import POESTask

task = POESTask(ClassificationTask(train_df, x_column="x", y_column="y"), n_steps=10, k=20)
optimizer = OPRO(predictor=predictor, task=task, meta_llm=llm,
                 initial_prompts=[...], callbacks=[task.callback()])
best = optimizer.optimize(n_steps=10)[0]
```

`POESTask` wraps a promptolution task and expects per-example scores in [0, 1]; we have tested it with `ClassificationTask`. Extra keyword arguments go to `POES`.

In any other loop:

```python
from poes import POES

poes = POES(train_inputs)                          # list of input strings
idx = poes.select(k=20, progress=step / n_steps)   # indices into train_inputs
outcomes = [score(prompt, train_inputs[i]) for i in idx]   # values in [0, 1]
poes.observe(prompt, idx, outcomes)
scores = poes.calibrated_scores(candidate_prompts)
```

## FAQ

**Q: Does POES make more LLM calls than a random subset?**

A: No. Every candidate is still scored on k examples. Selection and calibration only use the outcomes of those calls.

**Q: Which hyperparameters should I change?**

A: Usually only `k`. The other defaults (coverage weight 0.8 to 0.4, swap budget 25% to 10% of k, switching cost 0.05, warmup of 2 to 3 evaluations) are the ones used in the paper and are listed in the `POES` docstring.

**Q: Can I turn calibration off?**

A: Yes, `--no-calibration` in `run.py` or `POESTask(..., calibrate=False)`. Scores are then plain subset accuracies.

**Q: How do I use a different similarity, e.g. sentence embeddings?**

A: Pass a precomputed `(n, n)` similarity matrix: `POES(train_inputs, similarity=sim)`.

**Q: How do I add a new task type?**

A: Build the promptolution task and wrap it with `POESTask`. For a custom optimizer, call `select()` before scoring a prompt, `observe()` after, and use `calibrated_scores()` whenever prompts are compared.

## Acknowledgement

We are grateful for the following awesome projects:

- [promptolution](https://github.com/gepromptet/promptolution)
- [OPRO](https://github.com/google-deepmind/opro)
- [EvoPrompt](https://github.com/beeevita/EvoPrompt)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [vLLM](https://github.com/vllm-project/vllm)

## Citation

If you find this project helpful, please consider citing our work:

```bibtex
@misc{ma2026selectsmartermorepromptaware,
      title={Select Smarter, Not More: Prompt-Aware Evaluation Scheduling with Submodular Guarantees}, 
      author={Xiaoyu Ma and Yiwen Li and Haoyue Liu and Zhichao Wang and Ye Chen and Yongxin Guo and Xiaoying Tang},
      year={2026},
      eprint={2604.11328},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.11328}, 
}
```
