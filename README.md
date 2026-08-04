# Algorithmic Blindness in Large Language Models

[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=flat&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=OmJK0GK8_MI) [![ArXiv](https://img.shields.io/badge/arXiv-2602.21947-b31b1b.svg)](https://arxiv.org/abs/2602.21947)

Can a frontier LLM predict how an algorithm will perform on a given problem instance? Not "what does
quicksort do" — every model can recite that — but the quantitative question: how many comparisons,
with an interval that contains the truth.

Seven algorithms across three domains, chosen to span how knowable the answer is: causal discovery
(no closed form), shortest path (a closed form, exact for Dijkstra), and sorting (a closed form in
every textbook). **Sorting is the sharpest test.** Quicksort's expected comparison count on random
input is 2(n+1)H_n − 4n; on a sorted array under a Lomuto pivot it is exactly n(n−1)/2. A model that
cannot predict those cannot claim the task was unreasonable — which is the defence the causal-only
version of this study had no answer to.

**Status: rebuilt, not yet re-run.** v1 was rejected at UAI 2026 and accepted as a COLM 2026 SciFM
workshop poster. This repository is the rebuild that answers the reviews. **No number from the v1
paper should be cited from this repository until the pipeline is re-run** — see
[what changed and why](#what-changed-and-why).

## The design

Everything hangs off one condition grid and two tidy tables.

- A **condition** is `(domain, instance, variant, algorithm)` — the thing being predicted.
- A **prompt spec** is `(naming, metadata_level, formulation)` — how it gets asked.
- A **predictor** is a model, a baseline, or a human. They all emit the same schema.

| Table | One row per | Written by |
| --- | --- | --- |
| `ground_truth.jsonl` | domain × instance × variant × algorithm × metric | `build_ground_truth` |
| `predictions.jsonl` | predictor × condition × metric × prompt axes | `parse_predictions`, `score` |
| `scored.jsonl` | the join of the two | `score` |

Every analysis is a groupby over `scored.jsonl`. Adding a reviewer's experiment is adding a value to
an axis, not adding a script.

## What changed and why

Three defects in v1 were found during the rebuild. The first is the serious one.

### 1. The ground-truth confidence intervals were degenerate

v1 ran each algorithm 100 times on the **same array**. PC, FCI, DirectLiNGAM and `notears_linear`
are deterministic functions of their input, so all 100 runs returned the same graph. Measured on an
8-node synthetic DAG:

| | mean F1 | std | CI width | distinct values across runs |
| --- | --- | --- | --- | --- |
| v1 (no resampling) | 0.000 | 0.000 | 0.000 | 1 |
| rebuilt (bootstrap) | 0.231 | 0.124 | 0.401 | 6 |

The paper's methods section says "100 times with bootstrap resampling (Efron & Tibshirani, 1994)".
The code did no resampling. This bears directly on the claim that *LLM ranges are 8–27× wider than
true confidence intervals*: that ratio was computed against intervals of width zero.
`ground_truth_report.json` now counts degenerate intervals, and any above zero invalidates the run.

### 2. The evaluation stage read a format nothing wrote

It globbed `*_llm_comparison.json`; the pipeline produced `*_aggregated.json`. The chain was broken
in the middle.

### 3. Dataset sizes disagreed across three tables

`survey` was 43 variables in the baselines and 6 in the prompts. The `.bif` file says 6. There is
now one table, `BENCHMARK_NODES` in [conditions.py](src/conditions.py).

## What the reviews asked for, and where it now lives

| Review point | How it is addressed | Where |
| --- | --- | --- |
| Prompts too sparse to answer (o67S, u9TJ, u68Q) | `metadata_level` axis: `sparse` → `diagnostic` → `full` | [prompts.py](src/generation/prompts.py), [diagnostics.py](src/data/diagnostics.py) |
| Anonymized benchmarks never run (j7Mn) | `naming` axis: `real` vs `anonymized` | [conditions.py](src/conditions.py) |
| Gaussian synthetic confounds LiNGAM (KsDD) | `noise` axis: gaussian, uniform, laplace, exponential | [networks.py](src/data/networks.py) |
| No human baseline (KsDD) | elicitation sheet in, scored as a predictor | [predictors.py](src/metrics/predictors.py) |
| Coverage is not a proper scoring rule (j7Mn) | Winkler interval score on every prediction | [scoring.py](src/metrics/scoring.py) |
| Interval target ambiguous (j7Mn) | every prompt states it: the interval must contain the **mean** | [prompts.py](src/generation/prompts.py) |
| Random baseline underspecified (j7Mn) | both readings implemented and named | [predictors.py](src/metrics/predictors.py) |
| No bias analysis (j7Mn) | signed bias, oriented so positive is always optimistic | [scoring.py](src/metrics/scoring.py) |
| No CIs on coverage; 2.9pp gap untested | Wilson intervals, binomial and McNemar tests, Holm correction | [stats.py](src/metrics/stats.py) |
| Baselines too weak | added `marginal` (leave-one-out), `analytic` (textbook formula) and `oracle` | [predictors.py](src/metrics/predictors.py) |
| One domain can't support a domain-agnostic claim (j7Mn, o67S, u68Q) | three domains spanning no-closed-form to textbook-closed-form | [domains.py](src/domains.py) |

Two baselines carry the argument. **`marginal`** predicts from the overall spread of performance and
nothing about the specific instance — beating uniform random is trivial, beating `marginal` is what
"conditioning on problem structure" would have to mean. **`analytic`** is the published closed form,
and it is deliberately never fitted: it declines wherever no formula exists (duplicate-heavy sorting
input, `successful_relaxations`, Bellman-Ford without a measured hop count) rather than guessing. On
the conditions it does speak to it covers 100% of measured means, verified in
[test_algorithms.py](tests/test_algorithms.py).

## Project structure

```
algorithmic-blindness/
├── src/
│   ├── domains.py      the three domains: algorithms, metrics, instances, scales
│   ├── conditions.py   the experiment grid over them
│   ├── data/           benchmarks.py (.bif), networks.py, instances.py, diagnostics.py, io.py
│   ├── algorithms/     discovery, sorting, shortest_path, runner (dispatch), ground_truth
│   ├── generation/     models.py (registry), openrouter.py, prompts.py, query.py
│   ├── metrics/        extraction, scoring, stats, predictors, analysis, report, graph, io
│   ├── visualize/      style.py, figures.py
│   └── utils/          config.py, logging.py, seed.py, git.py
├── scripts/            one thin entry point per stage
├── configs/            smoke, ground_truth, predictions, replication
├── data/raw/           11 benchmark networks in .bif, never edited
├── results/            RESULTS.md (append-only record) + raw/<run>/ per run, never gitignored
├── docs/               design rationale, pre-registered decisions
└── tests/
```

## Domains and algorithms

| Domain | Algorithms | Metrics | Instances × variants |
| --- | --- | --- | --- |
| `causal_discovery` | `pc`, `lingam`, `notears` | precision, recall, f1, shd | 9 benchmark networks (native) + 4 synthetic × 4 noise families |
| `shortest_path` | `dijkstra`, `bellman_ford` | relaxations, successful_relaxations, nodes_settled | 3 graph sizes × sparse/dense |
| `sorting` | `quicksort`, `mergesort` | comparisons, moves, max_depth | 3 array sizes × random/sorted/reverse/few_unique |

FCI was dropped from causal discovery: it is a second constraint-based method alongside PC, and its
PAG output isn't directly comparable to a DAG adjacency matrix, so its low v1 coverage was partly a
measurement artifact. The three that remain are three distinct paradigms — constraint-based,
functional/non-Gaussian, and continuous optimisation.

Every metric is an exact operation count, never a timing. Counts are machine-independent, exactly
reproducible and analytically predictable; wall-clock would confound the measurement with the
hardware and hand a model a fair excuse for missing.

## Models

Seven models through OpenRouter — one key, one client, one retry path — spanning **six labs**, so a
shared failure can't be explained by shared lineage. One frontier model; every other is capped at
**$8/Mtok of output**:

| Key | OpenRouter id | Lab | Tier | $/Mtok out | Added | temp | seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `opus` | `anthropic/claude-opus-5` | Anthropic | frontier | 25.00 | 2026-07-24 | yes | **no** |
| `gemini` | `google/gemini-3.6-flash` | Google | mid | 7.50 | 2026-07-21 | yes | yes |
| `gpt` | `openai/gpt-5.6-terra` | OpenAI | mid | 6.00 | 2026-07-09 | **no** | yes |
| `grok` | `x-ai/grok-4.5` | xAI | mid | 6.00 | 2026-07-08 | yes | yes |
| `qwen` | `qwen/qwen3.8-max` | Alibaba | mid | 6.00 | 2026-08-03 | yes | yes |
| `haiku` | `anthropic/claude-haiku-4.5` | Anthropic | mid | 5.00 | 2025-10-15 | yes | **no** |
| `deepseek` | `deepseek/deepseek-v4-pro` | DeepSeek | mid | 0.87 | 2026-04-24 | yes | yes |

**`opus` and `haiku` are the tier contrast.** Same lab, same training lineage, same sampling
controls, one capability step apart — so a gap between them is a capability effect rather than a
difference between labs. That is a cleaner comparison than the aggregate frontier-vs-mid test,
which pools models whose price does not reliably track capability across the $0.87–$7.50 band
(Gemini 3.6 *Flash* is a small fast model at the top of it; DeepSeek V4 Pro is a 1.6T-parameter
flagship at the bottom). Both contrasts are reported; the within-lab pair is the one to trust.

`check_models` validates every slug against the live catalog and checks that the frontier model
costs more per output token than every mid-tier one. A test fails if any *mid-tier* model is priced
above $8/Mtok output or if a second frontier model appears, so cost cannot creep back in unnoticed.

The `temp` and `seed` columns matter more than they look. OpenRouter **accepts an unsupported
sampling control and silently drops it**, so sending `temperature=0.1` to a model that ignores it
buys nothing while looking like a control. Each model declares what it honours, `check_models`
verifies that against the live catalog, the client withholds the rest, and every response records
the controls actually applied. Six of seven honour temperature; `gpt` is pinned by `seed` alone.

Adding a model is a row in [models.py](src/generation/models.py).

## Setup

```bash
uv sync
cp .env.example .env      # OPENROUTER_API_KEY and HF_TOKEN
uv run -m scripts.check_models --probe true
```

**Run `check_models` before any run that spends money.** OpenRouter's catalog moves; a stale slug
otherwise fails at the first paid call, deep into a run.

## Where results land

One directory per run under `results/raw/`, named
`YYMMDD_<stage>[_<narrowed axes>]_v<N>/`:

```
results/raw/260804_ground_truth_v1/          the full grid
results/raw/260804_ground_truth_sorting_v1/  narrowed to one domain
results/raw/260804_predictions_sorting_gpt_v1/
```

An axis that ran in full contributes nothing to the name, so a bare
`260804_predictions_v1` means every domain and every model. Narrow it and the directory says so.

Three rules make this safe to rerun:

- **`{date}` in a config, never a literal date.** `configs/*.yaml` write
  `output_dir: results/raw/{date}_predictions`, stamped at run time. A hardcoded date silently
  reuses yesterday's directory.
- **`_vN` bumps past any directory that already holds a run.** Rerunning the same command writes
  `_v2`, never on top of `_v1`. `results/raw/` is append-only.
- **Input paths are never defaulted in a config.** `--ground_truth_dir` and `--predictions` are
  command-line arguments, so a run can't silently read a stale input.

Only the two run-creating stages — `build_ground_truth` and `run_predictions` — resolve a fresh
directory. `parse_predictions`, `score`, `analyze` and `plot` write where they are told, which is
normally back into the run directory they are derived from.

`tests/test_run_dir.py` enforces all of it.

## Running it

Start with the smoke config: one dataset, one algorithm, five runs.

```bash
uv run -m scripts.build_ground_truth --config_path configs/smoke.yaml
```

### 1. Ground truth

Runs each algorithm `--n_runs` times on **bootstrap resamples** and summarises them into an
empirical 95% interval.

**Input:** `data/raw/<dataset>.bif` for benchmarks; synthetic DAGs generated from `--seed`.
**Output:** `runs.jsonl` (one row per run), `ground_truth.jsonl` (`dataset`, `noise`, `algorithm`,
`metric`, `mean`, `std`, `ci_lower`, `ci_upper`, `ci_width`, `n_runs`, `n_failed`),
`diagnostics.jsonl` (the dataset properties the richer prompts expose), `ground_truth_report.json`.

```bash
uv run -m scripts.build_ground_truth \
  --output_dir results/raw/260804_ground_truth_v1 \
  --domains causal_discovery sorting \
  --instances asia synthetic_12 array_1k \
  --variants native gaussian random \
  --n_runs 100 \
  --seed 42
```

Check `n_degenerate_intervals` in the report before going further. Anything above zero means
nothing varied across runs.

### 2. Predictions

Queries every model over the grid. Preview a prompt first with
`uv run -m scripts.show_prompts --instance asia --algorithm lingam --metadata_level full`.

**Input:** `diagnostics.jsonl` from step 1 — required for any prompt above `sparse`.
**Output:** `responses.jsonl` (`id`, condition axes, prompt axes, `model`, `prompt`, `response`,
`error`), one raw `.txt` per call in `generations/` (gitignored), and `grid.json` recording the
scope of the run.

```bash
uv run -m scripts.run_predictions \
  --ground_truth_dir results/raw/260804_ground_truth_v1 \
  --output_dir results/raw/260804_predictions_v1 \
  --models gpt grok \
  --namings real anonymized \
  --metadata_levels sparse diagnostic full \
  --num_tasks 10 \
  --seed 42
```

The printed request count is `conditions × prompt specs × models`. Check it before a full run.

### 3. Parse

Pulls intervals out of the raw text and averages across wordings into a formulation-0 row.

**Input:** `responses.jsonl`.
**Output:** `predictions.jsonl` (`predictor`, `predictor_kind`, condition axes, prompt axes,
`lower`, `upper`, `confidence`, `parsed`), `parse_report.json`.

```bash
uv run -m scripts.parse_predictions \
  --dataset_path results/raw/260804_predictions_v1/responses.jsonl \
  --output_dir results/raw/260804_predictions_v1
```

Read the parse rate before trusting any coverage number: a model answering 70% of the time is not
comparable to one answering every time.

### 4. Score

Adds the baselines (and any human sheet), joins to ground truth, scores everything.

**Input:** `predictions.jsonl`, `ground_truth.jsonl`, optionally a filled elicitation csv.
**Output:** `scored.jsonl` with `covers_mean`, `covers_ci`, `coverage_score`, `interval_score`,
`width`, `width_ratio`, `bias`, `abs_error` — each also normalised — plus `score_summary.json`.

```bash
uv run -m scripts.score \
  --predictions results/raw/260804_predictions_v1/predictions.jsonl \
  --ground_truth_dir results/raw/260804_ground_truth_v1 \
  --output_dir results/raw/260804_scored_v1 \
  --seed 42
```

A prediction with no matching ground truth is an error, not a skip — it means the two halves ran
over different grids.

### 5. Analyse and plot

Appends one section to [results/RESULTS.md](results/RESULTS.md) — the chronological record of every
run, with its coverage table, parse-rate warnings and every contrast with its paired test.

**Output:** `analysis.json`, `tables/*.csv` and `.tex`, `figures/*.pdf` and `.png`, plus the appended
section in `results/RESULTS.md`.

```bash
uv run -m scripts.analyze \
  --scored results/raw/260804_scored_v1/scored.jsonl \
  --output_dir results/raw/260804_scored_v1 \
  --grid_path results/raw/260804_predictions_v1/grid.json

uv run -m scripts.plot --scored results/raw/260804_scored_v1/scored.jsonl --output_dir results/raw/260804_scored_v1
```

A figure needing an axis the run did not vary is skipped and logged, not faked.

## Where results live

**Nothing under `results/` is gitignored.** Every run artifact is committed — raw responses, per-run
logs, config and grid json — so any number in the paper traces to a file in git.

`results/RESULTS.md` is append-only and newest-last. A result that turns out to be wrong gets a new
section saying so; past sections are never edited or reordered.

The same tree is mirrored to the `algorithmic-blindness-results` Hugging Face dataset as the citable
copy a reader can pull without cloning the code. Uploads are additive — a published run is skipped
unless `--overwrite` is passed, so a re-sync never quietly replaces a published run.

```bash
uv run -m scripts.sync_results --dry_run true    # show what would upload
uv run -m scripts.sync_results --private true
```

### Human baseline

```bash
uv run -m scripts.make_elicitation_sheet \
  --ground_truth_dir results/raw/260804_ground_truth_v1 \
  --output_dir results/raw/260804_human_v1 \
  --n_conditions 20 --seed 42
```

An expert fills in `lower`/`upper`; pass the csv to `score --human_predictions`. Their rows are
scored by the same code as the models, on the same conditions.

## Reproducing the v1 cell

`configs/replication.yaml` pins the grid to what v1 actually ran: real names, sparse prompts, three
wordings, Gaussian synthetic only. Run it against bootstrapped ground truth to see how much of the
original result was the degenerate-interval bug.

## Tests

```bash
uv run -m pytest tests/ -v -s
```

Tests hitting a provider are skipped when the key is absent. Everything else runs unguarded.

## Known gaps

- **Two unused networks.** `data/raw/barley.bif` and `hailfinder.bif` are present but not in
  `BENCHMARK_DATASETS`, which is why the count is 9 benchmark + 4 synthetic.
- **`src/generation/cache.py` and `batch.py` are unused** — the template's async path, kept because
  `CLAUDE.md` documents them.
- **The OpenRouter slugs are unverified offline.** `scripts/check_models.py` is the check; run it
  before spending money.
- **`src/metrics/prompt_variance.py`** is legacy plotting code, superseded by
  `analysis.prompt_robustness`. Kept until its figures are confirmed unneeded.

## Citation

```bibtex
@article{venkatesh2026algorithmic,
  title={Algorithmic Blindness in Large Language Models: A Calibration Study of Performance Prediction},
  author={Venkatesh, Sohan and Kurapath, Ashish Mahendran and Melkote, Tejas},
  journal={arXiv preprint arXiv:2602.21947},
  year={2026}
}
```
