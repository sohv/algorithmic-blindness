# Experimental design

## Question

Can a frontier LLM predict how a causal discovery algorithm will actually perform on a given
dataset? Not "what does PC do" — the models can all recite that — but the quantitative question:
what precision, recall, F1 and SHD will PC reach on *this* dataset, with a range that contains the
true value.

## Design

Two halves, computed independently, compared at the end.

**Algorithmic ground truth.** Four algorithms (PC, LiNGAM, FCI, NOTEARS) x 13 datasets
(9 benchmark bayesian networks sampled from `.bif` files, 4 synthetic random DAGs), run 100 times
each. Each run's precision, recall, F1 and SHD are recorded; the 100 runs are summarized into a
mean and an empirical 95% interval. This is the value the LLM has to bracket.

**LLM predictions.** Each model is asked, for each (dataset, algorithm) pair, to give a predicted
range for the same four metrics. Asked three times, once per prompt formulation, and the three
ranges are averaged so no result rests on one wording.

**Score.** Calibrated coverage: the share of predictions whose range contains the true algorithmic
mean. Secondary: MAE of the range midpoint against the true mean, mean range width against the true
CI width, and the Spearman correlation between predicted and actual algorithm rankings per dataset.

## Controls

The headline result only means something against a floor and with the confounds ruled out, so:

- **Random baseline** — uniformly random ranges. If an LLM does not clear this, it carries no
  information. Expected to land near 15-20%.
- **Heuristic baseline** — two rules on (n_samples, n_variables): more samples is better, more
  variables is worse. Asks whether the LLM beats a lookup table.
- **Prompt robustness** — coefficient of variation of each model's estimate across the three
  formulations. A high CV means the result is a fact about the prompt, not about the model.
- **Perturbation test** — rename each benchmark dataset to a name with no lexical cues
  (`asia` -> `Network-A`) and re-ask. Algorithmic difficulty is unchanged, so a significant drop
  implicates recall of the dataset name rather than reasoning about the algorithm. One-tailed
  paired t-test on the precision pairs.
- **Real vs synthetic split** — coverage broken out by dataset type, per algorithm. A model that
  understood the algorithms should show a similar gap everywhere. A boost that varies by algorithm
  is algorithm-specific pattern matching.

## Prompt formulations

1. **Direct** — asks straight for the performance range.
2. **Step-by-step** — walks the model through the algorithm's assumptions first, then asks.
3. **Meta-knowledge** — frames the task as estimating a confidence interval.

All three carry the same dataset facts (domain, node count, sample count, edge count, data type).
Temperature is pinned at 0.1: the experiment measures movement across wordings, so sampling noise
has to be held down or it contaminates the signal.

## The grid

| Axis | Values | Answers |
| --- | --- | --- |
| dataset | 9 benchmark networks, 4 synthetic | is it memorable? |
| noise | native; gaussian, uniform, laplace, exponential | is LiNGAM's assumption met? |
| algorithm | pc, fci, lingam, notears | — |
| metric | precision, recall, f1, shd | — |
| naming | real, anonymized | does the name carry the answer? |
| metadata_level | sparse, diagnostic, full | was the prompt the problem? |
| formulation | 1 direct, 2 step-by-step, 3 interval estimation | is it about the wording? |
| predictor | 8 models, 4 baselines, humans | is it LLM-specific? |

## Where each piece lives

| Stage | Module | Entry point |
| --- | --- | --- |
| The grid itself | `src/conditions.py` | — |
| Sample benchmark networks | `src/data/benchmarks.py` | — |
| Synthetic DAGs per noise family | `src/data/networks.py` | — |
| Dataset diagnostics for the prompt | `src/data/diagnostics.py` | — |
| Bootstrap algorithm runs | `src/algorithms/discovery.py` | `scripts/build_ground_truth.py` |
| Score a learned graph | `src/metrics/graph.py` | — |
| Runs to confidence intervals | `src/algorithms/ground_truth.py` | `scripts/build_ground_truth.py` |
| Prompt assembly | `src/generation/prompts.py` | `scripts/show_prompts.py` |
| Query the models | `src/generation/query.py` | `scripts/run_predictions.py` |
| Parse and aggregate wordings | `src/metrics/extraction.py` | `scripts/parse_predictions.py` |
| Baselines and human ingestion | `src/metrics/predictors.py` | `scripts/make_elicitation_sheet.py` |
| Score a prediction | `src/metrics/scoring.py` | `scripts/score.py` |
| Intervals and tests | `src/metrics/stats.py` | — |
| Every analysis | `src/metrics/analysis.py` | `scripts/analyze.py` |
| Figures | `src/visualize/figures.py` | `scripts/plot.py` |
