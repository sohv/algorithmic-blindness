# Research log

One entry per run, appended immediately after the run finishes — not later. Pull the metrics from
the structured output file the script wrote, not from stdout. Newest entries at the top.

Read this before starting a new experiment. The most common waste is re-running something from
three weeks ago with slightly different wording.

## 260804 — rebuilt around a condition grid after the UAI reviews

**What:** not an experiment. The pipeline was rebuilt so every reviewer objection is an axis of one
condition grid rather than a separate script: metadata richness (sparse/diagnostic/full), dataset
naming (real/anonymized), synthetic noise family (gaussian and three non-Gaussian), plus human and
`marginal`/`oracle` predictors, the Winkler interval score, and Wilson intervals with paired tests
on every claim. Fourteen scripts became seven.
**Result:** the ground-truth intervals in v1 were degenerate. The 100 runs per condition refit the
same array, and PC/FCI/LiNGAM/NOTEARS are all deterministic, so every run returned an identical
graph. Measured on an 8-node synthetic DAG, PC gave std 0.000 and CI width 0.000 with one distinct
F1 value across runs; with bootstrap resampling, std 0.124 and CI width 0.401. The paper's methods
section claims bootstrap resampling that the code never did. The 8-27x width-ratio claim was
computed against a zero-width denominator.
**Command:** n/a — no experiment was run, only the pipeline rebuilt.
**Output:** n/a

## 260804 — repo reorganised onto the research template

**What:** not an experiment. The codebase was restructured onto the research template: `src/` split
into pipeline stages, entry points extracted into `scripts/`, `.bif` networks moved to `data/raw/`,
and every stage given a `--output_dir` that writes `config.json` and `run.log` beside its results.
**Result:** three real gaps surfaced and were closed — the missing run-summarization step, an
evaluation stage reading a file format nothing wrote, and three disagreeing dataset-size tables. See
`docs/project_design_decisions.md`. No results were re-computed, so nothing in the paper changes.
**Command:** n/a
**Output:** n/a

Runs before this date predate the template and have no `results/raw/YYMMDD_*/` entry. The numbers
in the README's abstract come from those runs; re-running the pipeline under the new layout is what
puts them back on a traceable footing.

## YYMMDD — short description

**What:** one sentence on what was tested
**Result:** one sentence on what was found
**Command:**
uv run -m scripts.run_predictions --config_path configs/predictions.yaml --output_dir results/raw/YYMMDD_description_v1 --seed 42
**Output:** results/raw/YYMMDD_description_v1/responses.jsonl
