# Project design decisions

Background on this project: the question it answers, the prior results it builds on, and the
reasoning behind the approach taken.

Distinct from `decisions.md`, which is a dated log of individual pre-registered thresholds and
cut-offs. This file is the narrative; that one is the record.

## Why this question

LLMs are increasingly used to *choose* methods — which causal discovery algorithm to run, whether a
result is worth trusting. That use assumes the model has calibrated knowledge of how the methods
behave, not just declarative knowledge of what they are. The two come apart, and asking for a
numeric range is the cheapest way to separate them: a model can describe PC's assumptions perfectly
and still have no idea what F1 it will reach.

## Why ranges rather than point estimates

A point estimate can only be scored on distance, which conflates two failures: being wrong, and not
knowing you might be wrong. A range separates them. Coverage answers "did it contain the truth" and
width answers "at what cost" — a model can trivially reach 100% coverage by predicting [0, 1], and
the width comparison against the true CI is what stops that from looking like success.

## Why 100 runs per algorithm

The algorithms are stochastic — resampling from the DAG, and in NOTEARS's case the optimisation
itself. Without an empirical interval there is no ground truth to bracket, only a single draw. 100
runs is where the interval stops moving materially and the cost is still tolerable.

## Why three prompt formulations

The obvious objection to any result of this shape is "you asked badly". Three formulations that
differ in framing (direct, reasoning-first, confidence-interval) turn that objection into a
measurement: if the estimates are stable across wordings, the result is about the model; if they
are not, the coefficient of variation says so and that instability is itself the finding.

## Why benchmark and synthetic datasets together

The benchmark networks (asia, alarm, sachs, …) are in every textbook and almost certainly in
training data. The synthetic DAGs are generated fresh and cannot be. Running both is what makes the
memorization question answerable: a model that only performs on the memorable half is doing
retrieval. The perturbation test is the sharper version of the same idea, holding the data fixed
and changing only the name.

## Why the two halves are computed independently

The algorithmic ground truth never touches an LLM, and the LLM predictions never touch the
algorithms' output. They meet only at the comparison stage. This keeps the ground truth
uncontaminated and means either half can be re-run without invalidating the other.

## Rebuild, 2026-08-04

After the UAI 2026 rejection the repository was rebuilt around a condition grid rather than a chain
of scripts glued together by filenames. The reason is structural: every criticism the reviewers
raised requires a *new axis* — anonymized naming, non-Gaussian noise, richer metadata, human
predictors, alternative scoring rules — and the old layout encoded exactly one value of each axis at
every layer. A filename like `asia_pc_f1_ranges.json` cannot express
`anonymized × laplace × diagnostic`.

### What the reviews actually converged on

Four reviewers, one meta-review and one workshop reviewer raised overlapping objections. Stripped of
detail they are:

1. The prompt withheld what any predictor would need, so failure under it is uninformative.
2. The memorization evidence is indirect, and the synthetic design confounds it — linear-Gaussian
   data structurally disadvantages LiNGAM and favours NOTEARS, which is the exact pattern read as
   memorization.
3. There is no human baseline, so LLM-specific failure cannot be separated from task difficulty.
4. Coverage of the mean is not a proper scoring rule and the interval target was never stated.

Points 1 and 2 are the load-bearing ones: if the task is unanswerable, "blindness" is not a finding,
and if the LiNGAM collapse has an assumption-violation explanation, the memorization claim loses its
sharpest evidence. Both are now experiments rather than arguments.

### Defects found during the rebuild

1. **The ground-truth intervals were degenerate.** The 100 runs per condition refit the same array,
   and every algorithm in the suite is deterministic. Measured directly: zero variance, zero-width
   interval, one distinct value across all runs. The methods section claims bootstrap resampling;
   the code did none. This undercuts the width-ratio claim, which was computed against a
   zero-width denominator.
2. **The evaluation stage read a file format nothing wrote** — `*_llm_comparison.json` against a
   pipeline producing `*_aggregated.json`.
3. **Dataset sizes disagreed across three tables** — `survey` was 43 variables in one and 6 in
   another; the `.bif` says 6.

### Why the baselines changed

v1 compared against uniform-random and a two-rule heuristic. Both are weak: clearing them says
almost nothing. The rebuild adds `marginal`, which predicts from the leave-one-out empirical spread
of algorithm performance and knows nothing about the specific dataset, and `oracle`, which is the
true interval. `marginal` is the honest floor for a claim about conditioning on problem structure,
and `oracle` fixes the width scale that the width ratio is expressed in.
