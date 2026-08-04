# Pre-registered decisions

Log every threshold, cut-order decision, or design choice fixed *before*
seeing results. Date each entry.

The entries below were read off the code during the 2026-08-04 reorganisation. They record what the
code currently does; they are not a claim about when each choice was originally made. Anything fixed
from here on gets dated at the time it is fixed.

## 260804 — thresholds carried over from the pre-reorganisation code

**Runs per algorithm:** 100.

**Confidence interval:** empirical 2.5th/97.5th percentile across runs, not a parametric interval.
The per-run metric distributions are not assumed normal.

**Calibrated coverage:** binary — the true mean falls inside the predicted range, or it does not.
The continuous 0-1 score alongside it is 1.0 at the range centre, 0.5 at either edge, and decays
linearly outside, hitting 0 at one half-width beyond the edge.

**Sampling temperature:** 0.1 for every provider. The experiment measures movement across prompt
wordings, so sampling noise is held down deliberately.

**Aggregation across formulations:** arithmetic mean of the three lower bounds and of the three
upper bounds, and only when at least 2 of the 3 formulations parsed. Aggregated confidence is the
worst of the three (`low` beats `medium` beats `high`).

**Memorization test:** one-tailed paired t-test, alternative `original > perturbed`, on the
precision pairs. Significance at α = 0.05. Effect size reported as Cohen's d on the paired
differences. Strength bands on the mean drop: <0.05 none, <0.10 or |d|<0.3 weak, <0.20 or |d|<0.6
moderate, above that strong.

**Perturbation parsing:** the first number in the response is read as precision, per the enforced
four-line output format. Pairs where either value falls outside [0, 1] are dropped rather than
clamped — out of range means the model ignored the format, and clamping would invent a datapoint.

**Pattern-matching flag:** a real-vs-synthetic coverage gap above 5 percentage points is treated as
a signal worth reporting. What the analysis turns on is the *spread* of that gap across algorithms,
not its sign.

**Baseline scoring:** the random and heuristic baselines are scored on exactly the set of
(experiment, metric) pairs the LLMs were scored on, seeded at 42. Scoring them on a different subset
would make "beats random" a comparison between two different populations.

**Failed runs and failed calls are recorded, not dropped.** A failed algorithm run is written with
`failed: true` and excluded from the summary statistics; a failed LLM call is written with a null
response. A shrunk output file would read as a smaller experiment rather than a partial failure.

## 260804 — decisions fixed by the rebuild

Set before re-running anything, so they are pre-registered rather than chosen after seeing results.

**Ground truth comes from bootstrap resamples.** Each run resamples rows with replacement at the
original row count and refits. Without this the algorithms are deterministic and the interval is a
point. `--bootstrap false` exists only to demonstrate the difference and must never produce a
reported number.

**A zero-width ground-truth interval invalidates the run.** `ground_truth_report.json` counts them.
A width ratio against a zero-width interval is undefined, not large.

**Winkler interval score at alpha = 0.05** is reported alongside coverage on every prediction.
Coverage alone is not a proper scoring rule: `[0, 1]` scores perfectly on every bounded metric.

**SHD is normalised by the maximum possible edge count** for every width, bias and error. A raw SHD
width of 20 means something different on 8 nodes than on 70.

**Bias is signed so positive always means optimistic**, including for SHD where lower is better.

**Every proportion carries a Wilson interval; every pairwise claim carries McNemar on the shared
conditions.** Families of tests are corrected with Holm.

**The prompt states the prediction target.** The interval must contain the mean over repeated runs,
not the spread of individual runs and not the model's uncertainty about itself.

**Anonymization changes the name and nothing else.** Removing the name entirely would change the
sentence structure too and confound the contrast. Neutral ids are assigned by sorted position.

**Domain descriptions are excluded from every prompt.** Domain is a name-linked retrieval cue, so
including it in the real arm only would contaminate the naming contrast.

**The noise family is scaled to unit variance in every case.** Only the distribution shape varies,
so the gaussian-vs-non-gaussian contrast is not confounded by noise scale.

**The graph is drawn from a separate random stream from the noise**, so the same structure is reused
across noise families and noise is the only thing that changes.

**`marginal` is the reference floor, not `uniform`.** Leave-one-out over every other condition.
Beating uniform random is trivial; beating marginal is the claim worth testing.

**A prediction with no ground truth is an error.** It means the two halves of the experiment ran
over different grids, which is never a thing to skip past.
