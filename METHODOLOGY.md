# Algorithmic Blindness Testing Methodology

## Overview
End-to-end pipeline to quantify whether LLMs can accurately predict causal discovery algorithm performance across datasets. The pipeline spans 9 phases: from computing algorithmic ground truth through LLM querying, extraction, aggregation, comparison, statistical analysis, baseline comparison, prompt robustness analysis, and pattern matching analysis.

---

## Pipeline Architecture

```
Phase 1: Algorithmic Ground Truth
  Run 4 algorithms × 13 datasets × 100 runs = 5,200 executions
  Output: 52 variance files with mean & 95% CI
                        ↓
Phase 2: LLM Simulation
  Query 8 LLMs × 13 datasets × 4 algorithms × 3 formulations = 1,248 API calls
  Output: 1,248 raw response text files
                        ↓
Phase 3: Extraction
  Parse raw LLM responses into structured metric ranges
  Output: 156 extracted range files (52 combos × 3 formulations)
                        ↓
Phase 4: Aggregation
  Average predictions across 3 prompt formulations per model
  Output: 52 aggregated range files
                        ↓
Phase 5: Comparison
  Compute calibrated coverage and secondary metrics
  Output: 1,664 total comparisons (52 experiments × 8 models × 4 metrics)
                        ↓
Phase 6: Statistical Analysis
  Aggregate coverage by model, algorithm, dataset, metric
  Pairwise significance testing between algorithms
                        ↓
Phase 7: Baseline Comparison
  Compare LLMs against random and heuristic baselines
  Output: Per-model and per-baseline coverage scores
                        ↓
Phase 8: Prompt Robustness Analysis
  Coefficient of variation across prompt formulations
  Output: CV% per model-metric-experiment
                        ↓
Phase 9: Pattern Matching Analysis
  Claude cross-algorithm breakdown + cross-algorithm analysis
  Output: Evidence for algorithm-specific pattern matching
```

---

## Detailed Component Breakdown

### Phase 1: Algorithmic Baseline Computation
**Script**: `src/experiments/run_experiments.py`
**Analysis**: `src/experiments/analyze_results.py`
**Output**: `src/experiments/results/*_variance.json` (48 files)

**What it does**:
- Run causal discovery algorithm 100 times per (dataset, algorithm) pair
- Record each metric: precision, recall, F1, SHD
- Compute: mean, std dev, 95% CI bounds
- Store results for algorithmic ground truth

**Why 100 runs?**
- Stochastic algorithms (PC, NOTEARS) have variance
- 100 runs gives stable 95% CI estimates
- Sufficient for statistical significance testing

**Output structure**:
```json
{
  "dataset": "alarm",
  "algorithm": "fci",
  "num_runs": 100,
  "results": {
    "precision": {
      "mean": 0.37,
      "std": 0.08,
      "ci_95_lower": 0.28,
      "ci_95_upper": 0.45,
      "values": [0.3, 0.4, 0.35, ...]
    }
  }
}
```

**Datasets** (13 total):
- **Benchmark** (9): alarm, asia, cancer, child, earthquake, hepar2, insurance, sachs, survey
- **Synthetic** (4): synthetic_12, synthetic_30, synthetic_50, synthetic_60

**Algorithms** (4): PC, FCI, LiNGAM, NOTEARS

---

### Phase 2: LLM Simulation
**Script**: `src/llm/query_all_llms.py`
**Prompts**: `prompts/prompt_templates.py`
**Output**: Raw response text files

**What it does**:
- For each (dataset, algorithm, formulation) triple:
  1. Load algorithmic variance file from Phase 1
  2. Construct prompt with 3 variants (f1, f2, f3)
  3. Query LLM
  4. Save raw response text

**Prompt design** (3 formulations for robustness):
- **f1**: Direct question with metric names
- **f2**: Expanded description with algorithm intuition
- **f3**: Alternative phrasing emphasizing uncertainty

**Why multiple formulations?**
- Tests consistency across prompt variations
- Captures LLM sensitivity to wording
- Provides basis for robustness scoring in Phase 8

**Models** (8): claude, deepseek, deepseekthink, gemini3, gpt5, llama, qwen, qwenthink

**LLM Coverage**:
- 8 models × 13 datasets × 4 algorithms × 3 formulations = 1,248 API calls

---

### Phase 3: Extraction (LLM Response Parsing)
**Script**: `src/llm/extract_llm_ranges.py`
**Output**: `src/llm/results/extracted_ranges/*_f{1,2,3}_ranges.json` (156 files)

**Algorithm**:
1. **Verbose Reasoning Stripping** — DeepSeek Reasoner outputs massive thinking traces; extract only final prediction lines
2. **Regex-Based Metric Extraction** — Pattern matching for `[lower, upper]` ranges per metric
3. **Confidence Classification** — high/medium/low based on response clarity
4. **Validation** — Bounds checking (lower < upper, precision/recall in [0,1], SHD >= 0)

**Challenges handled**:
- Multiple range formats: `[0.4, 0.6]`, `(0.4-0.6)`, `0.4 to 0.6`
- Unit inconsistencies: SHD as float vs integer
- Confidence clamping: Remove out-of-bound predictions

---

### Phase 4: Aggregation (Formulation Averaging)
**Script**: `src/llm/aggregate_formulations.py`
**Output**: `src/llm/results/aggregated_ranges/*_aggregated.json` (52 files)

**Pipeline**:
1. Load all f1/f2/f3 extracted files for a (dataset, algorithm) pair
2. Group by model
3. For each model-metric combo:
   - Collect: `[f1_lower, f2_lower, f3_lower]`
   - Compute: `aggregated_lower = mean(f1, f2, f3)`
   - Same for upper bound
4. Record formulation values for transparency

**Why aggregation instead of single formulation?**
- **Robustness**: Reduces variance from specific prompt wording
- **Fairness**: Each formulation gets equal weight (not cherry-picked)
- **Interpretability**: Can later analyze which LLMs are sensitive to prompts

**Confidence after aggregation**:
- "low" if ANY formulation had low confidence
- "high" only if ALL formulations had high confidence
- Else "medium"

---

### Phase 5: Comparison & Calibrated Coverage Computation
**Script**: `src/llm/compare_llm_vs_algorithmic.py`
**Output**: `src/llm/results/comparisons/comparison_results.json`

**Core Comparison Logic**:
```
For each (dataset, algorithm, model) triple:
  1. Load LLM aggregated range: [llm_lower, llm_upper]
  2. Load algorithmic CI: [algo_lower, algo_upper]
  3. Extract algorithmic ground truth mean: algo_mean

  4. PRIMARY METRIC - Calibrated Coverage:
     calibrated_coverage = bool(llm_lower <= algo_mean <= llm_upper)

  5. CONTINUOUS SCORE (0.0 to 1.0):
     If algo_mean inside range:  score = 1.0 - (dist_to_center / half_width) * 0.5
     If algo_mean outside range: score = max(0, 1.0 - (overshoot / half_width))

  6. Secondary metrics: overlap, containment, Jaccard fraction, center distance
```

**Why Calibrated Coverage as Primary?**
- **Interpretable**: "Does the LLM's range contain the true mean?"
- **Strict**: Requires actual containment, not just overlap
- **UAI-standard**: Commonly used in uncertainty quantification literature
- **Actionable**: 15.9% means 84.1% of predictions are fundamentally wrong

**Total comparisons**: 1,664 (52 experiments × 8 models × 4 metrics)

---

### Phase 6: Statistical Analysis
**Script**: `src/experiments/analyze_results.py`
**Output**: `src/experiments/results/statistical_analysis_report.txt`, `src/experiments/results/explanatory_theory_report.txt`

**What it does**:
- Aggregate calibrated coverage by model, algorithm, dataset, metric
- Pairwise significance testing between algorithms (with Bonferroni correction)
- Explanatory factor analysis (graph complexity, sample size, dimensionality)

---

### Phase 7: Baseline Comparison
**Scripts**: `src/baselines/simple_baselines.py`, `src/baselines/compare_baselines_full.py`
**Output**: `src/baselines/baseline_comparison_full_results.json`

**Baselines**:
1. **Random Baseline**: Uniformly random `[lower, upper]` ranges for each metric
2. **Heuristic Baseline**: Simple rules based on problem characteristics (sample size, variable count)

**Purpose**: Establish minimum performance thresholds. If LLMs perform at or below random, their predictions are no better than guessing.

---

### Phase 8: Prompt Robustness Analysis
**Script**: `src/llm/compute_prompt_robustness.py`
**Output**: `src/llm/results/robustness_analysis/robustness_summary.json`, `src/llm/results/robustness_analysis/robustness_report.txt`

**What it computes**:
- Coefficient of Variation (CV%) across the 3 prompt formulations per model-metric-experiment:
  ```
  CV% = (std_dev / mean) * 100
  ```
- Computed for both midpoint and width of predicted ranges
- Measures how sensitive each LLM is to prompt wording

---

### Phase 9: Pattern Matching Analysis
**Scripts**: `src/llm/analyze_pattern_matching.py`, `src/llm/cross_algorithm_analysis.py`
**Output**: `src/llm/results/comparisons/claude_pattern_matching_analysis.txt`, `src/llm/results/comparisons/cross_algorithm_analysis.txt`

**Claude Pattern Matching** (`analyze_pattern_matching.py`):
- Breaks down Claude's calibrated coverage by algorithm AND dataset type (real vs synthetic)
- Tests whether Claude's synthetic boost is algorithm-specific (evidence of pattern matching) or consistent (evidence of understanding)

**Cross-Algorithm Analysis** (`cross_algorithm_analysis.py`):
- Tracks coverage by algorithm, dataset type, and LLM for all models
- Computes average synthetic boost per algorithm across all LLMs
- Identifies which algorithms have exploitable patterns in synthetic data

**Key Insight**: If an LLM truly understood algorithms, its performance boost/drop on synthetic data would be consistent across all algorithms. Algorithm-specific variation proves pattern matching over genuine understanding.

---

## Metric Selection Justification: Precision, Recall, F1, SHD

These 4 metrics are the **standard evaluation suite in causal discovery literature** (UAI, ICML, JMLR):

| Metric | Measures | Why Important |
|--------|----------|---------------|
| Precision | Fraction of predicted edges that are correct | Avoids spurious causal claims |
| Recall | Fraction of true edges recovered | Measures discovery completeness |
| F1 | Harmonic mean of precision & recall | Balanced summary statistic |
| SHD | Edit distance from predicted to true DAG | Penalizes direction errors (crucial for causality) |

Together they form **complete and non-redundant coverage**: precision + recall for edge detection quality, F1 for balance, SHD for topology and directionality.

---

## Key Methodological Choices & Justifications

### Choice 1: Three Formulations (f1, f2, f3)
**Rationale**: Prompt engineering heavily affects LLM outputs. Three diverse formulations test consistency, sensitivity, and aggregation robustness.

### Choice 2: Aggregation via Mean
**Rationale**: Mean treats all formulations equally, avoids cherry-picking. More transparent and defensible than median or best-case.

### Choice 3: Calibrated Coverage as Primary Metric
**Rationale**: Directly answers "Can the LLM's range bound the truth?" Binary, strict, UAI-standard, and actionable.

### Choice 4: 100 Algorithm Runs per Experiment
13 datasets × 4 algorithms × 100 runs = 5,200 total algorithm executions. Ensures stable 95% CI estimates.

### Choice 5: Including Synthetic Datasets
Tests generalization to unseen data distributions. Benchmark networks are in literature (potential training data overlap); synthetics are procedurally generated and never in training data.

---

## Data Flow Summary

| Phase | Input | Processing | Output | Size |
|-------|-------|-----------|--------|------|
| 1 | 13 datasets | Run 100x each algorithm | Variance CIs | 48 variance files |
| 2 | Variance + prompts | Query 8 LLMs | Raw responses | 1,248 files |
| 3 | 1,248 raw files | Regex extract ranges | f1/f2/f3 ranges | 156 files |
| 4 | 156 formulation files | Average f1/f2/f3 | Aggregated ranges | 52 files |
| 5 | 52 aggregated + 48 variance | Compare LLM vs algo | Comparisons | 1,664 metrics |
| 6 | Comparison results | Statistical aggregation | Coverage analysis | Reports |
| 7 | Comparison + variance | Random & heuristic baselines | Baseline comparison | JSON |
| 8 | Raw responses | CV across formulations | Robustness analysis | JSON + report |
| 9 | Comparison results | Pattern matching tests | Cross-algo analysis | Reports |

---

## Reproducibility

```bash
# Phase 1: Compute algorithmic ground truth (~2h)
python src/experiments/run_experiments.py --runs 100

# Phase 2: Query LLMs (~3h with API delays)
python src/llm/query_all_llms.py

# Phase 3: Parse responses (~1m)
python src/llm/extract_llm_ranges.py

# Phase 4: Average formulations (<1s)
python src/llm/aggregate_formulations.py

# Phase 5: Compare LLM vs algorithmic (<1s)
python src/llm/compare_llm_vs_algorithmic.py

# Phase 6: Generate algorithmic analysis plots and reports
python src/experiments/analyze_results.py

# Phase 7: Baseline comparison
python src/baselines/compare_baselines_full.py

# Phase 8: Prompt robustness analysis
python src/llm/compute_prompt_robustness.py

# Phase 9: Pattern matching analysis
python src/llm/analyze_pattern_matching.py
python src/llm/cross_algorithm_analysis.py

# Generate all plots
python src/experiments/analyze_results.py       # Algorithmic performance plots
python src/llm/plot_llm_results.py              # LLM analysis plots
```

**Total runtime**: ~6 hours (mostly API calls and algorithm runs)

---

## Limitations & Caveats

1. **LLM Coverage**: 8 models tested; newer models not included
2. **Prompt Design**: Formulations are handcrafted; different prompts might yield different results
3. **Single Aggregation Method**: Only mean aggregation tested; median or weighted schemes not explored
4. **Reasoning Traces**: DeepSeek reasoning traces extracted but not deeply analyzed
5. **Metric Scope**: Only precision, recall, F1, SHD; other metrics (AUC, MCC) not tested
