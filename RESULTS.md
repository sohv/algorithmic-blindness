# Results

All results from the LLM algorithmic blindness study. 8 LLMs tested across 13 datasets and 4 causal discovery algorithms, with 5,200 algorithm runs establishing ground truth.

---

## 1. Primary Result: Calibrated Coverage

**Overall: 15.9% calibrated coverage** (264/1,664 comparisons). LLMs fail to contain the true algorithmic mean 84.1% of the time.

### Coverage by Model

| Model | Calibrated Coverage | Comparisons | Mean Score | Median Score |
|-------|-------------------:|------------:|----------:|-------------:|
| Claude | 39.4% | 82/208 | 0.442 | 0.531 |
| GPT-5 | 15.4% | 32/208 | 0.217 | 0.000 |
| DeepSeek-Think | 14.9% | 31/208 | 0.174 | 0.000 |
| DeepSeek | 14.4% | 30/208 | 0.198 | 0.000 |
| Qwen-Think | 13.9% | 29/208 | 0.191 | 0.000 |
| Gemini 3 | 13.0% | 27/208 | 0.182 | 0.000 |
| Llama | 10.1% | 21/208 | 0.152 | 0.000 |
| Qwen | 5.8% | 12/208 | 0.068 | 0.000 |

![Calibrated Coverage by Model](src/llm/results/plots/01_calibrated_coverage_primary.png)

### Coverage by Algorithm

| Algorithm | Coverage |
|-----------|--------:|
| NOTEARS | 20.7% |
| LiNGAM | 20.0% |
| PC | 11.5% |
| FCI | 11.3% |

### Coverage by Metric

| Metric | Coverage |
|--------|--------:|
| Recall | 18.8% |
| F1 | 16.3% |
| SHD | 14.9% |
| Precision | 13.5% |

### Coverage by Dataset

| Dataset | Type | Coverage |
|---------|------|--------:|
| Asia | Benchmark | 23.4% |
| Cancer | Benchmark | 21.9% |
| Synthetic-12 | Synthetic | 20.3% |
| Alarm | Benchmark | 18.8% |
| Insurance | Benchmark | 18.8% |
| Survey | Benchmark | 18.0% |
| Child | Benchmark | 17.2% |
| Sachs | Benchmark | 16.4% |
| Earthquake | Benchmark | 14.1% |
| Synthetic-30 | Synthetic | 13.3% |
| Hepar2 | Benchmark | 10.9% |
| Synthetic-50 | Synthetic | 7.0% |
| Synthetic-60 | Synthetic | 6.2% |

---

## 2. Benchmark vs Synthetic: Memorization Evidence

Coverage on benchmark networks (known in literature) vs synthetic networks (never in training data):

| Dataset Type | Coverage | Comparisons |
|-------------|--------:|-----------:|
| Benchmark (9 datasets) | 17.7% | 204/1,152 |
| Synthetic (4 datasets) | 11.7% | 60/512 |

![Benchmark vs Synthetic Coverage](src/llm/results/plots/02_real_vs_synthetic_ablation.png)

---

## 3. LLM Scalability: Coverage vs Network Size

Coverage collapses as network size increases, suggesting LLMs cannot reason about larger structures:

| Synthetic Network | Nodes | Coverage |
|------------------|------:|---------:|
| Synthetic-12 | 12 | 20.3% |
| Synthetic-30 | 30 | 13.3% |
| Synthetic-50 | 50 | 7.0% |
| Synthetic-60 | 60 | 6.2% |

Coverage drops from 20.3% to 6.2% as graph size scales from 12 to 60 nodes -- a 69% relative decline.

![Scalability Analysis](src/llm/results/plots/04_scalability_analysis.png)

---

## 4. LiNGAM Failure Mode Analysis

LiNGAM-specific analysis showing how LLMs mispredict performance for this functional causal model:

![LiNGAM Failure Mode](src/llm/results/plots/03_lingam_failure_mode.png)

---

## 5. Claude Pattern Matching Analysis

### Cross-Algorithm Breakdown (Claude Only)

Claude's calibrated coverage broken down by algorithm and dataset type:

| Algorithm | Real Coverage | Synthetic Coverage | Difference | Pattern Match? |
|-----------|------------:|------------------:|----------:|:--------------|
| FCI | 25.0% | 43.8% | +18.8% | YES |
| LiNGAM | 47.2% | 31.2% | -16.0% | No |
| NOTEARS | 44.4% | 68.8% | +24.3% | YES |
| PC | 27.8% | 43.8% | +16.0% | YES |

**Summary**:
- Average synthetic boost across algorithms: +10.8%
- Max boost: +24.3% (NOTEARS)
- Min boost: -16.0% (LiNGAM)
- Range of variation: 40.3%

**Key finding**: The algorithm-specific nature of Claude's synthetic boost (ranging from -16.0% to +24.3%) rules out a general synthetic-data simplicity effect and demonstrates exploitation of algorithm-specific patterns -- the hallmark of pattern matching, not understanding.

### Cross-Algorithm Analysis (All Models)

Average synthetic boost per algorithm across all 8 LLMs:

| Algorithm | Avg Synthetic Boost | Range Variation |
|-----------|-------------------:|---------------:|
| FCI | -0.5% | 34.7% |
| LiNGAM | -23.2% | 44.4% |
| NOTEARS | +1.7% | 56.9% |
| PC | -2.0% | 24.3% |

The large variation in synthetic boosts per algorithm proves LLMs learned algorithm-specific patterns, not general algorithmic principles.

---

## 6. Baseline Comparison

LLMs compared against random and heuristic baselines on the full 1,664-comparison dataset:

| Predictor | Calibrated Coverage | Mean Score |
|-----------|-------------------:|----------:|
| Random Baseline | 36.5% | 0.409 |
| Claude | 39.4% | 0.442 |
| Heuristic Baseline | 32.7% | 0.356 |
| GPT-5 | 15.4% | 0.217 |
| DeepSeek-Think | 14.9% | 0.174 |
| DeepSeek | 14.4% | 0.198 |
| Qwen-Think | 13.9% | 0.191 |
| Gemini 3 | 13.0% | 0.182 |
| Llama | 10.1% | 0.152 |
| Qwen | 5.8% | 0.068 |

**Key finding**: Only Claude exceeds the random baseline. All other LLMs perform worse than uniformly random range predictions, meaning 7 of 8 frontier LLMs provide predictions less useful than random guessing.

---

## 7. Prompt Robustness

Coefficient of Variation (CV%) across the 3 prompt formulations measures how sensitive each LLM is to prompt wording. Higher CV = less robust.

Example CV% ranges observed:
- Midpoint CV: 2.7% (very stable) to 44.4% (highly variable)
- Width CV: 0.0% (identical widths) to 50.8% (highly variable)

Full robustness data: `src/llm/results/robustness_analysis/robustness_summary.json`

---

## 8. Algorithmic Ground Truth

### Algorithm Performance (Mean F1 across all datasets)

| Algorithm | Mean F1 | Best Dataset |
|-----------|--------:|-------------|
| PC | 0.427 | Asia, Cancer, Earthquake, Child |
| FCI | 0.426 | Asia, Cancer, Earthquake, Child |
| LiNGAM | 0.301 | Survey (0.724) |
| NOTEARS | 0.299 | Synthetic-30 (0.681), Synthetic-12 (0.615) |

### Pairwise Statistical Significance (Bonferroni-corrected)

| Comparison | Mean Diff | P-value (corrected) | Cohen's d | Significant? |
|------------|----------:|--------------------:|----------:|:------------|
| FCI vs LiNGAM | +0.125 | 0.013 | 0.842 | YES |
| LiNGAM vs PC | -0.126 | 0.010 | -0.877 | YES |
| FCI vs NOTEARS | +0.127 | 1.000 | 0.489 | NO |
| FCI vs PC | -0.002 | 1.000 | -0.132 | NO |
| LiNGAM vs NOTEARS | +0.003 | 1.000 | 0.009 | NO |
| NOTEARS vs PC | -0.128 | 1.000 | -0.495 | NO |

2 of 6 pairwise comparisons are statistically significant (33.3%).

### Explanatory Factors

| Factor | Impact Score | Correlation |
|--------|------------:|----------:|
| Graph Complexity (Sparsity/Density) | 42.1% | 0.421 |
| Sample Size | 12.7% | 0.127 |
| Problem Dimensionality | 11.3% | 0.113 |
| Algorithm-Dataset Interaction | 0.5% | 0.000 |
| Noise Sensitivity | 0.0% | -0.000 |

Graph complexity is the primary driver of algorithm performance, explaining ~42% of variation.

---

## 9. Experimental Scale

| Component | Count |
|-----------|------:|
| Datasets | 13 (9 benchmark + 4 synthetic) |
| Algorithms | 4 (PC, FCI, LiNGAM, NOTEARS) |
| Algorithm runs | 5,200 (100 per dataset-algorithm) |
| LLM models | 8 |
| Prompt formulations | 3 |
| LLM API calls | 1,248 |
| Total comparisons | 1,664 |

---

## Figures

### Algorithmic Performance

| | |
|:---:|:---:|
| ![F1 Comparison](src/experiments/results/plots/01_f1_comparison.png) | ![Precision Recall F1](src/experiments/results/plots/02_precision_recall_f1.png) |
| F1 Score by Algorithm | Precision, Recall, F1 by Algorithm |
| ![SHD Comparison](src/experiments/results/plots/03_shd_comparison.png) | ![F1 Confidence Intervals](src/experiments/results/plots/04_f1_confidence_intervals.png) |
| Structural Hamming Distance | F1 with 95% Confidence Intervals |
| ![Dataset Heatmap](src/experiments/results/plots/05_dataset_heatmap.png) | ![F1 Distribution](src/experiments/results/plots/06_f1_distribution.png) |
| F1: Dataset vs Algorithm Heatmap | F1 Distribution by Algorithm |

### Multi-Metric Dataset View

![Metrics by Dataset](src/experiments/results/plots/05b_metrics_by_dataset.png)

### LLM Algorithmic Blindness

| | |
|:---:|:---:|
| ![Calibrated Coverage](src/llm/results/plots/01_calibrated_coverage_primary.png) | ![Real vs Synthetic](src/llm/results/plots/02_real_vs_synthetic_ablation.png) |
| Calibrated Coverage by Model (Primary Result) | Benchmark vs Synthetic Coverage |
| ![LiNGAM Failure](src/llm/results/plots/03_lingam_failure_mode.png) | ![Scalability](src/llm/results/plots/04_scalability_analysis.png) |
| LiNGAM Failure Mode Analysis | Coverage vs Network Size |

---

## Data Files

| File | Description |
|------|-------------|
| `src/experiments/results/*_variance.json` | Algorithmic ground truth (48 files) |
| `src/llm/results/extracted_ranges/*_f{1,2,3}_ranges.json` | Per-formulation extracted ranges (156 files) |
| `src/llm/results/aggregated_ranges/*_aggregated.json` | Aggregated LLM predictions (52 files) |
| `src/llm/results/comparisons/comparison_results.json` | Full comparison results |
| `src/llm/results/comparisons/comparison_results_all.json` | All comparison results (alternate format) |
| `src/baselines/baseline_comparison_full_results.json` | Baseline comparison data |
| `src/llm/results/robustness_analysis/robustness_summary.json` | Prompt robustness CV% data |
| `src/llm/results/robustness_analysis/robustness_report.txt` | Robustness report |
| `src/llm/results/comparisons/claude_pattern_matching_analysis.txt` | Claude pattern matching report |
| `src/llm/results/comparisons/cross_algorithm_analysis.txt` | Cross-algorithm analysis report |
| `src/experiments/results/statistical_analysis_report.txt` | Pairwise significance tests |
| `src/experiments/results/explanatory_theory_report.txt` | Explanatory factor analysis |
