# LLM Causal Discovery Evaluation: Project Documentation

## Executive Summary

This research project uses **causal discovery algorithms as a structured testbed** to understand whether Large Language Models (LLMs) can reason about algorithmic performance in rigorous, measurable domains. Rather than simply testing LLM accuracy, this work compares **LLM prediction ranges with algorithmic confidence intervals derived from 100+ rigorous experimental runs per algorithm**. This allows us to probe whether LLMs have genuine algorithmic understanding or merely pattern-match from training data.

**Central Research Question**: Can LLMs demonstrate genuine understanding of algorithmic performance characteristics, as evidenced by their ability to predict how different algorithms behave across diverse datasets and problem structures?

---

## Project Overview

### What the Project Intends to Do

The project tests a fundamental question about LLM reasoning: **Can LLMs predict algorithmic performance in a structured, measurable domain?** Using causal discovery algorithms as a testbed, the research aims to:

1. **Establish ground truth** through rigorous variance analysis of multiple algorithms on diverse datasets (100+ runs per algorithm)
2. **Query multiple LLMs** on algorithmic performance predictions using different prompt framings
3. **Compare LLM estimates** against algorithmic confidence intervals (CIs) to measure prediction accuracy
4. **Measure prediction robustness** across different prompt formulations (to distinguish understanding from prompt-sensitivity)
5. **Analyze algorithmic reasoning** by testing if LLMs:
   - Distinguish between fundamentally different algorithm types (constraint-based vs. functional)
   - Understand algorithm-specific assumptions and tradeoffs
   - Recognize how problem characteristics (size, density, dimensionality) affect performance
   - Adjust estimates appropriately across different datasets

This approach treats causal discovery algorithms as a **structured probe** for understanding LLM capabilities in domains requiring detailed technical knowledge.

---

## How It Works

### Methodology Overview

The project follows a four-phase experimental pipeline:

```
Phase 1: Variance Analysis (2-3 hours)
    ↓
    Run each algorithm 100+ times on each dataset
    Extract performance metrics from each run
    Compute 95% confidence intervals (bootstrap-based)
    ↓
Phase 2: Parse & Validate LLM Responses (1 minute)
    ↓
    Test LLM connection with small sample queries
    Verify response parsing accuracy
    ↓
Phase 3: Query LLMs (12-16 hours)
    ↓
    Test 5 LLMs × 11 datasets × 4 algorithms × 3 prompt formulations
    = 660 total LLM queries
    Extract numerical ranges from diverse response formats
    ↓
Phase 4: Visualize & Analyze Results (5 minutes)
    ↓
    Create publication-quality comparison plots
    Compute overlap statistics
    Generate summary tables
    ↓
Output: Confidence intervals, LLM predictions, overlap analysis
```

### Key Workflow Components

#### Phase 1: Algorithm Variance Analysis

**Purpose**: Establish statistical ground truth for algorithm performance.

**Process**:
- Load 11 datasets (real-world, benchmark networks, synthetic)
- For each dataset:
  - Run each of 4 algorithms N times (N=100+) with different random seeds
  - Extract 4 metrics per run:
    - **Precision**: Proportion of predicted edges that are correct
    - **Recall**: Proportion of true edges recovered
    - **F1**: Harmonic mean of precision and recall
    - **SHD** (Structural Hamming Distance): Total edge errors (missing + extra + reversed)
  - Compute statistics:
    - Sample mean and standard deviation
    - **95% Bootstrap Confidence Interval** (2.5th-97.5th percentile from 10,000 bootstrap samples)
    - Median, min, max values

**Output**: `results/variance/{dataset}_{algorithm}_variance.json` files containing full statistical summaries.

#### Phase 2: LLM Response Parsing

**Purpose**: Extract numerical estimates from diverse LLM response formats.

**Supported Formats**:
- Structured: `"Precision: (0.6, 0.8)"`
- Range: `"Precision: 0.6-0.8"` or `"Precision: 0.6 to 0.8"`
- Percentage: `"Precision: 60%-80%"`
- Natural language: `"I expect precision to be between 0.6 and 0.8"`
- JSON: `{"precision": [0.6, 0.8]}`

**Output**: Normalized metric ranges matching algorithmic CI format.

#### Phase 3: LLM Querying

**Purpose**: Obtain LLM predictions using different prompt framings.

**Process**:
- For each combination of:
  - Dataset (11 options)
  - Algorithm (4 options)
  - Prompt formulation (3 options)
  - LLM (5 models)
- Generate prompt with dataset/algorithm context
- Query LLM with specified temperature=0.1 (low variance mode)
- Parse response and extract ranges
- Store results by LLM model

**LLM Configuration**:
- `temperature=0.1`: Low randomness for reproducibility
- `max_tokens=1024`: Sufficient for structured responses
- `max_retries=3`: Resilience to API failures
- `retry_delay=5s`: Backoff between retries

#### Phase 4: Overlap Analysis & Visualization

**Comparison Metrics**:

Three types of overlap are tested:

1. **Simple Overlap**: Do algorithmic CI and LLM range intersect?
   ```
   Overlaps = NOT(CI_upper < LLM_lower OR CI_lower > LLM_upper)
   ```

2. **CI Contains Range**: Does algorithmic CI fully contain LLM range?
   ```
   CI_lower ≤ LLM_lower AND CI_upper ≥ LLM_upper
   ```

3. **Range Contains CI**: Does LLM range fully contain algorithmic CI?
   ```
   LLM_lower ≤ CI_lower AND LLM_upper ≥ CI_upper
   ```

**Visualizations**:
- Error bar plots: Algorithmic means with 95% CIs vs. LLM ranges
- Heatmaps: Overlap statistics across datasets and algorithms
- Distribution plots: Performance variance by algorithm complexity

---

## Causal Discovery Algorithms Evaluated

### 1. PC (Peter-Clark) Algorithm
- **Type**: Constraint-based (independence testing)
- **Principle**: Searches for causal structure by testing conditional independencies
- **Strengths**: Well-understood, interpretable, requires no functional form assumptions
- **Weaknesses**: Sensitive to independence test choice, limited by sample size
- **Assumptions**: Causal faithfulness, causal sufficiency, no selection bias
- **Runtime**: O(d^3) where d = number of variables

### 2. LiNGAM (Linear Non-Gaussian Acyclic Model)
- **Type**: Functional causal model
- **Principle**: Exploits non-Gaussianity to identify linear causal relationships
- **Strengths**: Identifies fully oriented edges (no ambiguity), handles linear relationships well
- **Weaknesses**: Requires non-Gaussian data, assumes linearity
- **Assumptions**: Acyclic structure, non-Gaussian noise, linear relationships
- **Special Feature**: Can produce fully directed graphs (no undirected edges)

### 3. FCI (Fast Causal Inference) Algorithm
- **Type**: Constraint-based (extension of PC)
- **Principle**: Relaxes causal sufficiency assumption, handles latent confounders
- **Strengths**: Handles latent variables, more general than PC
- **Weaknesses**: More computationally expensive, more ambiguous edges (partial directions)
- **Assumptions**: Causal faithfulness, no selection bias (latent confounders allowed)
- **Output**: Partially directed acyclic graphs (PDAGs)

### 4. NOTEARS (No Tears Acyclic Model)
- **Type**: Functional causal model with continuous optimization
- **Principle**: Solves optimization problem with acyclicity constraint
- **Strengths**: Modern approach, handles various data types, differentiable
- **Weaknesses**: May not converge to global optimum, assumes specific functional form
- **Assumptions**: Acyclic structure, differentiable score functions
- **Runtime**: Faster than constraint-based methods for large graphs

---

## Datasets (6 Total - UAI Submission Scope)

> **Note**: Original plan included 11 datasets. Reduced to 6 for UAI submission deadline.
> 
> **Dropped**: Titanic, Credit, Wine, Earthquake, Survey (5 datasets)  
> **Kept**: Asia, Sachs, Cancer, Child, Synthetic-12, Synthetic-30 (6 datasets)  
> **Rationale**: Focus on core testbed with mix of real and synthetic data; Child (20 vars) tests complex causal systems and fills gap between small (5-12 vars) and large (30 vars) networks.

### Benchmark Networks (from BNLearn)

| Network | Variables | Samples | Application | Notes |
|---------|-----------|---------|-------------|-------|
| **Asia** | 8 | 10,000 | Medical diagnosis | Classic pedagogical example |
| **Cancer** | 5 | 5,000 | Disease diagnosis | Simple causal structure |
| **Sachs** | 11 | ~1,000 | Protein signaling | Real experimental data |
| **Child** | 20 | 5,000 | Pediatric disease | Complex causal system - tests scalability |

### Synthetic Datasets

| Dataset | Variables | Nodes | Samples | Generation |
|---------|-----------|-------|---------|------------|
| **Synthetic-12** | 12 | 12 | 1,000 | Random DAG + linear causal model |
| **Synthetic-30** | 30 | 30 | 1,000 | Larger random DAG test |

**Selection Rationale** (UAI Submission Scope):
- **6 datasets** chosen for time constraints (UAI deadline)
- Mix of sizes (5-30 variables) tests scalability: Cancer (5), Asia (8), Sachs (11), Child (20), Synthetic-12 (12), Synthetic-30 (30)
- Mix of real (Asia, Cancer, Sachs, Child) and synthetic data
- Mix of densities tests algorithm sensitivity
- Sufficient for robust statistical testing (6 data points per comparison)

---

## Large Language Models (LLMs) Tested

| LLM | Provider | Context | Reasoning | Used |
|-----|----------|---------|-----------|------|
| **GPT-4** | OpenAI | Latest, strong reasoning | Excellent | ✓ |
| **DeepSeek** | DeepSeek | Open-weight, specialized | Strong | ✓ |
| **Claude 3.5 Sonnet** | Anthropic | Constitutional AI | Excellent | ✓ |
| **Gemini 1.5 Pro** | Google | Multimodal, large context | Very good | ✓ |
| **Llama 3** | Meta/Together | Open-weight model | Good | ✓ |

**Selection Rationale**:
- Mix of commercial (GPT-4, Claude, Gemini) and open-weight (DeepSeek, Llama) models
- Diverse providers and training approaches
- Representative of current state-of-the-art
- Different model sizes and capabilities

---

## Evaluation and Methodology

### Experimental Design

**Independent Variables**:
- Dataset (6)
- Algorithm (4)
- Prompt formulation (3)
- LLM (5)

**Total Queries**: 6 × 4 × 3 × 5 = **360 LLM queries**

**Total Algorithm Runs**: 6 × 4 × 100 = **2,400 algorithm runs**

### Metrics Evaluated

#### 1. Precision
- **Definition**: Proportion of predicted edges that are correct
- **Formula**: TP / (TP + FP)
- **Range**: [0, 1]
- **Interpretation**: Lower = more false positive edges predicted

#### 2. Recall
- **Definition**: Proportion of true edges that are recovered
- **Formula**: TP / (TP + FN)
- **Range**: [0, 1]
- **Interpretation**: Lower = more edges missed

#### 3. F1-Score
- **Definition**: Harmonic mean of precision and recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: [0, 1]
- **Interpretation**: Balanced performance metric

#### 4. Structural Hamming Distance (SHD)
- **Definition**: Number of edge differences between true and learned graphs
- **Calculation**: Missing edges + Extra edges + Reversed edges
- **Range**: [0, ∞)
- **Note**: Lower is better
- **Scale-aware**: Normalized by maximum possible edges (n×(n-1)/2)

### Three Prompt Formulations

Prompts are designed to test robustness of LLM predictions across different framings:

#### Formulation 1: Direct Question
```
You are an expert in causal discovery algorithms.

Dataset: {dataset_name}
- Domain: {domain}
- Variables: {n_nodes}
- Samples: {n_samples}

Algorithm: {algorithm_name}

Task: Estimate the algorithm's performance on this dataset.

Provide your estimates as ranges [lower, upper] for:
- Precision
- Recall
- F1-score
- SHD
```

**Goal**: Test straightforward performance estimation.

#### Formulation 2: Step-by-Step Reasoning
```
[Similar setup as Formulation 1, but includes:]

Reasoning steps:
1. Does the dataset satisfy the algorithm's assumptions?
2. How do sample size and complexity affect performance?
3. What performance range is realistic given these factors?
```

**Goal**: Test whether guided reasoning improves accuracy/calibration.

#### Formulation 3: Meta-Knowledge / Confidence Interval Framing
```
Task: Predict the 95% confidence interval for each metric 
based on your knowledge of {algorithm_name}'s behavior 
on similar datasets.

[Includes algorithm assumptions and dataset characterization]
```

**Goal**: Test if explicit CI framing aligns LLM thinking with statistical concepts.

**Expected Outcome: Variance Analysis**:
- **Robust predictions**: <20% variance across formulations (LLM estimates consistent)
- **Sensitive predictions**: >20% variance (LLM estimates depend on framing)
- **Significant finding**: Either pattern reveals important insights about LLM reasoning

---

## Statistical Tests and Analysis

### 1. Bootstrap Confidence Interval Computation

**Purpose**: Establish ground truth performance ranges from stochastic algorithm runs.

**Method**:
```python
For each algorithm on each dataset:
  1. Run algorithm N times with different random seeds (N = 100+)
  2. Extract metric values: M = [m₁, m₂, ..., m_N]
  3. For each metric:
     a. Compute sample mean: μ = mean(M)
     b. Compute sample std: σ = std(M)
     c. Bootstrap resample 10,000 times:
        - M_boot_i = random_sample(M, size=N, replace=True)
        - μ_boot_i = mean(M_boot_i)
     d. Extract percentiles: CI = [percentile(μ_boot, 2.5), 
                                    percentile(μ_boot, 97.5)]
  4. Output: {mean, std, CI_lower, CI_upper, median, min, max}
```

**Rationale**: Bootstrap CIs are distribution-free and robust to non-normality.

### 2. Overlap Analysis

**Purpose**: Quantify agreement between LLM predictions and algorithmic CIs.

**Metrics**:

a) **Intersection over Union (IoU)**:
```
IoU = (LLM_range ∩ CI) / (LLM_range ∪ CI)
```
- Range: [0, 1]
- Interpretation: Fraction of combined range that overlaps

b) **Coverage Fraction**:
```
Coverage = (LLM_range ∩ CI) / CI
```
- Interpretation: Fraction of algorithmic CI covered by LLM range

c) **Simple Binary Overlap**:
```
Overlaps = {true if ranges intersect, false otherwise}
```

### 3. Calibration Analysis

**Purpose**: Assess whether LLM predictions are appropriately confident.

**Method**:
- **Underconfident**: LLM range wider than algorithmic CI (LLM hedge too much)
- **Well-calibrated**: LLM range similar width to algorithmic CI
- **Overconfident**: LLM range narrower than algorithmic CI (risky predictions)

**Metric**:
```
Calibration_score = (LLM_range_width - CI_width) / CI_width
  Negative := overconfident
  Near zero := well-calibrated
  Positive := underconfident
```

### 4. Prompt Formulation Robustness

**Purpose**: Test prediction consistency across different prompt styles.

**Method**:
```
For each LLM on each (dataset, algorithm) pair:
  1. Generate 3 different prompts (formulations 1, 2, 3)
  2. Query LLM with each prompt
  3. Extract ranges: R₁, R₂, R₃
  4. Compute variance:
     V = std([width(R₁), width(R₂), width(R₃)])
     (or other stability metrics)
  5. Classify:
     V < 0.1 × mean(widths) := robust
     V > 0.2 × mean(widths) := unstable
```

### 5. Algorithm-Specific Analysis

**Purpose**: Determine if LLMs understand algorithm-specific properties.

**Expected Insights**:
- **PC vs. LiNGAM**: Does LLM recognize fundamentally different paradigms?
- **FCI vs. PC**: Does LLM account for latent variable handling?
- **NOTEARS vs. others**: Does LLM grasp modern optimization-based approaches?

**Analysis Method**:
- Compare LLM predictions across algorithms for same dataset
- Check if relative performance rankings match algorithmic rankings
- Assess if LLM mentions algorithm-specific assumptions

### 6. Dataset Complexity Effects

**Purpose**: Test if LLMs understand how problem properties affect performance.

**Factors**:
- **Size effect**: How do predictions change with variable count?
  - Compare synthetic-12 vs. synthetic-30
  - Compare asia (8 vars) vs. child (20 vars)
- **Density effect**: How do predictions change with edge density?
- **Dimension effect**: Do larger datasets hurt certain algorithms more?

**Statistical Test**:
```
Spearman correlation: 
  rho(dataset_size, algorithmic_F1)
  rho(dataset_size, LLM_predicted_F1)
  
Are these correlations similar?
(Would indicate LLM understands scalability)
```

---

## Research Hypotheses and What This Work Intends to Prove

### Primary Hypothesis
**H1**: *LLMs possess genuine understanding of algorithmic performance characteristics*

Evidence for this hypothesis requires LLMs to:
- Predict performance ranges that align with actual algorithmic confidence intervals
- Distinguish between fundamentally different algorithm types (distinguishing PC from LiNGAM, FCI from PC)
- Understand how dataset properties (size, density, Gaussianity) affect different algorithms differently
- Adjust predictions appropriately based on algorithm assumptions
- Generate robust predictions that don't change dramatically with prompt rewording

**Evidence Against H1**:
- LLM ranges are random/uncalibrated relative to algorithmic CIs
- Predictions change dramatically with prompt formulation (suggesting pattern-matching rather than understanding)
- LLM cannot distinguish between fundamentally different algorithms or their properties
- LLM misses expected scalability and complexity effects
- Predictions are the same regardless of algorithm type or dataset characteristics

### Secondary Hypotheses

**H2**: *LLM predictions are appropriately calibrated*

Tests whether LLMs avoid both overconfidence (range too narrow) and underconfidence (range too wide) in their estimates.

**H3**: *LLM predictions are robust to prompt formulation*

Tests whether estimates represent "genuine knowledge" vs. "prompt-sensitive confabulation." Robust predictions suggest understanding; highly variable predictions suggest the LLM is pattern-matching to prompt cues.

**H4**: *LLM algorithmic reasoning correlates with model capability*

Tests whether more capable models (GPT-4) demonstrate better algorithmic understanding than less capable ones (Llama), suggesting true reasoning rather than random effects.

**H5**: *LLMs understand algorithm-specific properties*

Tests whether LLMs recognize that different algorithms have fundamentally different behaviors:
- PC (constraint-based) vs. LiNGAM (functional): Different assumptions → different performance profiles
- FCI (handles latent variables) vs. PC (assumes no latent confounders): Subtle but important difference
- NOTEARS (optimization-based) vs. heuristic search: Different computational paradigms

---

## What This Research Proves/Disproves

### If Results Support H1-H5:
✓ **Conclusion**: LLMs have developed genuine domain knowledge of algorithm performance beyond simple pattern matching

✓ **Implications**:
- LLMs can reason about complex technical systems
- LLM consulting on algorithm selection could be justified
- LLM reasoning can be trustworthy in other technical domains
- LLMs exhibit transfer learning from training data to novel problem combinations

### If Results Reject H1-H5:
✗ **Conclusion**: LLM responses reflect superficial pattern-matching or hallucination

✗ **Implications**:
- LLM claims about algorithm performance should be viewed skeptically
- LLMs lack robust reasoning about technical systems
- Direct simulation/testing required instead of LLM consultation
- Caution needed when using LLMs for technical decision-making

---

## UAI Submission Scope Reduction

### ❌ Dropped (Time Constraints):
- **5 additional datasets**: Titanic, Credit, Wine, Earthquake, Survey
- **Expert baseline**: Human expert predictions (compare to heuristic baseline instead)
- **Failure mode analysis**: Systematic testing of edge cases (mention in limitations only)
- **Information ablation**: Test with/without dataset names (future work)
- **Training contamination**: Dataset overlap with LLM training data (future work)
- **Convergence analysis**: Compare performance across random seeds (future work)

### ✅ Kept (Essential for Acceptance):
- **6 datasets**: Asia, Sachs, Cancer, Child, Synthetic-12, Synthetic-30 (good statistical power)
- **4 algorithms**: PC, LiNGAM, FCI, NOTEARS (covers main algorithm paradigms)
- **5 LLMs**: GPT-4, Claude, Gemini, DeepSeek, Llama (representative sample)
- **3 formulations**: Tests prompt robustness
- **PRIMARY METRIC**: Calibrated Coverage (directly tests hypothesis)
- **Statistical tests**: Wilcoxon + FDR correction (establishes significance)
- **Baselines**: Random + Heuristic (necessary comparisons)

**Total Scope**: 6 × 4 × 5 × 3 = **360 queries** (doable in 7 hours)

---

## Project Structure

```
confidence-crisis/
├── config.json                          # Project configuration
├── requirements.txt                     # Python dependencies
├── README.md                            # Quick start guide
├── PROJECT_DOCUMENTATION.md             # This file
├── 9DAY_PLAN.md                         # UAI submission timeline
├── check_progress.py                    # Progress tracker
│
├── experiments/
│   └── run_experiments.py              # Phase 1: Run algorithms 100+ times
│
├── src/
│   ├── algorithms/
│   │   └── variance_analysis.py        # Statistical analysis engine
│   │
│   ├── datasets/
│   │   └── __init__.py                  # Dataset loaders (5 datasets)
│   │
│   ├── llm/
│   │   ├── llm_interface.py            # Unified LLM API wrapper
│   │   ├── query_all_llms.py           # Phase 3: Query LLMs
│   │   ├── parse_llm_responses.py      # Phase 2: Parse LLM responses
│   │   └── analyze_reasoning_traces.py # Extract & validate LLM reasoning
│   │
│   ├── evaluation/
│   │   ├── metrics.py                  # Calibrated coverage & evaluation
│   │   ├── compute_metrics.py          # Day 4: Compute all metrics
│   │   └── generate_tables.py          # Day 6: Publication tables
│   │
│   ├── baselines/
│   │   └── simple_baselines.py         # Random & heuristic baselines
│   │
│   ├── analysis/
│   │   └── statistical_tests.py        # Day 5: Wilcoxon + FDR
│   │
│   └── visualization/
│       └── visualize_results.py        # Phase 4: Generate plots
│
├── prompts/
│   ├── prompt_templates.py             # Three prompt formulations
│   ├── analyze_prompt_variance.py      # Analyze prompts for robustness
│   └── __init__.py
│
├── results/
│   ├── variance/                       # Phase 1 outputs
│   │   ├── asia_pc_variance.json
│   │   ├── asia_lingam_variance.json
│   │   ├── asia_fci_variance.json
│   │   ├── asia_notears_variance.json
│   │   └── ... (20 files: 5 datasets × 4 algorithms)
│   │
│   ├── llm_comparisons/                # Phase 3 outputs
│   │   ├── asia_pc_llm_comparison.json
│   │   ├── ... (20 files minimum)
│   │
│   ├── evaluation/                     # Day 4 outputs
│   │   ├── main_results.csv            # Table 1 for paper
│   │   ├── detailed_results.json
│   │   └── coverage_by_*.csv
│   │
│   └── statistics/                     # Day 5 outputs
│       ├── statistical_tests.csv
│       └── interpretation.txt
│
├── plots/                               # Phase 4 outputs
│   ├── fci_summary_table.csv
│   ├── results_summary.csv
│   └── results_summary.tex             # For LaTeX papers
│
├── paper/                               # UAI submission
│   ├── draft.tex                       # Paper draft
│   ├── final.pdf                       # Submission PDF
│   ├── tables/                         # Generated LaTeX tables
│   └── figures/                        # Generated figures
│
└── test_output/                        # Diagnostic outputs
```

---

## Execution Instructions (UAI Submission)

### Reduced Scope Run (~15 hours total)

> **Timeline**: 9-day plan for UAI submission (see `9DAY_PLAN.md`)

**Day 1 (COMPLETE)**: Infrastructure setup
**Days 2-6**: Experiments and analysis  
**Days 7-9**: Paper writing

### Sequential Commands

```bash
# Day 1: Test infrastructure (COMPLETE)
python src/evaluation/metrics.py
python src/baselines/simple_baselines.py

# Day 2: Run experiments (2-3 hours)
python experiments/run_experiments.py --runs 100 --experiments asia sachs cancer child synthetic_12 synthetic_30

# Day 3: Query LLMs (7 hours)
python src/llm/query_all_llms.py --datasets asia sachs cancer child synthetic_12 synthetic_30 \
       --algorithms pc lingam fci notears --formulations 1 2 3 \
       --models gpt5 claude gemini deepseek llama qwen

# Day 4: Compute metrics (PRIMARY)
python src/evaluation/compute_metrics.py

# Day 5: Statistical tests
python src/analysis/statistical_tests.py

# Day 6: Generate tables
python src/evaluation/generate_tables.py
```

### Original Full Experiment Run (Dropped for UAI)

```bash
# Step 1: Establish algorithmic ground truth (2-3 hours)
python experiments/run_experiments.py --runs 100  # Entire pipeline

# Step 2: Test LLM parsing (1 minute)
python test_parsing.py

# Step 3: Query all LLMs (12-16 hours)
# Requires API keys in environment or .env file
python src/llm/query_all_llms.py --all

# Step 4: Generate visualizations (5 minutes)
python src/visualization/visualize_results.py
```

### Quick Test Run (15 minutes)

```bash
# Test with single dataset (Titanic)
python experiments/run_experiments.py --runs 10 --experiments titanic

# Test LLM parsing
python test_parsing.py

# Test LLM querying with limited scope
python src/llm/query_all_llms.py --datasets titanic --algorithms pc --formulations 1

# Generate plots
python src/visualization/visualize_results.py
```

---

## Key Files and Their Roles

### `src/algorithms/variance_analysis.py`
- **Role**: Statistical engine for Phase 1
- **Key Classes**:
  - `MetricStats`: Stores mean, std, 95% CI, median, min, max
  - `AlgorithmResults`: Container for precision/recall/F1/SHD statistics
- **Key Methods**:
  - `compute_metric_stats()`: Bootstrap CI computation
  - `overlaps_with_range()`: Check overlap with LLM ranges
  - `contains_range()`: Check if CI contains LLM prediction

### `src/llm/llm_interface.py`
- **Role**: Unified API for all 5 LLM providers
- **Handles**: Authentication, rate limiting, retries, error handling
- **Providers**: OpenAI, Anthropic, Google, DeepSeek, Together (for Llama)

### `src/llm/parse_llm_responses.py`
- **Role**: Parse diverse LLM response formats
- **Regex Patterns**: Structured, range, percentage, natural language formats
- **Output**: Normalized metric ranges matching algorithmic CI format

### `prompts/prompt_templates.py`
- **Role**: Define 3 prompt formulations
- **Formulations**:
  1. **Direct**: Straightforward question
  2. **Step-by-Step**: Guided reasoning
  3. **Meta-Knowledge**: CI-focused framing

### `src/visualization/visualize_results.py`
- **Role**: Generate publication-quality plots
- **Outputs**:
  - Error bar plots: Algorithm CI vs. LLM ranges
  - Heatmaps: Overlap statistics
  - Summary tables: Performance comparison

---

## Dependencies

### Core Causal Discovery Libraries
- `causal-learn`: PC, FCI, and other constraint-based algorithms
- `lingam`: Linear Non-Gaussian Acyclic Model implementation
- `notears`: Modern optimization-based causal discovery

### Benchmark Datasets
- `pgmpy`: Bayesian network library (dataset generation)
- `bnlearn`: Causal learning library (with real datasets)

### Data Processing
- `pandas`: Data manipulation
- `numpy`: Numerical computation
- `scipy`: Statistical functions
- `scikit-learn`: Machine learning utilities

### LLM Interfaces
- `openai`: GPT-4 API
- `anthropic`: Claude API
- `google-generativeai`: Gemini API
- (DeepSeek and Llama via Together)

### Visualization
- `matplotlib`: Plotting
- `seaborn`: Statistical visualization

### Utilities
- `tqdm`: Progress bars
- `joblib`: Parallel processing

---

## Configuration

Edit `config.json` to customize:

```json
{
  "experiments": {
    "n_runs": 100,                    // Runs per algorithm per dataset
    "algorithms": ["PC", "LiNGAM", "FCI", "NOTEARS"],
    "datasets": [...],                // 11 datasets
    "llm_models": [...],              // 5 LLMs
    "prompt_formulations": [1, 2, 3]  // 3 formulations
  },
  "llm_settings": {
    "temperature": 0.1,               // Low randomness
    "max_tokens": 1024,
    "max_retries": 3,
    "retry_delay": 5
  }
}
```

---

## Expected Outputs and Interpretations

### Variance Results (`results/variance/`)
- **File**: `{dataset}_{algorithm}_variance.json`
- **Contents**: Mean, std, 95% CI, median, min, max for each metric
- **Interpretation**: Establishes what algorithms actually achieve

### LLM Comparisons
- **File**: `{dataset}_{algorithm}_llm_comparison.json`
- **Contents**: LLM predictions, overlap analysis, metric comparison
- **Interpretation**: Shows LLM accuracy relative to ground truth

### Visualizations (`plots/`)
- **Error bar plots**: Visual comparison of algorithm CIs vs. LLM ranges
- **Heatmaps**: Overlap percentages by dataset and algorithm
- **Summary tables**: Mean/std performance across all experiments

---

## Research Output and Publications

This research addresses fundamental questions about LLM reasoning and could result in:

1. **Primary Paper**: *"Do Language Models Understand Causal Discovery? A Rigorous Meta-Knowledge Evaluation"*
   - Comprehensive comparison of LLM predictions vs. algorithmic CIs
   - Analysis of calibration and robustness
   - Statistical significance testing

2. **Supplementary Analyses**:
   - Algorithm-specific insights (what do LLMs understand about PC vs. LiNGAM?)
   - Dataset complexity effects (do LLMs grasp scalability?)
   - Prompt robustness (how sensitive are predictions to wording?)

3. **Methodological Contribution**:
   - Framework for rigorous evaluation of LLM domain knowledge
   - Significance testing procedures for overlap analysis
   - Guidelines for prompt design in technical domains

---

## Troubleshooting and Known Issues

### Common Issues

**Issue**: Algorithm runs are very slow
- **Cause**: NOTEARS optimization can be expensive
- **Solution**: Reduce dataset size or use subset for testing

**Issue**: LLM API errors
- **Cause**: Rate limits, authentication issues, API downtime
- **Solution**: Check API keys, add delays between requests, use retries

**Issue**: Response parsing fails
- **Cause**: Unexpected LLM response format
- **Solution**: Manual inspection of response, update regex patterns

### Validation Checks

Before drawing conclusions, verify:
1. ✓ Algorithm ran at least 100 times per (dataset, algorithm) pair
2. ✓ All LLMs successfully queried (check error logs)
3. ✓ Confidence intervals have non-zero width (algorithms are stochastic)
4. ✓ LLM ranges are reasonable (not [0, 1] for all metrics)

---

## Contact and Contribution

This project demonstrates best practices for:
- Rigorous evaluation of stochastic algorithms
- Quantitative comparison of model predictions
- Publication-quality statistical analysis

For questions about methodology or results, refer to the docstrings in key files and the referenced research on causal discovery and LLM evaluation.
