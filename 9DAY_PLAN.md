# 9-Day Emergency UAI Submission Plan

**Status**: Ready to execute ⚡  
**Target**: UAI 2026 submission  
**Scope**: 5 datasets, 4 algorithms, 5 LLMs = 300 queries (reduced from original)

---

## Quick Start

```bash
# Check progress
python check_progress.py

# Day 1: Test infrastructure
python src/evaluation/metrics.py
python src/baselines/simple_baselines.py

# Day 2: Run experiments (5 datasets only)
python src/experiments/run_experiments.py --runs 100 --experiments asia sachs cancer synthetic_12 synthetic_30

# Day 3: Query LLMs
python src/llm/query_all_llms.py --datasets asia sachs cancer synthetic_12 synthetic_30 --all

# Day 4: Compute metrics (PRIMARY METRIC)
python src/evaluation/compute_metrics.py --ground_truth results/variance \
                                         --llm_results results/llm_comparisons \
                                         --output results/evaluation

# Day 5: Statistical tests
python src/analysis/statistical_tests.py --input results/evaluation \
                                         --output results/statistics

# Day 6: Generate tables
python src/evaluation/generate_tables.py --input results/evaluation \
                                         --output paper/tables

# Days 7-9: Write paper (manual)
```

---

## What Changed from Original Plan

### ❌ DROPPED (no time):
- Titanic, Credit, Wine datasets (3 datasets → 5 datasets only)
- Expert baseline
- Information ablation
- Training data contamination analysis
- Convergence analysis

### ✅ KEPT (essential):
- 5 datasets: **Asia, Sachs, Cancer, Synthetic-12, Synthetic-30**
- 4 algorithms: PC, LiNGAM, FCI, NOTEARS
- 5 LLMs: GPT-4, Claude, Gemini, DeepSeek, Llama
- 3 prompt formulations
- **PRIMARY METRIC**: Calibrated Coverage
- **Statistical tests**: Wilcoxon + FDR correction

**Total Queries**: 5 × 4 × 5 × 3 = **300 queries** (doable in 6 hours)

---

## Day-by-Day Breakdown

### Day 1: Critical Infrastructure (8 hours) ✅ COMPLETE
**Deliverables**:
- `src/evaluation/metrics.py` - Calibrated coverage metric
- `src/baselines/simple_baselines.py` - Random & heuristic baselines

**Verify**:
```bash
python src/evaluation/metrics.py  # Should print metric descriptions
python src/baselines/simple_baselines.py  # Should show demo predictions
```

---

### Day 2: Run Experiments (8 hours)
**Command**:
```bash
python src/experiments/run_experiments.py --runs 100 \
       --experiments asia sachs cancer synthetic_12 synthetic_30
```

**Expected Outputs**:
- `results/variance/asia_pc_variance.json`
- `results/variance/asia_lingam_variance.json`
- `results/variance/asia_fci_variance.json`
- `results/variance/asia_notears_variance.json`
- ... (20 files total: 5 datasets × 4 algorithms)

**Verify**:
```bash
ls results/variance/*.json | wc -l  # Should be 20
python -c "import json; print(json.load(open('results/variance/asia_pc_variance.json'))['results']['f1'])"
# Should show: {mean: X, std: Y, ci_95_lower: Z, ci_95_upper: W}
```

**Risk**: If NOTEARS fails, skip it (reduce to 3 algorithms = 240 queries on Day 3)

---

### Day 3: Query LLMs (8 hours)
**Command**:
```bash
# IMPORTANT: Set API keys first
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."

python src/llm/query_all_llms.py \
       --datasets asia sachs cancer synthetic_12 synthetic_30 \
       --algorithms pc lingam fci notears \
       --formulations 1 2 3 \
       --models gpt4 claude gemini deepseek llama
```

**Time Estimate**: 300 queries × 1 min/query ≈ 5 hours (with retries: ~6-7 hours)

**Expected Outputs**:
- `results/llm_comparisons/*_llm_comparison.json` (20 files minimum)

**Verify**:
```bash
ls results/llm_comparisons/*.json | wc -l  # Should be ≥20
python src/llm/parse_llm_responses.py  # Test parsing
```

**Fallback**: If time runs short, use only 3 LLMs (GPT-4, Claude, Gemini) = 180 queries

---

### Day 4: Compute Calibrated Coverage (8 hours) ⚠️ CRITICAL
**PRIMARY METRIC DAY**

**Command**:
```bash
python src/evaluation/compute_metrics.py \
       --ground_truth results/variance \
       --llm_results results/llm_comparisons \
       --output results/evaluation
```

**Expected Outputs**:
- `results/evaluation/main_results.csv` ← **Table 1 for paper**
- `results/evaluation/detailed_results.json`
- `results/evaluation/coverage_by_dataset.csv`
- `results/evaluation/coverage_by_algorithm.csv`

**Verify**:
```bash
cat results/evaluation/main_results.csv
# Should show:
# LLM, Calibrated_Coverage, MAE, Mean_Width, N_predictions
# gpt4, 0.72, 0.12, 0.20, 60
# claude, 0.68, 0.14, 0.22, 60
# ...
# random, 0.15, 0.28, 0.50, 60
```

**Success Criteria**:
- At least 1 LLM has coverage >0.60 (60%)
- All LLMs beat random (>0.15)
- If not: Check for parsing errors or re-query failed cases

---

### Day 5: Statistical Tests (8 hours)
**Command**:
```bash
python src/analysis/statistical_tests.py \
       --input results/evaluation \
       --output results/statistics \
       --baseline random
```

**Expected Outputs**:
- `results/statistics/statistical_tests.csv` ← **For paper**
- `results/statistics/interpretation.txt` ← **Copy to paper**

**What It Tests**:
- Wilcoxon signed-rank test: LLM vs random
- FDR correction (Benjamini-Hochberg)
- Cohen's d effect sizes

**Verify**:
```bash
cat results/statistics/interpretation.txt
# Should show p-values and significance for each LLM
```

---

### Day 6: Figures & Tables (8 hours)
**Commands**:
```bash
# Generate publication tables
python src/evaluation/generate_tables.py \
       --input results/evaluation \
       --output paper/tables

# Generate figures (use existing visualization script)
python src/visualization/visualize_results.py
```

**Expected Outputs**:
- `paper/tables/table1_main_results.tex` ← **Main table**
- `paper/tables/table2_dataset_breakdown.tex`
- `paper/figures/figure1_coverage.pdf` ← **Bar plot**

**Manual**: Create Figure 1 (calibration bar plot) using matplotlib if needed

---

### Days 7-9: Write Paper ⚠️ CRITICAL

#### Day 7: Structure & Methods (8 hours)
**Target**: 3-4 pages

**Sections**:
1. **Introduction** (1 page)
   - Problem: Need to test LLM algorithmic understanding
   - Testbed: Causal discovery = structured, measurable
   - Contributions: (1) testbed, (2) 5-LLM eval, (3) findings

2. **Related Work** (0.75 pages)
   - LLM capabilities (cite recent work)
   - Causal discovery algorithms (cite PC, LiNGAM, FCI, NOTEARS)
   - Gap: No work on LLM algorithm meta-knowledge

3. **Methodology** (1.5 pages)
   - 3.1 Datasets (5 datasets with citations)
   - 3.2 Algorithms (4 algorithms, 1 sentence each)
   - 3.3 Ground Truth (100 runs, bootstrap CIs)
  - 3.4 LLM Evaluation (5 LLMs, 3 formulations)
   - 3.5 Metrics (Calibrated Coverage = primary)

**Verify**: Draft exists, structure is clear

---

#### Day 8: Results & Discussion (8 hours)
**Target**: 7-8 pages total

**Sections**:
4. **Results** (2 pages)
   - 4.1 Overall Performance (Table 1, Figure 1)
     ```
     GPT-4 achieves 72% coverage vs 15% random (p<0.001)
     ```
   - 4.2 Statistical Significance (Wilcoxon tests, FDR)
   - 4.3 Algorithm Ranking (Spearman ρ)
   - 4.4 Dataset Breakdown (Table 2)

5. **Discussion** (1 page)
   - Interpretation: LLMs demonstrate understanding? Or pattern-matching?
   - Implications for algorithm selection
   - Comparison to heuristic baseline

**Verify**: Results section complete, all tables/figures inserted

---

#### Day 9: Polish & Submit (8 hours) ⚠️ DEADLINE
**Target**: Submission-ready PDF

**Sections**:
6. **Limitations** (0.5 pages)
   - Only 5 datasets (small scale)
   - Only causal discovery domain
   - No expert baseline
   - Dataset names may leak information

7. **Conclusion** (0.5 pages)
   - First rigorous testbed for LLM algorithm understanding
   - [Results summary]
   - Future work: More algorithms, domains, mechanistic analysis

8. **References**
   - Proper citations for all datasets, algorithms, LLMs

**Final Checks**:
- [ ] 8 pages (UAI limit)
- [ ] All figures/tables present
- [ ] References complete
- [ ] Proofread (no typos)
- [ ] UAI template formatting
- [ ] **SUBMIT**

---

## Key Metrics & Targets

### Primary Metric: Calibrated Coverage
**Definition**: % of predictions where `true_mean ∈ LLM_range`

**Targets**:
- **>60%**: Strong evidence of understanding
- **40-60%**: Moderate understanding
- **<40%**: Weak/no understanding
- **Random baseline**: ~15-20%

### Success Criteria for Paper Acceptance

**Minimum (60% acceptance chance)**:
- ✅ Clear research question
- ✅ Proper ground truth (100 runs per algorithm)
- ✅ Multiple LLMs tested (5)
- ✅ Baselines included (random + heuristic)
- ✅ Statistical tests (Wilcoxon + FDR)
- ✅ At least 1 LLM significantly beats baseline

**Competitive (70% acceptance chance)**:
- Above +
- ✅ Effect sizes reported (Cohen's d)
- ✅ Algorithm ranking analysis
- ✅ Per-dataset breakdown
- ✅ Clear interpretation

---

## Risk Mitigation

### If Day 3 LLM queries fail:
**Fallback**: Use only 3 LLMs (GPT-4, Claude, Gemini)
- Reduces to 180 queries (still publishable)
- Update: "We evaluate three leading LLMs..."

### If Day 5 stats are weak (no LLM beats baseline):
**Fallback**: Frame as negative result
- "Exploratory study reveals current LLMs lack robust algorithmic understanding"
- Still publishable (negative results matter)

### If Day 9 morning and paper incomplete:
**Nuclear option**: Submit 6-page short paper
- Focus on methodology (testbed contribution)
- Results as "preliminary findings"
- Submit to workshop instead (90% acceptance)

---

## Files Created

### Infrastructure (Day 1)
- `src/evaluation/metrics.py` - Core evaluation metrics
- `src/baselines/simple_baselines.py` - Baseline predictors

### Evaluation Scripts (Days 4-6)
- `src/evaluation/compute_metrics.py` - Compute all metrics
- `src/evaluation/generate_tables.py` - Generate LaTeX tables
- `src/analysis/statistical_tests.py` - Wilcoxon tests + FDR

### Planning
- `check_progress.py` - Track progress through 9 days
- `9DAY_PLAN.md` - This file

---

## Quick Commands Reference

```bash
# Check what's done
python check_progress.py

# Mark day complete
python check_progress.py --mark-complete "Day 1"

# Run full pipeline (after experiments complete)
python src/evaluation/compute_metrics.py && \
python src/analysis/statistical_tests.py && \
python src/evaluation/generate_tables.py

# Check results
cat results/evaluation/main_results.csv
cat results/statistics/interpretation.txt
```

---

## Contact / Questions

If stuck:
1. Check `check_progress.py` for what's missing
2. Verify deliverables exist (`ls results/*/`)
3. Check logs for errors
4. Run test scripts to verify setup

**You can do this.** 9 days is tight but achievable with focused execution.

Good luck! 🚀
