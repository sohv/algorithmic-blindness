# LLM Causal Discovery Evaluation

**Testing LLM Algorithmic Understanding Using Causal Discovery as a Testbed**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys (required for Day 3)
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

## Scope (UAI Submission)

**6 Datasets**: Asia, Sachs, Cancer, Child, Synthetic-12, Synthetic-30  
**4 Algorithms**: PC, LiNGAM, FCI, NOTEARS  
**5 LLMs**: GPT-4, Claude, Gemini, DeepSeek, Llama  
**3 Formulations**: Direct, Step-by-step, Meta-knowledge  
**Total**: 6 × 4 × 5 × 3 = **360 LLM queries**

---

## Sequential Experiment Commands

### Day 1: Setup Infrastructure (COMPLETE ✓)
```bash
python src/evaluation/metrics.py              # Test primary metric (calibrated coverage)
python src/baselines/simple_baselines.py      # Test baseline predictors
```

### Day 2: Run Algorithm Experiments (~2-3 hours)
```bash
python src/experiments/run_experiments.py --runs 100 --experiments asia sachs cancer child synthetic_12 synthetic_30
# Output: 24 variance files (6 datasets × 4 algorithms)
```

### Day 3: Query All LLMs (~7 hours)
```bash
python src/llm/query_all_llms.py --datasets asia sachs cancer child synthetic_12 synthetic_30 --algorithms pc lingam fci notears --formulations 1 2 3 --models gpt4 claude gemini deepseek llama
# Output: 360 LLM queries, ~24 comparison files
```

### Day 4: Compute Calibrated Coverage ⚠️ PRIMARY METRIC
```bash
python src/evaluation/compute_metrics.py --ground_truth results/variance --llm_results results/llm_comparisons --output results/evaluation
# Output: main_results.csv (Table 1 for paper)
```

### Day 5: Statistical Significance Tests
```bash
python src/analysis/statistical_tests.py --input results/evaluation --output results/statistics --baseline random
# Output: Wilcoxon tests + FDR correction
```

### Day 6: Generate Publication Tables
```bash
python src/evaluation/generate_tables.py --input results/evaluation --output paper/tables
# Output: LaTeX tables for paper
```

### Day 7-9: Write Paper (Manual)
See `9DAY_PLAN.md` for writing schedule

---

## Progress Tracking

```bash
python check_progress.py                      # Show completion status
python check_progress.py --mark-complete "Day 2"  # Mark day complete
```

---

## Outputs

- `results/variance/` - Algorithm performance with 95% CIs (Day 2)
- `results/evaluation/main_results.csv` - Table 1 for paper (Day 4)
- `results/statistics/` - Significance tests (Day 5)
- `paper/tables/` - Publication-ready LaTeX tables (Day 6)

## Documentation

- `PROJECT_DOCUMENTATION.md` - Full methodology and research design
- `9DAY_PLAN.md` - Day-by-day execution plan for UAI submission

## Key Metric

**Calibrated Coverage** = % predictions where `true_mean ∈ LLM_range`
- Target: >60% indicates genuine understanding
- Random baseline: ~15-20%
