# Quick Command Reference - UAI Submission

## All Experiments in Sequential Order

### Setup (One-time)
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### Day 1: Infrastructure (COMPLETE ✓)
```bash
python src/evaluation/metrics.py
python src/baselines/simple_baselines.py
python check_progress.py --mark-complete "Day 1"
```

### Day 2: Run Algorithm Experiments (~2-3 hours)
```bash
python src/experiments/run_experiments.py --runs 100 --experiments asia sachs cancer child synthetic_12 synthetic_30
python check_progress.py --mark-complete "Day 2"
```

### Day 3: Query All LLMs (~7 hours)
```bash
python src/llm/query_all_llms.py --datasets asia sachs cancer child synthetic_12 synthetic_30 --algorithms pc lingam fci notears --formulations 1 2 3 --models gpt5 claude gemini deepseek llama qwen
python check_progress.py --mark-complete "Day 3"
```

### Day 4: Compute Calibrated Coverage (PRIMARY METRIC)
```bash
python src/evaluation/compute_metrics.py --ground_truth results/variance --llm_results results/llm_comparisons --output results/evaluation
python check_progress.py --mark-complete "Day 4"
```

### Day 5: Statistical Significance Tests
```bash
python src/analysis/statistical_tests.py --input results/evaluation --output results/statistics --baseline random
python check_progress.py --mark-complete "Day 5"
```

### Day 6: Generate Publication Tables
```bash
python src/evaluation/generate_tables.py --input results/evaluation --output paper/tables
python check_progress.py --mark-complete "Day 6"
```

### Check Progress Anytime
```bash
python check_progress.py
```

---

## Total Time Estimate
- Day 1: Complete ✓
- Day 2: 2-3 hours (algorithm runs: 2,400 runs)
- Day 3: 7 hours (LLM queries: 360 queries)
- Day 4: 1 hour (metrics computation)
- Day 5: 1 hour (statistical tests)
- Day 6: 1 hour (table generation)
- **Total**: ~12 hours of computation

Days 7-9: Paper writing (manual - see 9DAY_PLAN.md)

---

## Verification Commands

```bash
# Day 2: Check variance files
ls results/variance/*.json | wc -l  # Should be 20

# Day 3: Check LLM comparison files
ls results/llm_comparisons/*.json | wc -l  # Should be ≥20

# Day 4: Check main results
cat results/evaluation/main_results.csv

# Day 5: Check significance
cat results/statistics/interpretation.txt

# Day 6: Check tables
ls paper/tables/*.tex
```
