# LLM Causal Discovery Evaluation

**Do LLMs understand causal discovery algorithms?**

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

## Experiments

### Step 1: Algorithm Variance Analysis (2-3 hours)
```bash
# Dry run: 1 dataset × 2 algorithms × 10 runs = 20 runs
python experiments/run_experiments.py --runs 10 --experiments titanic

# Full run: 11 datasets × 4 algorithms × 100 runs = 4,400 runs
python experiments/run_experiments.py --runs 100
```

### Step 2: Test LLM Parsing (~1 minute)
```bash
# Test all 5 LLMs with 1 query each = 5 queries
python test_parsing.py
```

### Step 3: LLM Predictions (12-16 hours)
```bash
# Dry run: 1 dataset × 1 algorithm × 5 LLMs × 1 formulation = 5 queries
python src/llm/query_all_llms.py --datasets titanic --algorithms pc --formulations 1

# Full run: 11 datasets × 4 algorithms × 5 LLMs × 3 formulations = 660 queries
python src/llm/query_all_llms.py --all
```

### Step 4: Generate Visualizations (~5 minutes)
```bash
# Create all plots from results
python src/visualization/visualize_results.py
```

## Outputs

- `results/variance/` - Algorithm performance with 95% CIs
- `results/llm_comparisons/` - LLM predictions
- `plots/` - Figures (CIs, heatmaps, overlap analysis)

## Configuration

Edit `config.json` to customize:
- Number of runs per algorithm
- Datasets to include
- Algorithms to test
- LLM models and formulations
