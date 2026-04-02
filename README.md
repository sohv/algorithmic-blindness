# Large Language Models Are Algorithmically Blind

[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=flat&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=OmJK0GK8_MI) [![ArXiv](https://img.shields.io/badge/arXiv-2602.21947-b31b1b.svg)](https://arxiv.org/abs/2602.21947)

Can LLMs predict how causal discovery algorithms actually perform? We test 8 frontier LLMs across 13 datasets and 4 algorithms, comparing their predicted performance ranges against ground truth from 5,200 algorithm runs. Result: **84.1% of LLM predictions fail to contain the true algorithmic performance** -- even with ranges 8-27x wider than actual confidence intervals.

## Project Structure

```
confidence-crisis/
├── src/
│   ├── prompts/
│   │   └── prompt_templates.py
│   ├── algorithms/
│   │   ├── run_experiments.py
│   │   ├── analyze_results.py
│   │   ├── compare_algo_vs_llm.py
│   │   ├── statistical_analysis.py
│   │   └── per_dataset_memorization_analysis.py
│   ├── llm/
│   │   ├── query_all_llms.py
│   │   ├── extract_llm_ranges.py
│   │   ├── aggregate_formulations.py
│   │   ├── compare_llm_vs_algorithmic.py
│   │   ├── compute_prompt_robustness.py
│   │   └── cross_algorithm_analysis.py
│   ├── baselines/
│   │   ├── simple_baselines.py
│   │   └── compare_baselines_full.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── compute_metrics.py
│   │   └── generate_tables.py
│   └── visualize/
│       ├── plot.py
│       └── generate_tables.py
├── datasets/
│   └── [13 benchmark and synthetic networks in .bif format]
├── METHODOLOGY.md
└── README.md
```

## Setup

### Prerequisites

```bash
# Install dependencies using uv
uv sync

# Create .env file from template
cp .env.example .env

# Add your API keys to .env
# Required keys:
# OPENAI_API_KEY=your-key
# ANTHROPIC_API_KEY=your-key
# GOOGLE_API_KEY=your-key
# DEEPSEEK_API_KEY=your-key (optional)
# TOGETHER_API_KEY=your-key (optional)
```

### Running Experiments

Execute the following commands in order:

```bash
# 1. Compute algorithmic ground truth (~2h)
python src/algorithms/run_experiments.py --runs 100

# 2. Query LLMs (~3h)
python src/llm/query_all_llms.py

# 3. Parse LLM responses into ranges (~1min)
python src/llm/extract_llm_ranges.py

# 4. Aggregate predictions across prompt formulations
python src/llm/aggregate_formulations.py

# 5. Compute calibrated coverage and comparison metrics
python src/llm/compare_llm_vs_algorithmic.py

# 6. Generate analysis plots and tables
python src/algorithms/analyze_results.py
```

**Total runtime**: ~5-6 hours (mostly API calls and algorithm runs)

## Citation

If you use this work, please cite:

```bibtex
@article{venkatesh2026large,
  title={Large Language Models are Algorithmically Blind},
  author={Venkatesh, Sohan and Kurapath, Ashish Mahendran and Melkote, Tejas},
  journal={arXiv preprint arXiv:2602.21947},
  year={2026}
}
```
