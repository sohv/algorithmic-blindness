#!/usr/bin/env python3
"""
Generate Publication-Ready Tables
==================================

Generate LaTeX and formatted tables for paper submission.

Usage:
    python generate_tables.py --input results/evaluation \
                              --output paper/tables
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


def format_main_table(results_df: pd.DataFrame) -> str:
    """
    Format main results table for paper (Table 1).
    
    Args:
        results_df: DataFrame from compute_metrics.py
        
    Returns:
        LaTeX table string
    """
    latex = r"""\begin{table}[t]
\centering
\caption{LLM Performance on Algorithm Meta-Knowledge Task}
\label{tab:main_results}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Coverage} & \textbf{MAE} & \textbf{Rank Corr.} \\
\midrule
"""
    
    for _, row in results_df.iterrows():
        llm = row['LLM']
        coverage = float(row['Calibrated_Coverage'])
        mae = float(row['MAE'])
        
        # Handle potential NaN in ranking correlation
        try:
            rank_corr = float(row['Ranking_Correlation'])
            rank_str = f"{rank_corr:.2f}" if not np.isnan(rank_corr) else "---"
        except:
            rank_str = "---"
        
        # Format model name
        model_display = {
            'gpt5': 'GPT-5',
            'claude': 'Claude',
            'deepseek': 'DeepSeek',
            'deepseekthink': 'DeepSeek-Think',
            'gemini3': 'Gemini 3',
            'llama': 'LLaMA',
            'qwen': 'Qwen',
            'qwenthink': 'Qwen-Think',
            'heuristic': r'\textit{Heuristic}',
            'random': r'\textit{Random}'
        }.get(llm, llm)
        
        # Add horizontal line before baselines
        if llm in ['heuristic', 'random'] and _ == results_df[results_df['LLM'] == 'heuristic'].index[0]:
            latex += r"\midrule" + "\n"
        
        latex += f"{model_display} & {coverage:.2f} & {mae:.2f} & {rank_str} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


def format_per_dataset_table(detailed_results: dict, datasets: list) -> str:
    """
    Format per-dataset coverage breakdown (Table 2 or appendix).
    
    Args:
        detailed_results: Detailed results from compute_metrics
        datasets: List of dataset names
        
    Returns:
        LaTeX table string
    """
    latex = r"""\begin{table}[t]
\centering
\caption{Coverage by Dataset}
\label{tab:dataset_breakdown}
\begin{tabular}{l""" + "c" * len(datasets) + r"""}
\toprule
\textbf{Model}"""
    
    # Header
    for ds in datasets:
        ds_display = ds.replace('_', ' ').title()
        latex += f" & \\textbf{{{ds_display}}}"
    latex += r" \\" + "\n" + r"\midrule" + "\n"
    
    # Rows
    for llm_name, results in detailed_results.items():
        if llm_name in ['random', 'heuristic']:
            continue  # Skip baselines in this table
        
        model_display = {
            'gpt5': 'GPT-5',
            'claude': 'Claude',
            'deepseek': 'DeepSeek',
            'deepseekthink': 'DeepSeek-Think',
            'gemini3': 'Gemini 3',
            'llama': 'LLaMA',
            'qwen': 'Qwen'
        }.get(llm_name, llm_name)
        for ds in datasets:
            coverage = by_dataset.get(ds, 0.0)
            latex += f" & {coverage:.2f}"
        
        latex += r" \\" + "\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


def format_per_algorithm_table(detailed_results: dict) -> str:
    """
    Format per-algorithm coverage breakdown.
    
    Args:
        detailed_results: Detailed results from compute_metrics
        
    Returns:
        LaTeX table string
    """
    algorithms = ['PC', 'LiNGAM', 'FCI', 'NOTEARS']
    
    latex = r"""\begin{table}[t]
\centering
\caption{Coverage by Algorithm Type}
\label{tab:algorithm_breakdown}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{PC} & \textbf{LiNGAM} & \textbf{FCI} & \textbf{NOTEARS} \\
\midrule
"""
    
    for llm_name, results in detailed_results.items():
        if llm_name in ['random', 'heuristic']:
            continue
        
        model_display = {
            'gpt5': 'GPT-5',
            'claude': 'Claude',
            'deepseek': 'DeepSeek',
            'deepseekthink': 'DeepSeek-Think',
            'gemini3': 'Gemini 3',
            'llama': 'LLaMA',
            'qwen': 'Qwen'
        }.get(llm_name, llm_name)
        
        latex += model_display
        
        by_algorithm = results['by_algorithm']
        for alg in algorithms:
            coverage = by_algorithm.get(alg, 0.0)
            latex += f" & {coverage:.2f}"
        
        latex += r" \\" + "\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


def main():
    parser = argparse.ArgumentParser(description='Generate publication tables')
    parser.add_argument('--input', type=str, default='results/evaluation',
                        help='Input directory with evaluation results')
    parser.add_argument('--output', type=str, default='paper/tables',
                        help='Output directory for tables')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("GENERATING PUBLICATION TABLES")
    print("="*70)
    
    # Load results
    main_results_csv = input_dir / 'main_results.csv'
    detailed_results_json = input_dir / 'detailed_results.json'
    
    if not main_results_csv.exists():
        print(f"Error: {main_results_csv} not found")
        print("Run compute_metrics.py first")
        return
    
    results_df = pd.read_csv(main_results_csv)
    
    print(f"\nLoaded: {len(results_df)} models")
    
    # Generate Table 1: Main results
    print("\n[1/4] Generating Table 1 (Main Results)...")
    table1_latex = format_main_table(results_df)
    table1_file = output_dir / 'table1_main_results.tex'
    with open(table1_file, 'w') as f:
        f.write(table1_latex)
    print(f"      Saved: {table1_file}")
    
    # Load detailed results for breakdown tables
    if detailed_results_json.exists():
        with open(detailed_results_json, 'r') as f:
            detailed_results = json.load(f)
        
        # Determine datasets
        first_llm = next(iter(detailed_results.values()))
        datasets = list(first_llm['by_dataset'].keys())
        
        # Generate Table 2: Per-dataset breakdown
        print("\n[2/4] Generating Table 2 (Per-Dataset Breakdown)...")
        table2_latex = format_per_dataset_table(detailed_results, datasets)
        table2_file = output_dir / 'table2_dataset_breakdown.tex'
        with open(table2_file, 'w') as f:
            f.write(table2_latex)
        print(f"      Saved: {table2_file}")
        
        # Generate Table 3: Per-algorithm breakdown
        print("\n[3/4] Generating Table 3 (Per-Algorithm Breakdown)...")
        table3_latex = format_per_algorithm_table(detailed_results)
        table3_file = output_dir / 'table3_algorithm_breakdown.tex'
        with open(table3_file, 'w') as f:
            f.write(table3_latex)
        print(f"      Saved: {table3_file}")
    else:
        print("\nWarning: detailed_results.json not found, skipping breakdown tables")
    
    # Generate Markdown summary
    print("\n[4/4] Generating README summary...")
    readme = f"""# Evaluation Results

Generated from: {input_dir}

## Main Results

{results_df.to_markdown(index=False)}

## Key Findings

"""
    
    best_row = results_df.iloc[0]
    readme += f"- **Best Model**: {best_row['LLM']} with {float(best_row['Calibrated_Coverage']):.1%} coverage\n"
    
    random_row = results_df[results_df['LLM'] == 'random']
    if not random_row.empty:
        random_coverage = float(random_row.iloc[0]['Calibrated_Coverage'])
        readme += f"- **Random Baseline**: {random_coverage:.1%}\n"
    
    readme += "\n## Files\n\n"
    readme += "- `table1_main_results.tex`: Main results table for paper\n"
    readme += "- `table2_dataset_breakdown.tex`: Performance by dataset\n"
    readme += "- `table3_algorithm_breakdown.tex`: Performance by algorithm\n"
    
    readme_file = output_dir / 'README.md'
    with open(readme_file, 'w') as f:
        f.write(readme)
    print(f"      Saved: {readme_file}")
    
    print("\n" + "="*70)
    print("TABLE GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nReady to include in paper!")


if __name__ == '__main__':
    main()
