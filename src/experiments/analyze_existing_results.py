"""
Analyze existing variance results using statistical analysis modules.

This script loads pre-computed variance analysis results from ./results/
and runs statistical analysis without needing to re-run experiments.

Usage:
    python analyze_existing_results.py
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent (src) to path for statistical_analysis import
sys.path.insert(0, str(Path(__file__).parent.parent))

from statistical_analysis import StatisticalTester, ExplanatoryAnalyzer


def load_variance_results() -> Dict[str, Dict]:
    """Load all variance JSON files from results/ folder."""
    results_dir = Path(__file__).parent / "results"
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    variance_files = sorted(results_dir.glob("*_variance.json"))
    
    if not variance_files:
        raise FileNotFoundError(f"No variance JSON files found in {results_dir}")
    
    print(f"Found {len(variance_files)} variance result files:")
    for f in variance_files:
        print(f"  • {f.name}")
    
    all_results = {}
    for file_path in variance_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            key = f"{data['dataset']}_{data['algorithm']}"
            all_results[key] = data
            print(f"  Loaded: {key}")
    
    return all_results


def extract_metric_by_algorithm(results: Dict[str, Dict], metric: str = 'f1') -> Dict[str, List[float]]:
    """Extract specified metric grouped by algorithm.
    
    Args:
        results: Dict of all variance results
        metric: Which metric to extract (f1, precision, recall, shd)
        
    Returns:
        Dict mapping algorithm name to list of mean values
    """
    algo_values = {}
    
    for key, data in results.items():
        parts = key.split('_')
        algorithm = parts[-1]  # Last part is algorithm
        
        if algorithm not in algo_values:
            algo_values[algorithm] = []
        
        if metric in data['results']:
            mean_value = data['results'][metric]['mean']
            algo_values[algorithm].append(mean_value)
    
    return algo_values


def main():
    print("=" * 80)
    print("STATISTICAL ANALYSIS OF EXISTING VARIANCE RESULTS")
    print("=" * 80)
    
    # Load results
    print("\n1. Loading variance results...")
    print("-" * 80)
    results = load_variance_results()
    print(f"\nTotal results loaded: {len(results)}")
    
    # ========================================================================
    # Part 1: Statistical Significance Testing
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. RUNNING STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)
    
    tester = StatisticalTester(alpha=0.05)
    
    # Extract F1 scores by algorithm
    algo_f1_scores = extract_metric_by_algorithm(results, 'f1')
    
    print(f"\nAlgorithms found: {list(algo_f1_scores.keys())}")
    for algo, scores in algo_f1_scores.items():
        print(f"  {algo}: {len(scores)} datasets, mean F1 = {sum(scores)/len(scores):.4f}")
    
    # Pairwise comparisons between algorithms
    statistical_results = []
    algo_list = sorted(algo_f1_scores.keys())
    
    print("\nPerforming pairwise algorithm comparisons...")
    for i in range(len(algo_list)):
        for j in range(i + 1, len(algo_list)):
            algo1, algo2 = algo_list[i], algo_list[j]
            
            if len(algo_f1_scores[algo1]) >= 2 and len(algo_f1_scores[algo2]) >= 2:
                result = tester.paired_t_test(
                    algo_f1_scores[algo1],
                    algo_f1_scores[algo2],
                    f"{algo1.upper()} vs {algo2.upper()}"
                )
                statistical_results.append(result)
                
                sig_marker = "✓ SIGNIFICANT" if result.is_significant else "  not significant"
                print(f"  {algo1} vs {algo2}: p={result.p_value:.4f}, d={result.effect_size:.3f} {sig_marker}")
    
    # Apply FDR correction
    if len(statistical_results) > 1:
        print(f"\nApplying FDR multiple comparison correction ({len(statistical_results)} tests)...")
        corrected_results = tester.multiple_comparison_correction(statistical_results, method='fdr')
        
        sig_before = sum(1 for r in statistical_results if r.is_significant)
        sig_after = sum(1 for r in corrected_results if r.corrected_is_significant)
        
        print(f"  Significant before correction: {sig_before}/{len(statistical_results)}")
        print(f"  Significant after FDR correction: {sig_after}/{len(corrected_results)}")
    else:
        corrected_results = statistical_results
    
    # Save statistical report
    report_path = Path(__file__).parent / "results" / "statistical_analysis_report.txt"
    tester.generate_statistical_report(corrected_results, str(report_path))
    print(f"\n✓ Statistical report saved: {report_path}")
    
    # ========================================================================
    # Part 2: Explanatory Factor Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. EXPLANATORY FACTOR ANALYSIS")
    print("=" * 80)
    
    analyzer = ExplanatoryAnalyzer()
    
    # Prepare data for explanatory analysis
    experimental_results = {}
    graph_structures = {}
    dataset_metadata = {}
    
    # Group results by dataset
    for key, data in results.items():
        parts = key.split('_')
        algorithm = parts[-1]
        dataset = '_'.join(parts[:-1])
        
        if dataset not in experimental_results:
            experimental_results[dataset] = {}
        
        experimental_results[dataset][algorithm] = {
            'accuracy': data['results']['f1']['mean'],
            'confidence_interval_width': data['results']['f1']['ci_95_upper'] - data['results']['f1']['ci_95_lower'],
            'calibration_error': abs(data['results']['precision']['mean'] - data['results']['recall']['mean'])
        }
        
        # Mock graph structures and metadata
        if dataset not in graph_structures:
            n_nodes_config = {
                'asia': 8,
                'sachs': 11,
                'synthetic_12': 12,
                'synthetic_30': 30
            }
            n_nodes = n_nodes_config.get(dataset, 10)
            
            # Create sparse DAG structure
            import numpy as np
            graph_structures[dataset] = np.random.binomial(1, 0.15, (n_nodes, n_nodes))
            
            sample_size_config = {
                'asia': 5000,
                'sachs': 7466,
                'synthetic_12': 1000,
                'synthetic_30': 1000
            }
            
            dataset_metadata[dataset] = {
                'sample_size': sample_size_config.get(dataset, 1000),
                'dimensionality': n_nodes,
                'noise_level': 0.05
            }
    
    # Run explanatory analysis
    mock_llm_results = {'Algorithm_Results': experimental_results}
    insights = analyzer.analyze_performance_factors(
        mock_llm_results, graph_structures, dataset_metadata
    )
    
    print(f"\nIdentified {len(insights)} explanatory factors:")
    for insight in sorted(insights, key=lambda x: x.impact_score, reverse=True):
        print(f"  • {insight.factor_name}")
        print(f"    Impact Score: {insight.impact_score:.1f}%")
        print(f"    Correlation: {insight.correlation:.3f}")
        print(f"    {insight.description}")
    
    # Save theory report
    theory_report_path = Path(__file__).parent / "results" / "explanatory_theory_report.txt"
    analyzer.generate_theory_report(insights, str(theory_report_path))
    print(f"\n✓ Explanatory theory report saved: {theory_report_path}")
    
    print(f"\nGenerated files:")
    print(f"  1. {report_path}")
    print(f"  2. {theory_report_path}")
    
    print(f"\nDatasets analyzed: {len(experimental_results)}")
    print(f"  {', '.join(sorted(experimental_results.keys()))}")
    
    print(f"\nAlgorithms analyzed: {len(algo_list)}")
    print(f"  {', '.join(sorted(algo_list))}")
    
    print(f"\nStatistical tests performed: {len(corrected_results)}")
    print(f"Significant findings (after FDR): {sum(1 for r in corrected_results if r.corrected_is_significant)}")


if __name__ == '__main__':
    main()
