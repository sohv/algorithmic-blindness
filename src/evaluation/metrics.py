#!/usr/bin/env python3
"""
Evaluation Metrics for LLM Algorithm Performance Prediction
============================================================

Primary Metric: Calibrated Coverage
- What: % of predictions where true algorithmic mean ∈ LLM predicted range
- Why: Directly tests if LLM can estimate algorithmic uncertainty
- Target: >60% indicates genuine understanding (random baseline ~15-20%)

Secondary Metrics:
- MAE: Mean Absolute Error of LLM midpoint vs true mean
- Width: Average prediction range width (calibration quality)
- Ranking Correlation: Spearman ρ for algorithm rankings per dataset
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluationResult:
    """Container for LLM evaluation results."""
    llm_name: str
    calibrated_coverage: float
    mae: float
    mean_width: float
    n_predictions: int
    coverage_by_metric: Dict[str, float]
    coverage_by_dataset: Dict[str, float]
    coverage_by_algorithm: Dict[str, float]
    ranking_correlation: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'LLM': self.llm_name,
            'Calibrated_Coverage': f"{self.calibrated_coverage:.3f}",
            'MAE': f"{self.mae:.3f}",
            'Mean_Width': f"{self.mean_width:.3f}",
            'N_predictions': self.n_predictions,
            'Ranking_Correlation': f"{self.ranking_correlation:.3f}" if self.ranking_correlation else "N/A"
        }


def compute_calibrated_coverage(llm_predictions: Dict, 
                                ground_truth: Dict,
                                metrics: List[str] = ['precision', 'recall', 'f1', 'shd']) -> Dict:
    """
    Compute calibrated coverage: % where true_mean ∈ LLM_range
    
    Args:
        llm_predictions: {dataset: {algorithm: {metric: (lower, upper)}}}
        ground_truth: {dataset: {algorithm: {metric: {'mean': X, 'ci_lower': Y, 'ci_upper': Z}}}}
        metrics: Metrics to evaluate
        
    Returns:
        Dict with overall and per-category coverage statistics
    """
    correct_total = 0
    total_count = 0
    
    # Track by category
    by_metric = {m: {'correct': 0, 'total': 0} for m in metrics}
    by_dataset = {}
    by_algorithm = {}
    
    for dataset in llm_predictions:
        if dataset not in ground_truth:
            continue
            
        if dataset not in by_dataset:
            by_dataset[dataset] = {'correct': 0, 'total': 0}
            
        for algorithm in llm_predictions[dataset]:
            if algorithm not in ground_truth[dataset]:
                continue
                
            if algorithm not in by_algorithm:
                by_algorithm[algorithm] = {'correct': 0, 'total': 0}
                
            for metric in metrics:
                if metric not in llm_predictions[dataset][algorithm]:
                    continue
                if metric not in ground_truth[dataset][algorithm]:
                    continue
                    
                llm_lower, llm_upper = llm_predictions[dataset][algorithm][metric]
                gt_mean = ground_truth[dataset][algorithm][metric]['mean']
                
                # Check if true mean falls within LLM range
                is_correct = llm_lower <= gt_mean <= llm_upper
                
                if is_correct:
                    correct_total += 1
                    by_metric[metric]['correct'] += 1
                    by_dataset[dataset]['correct'] += 1
                    by_algorithm[algorithm]['correct'] += 1
                
                total_count += 1
                by_metric[metric]['total'] += 1
                by_dataset[dataset]['total'] += 1
                by_algorithm[algorithm]['total'] += 1
    
    # Compute percentages
    overall_coverage = correct_total / total_count if total_count > 0 else 0.0
    
    coverage_by_metric = {
        m: stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        for m, stats in by_metric.items()
    }
    
    coverage_by_dataset = {
        d: stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        for d, stats in by_dataset.items()
    }
    
    coverage_by_algorithm = {
        a: stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        for a, stats in by_algorithm.items()
    }
    
    return {
        'overall': overall_coverage,
        'by_metric': coverage_by_metric,
        'by_dataset': coverage_by_dataset,
        'by_algorithm': coverage_by_algorithm,
        'n_predictions': total_count
    }


def compute_mae(llm_predictions: Dict,
                ground_truth: Dict,
                metrics: List[str] = ['precision', 'recall', 'f1']) -> float:
    """
    Mean Absolute Error between LLM midpoint and true mean.
    
    Note: Excludes SHD as it has different scale than [0,1] metrics.
    
    Args:
        llm_predictions: {dataset: {algorithm: {metric: (lower, upper)}}}
        ground_truth: {dataset: {algorithm: {metric: {'mean': X}}}}
        metrics: Metrics to include (default excludes SHD)
        
    Returns:
        Mean absolute error
    """
    errors = []
    
    for dataset in llm_predictions:
        if dataset not in ground_truth:
            continue
            
        for algorithm in llm_predictions[dataset]:
            if algorithm not in ground_truth[dataset]:
                continue
                
            for metric in metrics:
                if metric not in llm_predictions[dataset][algorithm]:
                    continue
                if metric not in ground_truth[dataset][algorithm]:
                    continue
                    
                llm_lower, llm_upper = llm_predictions[dataset][algorithm][metric]
                llm_midpoint = (llm_lower + llm_upper) / 2
                
                gt_mean = ground_truth[dataset][algorithm][metric]['mean']
                
                errors.append(abs(llm_midpoint - gt_mean))
    
    return np.mean(errors) if errors else np.nan


def compute_mean_width(llm_predictions: Dict,
                       metrics: List[str] = ['precision', 'recall', 'f1']) -> float:
    """
    Compute average width of LLM prediction ranges.
    Indicates calibration: too wide = underconfident, too narrow = overconfident.
    
    Args:
        llm_predictions: {dataset: {algorithm: {metric: (lower, upper)}}}
        metrics: Metrics to include
        
    Returns:
        Mean range width
    """
    widths = []
    
    for dataset in llm_predictions:
        for algorithm in llm_predictions[dataset]:
            for metric in metrics:
                if metric not in llm_predictions[dataset][algorithm]:
                    continue
                    
                llm_lower, llm_upper = llm_predictions[dataset][algorithm][metric]
                width = llm_upper - llm_lower
                widths.append(width)
    
    return np.mean(widths) if widths else np.nan


def compute_ranking_correlation(llm_predictions: Dict,
                                 ground_truth: Dict,
                                 metric: str = 'f1') -> float:
    """
    Compute Spearman correlation for algorithm rankings per dataset.
    
    Tests if LLM can rank algorithms correctly even if absolute values are off.
    
    Args:
        llm_predictions: {dataset: {algorithm: {metric: (lower, upper)}}}
        ground_truth: {dataset: {algorithm: {metric: {'mean': X}}}}
        metric: Which metric to use for ranking (default: F1)
        
    Returns:
        Average Spearman ρ across all datasets
    """
    correlations = []
    
    for dataset in llm_predictions:
        if dataset not in ground_truth:
            continue
        
        algorithms = []
        llm_rankings = []
        gt_rankings = []
        
        for algorithm in llm_predictions[dataset]:
            if algorithm not in ground_truth[dataset]:
                continue
            if metric not in llm_predictions[dataset][algorithm]:
                continue
            if metric not in ground_truth[dataset][algorithm]:
                continue
                
            # LLM ranking based on midpoint
            llm_lower, llm_upper = llm_predictions[dataset][algorithm][metric]
            llm_midpoint = (llm_lower + llm_upper) / 2
            
            # Ground truth ranking based on mean
            gt_mean = ground_truth[dataset][algorithm][metric]['mean']
            
            algorithms.append(algorithm)
            llm_rankings.append(llm_midpoint)
            gt_rankings.append(gt_mean)
        
        # Need at least 3 algorithms to compute meaningful correlation
        if len(algorithms) >= 3:
            rho, _ = spearmanr(llm_rankings, gt_rankings)
            if not np.isnan(rho):
                correlations.append(rho)
    
    return np.mean(correlations) if correlations else np.nan


def evaluate_llm(llm_name: str,
                 llm_predictions: Dict,
                 ground_truth: Dict) -> EvaluationResult:
    """
    Comprehensive evaluation of a single LLM.
    
    Args:
        llm_name: Name of LLM (e.g., 'gpt4', 'claude')
        llm_predictions: LLM's predictions
        ground_truth: Ground truth from algorithmic runs
        
    Returns:
        EvaluationResult object with all metrics
    """
    # Primary metric: Calibrated coverage
    coverage_stats = compute_calibrated_coverage(llm_predictions, ground_truth)
    
    # Secondary metrics
    mae = compute_mae(llm_predictions, ground_truth)
    mean_width = compute_mean_width(llm_predictions)
    ranking_corr = compute_ranking_correlation(llm_predictions, ground_truth)
    
    return EvaluationResult(
        llm_name=llm_name,
        calibrated_coverage=coverage_stats['overall'],
        mae=mae,
        mean_width=mean_width,
        n_predictions=coverage_stats['n_predictions'],
        coverage_by_metric=coverage_stats['by_metric'],
        coverage_by_dataset=coverage_stats['by_dataset'],
        coverage_by_algorithm=coverage_stats['by_algorithm'],
        ranking_correlation=ranking_corr
    )


def evaluate_all_llms(ground_truth: Dict,
                      llm_predictions_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Evaluate all LLMs and return comparison table.
    
    Args:
        ground_truth: Ground truth results
        llm_predictions_dict: {llm_name: predictions}
        
    Returns:
        DataFrame with results sorted by calibrated coverage
    """
    results = []
    
    for llm_name, predictions in llm_predictions_dict.items():
        eval_result = evaluate_llm(llm_name, predictions, ground_truth)
        results.append(eval_result.to_dict())
    
    df = pd.DataFrame(results)
    
    # Sort by calibrated coverage (descending)
    df['_sort_key'] = df['Calibrated_Coverage'].astype(float)
    df = df.sort_values('_sort_key', ascending=False)
    df = df.drop(columns=['_sort_key'])
    
    return df


if __name__ == '__main__':
    # Quick test
    print("Evaluation Metrics Module")
    print("=" * 60)
    print("\nPrimary Metric: Calibrated Coverage")
    print("  - Measures: % predictions where true_mean ∈ LLM_range")
    print("  - Target: >60% indicates understanding")
    print("  - Random baseline: ~15-20%")
    print("\nSecondary Metrics:")
    print("  - MAE: Mean absolute error of midpoints")
    print("  - Mean Width: Average range width (calibration)")
    print("  - Ranking Correlation: Spearman ρ for algorithm rankings")
