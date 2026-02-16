#!/usr/bin/env python3
"""
Simple Baselines for LLM Comparison
====================================

Two baselines to establish minimum performance thresholds:

1. Random Baseline: Uniformly random ranges
   - Purpose: Lower bound (if LLM ≤ random, it's useless)
   - Expected coverage: ~15-20%

2. Heuristic Baseline: Simple rules based on problem characteristics
   - Purpose: Can simple heuristics beat LLMs?
   - Rules: Larger N → better, more variables → worse
   - Expected coverage: ~35-50%
"""

import numpy as np
from typing import Dict, Tuple


class RandomBaseline:
    """
    Random baseline: Generate random prediction ranges.
    
    For each metric, samples random (lower, upper) pairs.
    Expected to capture true mean ~15-20% of time.
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.name = "random"
    
    def predict(self, dataset_name: str, algorithm_name: str, 
                n_samples: int, n_variables: int) -> Dict[str, Tuple[float, float]]:
        """
        Generate random predictions for all metrics.
        
        Args:
            dataset_name: Name of dataset (unused)
            algorithm_name: Name of algorithm (unused)
            n_samples: Number of samples (unused)
            n_variables: Number of variables (unused)
            
        Returns:
            Dict with random ranges for each metric
        """
        predictions = {}
        
        # For precision, recall, f1: random ranges in [0, 1]
        for metric in ['precision', 'recall', 'f1']:
            # Generate sorted pair for (lower, upper)
            vals = self.rng.uniform(0.0, 1.0, size=2)
            lower, upper = np.sort(vals)
            predictions[metric] = (float(lower), float(upper))
        
        # SHD: random range in [0, max_possible_errors]
        # max_possible_errors ≈ n_variables * (n_variables - 1) / 2
        max_shd = n_variables * (n_variables - 1) // 2
        vals = self.rng.uniform(0, max_shd, size=2)
        lower, upper = np.sort(vals)
        predictions['shd'] = (float(lower), float(upper))
        
        return predictions


class HeuristicBaseline:
    """
    Heuristic baseline: Simple rules based on algorithm/dataset properties.
    
    Rules:
    - Larger sample size → better performance
    - More variables → worse performance  
    - Algorithm assumptions matter (but simplified)
    - Base performance around 0.5-0.6 for most cases
    """
    
    def __init__(self):
        self.name = "heuristic"
        
        # Algorithm-specific base performance
        # (based on general knowledge, not fitted to data)
        self.algorithm_bases = {
            'PC': 0.55,      # Constraint-based, moderate performance
            'LiNGAM': 0.60,  # Often performs well on continuous data
            'FCI': 0.50,     # More conservative due to latent variables
            'NOTEARS': 0.58  # Modern, generally good
        }
    
    def predict(self, dataset_name: str, algorithm_name: str,
                n_samples: int, n_variables: int) -> Dict[str, Tuple[float, float]]:
        """
        Generate heuristic predictions based on problem characteristics.
        
        Args:
            dataset_name: Name of dataset
            algorithm_name: Name of algorithm
            n_samples: Number of samples
            n_variables: Number of variables
            
        Returns:
            Dict with heuristic ranges for each metric
        """
        # Get algorithm base performance
        base = self.algorithm_bases.get(algorithm_name, 0.55)
        
        # Adjustment based on sample size
        # More samples → better (logarithmic effect)
        sample_adjustment = 0.1 * np.log10(n_samples / 1000) if n_samples > 0 else 0
        sample_adjustment = np.clip(sample_adjustment, -0.15, 0.15)
        
        # Adjustment based on dimensionality
        # More variables → worse (linear penalty)
        dim_adjustment = -0.015 * n_variables
        dim_adjustment = np.clip(dim_adjustment, -0.20, 0)
        
        # Combined adjustment
        adjusted_base = base + sample_adjustment + dim_adjustment
        adjusted_base = np.clip(adjusted_base, 0.2, 0.9)
        
        # Create range with ±0.10 around base
        width = 0.20  # Conservative width
        
        predictions = {}
        
        # Precision, recall, f1: same heuristic
        for metric in ['precision', 'recall', 'f1']:
            lower = max(0.0, adjusted_base - width/2)
            upper = min(1.0, adjusted_base + width/2)
            predictions[metric] = (float(lower), float(upper))
        
        # SHD: inversely related to performance
        # Higher performance → fewer errors
        expected_f1 = adjusted_base
        # Rough heuristic: SHD ∝ (1 - F1) × max_edges
        max_edges = n_variables * (n_variables - 1) // 2
        expected_shd = (1 - expected_f1) * max_edges * 0.5  # Scale factor
        
        shd_lower = max(0, expected_shd - max_edges * 0.1)
        shd_upper = min(max_edges, expected_shd + max_edges * 0.1)
        
        predictions['shd'] = (float(shd_lower), float(shd_upper))
        
        return predictions


def generate_baseline_predictions(datasets_info: Dict[str, Dict],
                                   algorithms: list,
                                   baseline_type: str = 'both') -> Dict[str, Dict]:
    """
    Generate baseline predictions for all datasets and algorithms.
    
    Args:
        datasets_info: {dataset_name: {'n_samples': X, 'n_variables': Y}}
        algorithms: List of algorithm names
        baseline_type: 'random', 'heuristic', or 'both'
        
    Returns:
        {baseline_name: {dataset: {algorithm: {metric: (lower, upper)}}}}
    """
    baselines = {}
    
    if baseline_type in ['random', 'both']:
        random_bl = RandomBaseline()
        random_preds = {}
        
        for dataset_name, info in datasets_info.items():
            random_preds[dataset_name] = {}
            for algorithm in algorithms:
                random_preds[dataset_name][algorithm] = random_bl.predict(
                    dataset_name, algorithm, 
                    info['n_samples'], info['n_variables']
                )
        
        baselines['random'] = random_preds
    
    if baseline_type in ['heuristic', 'both']:
        heuristic_bl = HeuristicBaseline()
        heuristic_preds = {}
        
        for dataset_name, info in datasets_info.items():
            heuristic_preds[dataset_name] = {}
            for algorithm in algorithms:
                heuristic_preds[dataset_name][algorithm] = heuristic_bl.predict(
                    dataset_name, algorithm,
                    info['n_samples'], info['n_variables']
                )
        
        baselines['heuristic'] = heuristic_preds
    
    return baselines


if __name__ == '__main__':
    # Demo
    print("Baseline Predictors Demo")
    print("=" * 60)
    
    # Test case: Titanic dataset with PC algorithm
    dataset = 'titanic'
    algorithm = 'PC'
    n_samples = 891
    n_variables = 7
    
    # Random baseline
    random_bl = RandomBaseline(seed=42)
    random_pred = random_bl.predict(dataset, algorithm, n_samples, n_variables)
    
    print(f"\n{dataset} - {algorithm} (N={n_samples}, d={n_variables})")
    print("\nRandom Baseline:")
    for metric, (lower, upper) in random_pred.items():
        print(f"  {metric:10s}: [{lower:.3f}, {upper:.3f}]")
    
    # Heuristic baseline
    heuristic_bl = HeuristicBaseline()
    heuristic_pred = heuristic_bl.predict(dataset, algorithm, n_samples, n_variables)
    
    print("\nHeuristic Baseline:")
    for metric, (lower, upper) in heuristic_pred.items():
        print(f"  {metric:10s}: [{lower:.3f}, {upper:.3f}]")
    
    # Test with different dataset sizes
    print("\n" + "=" * 60)
    print("Effect of Sample Size on Heuristic Predictions:")
    print("=" * 60)
    
    for n in [100, 500, 1000, 5000]:
        pred = heuristic_bl.predict('test', 'PC', n, 10)
        f1_mid = (pred['f1'][0] + pred['f1'][1]) / 2
        print(f"N={n:5d}: F1 midpoint = {f1_mid:.3f}")
