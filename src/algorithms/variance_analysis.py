#!/usr/bin/env python3
"""
Robust Variance Analysis for Causal Discovery Algorithms
---------------------------------------------------------
This script addresses the methodological flaw in the original paper:
treating single algorithm runs as "ground truth" when algorithms are stochastic.

Key improvements:
1. Run each algorithm 100+ times with different random seeds
2. Report mean ± std for all metrics
3. Compute confidence intervals (95%)
4. Test statistical significance of LLM prediction accuracy
5. Measure overlap between LLM ranges and algorithmic CIs

Usage:
    python variance_analysis.py --dataset titanic --algorithm lingam --runs 100
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MetricStats:
    """Statistical summary of algorithm performance across multiple runs."""
    mean: float
    std: float
    ci_lower: float  # 95% confidence interval
    ci_upper: float
    median: float
    min_val: float
    max_val: float
    runs: int
    
    def to_dict(self) -> Dict:
        return {
            'mean': float(self.mean),
            'std': float(self.std),
            'ci_95_lower': float(self.ci_lower),
            'ci_95_upper': float(self.ci_upper),
            'median': float(self.median),
            'min': float(self.min_val),
            'max': float(self.max_val),
            'n_runs': int(self.runs)
        }
    
    def overlaps_with_range(self, lower: float, upper: float) -> bool:
        """Check if LLM prediction range overlaps with algorithmic CI."""
        return not (self.ci_upper < lower or self.ci_lower > upper)
    
    def contains_range(self, lower: float, upper: float) -> bool:
        """Check if algorithmic CI fully contains LLM range."""
        return self.ci_lower <= lower and self.ci_upper >= upper
    
    def range_within_ci(self, lower: float, upper: float) -> bool:
        """Check if LLM range fully contains algorithmic CI."""
        return lower <= self.ci_lower and upper >= self.ci_upper


@dataclass
class AlgorithmResults:
    """Results from multiple runs of a causal discovery algorithm."""
    precision: MetricStats
    recall: MetricStats
    f1: MetricStats
    shd: MetricStats
    accuracy: Optional[MetricStats] = None  # Only for Titanic
    
    def to_dict(self) -> Dict:
        result = {
            'precision': self.precision.to_dict(),
            'recall': self.recall.to_dict(),
            'f1': self.f1.to_dict(),
            'shd': self.shd.to_dict()
        }
        if self.accuracy:
            result['accuracy'] = self.accuracy.to_dict()
        return result


def compute_metric_stats(values: np.ndarray) -> MetricStats:
    """Compute statistical summary with 95% CI using bootstrap."""
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std
    
    # Bootstrap 95% CI
    n_bootstrap = 10000
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
    
    return MetricStats(
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        median=np.median(values),
        min_val=np.min(values),
        max_val=np.max(values),
        runs=len(values)
    )


def compute_shd(true_graph: np.ndarray, learned_graph: np.ndarray) -> int:
    """
    Compute Structural Hamming Distance between two adjacency matrices.
    
    SHD counts:
    - Missing edges (in true but not learned)
    - Extra edges (in learned but not true)
    - Reversed edges (wrong direction)
    """
    n = true_graph.shape[0]
    shd = 0
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            true_edge = true_graph[i, j]
            learned_edge = learned_graph[i, j]
            
            if true_edge == 1 and learned_edge == 0:
                # Missing edge
                shd += 1
            elif true_edge == 0 and learned_edge == 1:
                # Extra edge
                shd += 1
            elif true_edge == 1 and learned_edge == 1:
                # Check if reversed
                if true_graph[j, i] == 0 and learned_graph[j, i] == 1:
                    # Edge exists in both but reversed
                    shd += 1
    
    return shd


def compute_precision_recall_f1(true_graph: np.ndarray, 
                                  learned_graph: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute precision, recall, F1 for edge recovery.
    
    For directed graphs:
    - True positive: edge in same direction in both graphs
    - False positive: edge in learned but not in true
    - False negative: edge in true but not in learned
    """
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    n = true_graph.shape[0]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            true_edge = true_graph[i, j]
            learned_edge = learned_graph[i, j]
            
            if true_edge == 1 and learned_edge == 1:
                tp += 1
            elif true_edge == 0 and learned_edge == 1:
                fp += 1
            elif true_edge == 1 and learned_edge == 0:
                fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


class VarianceAnalyzer:
    """
    Runs causal discovery algorithms multiple times with proper variance analysis.
    """
    
    def __init__(self, n_runs: int = 100, output_dir: str = "results"):
        self.n_runs = n_runs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def run_lingam_multiple(self, 
                           data: pd.DataFrame, 
                           true_graph: np.ndarray,
                           prior_knowledge: Optional[List] = None) -> AlgorithmResults:
        """Run LiNGAM multiple times with different random initializations."""
        from causallearn.search.FCMBased import lingam
        
        precision_vals = []
        recall_vals = []
        f1_vals = []
        shd_vals = []
        
        for run in tqdm(range(self.n_runs), desc="LiNGAM runs"):
            try:
                # Set different random seed for each run
                np.random.seed(run)

                # Bootstrap sample: randomly sample rows with replacement
                n_samples = len(data)
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                bootstrap_data = data.iloc[bootstrap_indices].values

                # Run LiNGAM on bootstrap sample
                model = lingam.DirectLiNGAM()
                model.fit(bootstrap_data)
                learned_graph = (model.adjacency_matrix_ != 0).astype(int)
                
                # Compute metrics
                prec, rec, f1 = compute_precision_recall_f1(true_graph, learned_graph)
                shd = compute_shd(true_graph, learned_graph)
                
                precision_vals.append(prec)
                recall_vals.append(rec)
                f1_vals.append(f1)
                shd_vals.append(shd)
                
            except Exception as e:
                print(f"Run {run} failed: {e}")
                # Use zeros for failed runs
                precision_vals.append(0.0)
                recall_vals.append(0.0)
                f1_vals.append(0.0)
                shd_vals.append(len(true_graph) ** 2)  # Maximum possible SHD
        
        return AlgorithmResults(
            precision=compute_metric_stats(precision_vals),
            recall=compute_metric_stats(recall_vals),
            f1=compute_metric_stats(f1_vals),
            shd=compute_metric_stats(shd_vals)
        )
    
    def run_pc_multiple(self,
                       data: pd.DataFrame,
                       true_graph: np.ndarray,
                       alpha: float = 0.05) -> AlgorithmResults:
        """Run PC algorithm multiple times with different significance levels and bootstrap samples."""
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        precision_vals = []
        recall_vals = []
        f1_vals = []
        shd_vals = []

        # Vary both random seed AND significance level for true variance
        alphas = np.linspace(0.01, 0.1, self.n_runs)

        for run in tqdm(range(self.n_runs), desc="PC runs"):
            try:
                np.random.seed(run)

                # Bootstrap sample: randomly sample rows with replacement
                # This creates data variance while keeping the same underlying distribution
                n_samples = len(data)
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                bootstrap_data = data.iloc[bootstrap_indices].values

                # Run PC with varying alpha on bootstrap sample
                cg = pc(bootstrap_data, alpha=alphas[run], indep_test=fisherz)
                learned_graph = cg.G.graph

                # Convert to binary adjacency matrix
                learned_adj = (np.abs(learned_graph) > 0).astype(int)

                # Compute metrics
                prec, rec, f1 = compute_precision_recall_f1(true_graph, learned_adj)
                shd = compute_shd(true_graph, learned_adj)

                precision_vals.append(prec)
                recall_vals.append(rec)
                f1_vals.append(f1)
                shd_vals.append(shd)

            except Exception as e:
                print(f"Run {run} failed: {e}")
                precision_vals.append(0.0)
                recall_vals.append(0.0)
                f1_vals.append(0.0)
                shd_vals.append(len(true_graph) ** 2)
        
        return AlgorithmResults(
            precision=compute_metric_stats(precision_vals),
            recall=compute_metric_stats(recall_vals),
            f1=compute_metric_stats(f1_vals),
            shd=compute_metric_stats(shd_vals)
        )

    def run_fci_multiple(self,
                        data: pd.DataFrame,
                        true_graph: np.ndarray,
                        alpha: float = 0.05) -> AlgorithmResults:
        """Run FCI algorithm multiple times with different significance levels and bootstrap samples.

        FCI (Fast Causal Inference) handles latent confounders and selection bias,
        outputting a Partial Ancestral Graph (PAG) with various edge types.
        """
        from causallearn.search.ConstraintBased.FCI import fci
        from causallearn.utils.cit import fisherz

        precision_vals = []
        recall_vals = []
        f1_vals = []
        shd_vals = []

        # Vary both random seed AND significance level for true variance
        alphas = np.linspace(0.01, 0.1, self.n_runs)

        for run in tqdm(range(self.n_runs), desc="FCI runs"):
            try:
                np.random.seed(run)

                # Bootstrap sample: randomly sample rows with replacement
                n_samples = len(data)
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                bootstrap_data = data.iloc[bootstrap_indices].values

                # Run FCI with varying alpha on bootstrap sample
                G, edges = fci(bootstrap_data, independence_test_method=fisherz, alpha=float(alphas[run]))
                learned_graph = G.graph

                # Convert PAG to binary adjacency matrix (any edge type counts)
                learned_adj = (np.abs(learned_graph) > 0).astype(int)

                # Compute metrics
                prec, rec, f1 = compute_precision_recall_f1(true_graph, learned_adj)
                shd = compute_shd(true_graph, learned_adj)

                precision_vals.append(prec)
                recall_vals.append(rec)
                f1_vals.append(f1)
                shd_vals.append(shd)

            except Exception as e:
                print(f"Run {run} failed: {e}")
                precision_vals.append(0.0)
                recall_vals.append(0.0)
                f1_vals.append(0.0)
                shd_vals.append(len(true_graph) ** 2)

        return AlgorithmResults(
            precision=compute_metric_stats(precision_vals),
            recall=compute_metric_stats(recall_vals),
            f1=compute_metric_stats(f1_vals),
            shd=compute_metric_stats(shd_vals)
        )

    def run_notears_multiple(self,
                            data: pd.DataFrame,
                            true_graph: np.ndarray) -> AlgorithmResults:
        """Run NOTEARS algorithm multiple times with different regularization and bootstrap samples.

        NOTEARS uses continuous optimization with a differentiable acyclicity constraint
        (via matrix exponential) and L1 regularization for sparsity.
        """
        try:
            from notears_pytorch import notears_linear
        except ImportError:
            print("NOTEARS not installed. Install with: pip install notears-pytorch")
            raise

        precision_vals = []
        recall_vals = []
        f1_vals = []
        shd_vals = []

        # Vary L1 regularization lambda1 to create variance
        lambda1_values = np.linspace(0.01, 0.1, self.n_runs)

        for run in tqdm(range(self.n_runs), desc="NOTEARS runs"):
            try:
                np.random.seed(run)

                # Bootstrap sample: randomly sample rows with replacement
                n_samples = len(data)
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                bootstrap_data = data.iloc[bootstrap_indices].values

                # Run NOTEARS with varying lambda1 on bootstrap sample
                W_est = notears_linear(bootstrap_data, lambda1=lambda1_values[run])

                # Convert to binary adjacency matrix (threshold small weights)
                learned_graph = (np.abs(W_est) > 0.3).astype(int)

                # Compute metrics
                prec, rec, f1 = compute_precision_recall_f1(true_graph, learned_graph)
                shd = compute_shd(true_graph, learned_graph)

                precision_vals.append(prec)
                recall_vals.append(rec)
                f1_vals.append(f1)
                shd_vals.append(shd)

            except Exception as e:
                print(f"Run {run} failed: {e}")
                precision_vals.append(0.0)
                recall_vals.append(0.0)
                f1_vals.append(0.0)
                shd_vals.append(len(true_graph) ** 2)

        return AlgorithmResults(
            precision=compute_metric_stats(precision_vals),
            recall=compute_metric_stats(recall_vals),
            f1=compute_metric_stats(f1_vals),
            shd=compute_metric_stats(shd_vals)
        )

    def save_results(self, results: AlgorithmResults, 
                     dataset: str, algorithm: str):
        """Save results to JSON file."""
        output = {
            'dataset': dataset,
            'algorithm': algorithm,
            'n_runs': self.n_runs,
            'results': results.to_dict()
        }
        
        filename = self.output_dir / f"{dataset}_{algorithm}_variance.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def compare_with_llm_estimates(self, 
                                   results: AlgorithmResults,
                                   llm_estimates: Dict[str, Tuple[float, float]]) -> Dict:
        """
        Compare algorithmic variance with LLM prediction ranges.
        
        Args:
            results: AlgorithmResults with CIs
            llm_estimates: Dict mapping metric names to (lower, upper) tuples
        
        Returns:
            Dict with overlap analysis
        """
        comparison = {}
        
        for metric in ['precision', 'recall', 'f1', 'shd']:
            stats = getattr(results, metric)
            
            if metric not in llm_estimates:
                continue
            
            llm_lower, llm_upper = llm_estimates[metric]
            
            comparison[metric] = {
                'algorithmic_mean': float(stats.mean),
                'algorithmic_ci': (float(stats.ci_lower), float(stats.ci_upper)),
                'llm_range': (float(llm_lower), float(llm_upper)),
                'overlaps': bool(stats.overlaps_with_range(llm_lower, llm_upper)),
                'llm_contains_ci': bool(llm_lower <= stats.ci_lower and llm_upper >= stats.ci_upper),
                'ci_contains_llm': bool(stats.ci_lower <= llm_lower and stats.ci_upper >= llm_upper),
                'overlap_percentage': float(self._compute_overlap_percentage(
                    stats.ci_lower, stats.ci_upper, llm_lower, llm_upper
                ))
            }
        
        return comparison
    
    @staticmethod
    def _compute_overlap_percentage(ci_low, ci_high, llm_low, llm_high) -> float:
        """Compute percentage of overlap between two intervals."""
        overlap_low = max(ci_low, llm_low)
        overlap_high = min(ci_high, llm_high)
        
        if overlap_low >= overlap_high:
            return 0.0
        
        overlap_width = overlap_high - overlap_low
        ci_width = ci_high - ci_low
        llm_width = llm_high - llm_low
        
        # Return overlap as percentage of smaller interval
        return 100 * overlap_width / min(ci_width, llm_width)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Variance analysis for causal discovery")
    parser.add_argument('--runs', type=int, default=100, 
                       help='Number of runs per algorithm (default: 100)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    analyzer = VarianceAnalyzer(n_runs=args.runs, output_dir=args.output)
    
    print(f"Variance Analyzer initialized with {args.runs} runs per algorithm")
    print(f"Results will be saved to: {args.output}/")
    print("\nReady to run experiments. Use run_experiments.py to execute.")
