"""Statistical Rigor Testing Module."""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from pathlib import Path


@dataclass
class StatisticalResult:
    """Result of statistical test."""
    test_name: str
    group1_mean: float
    group2_mean: float
    p_value: float
    effect_size: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    corrected_p_value: Optional[float] = None
    corrected_is_significant: Optional[bool] = None
    
    def __str__(self):
        return (f"{self.test_name}: p={self.p_value:.4f}, "
                f"d={self.effect_size:.3f}, significant={self.is_significant}")


class StatisticalTester:
    """
    Comprehensive statistical testing for rigor enhancement.
    Implements paired t-tests, effect size calculations, and multiple comparison corrections.
    """
    
    def __init__(self, alpha: float = 0.05):
        """Initialize statistical tester.
        
        Args:
            alpha: Significance level (default 0.05)
        """
        self.alpha = alpha
        self.results: List[StatisticalResult] = []
    
    def paired_t_test(self, group1: List[float], group2: List[float], 
                     label: str = "Comparison") -> StatisticalResult:
        """Perform paired t-test between two groups.
        
        Args:
            group1: First group of values
            group2: Second group of values
            label: Test label/name
            
        Returns:
            StatisticalResult with test details
        """
        group1 = np.array(group1, dtype=float)
        group2 = np.array(group2, dtype=float)
        
        # Ensure equal lengths for paired test
        min_len = min(len(group1), len(group2))
        group1 = group1[:min_len]
        group2 = group2[:min_len]
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(group1, group2)
        
        # Cohen's d effect size
        diff = group1 - group2
        effect_size = np.mean(diff) / (np.std(diff) + 1e-10)
        
        # Confidence interval on mean difference
        mean_diff = np.mean(diff)
        sem_diff = stats.sem(diff)
        ci = stats.t.interval(0.95, len(diff) - 1, loc=mean_diff, scale=sem_diff)
        
        result = StatisticalResult(
            test_name=label,
            group1_mean=np.mean(group1),
            group2_mean=np.mean(group2),
            p_value=p_value,
            effect_size=effect_size,
            is_significant=p_value < self.alpha,
            confidence_interval=ci
        )
        
        self.results.append(result)
        return result
    
    def independent_t_test(self, group1: List[float], group2: List[float],
                          label: str = "Comparison") -> StatisticalResult:
        """Perform independent samples t-test.
        
        Args:
            group1: First group of values
            group2: Second group of values
            label: Test label/name
            
        Returns:
            StatisticalResult with test details
        """
        group1 = np.array(group1, dtype=float)
        group2 = np.array(group2, dtype=float)
        
        # Independent samples t-test
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        # Cohen's d effect size for independent samples
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        effect_size = (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)
        
        # Confidence interval
        mean_diff = np.mean(group1) - np.mean(group2)
        se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
        df = n1 + n2 - 2
        ci = stats.t.interval(0.95, df, loc=mean_diff, scale=se_diff)
        
        result = StatisticalResult(
            test_name=label,
            group1_mean=np.mean(group1),
            group2_mean=np.mean(group2),
            p_value=p_value,
            effect_size=effect_size,
            is_significant=p_value < self.alpha,
            confidence_interval=ci
        )
        
        self.results.append(result)
        return result
    
    def multiple_comparison_correction(self, results: List[StatisticalResult],
                                      method: str = 'bonferroni') -> List[StatisticalResult]:
        """Apply multiple comparison correction to p-values.
        
        Args:
            results: List of statistical results
            method: 'bonferroni' or 'fdr' (Benjamini-Hochberg)
            
        Returns:
            Corrected results with adjusted p-values
        """
        p_values = np.array([r.p_value for r in results])
        
        if method == 'bonferroni':
            # Bonferroni correction
            corrected_p = np.minimum(p_values * len(p_values), 1.0)
            
        elif method == 'fdr':
            # Benjamini-Hochberg FDR correction
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            n = len(p_values)
            
            # Find threshold
            threshold = None
            for i in range(n - 1, -1, -1):
                if sorted_p[i] <= (i + 1) / n * self.alpha:
                    threshold = sorted_p[i]
                    break
            
            corrected_p = p_values.copy()
            if threshold is not None:
                corrected_p = np.where(p_values <= threshold, p_values, 1.0)
            
        else:
            corrected_p = p_values
        
        # Update results with corrected p-values
        corrected_results = []
        for i, result in enumerate(results):
            result.corrected_p_value = float(corrected_p[i])
            result.corrected_is_significant = float(corrected_p[i]) < self.alpha
            corrected_results.append(result)
        
        return corrected_results
    
    def generate_statistical_report(self, results: List[StatisticalResult],
                                   output_file: str) -> None:
        """Generate comprehensive statistical report.
        
        Args:
            results: List of statistical results
            output_file: Path to save report
        """
        report_lines = [
            "STATISTICAL RIGOR ENHANCEMENT REPORT",
            "",
            f"Total Comparisons: {len(results)}",
            f"Significance Level (α): {self.alpha}",
            "",
            "PAIRWISE COMPARISON RESULTS",
            ""
        ]
        
        significant_count = 0
        for result in results:
            report_lines.append(f"Test: {result.test_name}")
            report_lines.append(f"  Group 1 Mean: {result.group1_mean:.4f}")
            report_lines.append(f"  Group 2 Mean: {result.group2_mean:.4f}")
            report_lines.append(f"  P-value: {result.p_value:.4f}")
            
            if result.corrected_p_value is not None:
                report_lines.append(f"  Corrected P-value: {result.corrected_p_value:.4f}")
                is_sig = result.corrected_is_significant
            else:
                is_sig = result.is_significant
            
            report_lines.append(f"  Effect Size (Cohen's d): {result.effect_size:.4f}")
            report_lines.append(f"  95% CI: ({result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f})")
            report_lines.append(f"  Significant: {'YES' if is_sig else 'NO'}")
            report_lines.append("")
            
            if is_sig:
                significant_count += 1
        
        report_lines.append("-" * 80)
        report_lines.append("SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Significant Results: {significant_count}/{len(results)}")
        report_lines.append(f"Significance Rate: {100*significant_count/max(len(results), 1):.1f}%")
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Save to file
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics of all stored results.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {}
        
        p_values = [r.p_value for r in self.results]
        effect_sizes = [r.effect_size for r in self.results]
        
        return {
            'n_comparisons': len(self.results),
            'mean_p_value': np.mean(p_values),
            'min_p_value': np.min(p_values),
            'max_p_value': np.max(p_values),
            'mean_effect_size': np.mean(effect_sizes),
            'significant_count': sum(1 for r in self.results if r.is_significant),
        }
