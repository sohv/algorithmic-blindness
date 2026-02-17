import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Lazy import of matplotlib to avoid compatibility issues
def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except (ImportError, AttributeError):
        return None

plt = None  # Will be initialized on demand


@dataclass
class PerformanceInsight:
    """Insight from explanatory analysis."""
    factor_name: str
    impact_score: float
    correlation: float
    description: str
    evidence: List[str]


class ExplanatoryAnalyzer:
    """
    Explanatory factor analysis for theoretical rigor enhancement.
    Analyzes which factors explain algorithm performance variation.
    """
    
    def __init__(self):
        """Initialize explanatory analyzer."""
        self.insights: List[PerformanceInsight] = []
        self.performance_data = None
        self.graph_structures = None
        self.metadata = None
    
    def analyze_performance_factors(self, llm_results: Dict[str, Any],
                                   graph_structures: Dict[str, np.ndarray],
                                   metadata: Dict[str, Dict[str, Any]]) -> List[PerformanceInsight]:
        """Analyze factors explaining performance variation.
        
        Args:
            llm_results: Results keyed by LLM/method name
            graph_structures: Graph structures keyed by dataset
            metadata: Dataset metadata
            
        Returns:
            List of performance insights
        """
        self.performance_data = llm_results
        self.graph_structures = graph_structures
        self.metadata = metadata
        
        insights = []
        
        # Factor 1: Graph Complexity (sparsity, density, etc.)
        complexity_insight = self._analyze_graph_complexity()
        if complexity_insight:
            insights.append(complexity_insight)
        
        # Factor 2: Sample Size Effect
        sample_size_insight = self._analyze_sample_size_effect()
        if sample_size_insight:
            insights.append(sample_size_insight)
        
        # Factor 3: Dimensionality Impact
        dimensionality_insight = self._analyze_dimensionality()
        if dimensionality_insight:
            insights.append(dimensionality_insight)
        
        # Factor 4: Algorithm-Dataset Interaction
        interaction_insight = self._analyze_algorithm_dataset_interaction()
        if interaction_insight:
            insights.append(interaction_insight)
        
        # Factor 5: Noise Sensitivity
        noise_insight = self._analyze_noise_sensitivity()
        if noise_insight:
            insights.append(noise_insight)
        
        self.insights = insights
        return insights
    
    def _analyze_graph_complexity(self) -> Optional[PerformanceInsight]:
        """Analyze how graph complexity affects performance."""
        if not self.graph_structures:
            return None
        
        complexities = []
        performances = []
        
        for dataset, graph in self.graph_structures.items():
            # Calculate graph sparsity (density)
            n = graph.shape[0]
            n_edges = np.sum(graph)
            max_edges = n * (n - 1)
            density = n_edges / max_edges if max_edges > 0 else 0
            complexities.append(density)
            
            # Get average performance for this dataset
            if dataset in self.performance_data.get('Algorithm_Results', {}):
                perf_values = []
                for algo_results in self.performance_data['Algorithm_Results'][dataset].values():
                    if isinstance(algo_results, dict) and 'accuracy' in algo_results:
                        perf_values.append(algo_results['accuracy'])
                
                if perf_values:
                    performances.append(np.mean(perf_values))
                else:
                    performances.append(0.5)
            else:
                performances.append(0.5)
        
        if len(complexities) >= 2:
            # Calculate correlation
            correlation = np.corrcoef(complexities, performances)[0, 1]
            
            # Impact score: how much variation in performance is explained
            complexity_impact = abs(correlation)
            
            insight = PerformanceInsight(
                factor_name="Graph Complexity (Sparsity/Density)",
                impact_score=min(complexity_impact * 100, 100),
                correlation=correlation,
                description=f"Graph density {'negatively' if correlation < 0 else 'positively'} correlates with performance",
                evidence=[
                    f"Graph sparsity range: {min(complexities):.3f} - {max(complexities):.3f}",
                    f"Performance range: {min(performances):.3f} - {max(performances):.3f}",
                    f"Correlation: {correlation:.3f}",
                    f"Sparser graphs perform {'better' if correlation < 0 else 'worse'}"
                ]
            )
            return insight
        
        return None
    
    def _analyze_sample_size_effect(self) -> Optional[PerformanceInsight]:
        """Analyze sample size effect on performance."""
        if not self.metadata:
            return None
        
        sample_sizes = []
        performances = []
        
        for dataset, meta in self.metadata.items():
            if 'sample_size' in meta:
                sample_sizes.append(np.log10(meta['sample_size'] + 1))  # Log scale
                
                if dataset in self.performance_data.get('Algorithm_Results', {}):
                    perf_values = []
                    for algo_results in self.performance_data['Algorithm_Results'][dataset].values():
                        if isinstance(algo_results, dict) and 'accuracy' in algo_results:
                            perf_values.append(algo_results['accuracy'])
                    
                    if perf_values:
                        performances.append(np.mean(perf_values))
                    else:
                        performances.append(0.5)
                else:
                    performances.append(0.5)
        
        if len(sample_sizes) >= 2:
            correlation = np.corrcoef(sample_sizes, performances)[0, 1]
            
            insight = PerformanceInsight(
                factor_name="Sample Size Effect",
                impact_score=min(abs(correlation) * 100, 100),
                correlation=correlation,
                description=f"Larger samples {'improve' if correlation > 0 else 'impair'} performance",
                evidence=[
                    f"Sample size log-scale range: {min(sample_sizes):.2f} - {max(sample_sizes):.2f}",
                    f"Performance range: {min(performances):.3f} - {max(performances):.3f}",
                    f"Correlation: {correlation:.3f}",
                    f"Effect: {'Positive' if correlation > 0.3 else 'Weak' if abs(correlation) < 0.3 else 'Negative'}"
                ]
            )
            return insight
        
        return None
    
    def _analyze_dimensionality(self) -> Optional[PerformanceInsight]:
        """Analyze dimensionality impact."""
        if not self.metadata:
            return None
        
        dims = []
        performances = []
        
        for dataset, meta in self.metadata.items():
            if 'dimensionality' in meta:
                dims.append(meta['dimensionality'])
                
                if dataset in self.performance_data.get('Algorithm_Results', {}):
                    perf_values = []
                    for algo_results in self.performance_data['Algorithm_Results'][dataset].values():
                        if isinstance(algo_results, dict) and 'accuracy' in algo_results:
                            perf_values.append(algo_results['accuracy'])
                    
                    if perf_values:
                        performances.append(np.mean(perf_values))
                    else:
                        performances.append(0.5)
                else:
                    performances.append(0.5)
        
        if len(dims) >= 2:
            correlation = np.corrcoef(dims, performances)[0, 1]
            
            insight = PerformanceInsight(
                factor_name="Problem Dimensionality",
                impact_score=min(abs(correlation) * 100, 100),
                correlation=correlation,
                description=f"Higher dimensionality {'helps' if correlation > 0 else 'hinders'} learning",
                evidence=[
                    f"Dimension range: {min(dims)} - {max(dims)} variables",
                    f"Performance on low-D: {np.mean([p for p, d in zip(performances, dims) if d <= 15]):.3f}",
                    f"Performance on high-D: {np.mean([p for p, d in zip(performances, dims) if d > 15]):.3f}",
                    f"Correlation: {correlation:.3f}"
                ]
            )
            return insight
        
        return None
    
    def _analyze_algorithm_dataset_interaction(self) -> Optional[PerformanceInsight]:
        """Analyze algorithm-dataset interaction effects."""
        if not self.performance_data:
            return None
        
        algo_performances = {}
        
        for dataset_results in self.performance_data.get('Algorithm_Results', {}).values():
            for algo, results in dataset_results.items():
                if isinstance(results, dict) and 'accuracy' in results:
                    if algo not in algo_performances:
                        algo_performances[algo] = []
                    algo_performances[algo].append(results['accuracy'])
        
        if len(algo_performances) >= 2:
            # Calculate variance in algorithm performance
            variances = []
            for algo, perfs in algo_performances.items():
                if len(perfs) > 1:
                    variances.append(np.var(perfs))
            
            if variances:
                max_variance = max(variances)
                mean_variance = np.mean(variances)
                
                insight = PerformanceInsight(
                    factor_name="Algorithm-Dataset Interaction",
                    impact_score=min((max_variance - mean_variance) * 100, 100),
                    correlation=0.0,
                    description="Different algorithms have different sensitivity to dataset properties",
                    evidence=[
                        f"Number of algorithms: {len(algo_performances)}",
                        f"Mean performance variance across datasets: {mean_variance:.4f}",
                        f"Max single-algorithm variance: {max_variance:.4f}",
                        f"Interaction effect: {sum(1 for v in variances if v > mean_variance)} algorithms show high sensitivity"
                    ]
                )
                return insight
        
        return None
    
    def _analyze_noise_sensitivity(self) -> Optional[PerformanceInsight]:
        """Analyze noise sensitivity."""
        if not self.metadata:
            return None
        
        noise_levels = []
        perfs_std = []
        
        for dataset, meta in self.metadata.items():
            if 'noise_level' in meta:
                noise_levels.append(meta['noise_level'])
                
                if dataset in self.performance_data.get('Algorithm_Results', {}):
                    perf_values = []
                    for algo_results in self.performance_data['Algorithm_Results'][dataset].values():
                        if isinstance(algo_results, dict) and 'accuracy' in algo_results:
                            perf_values.append(algo_results['accuracy'])
                    
                    if perf_values:
                        perfs_std.append(np.std(perf_values))
                    else:
                        perfs_std.append(0.1)
                else:
                    perfs_std.append(0.1)
        
        if len(noise_levels) >= 2:
            correlation = np.corrcoef(noise_levels, perfs_std)[0, 1]
            
            insight = PerformanceInsight(
                factor_name="Noise Sensitivity",
                impact_score=min(abs(correlation) * 100, 100),
                correlation=correlation,
                description=f"Algorithms Show {'high' if abs(correlation) > 0.5 else 'moderate' if abs(correlation) > 0.3 else 'low'} noise sensitivity",
                evidence=[
                    f"Noise level range: {min(noise_levels):.3f} - {max(noise_levels):.3f}",
                    f"Performance std range: {min(perfs_std):.3f} - {max(perfs_std):.3f}",
                    f"Correlation: {correlation:.3f}",
                    f"Insight: Noisy environments {'significantly' if abs(correlation) > 0.5 else 'mildly'} degrade performance"
                ]
            )
            return insight
        
        return None
    
    def generate_theory_report(self, insights: List[PerformanceInsight],
                              output_file: str) -> None:
        """Generate theoretical analysis report.
        
        Args:
            insights: List of performance insights
            output_file: Path to save report
        """
        report_lines = [
            "=" * 80,
            "EXPLANATORY THEORY ANALYSIS REPORT",
            "=" * 80,
            "",
            f"Total Factors Analyzed: {len(insights)}",
            "",
            "-" * 80,
            "KEY FINDINGS",
            "-" * 80,
            ""
        ]
        
        # Sort by impact
        sorted_insights = sorted(insights, key=lambda x: x.impact_score, reverse=True)
        
        for i, insight in enumerate(sorted_insights, 1):
            report_lines.append(f"{i}. {insight.factor_name}")
            report_lines.append(f"   Impact Score: {insight.impact_score:.1f}%")
            report_lines.append(f"   Correlation: {insight.correlation:.3f}")
            report_lines.append(f"   {insight.description}")
            report_lines.append("")
            report_lines.append("   Evidence:")
            for evidence in insight.evidence:
                report_lines.append(f"     • {evidence}")
            report_lines.append("")
        
        report_lines.append("-" * 80)
        report_lines.append("THEORETICAL IMPLICATIONS")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        # Generate implications
        if sorted_insights:
            top_factor = sorted_insights[0]
            report_lines.append(f"Primary Explanatory Factor: {top_factor.factor_name}")
            report_lines.append(f"  This factor explains ~{top_factor.impact_score:.0f}% of performance variation,")
            report_lines.append(f"  suggesting it is the primary driver of algorithm behavior in this domain.")
            report_lines.append("")
        
        
        # Save to file
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
    
    def create_explanatory_plots(self, insights: List[PerformanceInsight],
                                output_dir: str) -> None:
        """Create visualization of explanatory factors.
        
        Args:
            insights: List of performance insights
            output_dir: Directory to save plots
        """
        # Lazy import matplotlib
        plt = _import_matplotlib()
        if plt is None:
            print("  Warning: Matplotlib not available, skipping plot generation")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Impact scores
        fig, ax = plt.subplots(figsize=(10, 6))
        factors = [i.factor_name for i in insights]
        scores = [i.impact_score for i in insights]
        colors = ['green' if s > 50 else 'orange' if s > 30 else 'gray' for s in scores]
        
        ax.barh(factors, scores, color=colors)
        ax.set_xlabel('Impact Score (%)')
        ax.set_title('Explanatory Factor Importance')
        ax.set_xlim(0, 100)
        
        for i, (factor, score) in enumerate(zip(factors, scores)):
            ax.text(score + 2, i, f'{score:.0f}%', va='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'explanatory_factor_importance.png', dpi=300)
        plt.close()
        
        # Plot 2: Correlation heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        correlations = np.array([i.correlation for i in insights]).reshape(-1, 1)
        
        im = ax.imshow(correlations.T, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(insights)))
        ax.set_xticklabels([f.replace(' ', '\n') for f in factors], fontsize=9, rotation=45, ha='right')
        ax.set_yticks([0])
        ax.set_yticklabels(['Correlation'])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'explanatory_correlations.png', dpi=300)
        plt.close()
        
        # Plot 3: Summary statistics
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        summary_text = "EXPLANATORY ANALYSIS SUMMARY\n\n"
        summary_text += f"Factors Analyzed: {len(insights)}\n"
        summary_text += f"Mean Impact Score: {np.mean(scores):.1f}%\n"
        summary_text += f"Max Impact: {max(scores):.1f}% ({factors[scores.index(max(scores))]})\n"
        summary_text += f"Mean Absolute Correlation: {np.mean(np.abs([i.correlation for i in insights])):.3f}\n\n"
        
        summary_text += "Top 3 Factors:\n"
        for i, (factor, score) in enumerate(sorted(zip(factors, scores), key=lambda x: x[1], reverse=True)[:3], 1):
            summary_text += f"  {i}. {factor} ({score:.0f}%)\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'explanatory_summary.png', dpi=300)
        plt.close()
