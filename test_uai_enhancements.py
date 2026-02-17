import sys
import numpy as np
from pathlib import Path
from statistical_analysis import StatisticalTester, ExplanatoryAnalyzer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test 1: StatisticalTester
print("\n1. Testing StatisticalTester...")

tester = StatisticalTester(alpha=0.05)
print(f"✓ Created StatisticalTester instance (α={tester.alpha})")

# Create sample data
group1 = np.random.normal(5.0, 1.0, 30)
group2 = np.random.normal(5.2, 1.0, 30)

result = tester.paired_t_test(group1, group2, "Algorithm A vs B")
print(f"✓ Paired t-test completed:")
print(f"  - p-value: {result.p_value:.4f}")
print(f"  - Effect size: {result.effect_size:.4f}")
print(f"  - Significant: {result.is_significant}")

# Test multiple comparisons correction
results = [result]
group3 = np.random.normal(4.9, 1.0, 30)
result2 = tester.paired_t_test(group1, group3, "Algorithm A vs C")
results.append(result2)

corrected = tester.multiple_comparison_correction(results, method='fdr')
print(f"✓ Applied FDR correction to {len(corrected)} tests")
print(f"  - Significant before correction: {sum(1 for r in results if r.is_significant)}")
print(f"  - Significant after correction: {sum(1 for r in corrected if r.corrected_is_significant)}")

# Test report generation
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    tester.generate_statistical_report(corrected, f.name)
    report_path = f.name

with open(report_path, 'r') as f:
    report_content = f.read()
    report_lines = report_content.split('\n')
    
print(f" Generated statistical report ({len(report_lines)} lines)")
print(f"  Sample output (first 10 lines):")
for line in report_lines[:10]:
    print(f"    {line}")

# Cleanup
Path(report_path).unlink()

# Test 2: ExplanatoryAnalyzer
print("\n2. Testing ExplanatoryAnalyzer...")
print("-" * 80)

analyzer = ExplanatoryAnalyzer()
print(f"✓ Created ExplanatoryAnalyzer instance")

# Create mock data
mock_llm_results = {
    'Algorithm_Results': {
        'asia': {
            'PC': {'accuracy': 0.85, 'confidence_interval_width': 0.05},
            'LiNGAM': {'accuracy': 0.80, 'confidence_interval_width': 0.07},
            'FCI': {'accuracy': 0.78, 'confidence_interval_width': 0.08},
        },
        'sachs': {
            'PC': {'accuracy': 0.82, 'confidence_interval_width': 0.06},
            'LiNGAM': {'accuracy': 0.75, 'confidence_interval_width': 0.09},
            'FCI': {'accuracy': 0.80, 'confidence_interval_width': 0.07},
        },
    }
}

mock_graphs = {
    'asia': np.random.randint(0, 2, (8, 8)),
    'sachs': np.random.randint(0, 2, (11, 11)),
}

mock_metadata = {
    'asia': {'sample_size': 5000, 'dimensionality': 8, 'noise_level': 0.05},
    'sachs': {'sample_size': 7466, 'dimensionality': 11, 'noise_level': 0.08},
}

insights = analyzer.analyze_performance_factors(mock_llm_results, mock_graphs, mock_metadata)
print(f"✓ Analyzed performance factors: {len(insights)} insights generated")

for insight in insights:
    print(f"  - {insight.factor_name}: {insight.impact_score:.1f}% impact")

# Test report generation
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    analyzer.generate_theory_report(insights, f.name)
    theory_path = f.name

with open(theory_path, 'r') as f:
    theory_content = f.read()
    theory_lines = theory_content.split('\n')
    
print(f"✓ Generated theory report ({len(theory_lines)} lines)")
print(f"  Report sections that mention insights:")
for line in theory_lines:
    if 'Factor' in line or 'Impact' in line or 'Correlation' in line:
        print(f"    {line}")
        if line.strip() == '':
            break

# Cleanup
Path(theory_path).unlink()
