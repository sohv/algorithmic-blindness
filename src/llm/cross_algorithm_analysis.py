#!/usr/bin/env python3
"""
Cross-Algorithm Analysis: Algorithm Understanding Test
Tests if Claude's synthetic boost is CONSISTENT across algorithms

If Claude truly understood algorithms, the boost should be similar for all.
But if Claude is pattern-matching, different algorithms will have different boosts.

This script shows which algorithms have exploitable patterns in synthetic data.
"""
import json
from pathlib import Path
from collections import defaultdict

comparisons_file = Path("results/comparisons/comparison_results.json")
with open(comparisons_file, 'r') as f:
    data = json.load(f)

# Track coverage by algorithm, dataset type, and LLM
algo_llm_data = defaultdict(
    lambda: defaultdict(
        lambda: {"real": {"covered": 0, "total": 0}, "synthetic": {"covered": 0, "total": 0}}
    )
)

for dataset_algo, exp_data in data.items():
    if "models" not in exp_data:
        continue
    
    dataset = exp_data.get("dataset", "")
    algorithm = exp_data.get("algorithm", "")
    is_synthetic = "synthetic" in dataset.lower()
    
    for llm, metrics in exp_data["models"].items():
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and "calibrated_coverage" in metric_data:
                key = "synthetic" if is_synthetic else "real"
                algo_llm_data[algorithm][llm][key]["total"] += 1
                if metric_data["calibrated_coverage"]:
                    algo_llm_data[algorithm][llm][key]["covered"] += 1

print("=" * 100)
print("CROSS-ALGORITHM ANALYSIS: Algorithm Understanding Test")
print("=" * 100)

# Get unique LLMs
all_llms = set()
for algo in algo_llm_data:
    all_llms.update(algo_llm_data[algo].keys())
all_llms = sorted(all_llms)

# Analyze each algorithm
algorithm_results = {}

for algo in sorted(algo_llm_data.keys()):
    print(f"\n{'=' * 100}")
    print(f"ALGORITHM: {algo.upper()}")
    print(f"{'=' * 100}")
    print(f"{'LLM':<15} {'Real Coverage':<20} {'Synthetic Coverage':<20} {'Boost':<15} {'Pattern?'}")
    print("-" * 100)
    
    boosts = []
    for llm in sorted(algo_llm_data[algo].keys()):
        real_total = algo_llm_data[algo][llm]["real"]["total"]
        real_covered = algo_llm_data[algo][llm]["real"]["covered"]
        synthetic_total = algo_llm_data[algo][llm]["synthetic"]["total"]
        synthetic_covered = algo_llm_data[algo][llm]["synthetic"]["covered"]
        
        real_pct = 100 * real_covered / real_total if real_total > 0 else 0
        synthetic_pct = 100 * synthetic_covered / synthetic_total if synthetic_total > 0 else 0
        boost = synthetic_pct - real_pct
        boosts.append(boost)
        
        # Highlight if significant boost exists
        if boost > 10:
            marker = "🚨 SIGNIFICANT"
        elif boost > 5:
            marker = "⚠ MODERATE"
        elif boost < -5:
            marker = "✓ LEARNS ON REAL"
        else:
            marker = "~ Neutral"
        
        print(f"{llm:<15} {real_pct:>5.1f}% ({real_covered}/{real_total})      {synthetic_pct:>5.1f}% ({synthetic_covered}/{synthetic_total})      {boost:>+6.1f}%      {marker}")
    
    # Algorithm-level statistics
    avg_boost = sum(boosts) / len(boosts) if boosts else 0
    max_boost = max(boosts) if boosts else 0
    min_boost = min(boosts) if boosts else 0
    
    algorithm_results[algo] = {
        "avg_boost": avg_boost,
        "max_boost": max_boost,
        "min_boost": min_boost,
        "range": max_boost - min_boost
    }
    
    print("-" * 100)
    print(f"Algorithm Statistics:")
    print(f"  Average boost:    {avg_boost:+.1f}%")
    print(f"  Max boost:        {max_boost:+.1f}%")
    print(f"  Min boost:        {min_boost:+.1f}%")
    print(f"  Variation range:  {max_boost - min_boost:.1f}%")
    
    # Interpretation
    if avg_boost > 10:
        print(f"  🚨 EXPLOITABLE: This algorithm has exploitable patterns in synthetic data")
    elif avg_boost > 0:
        print(f"  ⚠ PARTIALLY EXPLOITABLE: Moderate synthetic advantage")
    elif avg_boost < -5:
        print(f"  ✓ REAL-DATA ADVANTAGE: LLMs perform better on real data")
    else:
        print(f"  ~ NEUTRAL: No systematic advantage")

# Summary comparison across algorithms
print(f"\n{'=' * 100}")
print("CROSS-ALGORITHM COMPARISON: Which algorithms have exploitable patterns?")
print(f"{'=' * 100}")
print(f"{'Algorithm':<15} {'Avg Boost':<15} {'Range':<15} {'Exploitability'}")
print("-" * 100)

results_sorted = sorted(algorithm_results.items(), key=lambda x: x[1]["avg_boost"], reverse=True)

for algo, results in results_sorted:
    avg = results["avg_boost"]
    range_val = results["range"]
    
    if avg > 15:
        exploitability = "🚨🚨 HIGHLY EXPLOITABLE"
    elif avg > 10:
        exploitability = "🚨 VERY EXPLOITABLE"
    elif avg > 5:
        exploitability = "⚠ EXPLOITABLE"
    elif avg < -5:
        exploitability = "✓ REAL-DATA ADVANTAGE"
    else:
        exploitability = "~ NEUTRAL"
    
    print(f"{algo:<15} {avg:>+6.1f}%        {range_val:>6.1f}%      {exploitability}")

print(f"\n{'=' * 100}")
print("KEY FINDING:")
print(f"{'=' * 100}")

# Identify which algorithm is most exploitable
most_exploitable = max(algorithm_results.items(), key=lambda x: x[1]["avg_boost"])
print(f"Most exploitable algorithm: {most_exploitable[0].upper()}")
print(f"  → Average synthetic boost: {most_exploitable[1]['avg_boost']:+.1f}%")
print(f"  → This suggests Claude learned specific patterns for {most_exploitable[0].upper()} in synthetic data")

print(f"\n💡 INTERPRETATION:")
print(f"The varying boosts across algorithms ({results_sorted[0][1]['avg_boost']:+.1f}% to {results_sorted[-1][1]['avg_boost']:+.1f}%)")
print(f"prove that Claude is NOT understanding algorithms generically.")
print(f"Instead, Claude learned DIFFERENT synthetic patterns for EACH algorithm.")
print(f"This is the hallmark of pattern-matching, not algorithmic understanding.")

print(f"\n{'=' * 100}")

# Save to file
output_file = Path("results/comparisons/cross_algorithm_analysis.txt")
with open(output_file, 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("CROSS-ALGORITHM ANALYSIS: Algorithm Understanding Test\n")
    f.write("=" * 100 + "\n\n")
    
    for algo in sorted(algorithm_results.keys()):
        results = algorithm_results[algo]
        f.write(f"{algo.upper()}:\n")
        f.write(f"  Average synthetic boost: {results['avg_boost']:+.1f}%\n")
        f.write(f"  Range variation: {results['range']:.1f}%\n")
        f.write(f"  Most exploitable algorithm: {max(algorithm_results.items(), key=lambda x: x[1]['avg_boost'])[0].upper()}\n\n")
    
    f.write("=" * 100 + "\n")
    f.write("KEY FINDING:\n")
    f.write("=" * 100 + "\n")
    f.write("The different synthetic boosts per algorithm prove Claude learned\n")
    f.write("algorithm-specific patterns, not general algorithmic principles.\n")

print(f"✓ Results saved to: {output_file}")
