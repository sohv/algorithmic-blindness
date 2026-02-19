#!/usr/bin/env python3
"""
Aggregate LLM Predictions Across Formulations
==============================================
Takes extracted ranges from f1, f2, f3 and averages them.

Usage:
    python aggregate_formulations.py --input_dir variance/extracted_ranges --output_dir variance/aggregated_ranges

This ensures:
1. All 3 formulations are weighted equally
2. Aggregated range captures formulation variance
3. Final comparison uses averaged LLM predictions
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict
import statistics


def aggregate_formulation_ranges(extracted_ranges_dir: Path) -> Dict:
    """
    Load all f1, f2, f3 extracted ranges and aggregate them.
    
    Returns:
        Dict mapping (dataset, algorithm, model) -> aggregated ranges
    """
    aggregated = defaultdict(lambda: defaultdict(dict))
    
    # Load all extracted range files
    # ONLY load formulation-specific files (f1/f2/f3), skip files without formulation suffix
    range_files = sorted(extracted_ranges_dir.glob("*_ranges.json"))
    
    if not range_files:
        print(f"❌ No range files found in {extracted_ranges_dir}")
        return {}
    
    # Group by (dataset, algorithm, model) across formulations
    by_experiment = defaultdict(lambda: defaultdict(lambda: {}))
    
    for range_file in range_files:
        # Parse: {dataset}_{algorithm}_f{formulation}_ranges.json
        # stem = "alarm_fci_f1_ranges" → remove "_ranges" suffix first
        stem = range_file.stem
        if not stem.endswith('_ranges'):
            continue
        
        stem_without_suffix = stem[:-7]  # Remove "_ranges"
        parts = stem_without_suffix.split('_')
        
        if len(parts) < 3:
            continue
        
        # Last part must be f1, f2, or f3 (formulation)
        formulation = parts[-1]
        if formulation not in ['f1', 'f2', 'f3']:
            continue  # Skip non-formulation files
        
        # Second-to-last is algorithm, everything before that is dataset
        algorithm = parts[-2]
        dataset = '_'.join(parts[:-2])
        
        with open(range_file, 'r') as f:
            data = json.load(f)
        
        exp_key = (dataset, algorithm)
        
        for model, model_estimates in data['llm_estimates'].items():
            by_experiment[exp_key][model][formulation] = model_estimates
    
    # Now aggregate across formulations
    for (dataset, algorithm), models in by_experiment.items():
        for model, formulations in models.items():
            aggregated_metrics = {}
            
            # Process each metric
            for metric in ['precision', 'recall', 'f1', 'shd']:
                lowers = []
                uppers = []
                confidences = []
                
                # Collect across f1, f2, f3
                for form_key in ['f1', 'f2', 'f3']:
                    if form_key in formulations and metric in formulations[form_key]:
                        metric_data = formulations[form_key][metric]
                        lowers.append(metric_data['lower'])
                        uppers.append(metric_data['upper'])
                        confidences.append(metric_data.get('confidence', 'medium'))
                
                # Only aggregate if we have data from at least 2 formulations
                if len(lowers) >= 2:
                    agg_lower = statistics.mean(lowers)
                    agg_upper = statistics.mean(uppers)
                    
                    # Confidence: "low" if any formulation is low, else "medium" if any medium, else "high"
                    if 'low' in confidences:
                        agg_confidence = 'low'
                    elif 'medium' in confidences:
                        agg_confidence = 'medium'
                    else:
                        agg_confidence = 'high'
                    
                    aggregated_metrics[metric] = {
                        'lower': float(agg_lower),
                        'upper': float(agg_upper),
                        'confidence': agg_confidence,
                        'num_formulations': len(lowers),
                        'formulation_values': {
                            'lowers': [float(x) for x in lowers],
                            'uppers': [float(x) for x in uppers],
                        }
                    }
            
            aggregated[dataset][algorithm][model] = aggregated_metrics
    
    return dict(aggregated)


def main():
    parser = argparse.ArgumentParser(description="Aggregate LLM predictions across formulations")
    parser.add_argument('--input_dir', type=str, default='variance/extracted_ranges',
        help='Directory with extracted LLM ranges (per formulation)')
    parser.add_argument('--output_dir', type=str, default='variance/aggregated_ranges',
        help='Output directory for aggregated ranges')
    parser.add_argument('--verbose', action='store_true', help='Show aggregation details')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("AGGREGATING LLM PREDICTIONS ACROSS FORMULATIONS")
    print("="*80)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Aggregate
    aggregated = aggregate_formulation_ranges(input_dir)
    
    if not aggregated:
        print("❌ No data to aggregate")
        return
    
    # Save aggregated results
    total_aggregated = 0
    
    for dataset, algorithms in aggregated.items():
        for algorithm, models in algorithms.items():
            output_file = output_dir / f"{dataset}_{algorithm}_aggregated.json"
            
            output_data = {
                'dataset': dataset,
                'algorithm': algorithm,
                'aggregation_method': 'mean of f1, f2, f3',
                'llm_estimates': models
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            num_models = len(models)
            total_aggregated += num_models
            
            if args.verbose:
                print(f"✓ {output_file.name}: {num_models} models")
                for model, metrics in models.items():
                    print(f"  {model}: {len(metrics)} metrics")
    
    print()
    print("="*80)
    print(f"Aggregated {total_aggregated} model predictions")
    print(f"Saved to: {output_dir}/")
    print(f"Next step: python compare_llm_vs_algorithmic.py --llm_dir {output_dir} --algorithmic_dir ../experiments/results --aggregated")
    print("="*80)


if __name__ == "__main__":
    main()
