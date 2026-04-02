#!/usr/bin/env python3
"""

Take real benchmark datasets and rename them completely (e.g., "Asia" → "Network-A"),
then re-query the LLMs. If coverage drops significantly on the renamed version despite 
identical algorithmic difficulty, that implicates surface-level name matching rather than 
reasoning about algorithm properties.

Usage:
    python src/experiments/perturbed_benchmark_test.py \
      --models claude gpt5 deepseek \
      --datasets asia cancer child earthquake
    python src/algorithms/perturbed_benchmark_test.py --output_dir src/results/algorithms/memorization --all
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file in root directory
load_dotenv(Path(__file__).parent.parent.parent / '.env')

from src.llm.llm_interface import LLMQueryInterface


class PerturbedBenchmarkTest:
    """Test LLM memorization via dataset renaming."""

    def __init__(self):
        self.results = []
        # Use completely unrelated names that give zero lexical cues
        # Original name → disguised name with no retrieval hints
        self.dataset_perturbations = {
            'asia': 'Network-A',
            'cancer': 'Dataset-7',
            'child': 'Crestwood',
            'earthquake': 'Greyfield',
            'alarm': 'System-B',
            'sachs': 'Structure-Q'
        }

    def generate_performance_estimation_prompt(self, dataset_name: str, algorithm_name: str, 
                                              is_perturbed: bool = False, model_name: str = None) -> str:
        """Generate prompt for estimating algorithm performance.
        
        Args:
            dataset_name: Name of the dataset
            algorithm_name: Name of the algorithm
            is_perturbed: Whether this is a perturbed (renamed) dataset
            model_name: Name of the model (used for stronger prompting if needed)
        """
        
        # Hybrid prompt for LLaMA: dataset-awareness + strict format
        if model_name == 'llama':
            prompt = f"""You are an expert in causal discovery algorithms.
Your answer MUST change if the dataset changes - different datasets produce DIFFERENT performance.
The dataset "{dataset_name}" has specific properties that affect {algorithm_name}'s performance.

Respond with EXACTLY these four lines only. No explanations, no text before/after:
Precision: X.XX
Recall: X.XX
F1: X.XX
SHD: X

For "{dataset_name}" with {algorithm_name.upper()}:"""
            return prompt
        
        # Standard strict prompt for all other models
        prompt = f"""RESPOND WITH EXACTLY THESE FOUR LINES ONLY. NO OTHER TEXT.

Precision: X.XX
Recall: X.XX
F1: X.XX
SHD: X

Estimate for {algorithm_name.upper()} on {dataset_name}. Single decimal number per line. Nothing else."""

        return prompt

    def parse_performance_estimate(self, response_text: str) -> Dict:
        """Extract metrics from simple format responses."""
        import re
        result = {
            'raw_response': response_text,
            'extracted_numbers': []
        }
        
        # Extract all numbers from the response
        numbers = re.findall(r'\d+\.?\d*', response_text)
        try:
            result['extracted_numbers'] = [float(n) for n in numbers if n][:10]
        except:
            pass
        
        return result

    def calculate_perturbation_impact(self, original: Dict, perturbed: Dict) -> Dict:
        """Calculate how much performance estimates changed due to naming."""
        impact = {
            'precision_shift': None,
            'recall_shift': None,
            'f1_shift': None,
            'shd_shift': None,
            'midpoint_shifts': {}
        }

        for metric in ['precision', 'recall', 'f1']:
            if original[metric] and perturbed[metric]:
                orig_mid = (original[metric][0] + original[metric][1]) / 2
                pert_mid = (perturbed[metric][0] + perturbed[metric][1]) / 2
                shift = pert_mid - orig_mid
                impact[metric + '_shift'] = shift
                impact['midpoint_shifts'][metric] = {
                    'original': orig_mid,
                    'perturbed': pert_mid,
                    'absolute_change': abs(shift),
                    'percent_change': (shift / max(abs(orig_mid), 0.01)) * 100
                }
        
        # For SHD (lower is better), calculate similar metrics
        if original['shd'] and perturbed['shd']:
            orig_mid = (original['shd'][0] + original['shd'][1]) / 2
            pert_mid = (perturbed['shd'][0] + perturbed['shd'][1]) / 2
            shift = pert_mid - orig_mid
            impact['shd_shift'] = shift
            impact['midpoint_shifts']['shd'] = {
                'original': orig_mid,
                'perturbed': pert_mid,
                'absolute_change': abs(shift),
                'percent_change': (shift / max(abs(orig_mid), 0.01)) * 100
            }

        return impact

    def run_test(self, models: List[str], datasets: List[str], algorithms: List[str], 
                 output_dir: str = None):
        """Run perturbed benchmark test."""
        # Use default path relative to script location if not specified
        if output_dir is None:
            output_path = Path(__file__).parent.parent / 'results' / 'algorithms' / 'memorization'
        else:
            output_path = Path(output_dir)
            if not output_path.is_absolute():
                # Interpret relative paths as relative to script location
                output_path = Path(__file__).parent / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"PERTURBED BENCHMARK TEST - Name-Based Memorization Detection")
        print(f"{'='*80}")
        print(f"Testing {len(models)} models × {len(datasets)} datasets × {len(algorithms)} algorithms")
        print(f"Each dataset tested in ORIGINAL and PERTURBED (renamed) versions\n")

        for model_name in models:
            print(f"\n{'─'*80}")
            print(f"Model: {model_name.upper()}")
            print(f"{'─'*80}")

            try:
                llm = LLMQueryInterface(model_name)
            except Exception as e:
                print(f"  ✗ Failed to initialize model: {e}")
                continue

            model_results = []

            for dataset_name in datasets:
                if dataset_name not in self.dataset_perturbations:
                    print(f"  ⊘ Skipping {dataset_name}: No perturbation defined")
                    continue

                perturbed_name = self.dataset_perturbations[dataset_name]
                print(f"\n  Dataset: {dataset_name.upper()}")

                for algorithm_name in algorithms:
                    print(f"    Algorithm: {algorithm_name.upper()}")

                    # Query original dataset name
                    print(f"      Querying ORIGINAL ({dataset_name})...", end=' ', flush=True)
                    original_prompt = self.generate_performance_estimation_prompt(
                        dataset_name, algorithm_name, is_perturbed=False, model_name=model_name
                    )

                    try:
                        original_response = llm.query(original_prompt)
                        if not original_response.success:
                            print(f"\n        ✗ Query failed")
                            continue

                        original_parsed = self.parse_performance_estimate(original_response.content)
                        print(f"✓")

                    except Exception as e:
                        print(f"\n        ✗ Error - {str(e)[:50]}")
                        continue

                    # Query perturbed (renamed) dataset
                    print(f"      Querying PERTURBED ({perturbed_name})...", end=' ', flush=True)
                    perturbed_prompt = self.generate_performance_estimation_prompt(
                        perturbed_name, algorithm_name, is_perturbed=True, model_name=model_name
                    )

                    try:
                        perturbed_response = llm.query(perturbed_prompt)
                        if not perturbed_response.success:
                            print(f"\n        ✗ Query failed")
                            continue

                        perturbed_parsed = self.parse_performance_estimate(perturbed_response.content)
                        print(f"✓")

                    except Exception as e:
                        print(f"\n        ✗ Error - {str(e)[:50]}")
                        continue

                    # Store result with raw responses for manual analysis
                    result = {
                        'model': model_name,
                        'dataset': dataset_name,
                        'perturbed_name': perturbed_name,
                        'algorithm': algorithm_name,
                        'original_response': original_parsed['raw_response'],
                        'original_extracted_numbers': original_parsed['extracted_numbers'],
                        'perturbed_response': perturbed_parsed['raw_response'],
                        'perturbed_extracted_numbers': perturbed_parsed['extracted_numbers'],
                        'timestamp': datetime.now().isoformat()
                    }
                    model_results.append(result)
                    self.results.append(result)

                    # Display extraction summary
                    orig_count = len(original_parsed['extracted_numbers'])
                    pert_count = len(perturbed_parsed['extracted_numbers'])
                    print(f"      Extracted {orig_count} numbers (original), {pert_count} numbers (perturbed)")

            # Save per-model results
            if model_results:
                model_output = output_path / f"perturbation_{model_name}.json"
                with open(model_output, 'w') as f:
                    json.dump(model_results, f, indent=2, default=str)
                print(f"\n  Saved results to {model_output}")

    def generate_summary_report(self, output_dir: str = None):
        """Save raw LLM responses for manual perturbation analysis."""
        # Use default path relative to script location if not specified
        if output_dir is None:
            output_path = Path(__file__).parent.parent / 'results' / 'algorithms' / 'memorization'
        else:
            output_path = Path(output_dir)
            if not output_path.is_absolute():
                # Interpret relative paths as relative to script location
                output_path = Path(__file__).parent / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.results:
            print("  No results to summarize")
            return

        print(f"\n{'='*80}")
        print(f"PERTURBATION TEST - Raw Response Data")
        print(f"{'='*80}\n")

        # Save detailed report with raw responses for manual analysis
        report_path = output_path / 'perturbation_raw_responses.json'
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'description': 'Raw LLM responses comparing original vs perturbed dataset names',
            'n_total_tests': len(self.results),
            'dataset_perturbations': self.dataset_perturbations,
            'results': self.results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"✓ Saved raw responses to {report_path}")

        # Show query counts by model and dataset
        df = pd.DataFrame(self.results)
        
        print(f"\n{'='*80}")
        print(f"QUERIES BY MODEL AND DATASET")
        print(f"{'='*80}\n")
        
        summary = df.groupby(['model', 'dataset']).size().reset_index(name='tests')
        for _, row in summary.iterrows():
            print(f"  {row['model']:15s} × {row['dataset']:15s}: {row['tests']:3d} tests")

        print(f"\n{'='*80}")
        print(f"MANUAL ANALYSIS WORKFLOW")
        print(f"{'='*80}\n")
        print("Each result includes:")
        print("  - original_response: Full LLM response for original dataset name")
        print("  - original_extracted_numbers: All numbers found in original response")
        print("  - perturbed_response: Full LLM response for perturbed dataset name")
        print("  - perturbed_extracted_numbers: All numbers found in perturbed response")
        print("  - dataset_perturbations: Original → Perturbed name mappings")
        print(f"\nTo detect memorization:")
        print("  1. Compare extracted_numbers between original and perturbed")
        print("  2. Calculate midpoint shifts for each metric pair")
        print("  3. >15% change = memorization signal (name-based matching)")
        print(f"  4. <5% change = algorithmic reasoning (dataset-agnostic)")
        print(f"\nResults saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Perturbed Benchmark Test for Memorization Detection")
    parser.add_argument('--models', nargs='+', default=['claude', 'gpt5'],
        choices=['gpt5', 'deepseek', 'deepseekthink', 'claude', 'gemini3', 'llama', 'qwen', 'qwenthink', 'all'],
        help='LLMs to test (default: claude gpt5)')
    parser.add_argument('--datasets', nargs='+', default=['asia', 'cancer', 'child', 'earthquake', 'alarm', 'sachs'],
        choices=['asia', 'cancer', 'child', 'earthquake', 'alarm', 'sachs', 'all'],
        help='Benchmark datasets only (default: asia cancer child earthquake alarm sachs). Synthetic datasets cannot be perturbed.')
    parser.add_argument('--algorithms', nargs='+', default=['pc', 'lingam', 'fci', 'notears'],
        choices=['pc', 'lingam', 'fci', 'notears', 'all'],
        help='Algorithms to test (default: pc lingam fci notears)')
    parser.add_argument('--output_dir', type=str, default=None,
        help='Output directory for results (default: src/results/algorithms/memorization)')
    parser.add_argument('--all', action='store_true', help='Test all models on all benchmark datasets')

    args = parser.parse_args()

    if args.all:
        models = ['gpt5', 'deepseek', 'deepseekthink', 'claude', 'gemini3', 'llama', 'qwen', 'qwenthink']
        datasets = ['asia', 'cancer', 'child', 'earthquake', 'alarm', 'sachs']
    else:
        models = args.models
        datasets = args.datasets

    algorithms = args.algorithms

    # Run test
    test = PerturbedBenchmarkTest()
    test.run_test(models, datasets, algorithms, output_dir=args.output_dir)
    test.generate_summary_report(output_dir=args.output_dir)


if __name__ == '__main__':
    main()
