#!/usr/bin/env python3
"""
LLaMA Family Variants Testing
==============================

Test multiple LLaMA model variants to determine if hallucination/memorization 
patterns observed in meta-llama/Llama-3.3-70B-Instruct-Turbo apply across 
the LLaMA family.

Models tested:
- llama1: meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
- llama2: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
- llama3: meta-llama/Llama-3.2-3B-Instruct-Turbo
- llama4: meta-llama/Meta-Llama-3-8B-Instruct-Lite

These are tested using the same perturbed benchmark methodology as the main test.

Usage:
    python llama_variants_test.py --datasets asia cancer child earthquake alarm sachs
"""

import sys
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv(Path(__file__).parent.parent.parent / '.env')

from src.llm.llm_interface import LLMQueryInterface


class LLaMAVariantsTest:
    """Test LLaMA model variants for memorization patterns."""

    # LLaMA model variants with shorthand names
    LLAMA_VARIANTS = {
        'llama1': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        'llama2': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        'llama3': 'meta-llama/Llama-3.2-3B-Instruct-Turbo',
        'llama4': 'meta-llama/Meta-Llama-3-8B-Instruct-Lite'
    }

    def __init__(self):
        self.results = []
        self.dataset_perturbations = {
            'asia': 'Network-A',
            'cancer': 'Dataset-7',
            'child': 'Crestwood',
            'earthquake': 'Greyfield',
            'alarm': 'System-B',
            'sachs': 'Structure-Q'
        }

    def generate_performance_estimation_prompt(self, dataset_name: str, algorithm_name: str) -> str:
        """Generate prompt for LLaMA variants."""
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

    def parse_performance_estimate(self, response_text: str) -> Dict:
        """Extract metrics from response."""
        import re
        result = {
            'raw_response': response_text,
            'extracted_numbers': []
        }
        
        numbers = re.findall(r'\d+\.?\d*', response_text)
        try:
            result['extracted_numbers'] = [float(n) for n in numbers if n][:10]
        except:
            pass
        
        return result

    def run_variant_test(self, model_shorthand: str, model_id: str, datasets: List[str], 
                        algorithms: List[str], output_dir: Path):
        """Test a single LLaMA variant."""
        
        print(f"\n{'─'*80}")
        print(f"Model: {model_shorthand.upper()} ({model_id})")
        print(f"{'─'*80}")

        # Create temporary model entry in SUPPORTED_MODELS
        original_models = LLMQueryInterface.SUPPORTED_MODELS.copy()
        LLMQueryInterface.SUPPORTED_MODELS[model_shorthand] = model_id

        try:
            llm = LLMQueryInterface(model_shorthand)
        except Exception as e:
            print(f"  ✗ Failed to initialize model: {e}")
            LLMQueryInterface.SUPPORTED_MODELS = original_models
            return []

        model_results = []

        for dataset_name in datasets:
            if dataset_name not in self.dataset_perturbations:
                print(f"  ⊘ Skipping {dataset_name}: No perturbation defined")
                continue

            perturbed_name = self.dataset_perturbations[dataset_name]
            print(f"\n  Dataset: {dataset_name.upper()}")

            for algorithm_name in algorithms:
                print(f"    Algorithm: {algorithm_name.upper()}", end='')

                # Query original
                print(f" [Original]...", end=' ', flush=True)
                original_prompt = self.generate_performance_estimation_prompt(dataset_name, algorithm_name)

                try:
                    original_response = llm.query(original_prompt)
                    if not original_response.success:
                        print(f"✗")
                        continue
                    original_parsed = self.parse_performance_estimate(original_response.content)
                    print(f"✓", end='')
                except Exception as e:
                    print(f"✗")
                    continue

                # Query perturbed
                print(f" [Perturbed]...", end=' ', flush=True)
                perturbed_prompt = self.generate_performance_estimation_prompt(perturbed_name, algorithm_name)

                try:
                    perturbed_response = llm.query(perturbed_prompt)
                    if not perturbed_response.success:
                        print(f"✗")
                        continue
                    perturbed_parsed = self.parse_performance_estimate(perturbed_response.content)
                    print(f"✓")
                except Exception as e:
                    print(f"✗")
                    continue

                # Store result
                result = {
                    'model': model_shorthand,
                    'model_id': model_id,
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

        # Save results for this variant
        if model_results:
            output_file = output_dir / f"llama_variant_{model_shorthand}.json"
            with open(output_file, 'w') as f:
                json.dump(model_results, f, indent=2, default=str)
            print(f"\n  ✓ Saved {len(model_results)} results to {output_file}")

        # Restore original SUPPORTED_MODELS
        LLMQueryInterface.SUPPORTED_MODELS = original_models
        return model_results

    def run_all_variants(self, datasets: List[str], algorithms: List[str], output_dir: str = None):
        """Test all LLaMA variants."""
        
        if output_dir is None:
            output_path = Path(__file__).parent / 'results' / 'memorization'
        else:
            output_path = Path(output_dir)
            if not output_path.is_absolute():
                output_path = Path(__file__).parent / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"LLAMA VARIANTS MEMORIZATION TEST")
        print(f"{'='*80}")
        print(f"Testing {len(self.LLAMA_VARIANTS)} LLaMA variants × {len(datasets)} datasets × {len(algorithms)} algorithms")
        print(f"Comparing results to detect if hallucination patterns apply across LLaMA family\n")

        all_results = []
        for shorthand, model_id in self.LLAMA_VARIANTS.items():
            results = self.run_variant_test(shorthand, model_id, datasets, algorithms, output_path)
            all_results.extend(results)

        # Save combined results
        combined_file = output_path / 'llama_variants_combined.json'
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n{'='*80}")
        print(f"✓ All variants tested. Combined results saved to {combined_file}")
        print(f"{'='*80}")

        return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLaMA model variants for memorization")
    parser.add_argument('--datasets', nargs='+', default=['asia', 'cancer', 'child', 'earthquake'],
        choices=['asia', 'cancer', 'child', 'earthquake', 'alarm', 'sachs'],
        help='Datasets to test (default: asia cancer child earthquake)')
    parser.add_argument('--algorithms', nargs='+', default=['pc', 'lingam', 'fci', 'notears'],
        choices=['pc', 'lingam', 'fci', 'notears'],
        help='Algorithms to test (default: pc lingam fci notears)')
    parser.add_argument('--output_dir', type=str, default=None,
        help='Output directory (default: src/experiments/results/memorization)')

    args = parser.parse_args()

    test = LLaMAVariantsTest()
    test.run_all_variants(args.datasets, args.algorithms, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
