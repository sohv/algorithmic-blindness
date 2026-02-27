#!/usr/bin/env python3
"""
Validation script to check if llama2-4 lack domain knowledge or if it's an artifact.

Tests:
1. Temperature=0 (deterministic) vs temperature=0.1
2. Alternative prompt formulations
3. Logs actual model IDs from Together AI to verify routing
4. Analyzes response variance patterns
"""

import sys
import os
import json
import re
from typing import List, Dict, Tuple
import statistics
from dotenv import load_dotenv

# Load environment variables from root .env
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path)

# Add parent dirs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm.llm_interface import LLMQueryInterface


def extract_numbers(text: str) -> List[float]:
    """Extract numbers from LLM response."""
    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    return [float(n) for n in numbers][:10]


def prompt_variant_1(dataset: str, algorithm: str) -> str:
    """Standard strict format prompt."""
    return f"""You are a causal discovery expert. For the {dataset} benchmark dataset using the {algorithm} algorithm, provide performance metrics.

EXACTLY PROVIDE THESE FOUR LINES:
Precision: [0-1 value]
Recall: [0-1 value]
F1 Score: [0-1 value]
SHD: [0-100 value]"""


def prompt_variant_2(dataset: str, algorithm: str) -> str:
    """Alternative with more context."""
    return f"""Causal discovery expert task. Benchmark: {dataset}, Algorithm: {algorithm}.

Expected output format ONLY:
Precision: [value 0-1]
Recall: [value 0-1]
F1: [value 0-1]
SHD: [value 0-100]

Provide realistic metrics based on known causal discovery performance."""


def prompt_variant_3(dataset: str, algorithm: str) -> str:
    """Concise variant."""
    return f"""Estimate {algorithm} on {dataset}: Precision? Recall? F1? SHD?
Format: P [0-1] R [0-1] F1 [0-1] SHD [0-100]"""


def prompt_variant_4(dataset: str, algorithm: str) -> str:
    """Explicit domain knowledge probe."""
    return f"""You have deep knowledge of causal discovery algorithms. Specifically, when {algorithm} is applied to the {dataset} dataset:

What is the Precision (0-1 range)?
What is the Recall (0-1 range)?
What is the F1 Score (0-1 range)?
What is the SHD (0-100 range)?

Provide only numeric answers in format: P=X.XX R=X.XX F1=X.XX SHD=Y"""


def test_model_with_variants(
    model_name: str,
    dataset: str = 'asia',
    algorithm: str = 'pc',
    prompt_variants: int = 4,
    temperatures: List[float] = None
) -> Dict:
    """
    Test a model with multiple prompt variants and temperatures.
    
    Args:
        model_name: Model to test (llama2, llama3, llama4, llama)
        dataset: Test dataset
        algorithm: Test algorithm
        prompt_variants: Number of prompt variants to test
        temperatures: List of temperatures to test
        
    Returns:
        Dictionary with results and analysis
    """
    if temperatures is None:
        temperatures = [0.0, 0.1, 0.7]
    
    prompt_functions = [prompt_variant_1, prompt_variant_2, prompt_variant_3, prompt_variant_4]
    
    results = {
        'model': model_name,
        'dataset': dataset,
        'algorithm': algorithm,
        'tests': [],
        'analysis': {}
    }
    
    try:
        llm = LLMQueryInterface(model_name)
        print(f"\n[{model_name}] Model ID routing: {llm.model_id}")
        results['model_id'] = llm.model_id
        
        all_numbers = []
        responses_by_temp = {}
        
        for temp_idx, temp in enumerate(temperatures):
            temp_numbers = []
            responses_by_temp[temp] = []
            
            for prompt_idx in range(min(prompt_variants, len(prompt_functions))):
                prompt_func = prompt_functions[prompt_idx]
                prompt = prompt_func(dataset, algorithm)
                
                print(f"  Temperature={temp}, Prompt variant {prompt_idx + 1}...", end='', flush=True)
                
                response = llm.query(prompt, temperature=temp)
                numbers = extract_numbers(response.content)
                
                all_numbers.extend(numbers)
                temp_numbers.extend(numbers)
                responses_by_temp[temp].append({
                    'prompt_idx': prompt_idx,
                    'response_preview': response.content[:100],
                    'extracted_numbers': numbers[:5],  # First 5 for summary
                    'num_count': len(numbers)
                })
                
                print(f" → {len(numbers)} numbers extracted: {numbers[:3] if numbers else 'NONE'}")
                
                results['tests'].append({
                    'temperature': temp,
                    'prompt_variant': prompt_idx + 1,
                    'response': response.content,
                    'extracted_numbers': numbers
                })
        
        # Analysis
        results['analysis'] = {
            'total_unique_responses': len(set(
                r['response_preview'] for test in results['tests'] for r in [{'response_preview': test['response'][:50]}]
            )),
            'unique_precision_values': len(set(
                n for test in results['tests'] for n in (test['extracted_numbers'][:1] if test['extracted_numbers'] else [])
            )),
            'all_extracted_numbers': all_numbers if all_numbers else [None],
            'number_stats': {
                'total_count': len(all_numbers),
                'unique_count': len(set(all_numbers)) if all_numbers else 0,
                'mean': statistics.mean(all_numbers) if all_numbers else None,
                'stdev': statistics.stdev(all_numbers) if len(all_numbers) > 1 else None,
                'min': min(all_numbers) if all_numbers else None,
                'max': max(all_numbers) if all_numbers else None,
                'range': (max(all_numbers) - min(all_numbers)) if all_numbers else None
            },
            'temperature_consistency': {
                temp: len(set(n for resp in responses_by_temp[temp] for n in resp['extracted_numbers']))
                for temp in temperatures
            },
            'prompt_sensitivity': {
                f"variant_{i+1}": len(set(
                    tuple(r['extracted_numbers']) for r in results['tests'] if r['prompt_variant'] == i + 1
                ))
                for i in range(min(prompt_variants, len(prompt_functions)))
            }
        }
        
    except Exception as e:
        results['error'] = str(e)
        print(f"  ERROR: {e}")
    
    return results


def main():
    """Run full validation suite."""
    print("=" * 80)
    print("LLAMA VARIANTS VALIDATION: Domain Knowledge vs Prompt/Temperature Sensitivity")
    print("=" * 80)
    
    # Test llama2, llama3, llama4 (the "generic" ones) plus llama (3.3-70B) as reference
    models_to_test = ['llama2', 'llama3', 'llama4', 'llama']
    datasets_and_algos = [
        ('asia', 'pc'),
        ('cancer', 'lingam'),
        ('child', 'fci'),
        ('earthquake', 'notears')
    ]
    
    all_results = {}
    
    for model in models_to_test:
        print(f"\n{'=' * 80}")
        print(f"Testing: {model}")
        print(f"{'=' * 80}")
        
        model_results = []
        
        # Test each dataset/algo combo with all prompt variants and temperatures
        for dataset, algo in datasets_and_algos:
            result = test_model_with_variants(
                model,
                dataset=dataset,
                algorithm=algo,
                prompt_variants=4,
                temperatures=[0.0, 0.1, 0.7]
            )
            model_results.append(result)
        
        all_results[model] = model_results
    
    # Save results
    output_file = '/home/ece/hdd/sohan/algorithmic-blindness/src/experiments/results/memorization/llama_validation_results.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 80}\n")
    
    # Print summary
    print("\nSUMMARY BY MODEL:")
    print("-" * 80)
    
    for model, results in all_results.items():
        print(f"\n{model.upper()}:")
        for test_result in results:
            if 'error' in test_result:
                print(f"  ERROR: {test_result['error']}")
                continue
            
            analysis = test_result['analysis']
            dataset = test_result['dataset']
            algo = test_result['algorithm']
            
            print(f"  {dataset}+{algo}:")
            print(f"    - Numbers extracted: {analysis['number_stats']['total_count']}")
            print(f"    - Unique values: {analysis['number_stats']['unique_count']}")
            if analysis['number_stats']['mean'] is not None:
                print(f"    - Mean: {analysis['number_stats']['mean']:.3f}, Range: {analysis['number_stats']['min']:.3f}-{analysis['number_stats']['max']:.3f}")
            print(f"    - Prompt sensitivity (variants): {analysis['prompt_sensitivity']}")
            print(f"    - Temperature consistency: {analysis['temperature_consistency']}")


if __name__ == '__main__':
    main()
