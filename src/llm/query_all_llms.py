#!/usr/bin/env python3
"""
Query All LLMs on Causal Discovery Experiments
===============================================

Queries 5 LLMs (GPT-4, DeepSeek, Claude, Gemini, Llama) on all variance experiments
and compares their predictions with algorithmic confidence intervals.

Usage:
    # Query all LLMs on all experiments
    python query_all_llms.py --all

    # Query specific LLMs
    python query_all_llms.py --models claude gemini llama

    # Query specific datasets
    python query_all_llms.py --datasets titanic sachs --models all

    # Use specific prompt formulation
    python query_all_llms.py --formulation 1  # Direct (default)
    python query_all_llms.py --formulation 2  # Step-by-step
    python query_all_llms.py --formulation 3  # Meta-knowledge
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import time

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from variance.llm_queries.llm_interface import LLMQueryInterface
from variance.llm_queries.parse_llm_responses import parse_llm_response
from prompts.prompt_templates import generate_prompt, get_all_formulations, FORMULATION_1_DIRECT
from variance.variance_analysis import VarianceAnalyzer


def load_variance_results(results_dir: Path) -> Dict:
    """
    Load all variance analysis results.

    Args:
        results_dir: Directory containing variance JSON files

    Returns:
        Dictionary mapping (dataset, algorithm) -> results
    """
    variance_results = {}

    for result_file in results_dir.glob("*_variance.json"):
        with open(result_file, 'r') as f:
            data = json.load(f)

        dataset = data['dataset']
        algorithm = data['algorithm']
        variance_results[(dataset, algorithm)] = data

    print(f"Loaded {len(variance_results)} variance results")
    return variance_results


def query_llm_for_experiment(
    llm: LLMQueryInterface,
    dataset_name: str,
    algorithm_name: str,
    formulation,
    n_samples: int = 1000
) -> Optional[Dict]:
    """
    Query a single LLM for one experiment.

    Args:
        llm: LLMQueryInterface instance
        dataset_name: Name of dataset (e.g., 'titanic')
        algorithm_name: Name of algorithm (e.g., 'PC', 'LiNGAM')
        formulation: PromptTemplate to use
        n_samples: Number of samples in dataset

    Returns:
        Dictionary with parsed LLM estimates or None if failed
    """
    # Generate prompt
    prompt = generate_prompt(
        dataset_name=dataset_name,
        algorithm_name=algorithm_name,
        formulation=formulation,
        n_samples=n_samples
    )

    # Query LLM
    response = llm.query(prompt)

    if not response.success:
        print(f"  ✗ Query failed: {response.error}")
        return None

    # Parse response
    parsed_estimates = parse_llm_response(response.content)

    if not parsed_estimates:
        print(f"  ✗ Failed to parse response")
        print(f"  Raw response: {response.content[:200]}...")
        return None

    return parsed_estimates


def create_llm_comparison(
    variance_result: Dict,
    llm_estimates: Dict,
    llm_name: str
) -> Dict:
    """
    Create comparison between algorithmic CI and LLM estimates.

    Args:
        variance_result: Variance analysis results
        llm_estimates: Parsed LLM estimates
        llm_name: Name of the LLM

    Returns:
        Dictionary with overlap analysis
    """
    analyzer = VarianceAnalyzer()

    # Convert variance results to AlgorithmResults format
    from variance.variance_analysis import MetricStats, AlgorithmResults

    def dict_to_metric_stats(metric_dict: Dict) -> MetricStats:
        return MetricStats(
            mean=metric_dict['mean'],
            std=metric_dict['std'],
            ci_lower=metric_dict['ci_95_lower'],
            ci_upper=metric_dict['ci_95_upper'],
            median=metric_dict['median'],
            min_val=metric_dict['min'],
            max_val=metric_dict['max'],
            runs=metric_dict['n_runs']
        )

    results_obj = AlgorithmResults(
        precision=dict_to_metric_stats(variance_result['results']['precision']),
        recall=dict_to_metric_stats(variance_result['results']['recall']),
        f1=dict_to_metric_stats(variance_result['results']['f1']),
        shd=dict_to_metric_stats(variance_result['results']['shd'])
    )

    # Create comparison
    comparison = analyzer.compare_with_llm_estimates(results_obj, llm_estimates)

    return {
        f'{llm_name}_comparison': comparison
    }


def save_llm_comparison(
    dataset: str,
    algorithm: str,
    llm_name: str,
    comparison: Dict,
    output_dir: Path
):
    """Save LLM comparison to JSON file."""
    output_file = output_dir / f"{dataset}_{algorithm}_llm_comparison.json"

    # Load existing comparisons if file exists
    if output_file.exists():
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {
            'dataset': dataset,
            'algorithm': algorithm
        }

    # Add new LLM comparison
    existing_data.update(comparison)

    # Save updated file
    with open(output_file, 'w') as f:
        json.dump(existing_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Query LLMs on causal discovery experiments"
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['claude', 'gemini', 'llama'],
        choices=['gpt4', 'deepseek', 'claude', 'gemini', 'llama', 'all'],
        help='LLMs to query (default: claude gemini llama)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='Datasets to query (default: all)'
    )
    parser.add_argument(
        '--algorithms',
        nargs='+',
        default=['pc', 'lingam'],
        choices=['pc', 'lingam', 'all'],
        help='Algorithms to query (default: pc lingam)'
    )
    parser.add_argument(
        '--formulation',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help='Prompt formulation to use (1=Direct, 2=Step-by-step, 3=Meta-knowledge)'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='variance/results_full',
        help='Directory with variance results (default: variance/results_full)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='variance/results_full',
        help='Output directory (default: variance/results_full)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Query all LLMs on all experiments'
    )

    args = parser.parse_args()

    # Resolve 'all' flags
    if args.all or 'all' in args.models:
        models = ['gpt4', 'deepseek', 'claude', 'gemini', 'llama']
    else:
        models = args.models

    if 'all' in args.algorithms:
        algorithms = ['pc', 'lingam', 'fci', 'notears']
    else:
        algorithms = args.algorithms

    # Load variance results
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    variance_results = load_variance_results(results_dir)

    # Filter by dataset if specified
    if args.datasets:
        variance_results = {
            k: v for k, v in variance_results.items()
            if k[0] in args.datasets
        }

    # Filter by algorithm
    variance_results = {
        k: v for k, v in variance_results.items()
        if k[1] in algorithms
    }

    # Get prompt formulation
    formulations = get_all_formulations()
    formulation = formulations[args.formulation - 1]

    print("="*80)
    print("QUERYING LLMs ON CAUSAL DISCOVERY EXPERIMENTS")
    print("="*80)
    print(f"Models: {', '.join(models)}")
    print(f"Experiments: {len(variance_results)}")
    print(f"Formulation: {formulation.name}")
    print(f"Output: {output_dir}")
    print()

    # Initialize LLM clients
    llm_clients = {}
    for model in models:
        try:
            llm_clients[model] = LLMQueryInterface(model)
            print(f"✓ {model.upper()} initialized")
        except Exception as e:
            print(f"✗ {model.upper()} failed to initialize: {e}")

    print()

    # Query each LLM on each experiment
    total_queries = len(llm_clients) * len(variance_results)
    successful_queries = 0
    failed_queries = 0

    with tqdm(total=total_queries, desc="Querying LLMs") as pbar:
        for (dataset, algorithm), variance_data in variance_results.items():
            for llm_name, llm in llm_clients.items():
                pbar.set_description(f"{llm_name.upper()} on {dataset}_{algorithm}")

                try:
                    # Query LLM
                    estimates = query_llm_for_experiment(
                        llm=llm,
                        dataset_name=dataset,
                        algorithm_name=algorithm.upper(),
                        formulation=formulation,
                        n_samples=1000  # Default sample size
                    )

                    if estimates:
                        # Create comparison
                        comparison = create_llm_comparison(
                            variance_result=variance_data,
                            llm_estimates=estimates,
                            llm_name=llm_name
                        )

                        # Save comparison
                        save_llm_comparison(
                            dataset=dataset,
                            algorithm=algorithm,
                            llm_name=llm_name,
                            comparison=comparison,
                            output_dir=output_dir
                        )

                        successful_queries += 1
                    else:
                        failed_queries += 1

                except Exception as e:
                    print(f"\n✗ Error querying {llm_name} on {dataset}_{algorithm}: {e}")
                    failed_queries += 1

                pbar.update(1)

                # Rate limiting
                time.sleep(1)

    print("\n" + "="*80)
    print("QUERY RESULTS")
    print("="*80)
    print(f"Total queries: {total_queries}")
    print(f"Successful: {successful_queries} ({100*successful_queries/total_queries:.1f}%)")
    print(f"Failed: {failed_queries} ({100*failed_queries/total_queries:.1f}%)")
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
