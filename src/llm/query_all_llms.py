"""
Query All LLMs on Causal Discovery Experiments
===============================================
Queries 6 LLMs (GPT-5.2, Claude Opus 4.6, Gemini 2.5 Pro, DeepSeek R1, Llama 3.3, Qwen 3) on variance experiments
and compares their predictions with algorithmic confidence intervals.

Usage:
    # Query all LLMs on all experiments with ALL formulations
    python query_all_llms.py --all --formulation 1 2 3

    # Query specific formulations
    python query_all_llms.py --all --formulation 1 2       # Formulations 1 and 2
    python query_all_llms.py --all --formulation 3         # Formulation 3 only
    python query_all_llms.py --all                         # Default: Formulation 1 only

    # Query specific LLMs
    python query_all_llms.py --models claude gemini llama

    # Query specific datasets
    python query_all_llms.py --datasets titanic sachs cancer

    # Query specific algorithms
    python query_all_llms.py --algorithms pc fci

    # Query specific combination with all formulations
    python query_all_llms.py --models gpt5 claude --datasets asia sachs cancer --algorithms pc lingam fci --formulation 1 2 3

Formulation Details:
    1 = Direct Question - Straightforward question about algorithm performance
    2 = Step-by-Step - Guides LLM through reasoning process
    3 = Meta-Knowledge - Frames as confidence interval estimation task
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm.llm_interface import LLMQueryInterface
from src.llm.parse_llm_responses import parse_llm_response
from prompts.prompt_templates import generate_prompt, get_all_formulations, FORMULATION_1_DIRECT
from src.algorithms.variance_analysis import VarianceAnalyzer


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
        choices=['gpt5', 'deepseek', 'claude', 'gemini', 'llama', 'qwen', 'all'],
        help='LLMs to query (default: claude gemini llama)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        choices=['asia', 'sachs', 'cancer', 'child', 'synthetic_12', 'synthetic_30', 'titanic', 'survey', 'earthquake', 'wine_quality', 'credit_approval'],
        help='Datasets to query (default: all)'
    )
    parser.add_argument(
        '--algorithms',
        nargs='+',
        default=['pc', 'lingam'],
        choices=['pc', 'lingam', 'fci', 'notears', 'all'],
        help='Algorithms to query (default: pc lingam)'
    )
    parser.add_argument(
        '--formulation',
        type=int,
        nargs='+',
        default=[1],
        choices=[1, 2, 3],
        help='Prompt formulations to use (default: 1) - can specify multiple: 1 2 3'
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
        models = ['gpt5', 'deepseek', 'claude', 'gemini', 'llama', 'qwen']
    else:
        models = args.models

    if args.datasets is None or 'all' in args.datasets:
        datasets = None  # Use all datasets
    else:
        datasets = args.datasets

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
    if datasets is not None:
        variance_results = {
            k: v for k, v in variance_results.items()
            if k[0] in datasets
        }

    # Filter by algorithm
    variance_results = {
        k: v for k, v in variance_results.items()
        if k[1] in algorithms
    }

    # Get prompt formulations
    all_formulations = get_all_formulations()
    formulations_to_run = [all_formulations[f - 1] for f in args.formulation]

    print("="*80)
    print("QUERYING LLMs ON CAUSAL DISCOVERY EXPERIMENTS")
    print("="*80)
    print(f"Models: {', '.join(models)}")
    print(f"Datasets: {', '.join(datasets) if datasets else 'all'}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Formulations: {', '.join([f.name for f in formulations_to_run])}")
    print(f"Experiments per formulation: {len(variance_results)}")
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

    # Query each LLM on each experiment with each formulation
    total_queries = len(llm_clients) * len(variance_results) * len(formulations_to_run)
    successful_queries = 0
    failed_queries = 0

    with tqdm(total=total_queries, desc="Querying LLMs") as pbar:
        for formulation in formulations_to_run:
            print(f"\n{'='*80}")
            print(f"FORMULATION: {formulation.name} (#{formulation.formulation_id})")
            print(f"{'='*80}\n")
            
            for (dataset, algorithm), variance_data in variance_results.items():
                for llm_name, llm in llm_clients.items():
                    pbar.set_description(f"{llm_name.upper()} | F{formulation.formulation_id} | {dataset}_{algorithm}")

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

                            # Save comparison with formulation suffix
                            output_file = output_dir / f"{dataset}_{algorithm}_f{formulation.formulation_id}_llm_comparison.json"
                            
                            # Load existing comparisons if file exists
                            if output_file.exists():
                                with open(output_file, 'r') as f:
                                    existing_data = json.load(f)
                            else:
                                existing_data = {
                                    'dataset': dataset,
                                    'algorithm': algorithm,
                                    'formulation': formulation.formulation_id
                                }

                            # Add new LLM comparison
                            existing_data.update(comparison)

                            # Save updated file
                            with open(output_file, 'w') as f:
                                json.dump(existing_data, f, indent=2)

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
    if total_queries > 0:
        print(f"Successful: {successful_queries} ({100*successful_queries/total_queries:.1f}%)")
        print(f"Failed: {failed_queries} ({100*failed_queries/total_queries:.1f}%)")
    else:
        print("No queries executed. Check that variance results exist in the results directory.")
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
