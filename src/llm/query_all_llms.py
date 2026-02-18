"""
Query All LLMs on Causal Discovery Experiments (v2)
======================================================
Queries LLMs and saves RAW RESPONSES ONLY (no parsing).
Parsing happens in a separate offline step.

Usage:
    python query_all_llms.py --datasets asia --algorithms pc --formulation 1 2 3 --models gpt5 claude
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm.llm_interface import LLMQueryInterface
from prompts.prompt_templates import generate_prompt, get_all_formulations


def query_llm_for_experiment(
    llm: LLMQueryInterface,
    dataset_name: str,
    algorithm_name: str,
    formulation,
    n_samples: int = 1000
) -> Optional[str]:
    """Query a single LLM and return raw response."""
    prompt = generate_prompt(
        dataset_name=dataset_name,
        algorithm_name=algorithm_name,
        formulation=formulation,
        n_samples=n_samples
    )
    response = llm.query(prompt)
    if not response.success:
        print(f"  ✗ Query failed: {response.error}")
        return None
    return response.content


def main():
    parser = argparse.ArgumentParser(description="Query LLMs on causal discovery experiments (save raw responses)")
    parser.add_argument('--models', nargs='+', default=['claude', 'geminiflash', 'llama'],
        choices=['gpt5', 'deepseek', 'deepseekthink', 'claude', 'geminiflash', 'gemini3', 'llama', 'qwen', 'qwenthink', 'all'],
        help='LLMs to query (default: claude geminiflash llama)')
    parser.add_argument('--datasets', nargs='+', default=None,
        choices=['asia', 'sachs', 'cancer', 'child', 'alarm', 'earthquake', 'hepar2', 'insurance', 'survey', 'synthetic_12', 'synthetic_30', 'synthetic_50', 'synthetic_60'],
        help='Datasets to query (default: all)')
    parser.add_argument('--algorithms', nargs='+', default=['pc', 'lingam'],
        choices=['pc', 'lingam', 'fci', 'notears', 'all'],
        help='Algorithms to query (default: pc lingam)')
    parser.add_argument('--formulation', type=int, nargs='+', default=[1], choices=[1, 2, 3],
        help='Prompt formulations to use (default: 1)')
    parser.add_argument('--output_dir', type=str, default='variance/raw_responses',
        help='Output directory for raw responses')
    parser.add_argument('--all', action='store_true', help='Query all LLMs on all experiments')

    args = parser.parse_args()

    # Resolve flags
    if args.all or 'all' in args.models:
        models = ['gpt5', 'deepseek', 'deepseekthink', 'claude', 'geminiflash', 'gemini3', 'llama', 'qwen', 'qwenthink']
    else:
        models = args.models

    if args.datasets is None or 'all' in args.datasets:
        datasets = None
    else:
        datasets = args.datasets

    if 'all' in args.algorithms:
        algorithms = ['pc', 'lingam', 'fci', 'notears']
    else:
        algorithms = args.algorithms

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate experiment pairs: (dataset, algorithm)
    all_datasets = ['asia', 'sachs', 'cancer', 'child', 'alarm', 'earthquake', 'hepar2', 'insurance', 'survey', 'synthetic_12', 'synthetic_30', 'synthetic_50', 'synthetic_60']
    datasets_to_query = datasets if datasets is not None else all_datasets
    experiment_pairs = [(d, a) for d in datasets_to_query for a in algorithms]

    all_formulations = get_all_formulations()
    formulations_to_run = [all_formulations[f - 1] for f in args.formulation]

    print("="*80)
    print("QUERYING LLMs (SAVING RAW RESPONSES ONLY)")
    print("="*80)
    print(f"Models: {', '.join(models)}")
    print(f"Datasets: {', '.join(datasets_to_query)}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Formulations: {', '.join([f.name for f in formulations_to_run])}")
    print(f"Experiment pairs: {len(experiment_pairs)}")
    print(f"Output dir: {output_dir}")
    print()

    # Initialize LLM clients
    llm_clients = {}
    for model in models:
        try:
            llm_clients[model] = LLMQueryInterface(model)
            print(f"✓ {model.upper()} initialized")
        except Exception as e:
            print(f"✗ {model.upper()} failed: {e}")

    print()

    # Query each LLM
    total_queries = len(llm_clients) * len(experiment_pairs) * len(formulations_to_run)
    successful_queries = 0
    failed_queries = 0

    with tqdm(total=total_queries, desc="Querying LLMs") as pbar:
        for formulation in formulations_to_run:
            print(f"\n{'='*80}")
            print(f"FORMULATION: {formulation.name} (#{formulation.formulation_id})")
            print(f"{'='*80}\n")
            
            for dataset, algorithm in experiment_pairs:
                for llm_name, llm in llm_clients.items():
                    pbar.set_description(f"{llm_name.upper()} | F{formulation.formulation_id} | {dataset}_{algorithm}")

                    try:
                        # Query LLM - get raw response
                        response = query_llm_for_experiment(
                            llm=llm,
                            dataset_name=dataset,
                            algorithm_name=algorithm.upper(),
                            formulation=formulation,
                            n_samples=1000
                        )

                        if response:
                            # Save raw response
                            output_file = output_dir / f"{dataset}_{algorithm}_f{formulation.formulation_id}_{llm_name}_raw.txt"
                            with open(output_file, 'w') as f:
                                f.write(response)
                            successful_queries += 1
                        else:
                            failed_queries += 1

                    except Exception as e:
                        print(f"\n✗ Error querying {llm_name}: {e}")
                        failed_queries += 1

                    pbar.update(1)
                    time.sleep(1)  # Rate limiting

    print("\n" + "="*80)
    print("QUERY RESULTS")
    print("="*80)
    print(f"Total queries: {total_queries}")
    print(f"Successful: {successful_queries} ({100*successful_queries/total_queries:.1f}%)")
    print(f"Failed: {failed_queries} ({100*failed_queries/total_queries:.1f}%)")
    print(f"\nRaw responses saved to: {output_dir}/")
    print(f"Next step: python extract_llm_ranges.py --input_dir {output_dir} --output_dir variance/extracted_ranges")


if __name__ == "__main__":
    main()
