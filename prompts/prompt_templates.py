#!/usr/bin/env python3
"""
Three Prompt Formulations for LLM Meta-Knowledge Testing
=========================================================

This module defines three distinct prompt formulations to test
whether LLM estimates are robust across different question styles.

Formulations:
1. DIRECT: Straightforward question about algorithm performance
2. STEP-BY-STEP: Guides LLM through reasoning process
3. META-KNOWLEDGE: Frames as confidence interval estimation task

Why this matters:
- If variance across prompts is LOW (<20%): Results are robust
- If variance is HIGH (>20%): Need to discuss sensitivity
- Either way: Shows methodological thoroughness
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Container for a prompt formulation."""
    name: str
    description: str
    template: str
    formulation_id: int


# =============================================================================
# FORMULATION 1: Direct Question
# =============================================================================

FORMULATION_1_DIRECT = PromptTemplate(
    name="Direct Question",
    description="Straightforward, concise question about algorithm performance",
    formulation_id=1,
    template="""You are an expert in causal discovery algorithms.

Dataset: {dataset_name}
- Domain: {domain}
- Variables: {n_nodes}
- Samples: {n_samples}

Algorithm: {algorithm_name}

Task: Estimate the algorithm's performance on this dataset.

Provide your estimates as ranges [lower, upper] for:
- Precision (proportion of predicted edges that are correct)
- Recall (proportion of true edges that are recovered)
- F1-score (harmonic mean of precision and recall)
- SHD (Structural Hamming Distance - total edge errors)

Format your response as:
Precision: [X.XX, X.XX]
Recall: [X.XX, X.XX]
F1: [X.XX, X.XX]
SHD: [X, X]

Provide only numerical ranges."""
)


# =============================================================================
# FORMULATION 2: Step-by-Step Reasoning
# =============================================================================

FORMULATION_2_STEPBYSTEP = PromptTemplate(
    name="Step-by-Step Reasoning",
    description="Guides LLM through systematic reasoning about algorithm properties",
    formulation_id=2,
    template="""You are an expert in causal discovery algorithms.

Dataset: {dataset_name}
- Domain: {domain}
- Variables: {n_nodes}
- Samples: {n_samples}
- Sample adequacy: {sample_adequacy}

Algorithm: {algorithm_name}

Key assumptions:
{algorithm_assumptions}

Reasoning steps:
1. Does the dataset satisfy the algorithm's assumptions?
2. How do sample size and complexity affect performance?
3. What performance range is realistic given these factors?

Provide your estimates as ranges [lower, upper] for:
- Precision (proportion of predicted edges that are correct)
- Recall (proportion of true edges that are recovered)
- F1-score (overall edge recovery quality)
- SHD (total edge errors: missing, extra, reversed)

Format your response as:
Precision: [X.XX, X.XX]
Recall: [X.XX, X.XX]
F1: [X.XX, X.XX]
SHD: [X, X]

Provide only numerical ranges."""
)


# =============================================================================
# FORMULATION 3: Meta-Knowledge / Confidence Interval Framing
# =============================================================================

FORMULATION_3_METAKNOWLEDGE = PromptTemplate(
    name="Meta-Knowledge Framing",
    description="Frames task as predicting algorithm variance and confidence intervals",
    formulation_id=3,
    template="""You are an expert in causal discovery algorithms.

Scenario: A researcher will run {algorithm_name} on {dataset_name} 100 times with different random seeds to measure performance distribution.

Dataset: {dataset_name}
- Domain: {domain}
- Variables: {n_nodes}
- Samples: {n_samples}
- Complexity: {complexity}

Task: Predict the 95% confidence interval for each metric based on your knowledge of {algorithm_name}'s behavior on similar datasets.

Consider:
- {algorithm_name}'s algorithmic properties
- Dataset characteristics
- Typical performance ranges on similar data

Provide estimates as ranges [lower, upper] for:
- Precision (proportion of predicted edges that are correct)
- Recall (proportion of true edges that are recovered)
- F1-score (overall edge recovery quality)
- SHD (total edge errors)

Format your response as:
Precision: [X.XX, X.XX]
Recall: [X.XX, X.XX]
F1: [X.XX, X.XX]
SHD: [X, X]

Provide only numerical ranges."""
)


# =============================================================================
# Algorithm-Specific Information
# =============================================================================

ALGORITHM_ASSUMPTIONS = {
    'PC': """
- Causal Sufficiency: No unmeasured confounders
- Faithfulness: Graph structure is identifiable from data
- Conditional independence tests work correctly
- Adequate sample size for reliable independence tests
    """,
    'LiNGAM': """
- Linearity: Relationships are linear
- Non-Gaussianity: Error terms are non-Gaussian
- Acyclicity: No feedback loops
- No latent confounders
- Continuous data (or can treat discrete as continuous)
    """,
    'FCI': """
- Faithfulness: Graph structure is identifiable
- Allows latent confounders (weaker assumption than PC)
- Conditional independence tests work correctly
- Can return partial orientation (with o-> edges)
    """,
    'NOTEARS': """
- Linearity: Relationships are linear
- Acyclicity: No cycles in the graph
- Continuous data
- Convex optimization assumptions
- L1 regularization for sparsity
    """,
    'GES': """
- Score-based approach (uses BIC/BDeu scoring)
- Searches over equivalence classes of DAGs
- Greedy forward (add edges) and backward (remove edges) phases
- Consistency: recovers true graph with infinite data
- No assumption about functional form (unlike LiNGAM)
- Causal sufficiency assumed
    """,
    'GRaSP': """
- Permutation-based approach
- Searches over variable orderings for sparsest graph
- Greedy relaxation strategy
- Modern algorithm with strong theoretical guarantees
- Handles both discrete and continuous data
- Causal sufficiency assumed
    """
}


DATASET_PROPERTIES = {
    'titanic': {
        'domain': 'Social Science (survival prediction)',
        'data_type': 'Mixed (categorical + continuous)',
        'complexity': 'Low (7 variables)',
        'n_nodes': 7,
        'n_edges': 5
    },
    'credit': {
        'domain': 'Finance (credit risk assessment)',
        'data_type': 'Mixed (categorical + continuous)',
        'complexity': 'Medium (16 variables, 7 edges)',
        'n_nodes': 16,
        'n_edges': 7
    },
    'wine': {
        'domain': 'Chemistry (wine physicochemical properties)',
        'data_type': 'Continuous',
        'complexity': 'Medium (12 variables, 11 edges)',
        'n_nodes': 12,
        'n_edges': 11
    },
    'sachs': {
        'domain': 'Biology (protein signaling)',
        'data_type': 'Continuous (flow cytometry)',
        'complexity': 'Medium (11 variables, 17 edges)',
        'n_nodes': 11,
        'n_edges': 17
    },
    'alarm': {
        'domain': 'Medical (intensive care monitoring)',
        'data_type': 'Discrete (patient monitoring)',
        'complexity': 'High (37 variables, 46 edges)',
        'n_nodes': 37,
        'n_edges': 46
    },
    'stock_market': {
        'domain': 'Finance (market relationships)',
        'data_type': 'Continuous (time series)',
        'complexity': 'Medium (10 variables, 18 edges)',
        'n_nodes': 10,
        'n_edges': 18
    },
    'asia': {
        'domain': 'Medical (diagnosis)',
        'data_type': 'Discrete',
        'complexity': 'Low (8 variables)',
        'n_nodes': 8,
        'n_edges': 8
    },
    'alarm': {
        'domain': 'Medical (ICU Monitoring)',
        'data_type': 'Discrete',
        'complexity': 'High (37 variables)',
        'n_nodes': 37,
        'n_edges': 46
    },
    'survey': {
        'domain': 'Social Science',
        'data_type': 'Discrete',
        'complexity': 'Low (6 variables)',
        'n_nodes': 6,
        'n_edges': 6
    },
    'child': {
        'domain': 'Medical',
        'data_type': 'Discrete',
        'complexity': 'Medium (20 variables)',
        'n_nodes': 20,
        'n_edges': 25
    },
    'hepar2': {
        'domain': 'Medical (Hepatitis diagnosis)',
        'data_type': 'Discrete',
        'complexity': 'High (70 variables)',
        'n_nodes': 70,
        'n_edges': 120
    },
    'insurance': {
        'domain': 'Insurance (risk assessment)',
        'data_type': 'Discrete',
        'complexity': 'High (27 variables, 52 edges)',
        'n_nodes': 27,
        'n_edges': 52
    },
    'barley': {
        'domain': 'Agriculture (crop production)',
        'data_type': 'Discrete',
        'complexity': 'High (48 variables, 84 edges)',
        'n_nodes': 48,
        'n_edges': 84
    }
}


def generate_prompt(dataset_name: str,
                   algorithm_name: str,
                   formulation: PromptTemplate,
                   n_samples: int = 1000) -> str:
    """
    Generate a prompt from a template.

    Args:
        dataset_name: Name of the dataset (e.g., 'titanic')
        algorithm_name: Name of algorithm (e.g., 'PC', 'LiNGAM')
        formulation: PromptTemplate to use
        n_samples: Number of samples in dataset

    Returns:
        Formatted prompt string
    """
    # Calculate sample adequacy for Formulation 2
    sample_adequacy = "adequate" if n_samples > 500 else "limited"

    # Get dataset properties
    dataset_key = dataset_name.lower().replace('_', '').replace('-', '')

    # Handle synthetic datasets
    if 'synthetic' in dataset_key:
        if '12' in dataset_key:
            props = {
                'domain': 'Synthetic (linear Gaussian)',
                'data_type': 'Continuous',
                'complexity': 'Medium',
                'n_nodes': 12,
                'n_edges': 14
            }
        elif '30' in dataset_key:
            props = {
                'domain': 'Synthetic (linear Gaussian)',
                'data_type': 'Continuous',
                'complexity': 'High',
                'n_nodes': 30,
                'n_edges': 45
            }
        else:
            props = {
                'domain': 'Synthetic (linear Gaussian)',
                'data_type': 'Continuous',
                'complexity': 'Medium',
                'n_nodes': 12,
                'n_edges': 14
            }
    else:
        props = DATASET_PROPERTIES.get(dataset_key, {
            'domain': 'Unknown',
            'data_type': 'Mixed',
            'complexity': 'Medium',
            'n_nodes': 10,
            'n_edges': 'unknown'
        })

    # Handle algorithm name variations
    algo_key = algorithm_name.upper().replace('-', '').replace('_', '')
    if 'LINGAM' in algo_key:
        algo_key = 'LiNGAM'
    elif algo_key == 'FCI':
        algo_key = 'FCI'
    elif 'NOTEARS' in algo_key:
        algo_key = 'NOTEARS'
    elif algo_key == 'GES':
        algo_key = 'GES'
    elif 'GRASP' in algo_key:
        algo_key = 'GRaSP'
    else:
        algo_key = 'PC'

    assumptions = ALGORITHM_ASSUMPTIONS.get(algo_key, "Standard causal discovery assumptions")

    # Get n_nodes from props (all datasets now have this defined)
    n_nodes = props.get('n_nodes', 10)  # Default only for truly unknown datasets

    # Format the template
    prompt = formulation.template.format(
        dataset_name=dataset_name.title(),
        algorithm_name=algorithm_name,
        domain=props.get('domain', 'General'),
        n_nodes=n_nodes,
        n_samples=n_samples,
        n_edges=props.get('n_edges', 'unknown'),
        data_type=props.get('data_type', 'Mixed'),
        complexity=props.get('complexity', 'Medium'),
        algorithm_assumptions=assumptions,
        sample_adequacy=sample_adequacy
    )

    return prompt


def get_all_formulations() -> List[PromptTemplate]:
    """Return all three prompt formulations."""
    return [
        FORMULATION_1_DIRECT,
        FORMULATION_2_STEPBYSTEP,
        FORMULATION_3_METAKNOWLEDGE
    ]


if __name__ == "__main__":
    # Example usage
    print("="*80)
    print("PROMPT FORMULATION EXAMPLES")
    print("="*80)

    dataset = "titanic"
    algorithm = "PC"

    for formulation in get_all_formulations():
        print(f"\n{'='*80}")
        print(f"FORMULATION {formulation.formulation_id}: {formulation.name}")
        print(f"{'='*80}")
        print(formulation.description)
        print(f"\n{'-'*80}")
        prompt = generate_prompt(dataset, algorithm, formulation)
        print(prompt)
        print()
