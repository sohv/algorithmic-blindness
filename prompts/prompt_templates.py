"""
Three Prompt Formulations for LLM Meta-Knowledge Testing
=========================================================

This module defines three distinct prompt formulations to test
whether LLM estimates are robust across different question styles.

Formulations:
1. DIRECT: Straightforward question about algorithm performance
2. STEP-BY-STEP: Guides LLM through reasoning process (asks LLM to recall assumptions)
3. META-KNOWLEDGE: Frames as confidence interval estimation task

Why this matters:
- If variance across prompts is LOW (<20%): Results are robust
- If variance is HIGH (>20%): Need to discuss sensitivity
- Either way: Shows methodological thoroughness

IMPORTANT: Before running full experiments, pilot test parsing with:
    python src/llm/query_all_llms.py --datasets asia --algorithms pc --formulation 1 2 3 --models gpt5 claude gemini deepseek llama qwen
    Then verify with: python test_parsing.py --verbose
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
- Variables: {n_nodes}
- Samples: {n_samples}
- Data type: {data_type}

Algorithm: {algorithm_name}

Estimate the algorithm's performance ranges [lower, upper] for:

Precision: [X.XX, X.XX]
Recall: [X.XX, X.XX]
F1: [X.XX, X.XX]
SHD: [X, X]

---CRITICAL INSTRUCTION---
You MUST output ONLY these four lines of numerical ranges.
Do NOT include any reasoning, explanations, context, labels, or additional text.
Your entire response must be EXACTLY:

Precision: [X.XX, X.XX]
Recall: [X.XX, X.XX]
F1: [X.XX, X.XX]
SHD: [X, X]

Nothing else. No preamble. No conclusions. Just these four lines."""
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
- Variables: {n_nodes}
- Samples: {n_samples}
- Data type: {data_type}
- Complexity: {complexity}

Algorithm: {algorithm_name}

Before predicting, reason through these questions:
1. What are {algorithm_name}'s core assumptions and requirements?
2. Does {dataset_name} (with {n_samples} samples, {data_type} data) satisfy these assumptions?
3. How does the dataset complexity affect {algorithm_name}'s reliability?
4. What performance range is realistic given these factors?

---CRITICAL INSTRUCTION---
After your internal reasoning, you MUST output ONLY these four lines of numerical ranges.
DO NOT INCLUDE any reasoning, summary, explanations, context, labels, or additional text.
Your response must contain EXACTLY these four lines and nothing else:

Precision: [X.XX, X.XX]
Recall: [X.XX, X.XX]
F1: [X.XX, X.XX]
SHD: [X, X]

No preamble. No conclusions. No summary. No reasoning. Just the four lines."""
)


# =============================================================================
# FORMULATION 3: Meta-Knowledge / Confidence Interval Framing
# =============================================================================

FORMULATION_3_METAKNOWLEDGE = PromptTemplate(
    name="Meta-Knowledge Framing",
    description="Frames task as predicting algorithm variance and confidence intervals",
    formulation_id=3,
    template="""You are a statistician evaluating causal discovery algorithms.

A researcher repeatedly runs {algorithm_name} on {dataset_name} with different random seeds.

Dataset characteristics:
- Variables: {n_nodes}
- Samples: {n_samples}
- Data type: {data_type}

Based on {algorithm_name}'s known behavior and {dataset_name}'s properties,
predict the expected performance distribution.

What ranges capture 95% of typical outcomes?

Precision: [X.XX, X.XX]
Recall: [X.XX, X.XX]
F1: [X.XX, X.XX]
SHD: [X, X]

---CRITICAL INSTRUCTION---
You MUST output ONLY these four lines of numerical ranges.
Do NOT include any reasoning, thoughts, context, labels, or additional text.
Your entire response must be EXACTLY:

Precision: [X.XX, X.XX]
Recall: [X.XX, X.XX]
F1: [X.XX, X.XX]
SHD: [X, X]

NOTHING ELSE. NOT A SINGLE ADDITIONAL WORD."""
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
    'asia': {
        'domain': 'Medical (lung cancer diagnosis)',
        'data_type': 'Discrete',
        'complexity': 'Small (8 variables, 8 edges)',
        'n_nodes': 8,
        'n_edges': 8
    },
    'alarm': {
        'domain': 'Medical (ICU alarm monitoring system)',
        'data_type': 'Discrete',
        'complexity': 'Medium (37 variables, 46 edges)',
        'n_nodes': 37,
        'n_edges': 46
    },
    'sachs': {
        'domain': 'Biology (causal protein-signaling networks)',
        'data_type': 'Continuous (flow cytometry)',
        'complexity': 'Small (11 variables, 17 edges)',
        'n_nodes': 11,
        'n_edges': 17
    },
    'survey': {
        'domain': 'Social Science (attitudes survey)',
        'data_type': 'Discrete',
        'complexity': 'Small (6 variables, 6 edges)',
        'n_nodes': 6,
        'n_edges': 6
    },
    'child': {
        'domain': 'Medical (pediatric diagnosis expert system)',
        'data_type': 'Discrete',
        'complexity': 'Small (20 variables, 25 edges)',
        'n_nodes': 20,
        'n_edges': 25
    },
    'cancer': {
        'domain': 'Medical (lung cancer diagnosis)',
        'data_type': 'Discrete',
        'complexity': 'Small (5 variables, 4 edges)',
        'n_nodes': 5,
        'n_edges': 4
    },
    'hepar2': {
        'domain': 'Medical (Hepatitis diagnosis)',
        'data_type': 'Discrete',
        'complexity': 'Large (70 variables, 120 edges)',
        'n_nodes': 70,
        'n_edges': 120
    },
    'earthquake': {
        'domain': 'Seismology (earthquake causes triggering of alarms and reports)',
        'data_type': 'Discrete',
        'complexity': 'Small (5 variables, 4 edges)',
        'n_nodes': 5,
        'n_edges': 4
    },
    'insurance': {
        'domain': 'Insurance (risk assessment with hidden variables)',
        'data_type': 'Discrete',
        'complexity': 'Medium (27 variables, 52 edges)',
        'n_nodes': 27,
        'n_edges': 52
    },
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
                'domain': 'Synthetic linear Gaussian',
                'data_type': 'Continuous',
                'complexity': 'Small',
                'n_nodes': 12,
                'n_edges': 13
            }
        elif '30' in dataset_key:
            props = {
                'domain': 'Synthetic linear Gaussian',
                'data_type': 'Continuous',
                'complexity': 'Medium',
                'n_nodes': 30,
                'n_edges': 87
            }
        elif '50' in dataset_key:
            props = {
                'domain': 'Synthetic linear Gaussian',
                'data_type': 'Continuous',
                'complexity': 'High',
                'n_nodes': 50,
                'n_edges': 368
            }
        elif '60' in dataset_key:
            props = {
                'domain': 'Synthetic linear Gaussian',
                'data_type': 'Continuous',
                'complexity': 'Very High',
                'n_nodes': 60,
                'n_edges': 531
            }
        else:
            props = {
                'domain': 'Synthetic linear Gaussian',
                'data_type': 'Continuous',
                'complexity': 'Small',
                'n_nodes': 12,
                'n_edges': 13
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
