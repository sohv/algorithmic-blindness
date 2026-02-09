"""
Prompt Variation Study for LLM Meta-Knowledge Evaluation
=========================================================

This module implements the HIGHEST IMPACT critical fix:
testing robustness of LLM estimates across different prompt formulations.

Key components:
1. Three distinct prompt templates (direct, step-by-step, meta-knowledge)
2. Variance analysis across prompts
3. Robustness metrics and visualizations

Impact: Addresses the "results are prompt-dependent" criticism
Expected improvement: +15% acceptance probability (40% -> 55%)
"""

from .prompt_templates import (
    PromptTemplate,
    FORMULATION_1_DIRECT,
    FORMULATION_2_STEPBYSTEP,
    FORMULATION_3_METAKNOWLEDGE,
    generate_prompt
)

from .analyze_prompt_variance import (
    analyze_prompt_variance,
    compare_prompt_formulations,
    compute_prompt_robustness_score
)

__all__ = [
    'PromptTemplate',
    'FORMULATION_1_DIRECT',
    'FORMULATION_2_STEPBYSTEP',
    'FORMULATION_3_METAKNOWLEDGE',
    'generate_prompt',
    'analyze_prompt_variance',
    'compare_prompt_formulations',
    'compute_prompt_robustness_score'
]
