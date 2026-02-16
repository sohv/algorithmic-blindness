#!/usr/bin/env python3
"""
Analyze LLM Reasoning Traces and Explanations
==============================================

Extracts and validates reasoning from LLM responses to distinguish genuine 
algorithmic understanding from pattern-matching.

Three analysis methods:
1. Reasoning Extraction: Parse what factors LLM cites in explanations
2. Consistency Checks: Verify predictions align with stated reasoning
3. Counterfactual Validation: Test if LLM adjusts reasoning for hypothetical scenarios
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np


@dataclass
class ReasoningTrace:
    """Extract and validate LLM reasoning."""
    dataset_name: str
    algorithm_name: str
    llm_model: str
    
    # Predictions
    precision_estimate: Tuple[float, float]  # (lower, upper)
    recall_estimate: Tuple[float, float]
    f1_estimate: Tuple[float, float]
    shd_estimate: Tuple[float, float]
    
    # Reasoning
    explanation: str
    mentioned_factors: List[str]  # e.g., ["dataset_size", "algorithm_assumptions", "sample_adequacy"]
    
    # Validation
    reasoning_coherent: Optional[bool] = None  # Reasoning logically consistent
    predictions_match_reasoning: Optional[bool] = None  # Estimates align with reasoning
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ReasoningAnalyzer:
    """Analyze LLM reasoning traces for consistency and coherence."""
    
    # Factors that should influence algorithm performance
    KNOWN_FACTORS = {
        'dataset_size': ['large dataset', 'small dataset', 'sample size', 'n_samples', 'samples', 'observations'],
        'algorithm_assumptions': ['assumptions', 'assumes', 'requirement', 'requires', 'supposing', 'given that'],
        'sample_adequacy': ['adequate', 'inadequate', 'sufficient', 'sample'],
        'dimensionality': ['variables', 'dimensions', 'features', 'high-dimensional', 'low-dimensional', 'nodes'],
        'graph_density': ['sparse', 'dense', 'edges', 'connectivity'],
        'data_type': ['gaussian', 'non-gaussian', 'linear', 'nonlinear', 'categorical', 'continuous'],
        'algorithm_type': ['constraint-based', 'functional', 'optimization', 'heuristic', 'linear', 'independence test'],
        'confounder_handling': ['latent', 'confound', 'hidden variable', 'unobserved'],
    }
    
    # Expected relationships: if factor mentioned, how should performance change?
    EXPECTED_EFFECTS = {
        # Larger datasets generally help
        'larger_dataset': {
            'precision': 'increases',     # Fewer spurious relationships
            'recall': 'increases',        # More power to detect edges
            'f1': 'increases',
            'shd': 'decreases',           # Fewer errors
        },
        # Higher dimensionality generally hurts
        'higher_dimensionality': {
            'precision': 'decreases',
            'recall': 'decreases',
            'f1': 'decreases',
            'shd': 'increases',
        },
        # Sparse graphs are easier
        'sparser_graph': {
            'precision': 'increases',
            'recall': 'increases',
            'f1': 'increases',
            'shd': 'decreases',
        },
        # Algorithm assumptions met helps
        'assumptions_met': {
            'precision': 'increases',
            'recall': 'increases',
            'f1': 'increases',
            'shd': 'decreases',
        },
    }
    
    def __init__(self):
        self.factor_patterns = {
            factor: re.compile('|'.join(keywords), re.IGNORECASE)
            for factor, keywords in self.KNOWN_FACTORS.items()
        }
    
    def extract_reasoning(self, explanation_text: str) -> Tuple[List[str], Dict]:
        """
        Extract factors mentioned in LLM explanation.
        
        Args:
            explanation_text: LLM's written explanation
            
        Returns:
            (mentioned_factors, factor_details)
        """
        mentioned_factors = []
        factor_details = {}
        
        for factor, pattern in self.factor_patterns.items():
            if pattern.search(explanation_text):
                mentioned_factors.append(factor)
                # Extract the sentence containing this factor
                sentences = explanation_text.split('.')
                for sent in sentences:
                    if pattern.search(sent):
                        factor_details[factor] = sent.strip()
                        break
        
        return mentioned_factors, factor_details
    
    def check_consistency(self, 
                         mentioned_factors: List[str],
                         factor_details: Dict[str, str],
                         predictions: Dict[str, Tuple[float, float]]) -> Dict:
        """
        Check if predictions are consistent with stated reasoning.
        
        Args:
            mentioned_factors: Factors LLM mentioned
            factor_details: Details about each factor
            predictions: {metric: (lower, upper)} predictions
            
        Returns:
            Consistency analysis report
        """
        consistency_issues = []
        coherence_score = 1.0
        
        # For each mentioned factor, check if predictions make sense
        for factor in mentioned_factors:
            # Check if text suggests positive or negative impact
            text = factor_details.get(factor, "")
            
            if factor == 'dataset_size':
                if 'small' in text.lower() or 'limited' in text.lower():
                    # Small dataset should hurt performance
                    if predictions['f1'][1] > 0.8:  # Too optimistic
                        consistency_issues.append(
                            f"Mentioned small dataset but predicted high F1 ({predictions['f1']})"
                        )
                        coherence_score -= 0.2
                elif 'large' in text.lower():
                    # Large dataset should help
                    if predictions['f1'][0] < 0.3:  # Too pessimistic
                        consistency_issues.append(
                            f"Mentioned large dataset but predicted low F1 ({predictions['f1']})"
                        )
                        coherence_score -= 0.2
            
            elif factor == 'dimensionality':
                if 'high' in text.lower() or 'many' in text.lower():
                    # High dimensionality should hurt
                    if predictions['f1'][1] > 0.8:
                        consistency_issues.append(
                            f"Mentioned high dimensionality but predicted high F1 ({predictions['f1']})"
                        )
                        coherence_score -= 0.2
            
            elif factor == 'assumptions_met':
                if 'met' in text.lower() or 'satisfied' in text.lower():
                    # Assumptions met should help
                    if predictions['f1'][0] < 0.4:
                        consistency_issues.append(
                            f"Mentioned assumptions met but predicted low F1 ({predictions['f1']})"
                        )
                        coherence_score -= 0.2
                elif 'violated' in text.lower() or 'not' in text.lower():
                    # Assumptions violated should hurt
                    if predictions['f1'][1] > 0.8:
                        consistency_issues.append(
                            f"Mentioned assumptions violated but predicted high F1 ({predictions['f1']})"
                        )
                        coherence_score -= 0.2
        
        return {
            'coherence_score': max(0.0, coherence_score),
            'consistency_issues': consistency_issues,
            'has_major_issues': len(consistency_issues) > 0
        }
    
    def generate_counterfactual_prompt(self,
                                       dataset_name: str,
                                       algorithm_name: str,
                                       n_samples: int,
                                       n_nodes: int,
                                       modification: str) -> str:
        """
        Generate counterfactual prompt to test reasoning robustness.
        
        Modifications:
        - "double_size": What if dataset doubled?
        - "half_size": What if dataset halved?
        - "double_nodes": What if 2x more variables?
        - "assumptions_violated": What if key assumption violated?
        
        Args:
            dataset_name: Original dataset
            algorithm_name: Algorithm
            n_samples: Original sample count
            n_nodes: Original node count
            modification: Type of modification
            
        Returns:
            Counterfactual prompt
        """
        modifications = {
            "double_size": {
                "new_samples": int(n_samples * 2),
                "description": "double the sample size"
            },
            "half_size": {
                "new_samples": max(100, int(n_samples / 2)),
                "description": "halve the sample size"
            },
            "double_nodes": {
                "new_nodes": int(n_nodes * 2),
                "description": "double the number of variables"
            },
            "assumptions_violated": {
                "description": "key assumptions violated"
            }
        }
        
        if modification not in modifications:
            raise ValueError(f"Unknown modification: {modification}")
        
        mod = modifications[modification]
        
        prompt = f"""You previously estimated {algorithm_name}'s performance on {dataset_name}.

Now consider a counterfactual scenario: what if we {mod['description']}?

Original dataset: {n_samples} samples, {n_nodes} variables
Modified dataset: """
        
        if "new_samples" in mod:
            prompt += f"{mod['new_samples']} samples, {n_nodes} variables"
        elif "new_nodes" in mod:
            prompt += f"{n_samples} samples, {mod['new_nodes']} variables"
        else:
            prompt += f"{n_samples} samples, {n_nodes} variables (with violated assumptions)"
        
        prompt += f"""

Algorithm: {algorithm_name}

Would your performance estimates change? If so, how much?
- New Precision range: [X, X]
- New Recall range: [X, X]
- New F1 range: [X, X]
- New SHD range: [X, X]

Explain the reasoning for any changes."""
        
        return prompt
    
    def validate_counterfactual_reasoning(self,
                                         original_predictions: Dict[str, Tuple[float, float]],
                                         counterfactual_predictions: Dict[str, Tuple[float, float]],
                                         modification: str) -> Dict:
        """
        Check if counterfactual predictions are logically consistent.
        
        Args:
            original_predictions: Original estimate ranges
            counterfactual_predictions: Counterfactual estimate ranges
            modification: Type of modification applied
            
        Returns:
            Validation report
        """
        issues = []
        reasoning_quality = 1.0
        
        if modification == "double_size":
            # Doubling dataset should improve (or stay same), not hurt
            for metric in ['precision', 'recall', 'f1']:
                if counterfactual_predictions[metric][1] < original_predictions[metric][0]:
                    issues.append(f"{metric}: Got worse with larger dataset")
                    reasoning_quality -= 0.25
            
            if counterfactual_predictions['shd'][0] > original_predictions['shd'][1]:
                issues.append("SHD: Got worse with larger dataset")
                reasoning_quality -= 0.25
        
        elif modification == "half_size":
            # Halving dataset should hurt (or stay same), not improve
            for metric in ['precision', 'recall', 'f1']:
                if counterfactual_predictions[metric][0] > original_predictions[metric][1]:
                    issues.append(f"{metric}: Got better with smaller dataset")
                    reasoning_quality -= 0.25
            
            if counterfactual_predictions['shd'][1] < original_predictions['shd'][0]:
                issues.append("SHD: Got better with smaller dataset")
                reasoning_quality -= 0.25
        
        elif modification == "double_nodes":
            # More variables (generally) should hurt
            for metric in ['precision', 'recall', 'f1']:
                if counterfactual_predictions[metric][0] > original_predictions[metric][1]:
                    issues.append(f"{metric}: Got better with more variables")
                    reasoning_quality -= 0.25
        
        return {
            'reasoning_quality': max(0.0, reasoning_quality),
            'logical_errors': issues,
            'is_logically_consistent': len(issues) == 0
        }
    
    def analyze_trace(self, 
                     dataset_name: str,
                     algorithm_name: str,
                     llm_model: str,
                     predictions: Dict[str, Tuple[float, float]],
                     explanation: str) -> ReasoningTrace:
        """
        Full analysis of LLM reasoning trace.
        
        Args:
            dataset_name: Dataset used
            algorithm_name: Algorithm tested
            llm_model: Which LLM
            predictions: Metric predictions
            explanation: LLM's explanation
            
        Returns:
            ReasoningTrace with analysis results
        """
        mentioned_factors, factor_details = self.extract_reasoning(explanation)
        consistency = self.check_consistency(mentioned_factors, factor_details, predictions)
        
        trace = ReasoningTrace(
            dataset_name=dataset_name,
            algorithm_name=algorithm_name,
            llm_model=llm_model,
            precision_estimate=predictions.get('precision', (0, 1)),
            recall_estimate=predictions.get('recall', (0, 1)),
            f1_estimate=predictions.get('f1', (0, 1)),
            shd_estimate=predictions.get('shd', (0, 1)),
            explanation=explanation,
            mentioned_factors=mentioned_factors,
            reasoning_coherent=not consistency['has_major_issues'],
            predictions_match_reasoning=consistency['coherence_score'] > 0.7
        )
        
        return trace


def main():
    """Demo: Analyze sample LLM responses."""
    analyzer = ReasoningAnalyzer()
    
    # Example 1: Good reasoning
    explanation_1 = """
    The Titanic dataset has 7 variables and ~900 samples. PC algorithm uses independence 
    tests, so with adequate sample size this should work reasonably well. However, 7 variables 
    is moderate complexity. I'd expect precision around 0.65-0.75 since some false positives 
    are likely with this sample size. Recall similar range since the dataset is reasonably sized.
    """
    
    predictions_1 = {
        'precision': (0.65, 0.75),
        'recall': (0.60, 0.70),
        'f1': (0.62, 0.72),
        'shd': (2, 5)
    }
    
    trace_1 = analyzer.analyze_trace(
        dataset_name='titanic',
        algorithm_name='PC',
        llm_model='gpt4',
        predictions=predictions_1,
        explanation=explanation_1
    )
    
    print("=== Example 1: Good Reasoning ===")
    print(f"Mentioned factors: {trace_1.mentioned_factors}")
    print(f"Reasoning coherent: {trace_1.reasoning_coherent}")
    print(f"Predictions match reasoning: {trace_1.predictions_match_reasoning}")
    print()
    
    # Example 2: Incoherent reasoning
    explanation_2 = """
    This is a causal discovery problem. The algorithm will definitely achieve high accuracy.
    I expect very high precision and recall.
    """
    
    predictions_2 = {
        'precision': (0.95, 0.99),  # Unrealistic given small dataset
        'recall': (0.95, 0.99),
        'f1': (0.95, 0.99),
        'shd': (0, 2)
    }
    
    trace_2 = analyzer.analyze_trace(
        dataset_name='titanic',
        algorithm_name='PC',
        llm_model='claude',
        predictions=predictions_2,
        explanation=explanation_2
    )
    
    print("=== Example 2: Poor Reasoning (Pattern-Matching) ===")
    print(f"Mentioned factors: {trace_2.mentioned_factors}")
    print(f"Reasoning coherent: {trace_2.reasoning_coherent}")
    print(f"Predictions match reasoning: {trace_2.predictions_match_reasoning}")
    print()
    
    # Example 3: Counterfactual test
    print("=== Example 3: Counterfactual Reasoning ===")
    counterfactual_prompt = analyzer.generate_counterfactual_prompt(
        dataset_name='titanic',
        algorithm_name='PC',
        n_samples=900,
        n_nodes=7,
        modification='double_size'
    )
    print(counterfactual_prompt)
    print()
    
    # Simulate counterfactual response
    cf_predictions = {  # Should improve with larger dataset
        'precision': (0.70, 0.80),  # Improved from 0.65-0.75
        'recall': (0.68, 0.78),     # Improved from 0.60-0.70
        'f1': (0.69, 0.79),
        'shd': (1, 4)               # Improved from 2-5
    }
    
    cf_validation = analyzer.validate_counterfactual_reasoning(
        original_predictions=predictions_1,
        counterfactual_predictions=cf_predictions,
        modification='double_size'
    )
    
    print(f"Logical consistency: {cf_validation['is_logically_consistent']}")
    print(f"Reasoning quality: {cf_validation['reasoning_quality']:.2f}")
    if cf_validation['logical_errors']:
        print(f"Issues: {cf_validation['logical_errors']}")


if __name__ == '__main__':
    main()
