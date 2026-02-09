#!/usr/bin/env python3
"""
LLM Response Parser for Causal Discovery Performance Estimates
===============================================================

Extracts metric ranges from various LLM response formats:
- JSON formatted responses
- Natural language descriptions
- Range notation (e.g., "0.6-0.8", "60%-80%", "(0.6, 0.8)")
- Mixed formats with explanations

Output format matches algorithmic variance results for direct comparison.
"""

import re
import json
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MetricRange:
    """Container for a metric's predicted range."""
    lower: float
    upper: float
    metric_name: str


@dataclass
class ParsedLLMEstimate:
    """Parsed LLM performance estimates."""
    precision: Optional[MetricRange] = None
    recall: Optional[MetricRange] = None
    f1: Optional[MetricRange] = None
    shd: Optional[MetricRange] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary format matching variance results."""
        result = {}
        for metric in ['precision', 'recall', 'f1', 'shd']:
            range_obj = getattr(self, metric)
            if range_obj:
                result[metric] = (range_obj.lower, range_obj.upper)
        return result


class LLMResponseParser:
    """
    Parse LLM responses to extract performance metric ranges.

    Handles multiple formats:
    1. Structured format: "Precision: (0.6, 0.8)"
    2. Range format: "Precision: 0.6-0.8"
    3. Percentage format: "Precision: 60%-80%"
    4. JSON format: {"precision": [0.6, 0.8]}
    5. Natural language: "I expect precision to be between 0.6 and 0.8"
    """

    # Regex patterns for extracting ranges
    PATTERNS = {
        # Structured format: "Precision: (0.6, 0.8)" or "Precision: [0.6, 0.8]"
        'structured': r'(precision|recall|f1|shd)[:\s]+[\(\[]?\s*([0-9.]+)\s*,\s*([0-9.]+)\s*[\)\]]?',

        # Range format: "Precision: 0.6-0.8" or "Precision: 0.6 to 0.8"
        'range': r'(precision|recall|f1|shd)[:\s]+([0-9.]+)\s*[-–to]+\s*([0-9.]+)',

        # Percentage format: "Precision: 60%-80%"
        'percentage': r'(precision|recall|f1|shd)[:\s]+([0-9.]+)%\s*[-–to]+\s*([0-9.]+)%',

        # Natural language: "between X and Y"
        'natural': r'(precision|recall|f1|shd)\s+(?:to be\s+)?between\s+([0-9.]+)\s+and\s+([0-9.]+)',
    }

    def __init__(self):
        self.patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }

    def parse(self, response_text: str) -> ParsedLLMEstimate:
        """
        Parse LLM response to extract metric ranges.

        Args:
            response_text: Raw text response from LLM

        Returns:
            ParsedLLMEstimate object with extracted ranges
        """
        # Try JSON parsing first
        estimate = self._try_parse_json(response_text)
        if estimate:
            return estimate

        # Try regex patterns
        estimate = self._parse_with_regex(response_text)

        return estimate

    def _try_parse_json(self, text: str) -> Optional[ParsedLLMEstimate]:
        """Try to parse response as JSON."""
        try:
            # Find JSON object in text
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            estimate = ParsedLLMEstimate()

            for metric in ['precision', 'recall', 'f1', 'shd']:
                if metric in data:
                    value = data[metric]

                    # Handle different JSON formats
                    if isinstance(value, list) and len(value) == 2:
                        lower, upper = value
                    elif isinstance(value, dict):
                        lower = value.get('lower', value.get('min'))
                        upper = value.get('upper', value.get('max'))
                    else:
                        continue

                    # Convert percentages if needed
                    if lower > 1.0 and metric != 'shd':
                        lower /= 100.0
                        upper /= 100.0

                    setattr(estimate, metric, MetricRange(lower, upper, metric))

            return estimate if any([estimate.precision, estimate.recall, estimate.f1, estimate.shd]) else None

        except (json.JSONDecodeError, ValueError, KeyError):
            return None

    def _parse_with_regex(self, text: str) -> ParsedLLMEstimate:
        """Parse response using regex patterns."""
        estimate = ParsedLLMEstimate()

        # Try each pattern type
        for pattern_name, pattern in self.patterns.items():
            matches = pattern.findall(text)

            for match in matches:
                metric_name, lower_str, upper_str = match

                # Normalize metric name
                metric_name = metric_name.lower().replace('-', '').replace('_', '')
                if 'f1' in metric_name or 'f-1' in metric_name:
                    metric_name = 'f1'

                try:
                    lower = float(lower_str)
                    upper = float(upper_str)

                    # Convert percentages to decimals (except for SHD)
                    if pattern_name == 'percentage' and metric_name != 'shd':
                        lower /= 100.0
                        upper /= 100.0

                    # Store the range
                    if metric_name in ['precision', 'recall', 'f1', 'shd']:
                        setattr(estimate, metric_name, MetricRange(lower, upper, metric_name))

                except ValueError:
                    continue

        return estimate

    def validate_estimate(self, estimate: ParsedLLMEstimate) -> bool:
        """
        Validate that extracted ranges are reasonable.

        Args:
            estimate: ParsedLLMEstimate object

        Returns:
            True if estimate passes validation, False otherwise
        """
        for metric in ['precision', 'recall', 'f1']:
            range_obj = getattr(estimate, metric)
            if range_obj:
                # Check bounds
                if range_obj.lower < 0 or range_obj.upper > 1:
                    return False
                if range_obj.lower > range_obj.upper:
                    return False

        # SHD should be non-negative
        if estimate.shd:
            if estimate.shd.lower < 0 or estimate.shd.lower > estimate.shd.upper:
                return False

        return True


def parse_llm_response(response_text: str, validate: bool = True) -> Optional[Dict]:
    """
    Convenience function to parse LLM response and return as dictionary.

    Args:
        response_text: Raw text response from LLM
        validate: Whether to validate ranges (default: True)

    Returns:
        Dictionary mapping metric names to (lower, upper) tuples, or None if parsing fails
    """
    parser = LLMResponseParser()
    estimate = parser.parse(response_text)

    if validate and not parser.validate_estimate(estimate):
        print("Warning: Parsed estimates failed validation")
        return None

    result = estimate.to_dict()
    return result if result else None


if __name__ == "__main__":
    # Test cases
    print("="*80)
    print("LLM RESPONSE PARSER TESTS")
    print("="*80)

    test_cases = [
        # Case 1: Structured format
        """
        Precision: (0.65, 0.78)
        Recall: (0.55, 0.70)
        F1: (0.60, 0.74)
        SHD: (5, 12)
        """,

        # Case 2: Range format
        """
        Based on the dataset characteristics:
        - Precision: 0.6-0.8
        - Recall: 0.5-0.7
        - F1-score: 0.55-0.75
        - SHD: 3 to 10
        """,

        # Case 3: Percentage format
        """
        Expected performance:
        Precision: 60%-80%
        Recall: 50%-70%
        F1: 55%-75%
        SHD: 5-12 edges
        """,

        # Case 4: JSON format
        """
        {
            "precision": [0.6, 0.8],
            "recall": [0.5, 0.7],
            "f1": [0.55, 0.75],
            "shd": [5, 12]
        }
        """,

        # Case 5: Natural language
        """
        I expect precision to be between 0.65 and 0.80,
        recall to be between 0.55 and 0.70,
        F1-score to be between 0.60 and 0.75,
        and SHD to be between 5 and 12.
        """
    ]

    parser = LLMResponseParser()

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print("-" * 40)
        print(test_case.strip())
        print("\nParsed Result:")
        result = parse_llm_response(test_case)
        if result:
            for metric, (lower, upper) in result.items():
                print(f"  {metric}: ({lower:.3f}, {upper:.3f})")
        else:
            print("  Failed to parse")
        print()
