#!/usr/bin/env python3
"""
LLM Response Parsing Test
===========================

Tests that parsing works for ALL 5 LLMs before running 660 queries.

Tests:
1. Query each LLM with one test prompt
2. Parse the response using parse_llm_responses.py
3. Validate that ranges are reasonable
4. Report success/failure for each LLM

Usage:
    python test_parsing.py
    python test_parsing.py --llm claude  # Test specific LLM
    python test_parsing.py --verbose     # Show full responses
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'llm'))
sys.path.insert(0, str(Path(__file__).parent / 'prompts'))

import argparse
from llm_interface import LLMQueryInterface
from parse_llm_responses import parse_llm_response
from prompt_templates import generate_prompt, get_all_formulations

# Test configuration
TEST_DATASET = 'titanic'
TEST_ALGORITHM = 'PC'
TEST_N_SAMPLES = 891
ALL_LLMS = ['gpt4', 'deepseek', 'claude', 'gemini', 'llama']


def test_single_llm(llm_name: str, verbose: bool = False) -> tuple[bool, dict]:
    """
    Test parsing for a single LLM.

    Args:
        llm_name: Name of LLM to test
        verbose: If True, print full response

    Returns:
        (success, results_dict)
    """
    print(f"\n{'='*70}")
    print(f"Testing: {llm_name.upper()}")
    print(f"{'='*70}")

    try:
        # Initialize LLM
        print(f"[1/4] Initializing {llm_name}...")
        llm = LLMQueryInterface(llm_name)
        print(f"      ✓ Client initialized")

        # Generate prompt (use Formulation 1 - Direct)
        print(f"[2/4] Generating prompt...")
        formulation = get_all_formulations()[0]
        prompt = generate_prompt(TEST_DATASET, TEST_ALGORITHM, formulation, TEST_N_SAMPLES)
        print(f"      ✓ Prompt generated ({len(prompt)} chars)")

        # Query LLM
        print(f"[3/4] Querying {llm_name}...")
        response = llm.query(prompt)

        if not response.success:
            print(f"      ✗ Query failed: {response.error}")
            return False, {'error': response.error}

        print(f"      ✓ Response received ({len(response.content)} chars)")

        if verbose:
            print(f"\n--- Raw Response ---")
            print(response.content[:500])
            if len(response.content) > 500:
                print(f"... (truncated, total {len(response.content)} chars)")
            print(f"--- End Response ---\n")

        # Parse response
        print(f"[4/4] Parsing response...")
        metrics = parse_llm_response(response.content, validate=True)

        if not metrics:
            print(f"      ✗ Parsing failed")
            print(f"      Raw response preview:")
            print(f"      {response.content[:200]}...")
            return False, {'error': 'parsing_failed', 'response': response.content[:500]}

        print(f"      ✓ Parsing successful")

        # Display parsed metrics
        print(f"\n📊 Parsed Metrics:")
        for metric, (lower, upper) in metrics.items():
            if metric == 'shd':
                print(f"   {metric.upper():10s}: [{lower:.0f}, {upper:.0f}]")
            else:
                print(f"   {metric.upper():10s}: [{lower:.3f}, {upper:.3f}]")

        # Validate ranges
        print(f"\n✅ Validation:")
        all_valid = True

        for metric, (lower, upper) in metrics.items():
            # Check bounds
            if metric in ['precision', 'recall', 'f1']:
                if not (0 <= lower <= 1 and 0 <= upper <= 1):
                    print(f"   ✗ {metric}: values out of range [0, 1]")
                    all_valid = False
                    continue

            if metric == 'shd' and lower < 0:
                print(f"   ✗ {metric}: negative value")
                all_valid = False
                continue

            if lower > upper:
                print(f"   ✗ {metric}: lower > upper")
                all_valid = False
                continue

            print(f"   ✓ {metric}: valid")

        if all_valid:
            print(f"\n{'='*70}")
            print(f"✅ {llm_name.upper()} PASSED")
            print(f"{'='*70}")
            return True, metrics
        else:
            print(f"\n{'='*70}")
            print(f"⚠️  {llm_name.upper()} PARSED BUT VALIDATION FAILED")
            print(f"{'='*70}")
            return False, metrics

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ {llm_name.upper()} FAILED WITH ERROR")
        print(f"{'='*70}")
        print(f"Error: {str(e)}")
        import traceback
        if verbose:
            traceback.print_exc()
        return False, {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Test LLM response parsing")
    parser.add_argument('--llm', type=str, choices=ALL_LLMS,
                       help='Test specific LLM only')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show full responses')
    parser.add_argument('--formulation', type=int, choices=[1, 2, 3],
                       help='Test specific formulation (1=Direct, 2=Step-by-Step, 3=Meta-Knowledge)')

    args = parser.parse_args()

    print("="*70)
    print("LLM RESPONSE PARSING TEST")
    print("="*70)
    print(f"Dataset: {TEST_DATASET}")
    print(f"Algorithm: {TEST_ALGORITHM}")
    print(f"Formulation: 1 (Direct Question)")
    print("="*70)

    # Select LLMs to test
    llms_to_test = [args.llm] if args.llm else ALL_LLMS

    # Run tests
    results = {}
    for llm_name in llms_to_test:
        success, metrics = test_single_llm(llm_name, verbose=args.verbose)
        results[llm_name] = {
            'success': success,
            'metrics': metrics
        }

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)

    print(f"\nResults: {passed}/{total} LLMs passed\n")

    for llm_name, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {status}  {llm_name.upper():10s}")

        if result['success']:
            metrics = result['metrics']
            if 'precision' in metrics:
                p_lower, p_upper = metrics['precision']
                print(f"          Precision: [{p_lower:.3f}, {p_upper:.3f}]")

    print(f"\n{'='*70}")

    if passed == total:
        print("✅ ALL TESTS PASSED - Ready to run full experiment!")
        print(f"{'='*70}")
        return 0
    else:
        print(f"⚠️  {total - passed} LLM(s) FAILED - Fix parsing before running full experiment")
        print(f"{'='*70}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
