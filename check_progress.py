#!/usr/bin/env python3
"""
9-Day UAI Submission Plan - Master Checklist
=============================================

Tracks progress through the 9-day emergency plan.

Usage:
    python check_progress.py --show-status
    python check_progress.py --mark-complete "Day 1: Metrics & Baselines"
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List


# 9-Day Plan Structure
PLAN = {
    "Day 1": {
        "title": "Critical Infrastructure - Metrics & Baselines",
        "tasks": [
            "Create src/evaluation/metrics.py with calibrated_coverage()",
            "Create src/baselines/simple_baselines.py",
            "Test metrics on 1 sample dataset",
            "Verify baselines generate reasonable predictions"
        ],
        "deliverables": [
            "src/evaluation/metrics.py",
            "src/baselines/simple_baselines.py"
        ],
        "estimated_hours": 8
    },
    "Day 2": {
        "title": "Run Algorithm Experiments (5 datasets only)",
        "tasks": [
            "Run algorithms on 5 datasets: asia, sachs, cancer, synthetic_12, synthetic_30",
            "100 runs per (dataset, algorithm) = 2000 total runs",
            "Generate variance results with 95% CIs",
            "Verify all variance files saved correctly",
            "Spot-check: CIs look reasonable (non-zero width)"
        ],
        "deliverables": [
            "results/variance/asia_*_variance.json (4 algorithms)",
            "results/variance/sachs_*_variance.json (4 algorithms)",
            "results/variance/cancer_*_variance.json (4 algorithms)",
            "results/variance/synthetic_12_*_variance.json (4 algorithms)",
            "results/variance/synthetic_30_*_variance.json (4 algorithms)"
        ],
        "estimated_hours": 8
    },
    "Day 3": {
        "title": "Query All LLMs (300 queries)",
        "tasks": [
            "Query 5 LLMs × 5 datasets × 4 algorithms × 3 formulations = 300 queries",
            "Parse all LLM responses",
            "Handle parsing failures (log and retry)",
            "Verify: No critical parsing errors",
            "Save all LLM comparison files"
        ],
        "deliverables": [
            "results/llm_comparisons/*_llm_comparison.json (20 files minimum)",
            "logs/llm_queries.log"
        ],
        "estimated_hours": 8
    },
    "Day 4": {
        "title": "Compute Calibrated Coverage (PRIMARY METRIC)",
        "tasks": [
            "Run compute_metrics.py on all LLMs + baselines",
            "Compute calibrated coverage for all",
            "Compute MAE for all",
            "Generate main_results.csv (Table 1)",
            "Generate per-dataset and per-algorithm breakdowns",
            "Verify: At least 1 LLM beats random significantly"
        ],
        "deliverables": [
            "results/evaluation/main_results.csv",
            "results/evaluation/detailed_results.json",
            "results/evaluation/coverage_by_dataset.csv",
            "results/evaluation/coverage_by_algorithm.csv"
        ],
        "estimated_hours": 8,
        "critical": True
    },
    "Day 5": {
        "title": "Statistical Tests & Significance",
        "tasks": [
            "Run Wilcoxon signed-rank tests (LLM vs random)",
            "Apply FDR correction (Benjamini-Hochberg)",
            "Compute Cohen's d effect sizes",
            "Generate interpretation text for paper",
            "Create statistical_tests.csv"
        ],
        "deliverables": [
            "results/statistics/statistical_tests.csv",
            "results/statistics/interpretation.txt"
        ],
        "estimated_hours": 8
    },
    "Day 6": {
        "title": "Generate Figures & Tables",
        "tasks": [
            "Create Figure 1: Calibrated coverage bar plot",
            "Generate Table 1: Main results (LaTeX)",
            "Generate Table 2: Dataset breakdown (optional)",
            "Generate Table 3: Algorithm breakdown (optional)",
            "Verify all figures/tables render correctly"
        ],
        "deliverables": [
            "paper/figures/figure1_coverage.pdf",
            "paper/tables/table1_main_results.tex",
            "paper/tables/table2_dataset_breakdown.tex"
        ],
        "estimated_hours": 8
    },
    "Day 7": {
        "title": "Write Paper - Structure & Methods",
        "tasks": [
            "Introduction (1 page): Problem, testbed, contributions",
            "Related Work (0.75 pages): LLM capabilities, causal discovery",
            "Methodology (1.5 pages): Datasets, algorithms, ground truth, metrics",
            "Target: 3-4 pages written"
        ],
        "deliverables": [
            "paper/draft.tex (sections 1-3)"
        ],
        "estimated_hours": 8,
        "critical": True
    },
    "Day 8": {
        "title": "Write Paper - Results & Analysis",
        "tasks": [
            "Results section (2 pages): Overall performance, significance, ranking",
            "Insert all tables and figures",
            "Discussion (1 page): Interpretation, implications",
            "Target: 7-8 pages total"
        ],
        "deliverables": [
            "paper/draft.tex (sections 4-5, 7-8 pages)"
        ],
        "estimated_hours": 8,
        "critical": True
    },
    "Day 9": {
        "title": "Polish & Submit",
        "tasks": [
            "Write Limitations section (0.5 pages)",
            "Write Conclusion (0.5 pages)",
            "Add References (proper citations)",
            "Proofread entire paper",
            "Check formatting (UAI template)",
            "SUBMIT to UAI"
        ],
        "deliverables": [
            "paper/final.pdf",
            "UAI submission confirmation"
        ],
        "estimated_hours": 8,
        "critical": True
    }
}


def get_status_file() -> Path:
    """Get path to status tracking file."""
    return Path('9day_plan_status.json')


def load_status() -> Dict:
    """Load current progress status."""
    status_file = get_status_file()
    
    if status_file.exists():
        with open(status_file, 'r') as f:
            return json.load(f)
    
    # Initialize fresh status
    status = {
        "started": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "days": {}
    }
    
    for day_name in PLAN.keys():
        status["days"][day_name] = {
            "completed": False,
            "tasks_completed": [],
            "completion_date": None
        }
    
    return status


def save_status(status: Dict):
    """Save progress status."""
    status["last_updated"] = datetime.now().isoformat()
    status_file = get_status_file()
    
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


def check_deliverables_exist(deliverables: List[str]) -> Dict[str, bool]:
    """Check which deliverables exist on disk."""
    results = {}
    
    for deliverable in deliverables:
        # Handle wildcards
        if '*' in deliverable:
            pattern = deliverable
            base_path = Path(pattern.split('*')[0]).parent
            
            if base_path.exists():
                matching_files = list(base_path.glob(Path(pattern).name))
                results[deliverable] = len(matching_files) > 0
            else:
                results[deliverable] = False
        else:
            results[deliverable] = Path(deliverable).exists()
    
    return results


def show_status():
    """Display current progress."""
    status = load_status()
    
    print("\n" + "="*70)
    print("9-DAY UAI SUBMISSION PLAN - STATUS")
    print("="*70)
    print(f"Started: {status['started']}")
    print(f"Last Updated: {status['last_updated']}")
    print()
    
    completed_days = sum(1 for day_status in status['days'].values() if day_status['completed'])
    
    print(f"Progress: {completed_days}/9 days completed")
    print()
    
    for day_name, day_plan in PLAN.items():
        day_status = status['days'][day_name]
        
        # Status symbol
        if day_status['completed']:
            symbol = "✓"
            status_text = f"[DONE on {day_status['completion_date']}]"
        else:
            symbol = "○"
            status_text = "[TODO]"
        
        critical_marker = " ⚠️ CRITICAL" if day_plan.get('critical', False) else ""
        
        print(f"{symbol} {day_name}: {day_plan['title']}{critical_marker}")
        print(f"   {status_text}")
        print(f"   Estimated: {day_plan['estimated_hours']} hours")
        
        # Check deliverables
        deliverable_status = check_deliverables_exist(day_plan['deliverables'])
        n_complete = sum(1 for exists in deliverable_status.values() if exists)
        n_total = len(deliverable_status)
        
        print(f"   Deliverables: {n_complete}/{n_total} exist")
        
        for deliverable, exists in deliverable_status.items():
            marker = "✓" if exists else "✗"
            print(f"     {marker} {deliverable}")
        
        print()


def mark_complete(day_name: str):
    """Mark a day as complete."""
    if day_name not in PLAN:
        print(f"Error: Unknown day '{day_name}'")
        print(f"Valid days: {', '.join(PLAN.keys())}")
        return
    
    status = load_status()
    status['days'][day_name]['completed'] = True
    status['days'][day_name]['completion_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    save_status(status)
    
    print(f"✓ Marked {day_name} as complete")
    print()
    
    # Show updated status
    show_status()


def main():
    parser = argparse.ArgumentParser(description='9-Day Plan Progress Tracker')
    parser.add_argument('--show-status', action='store_true',
                        help='Show current progress')
    parser.add_argument('--mark-complete', type=str, metavar='DAY',
                        help='Mark day as complete (e.g., "Day 1")')
    parser.add_argument('--reset', action='store_true',
                        help='Reset progress tracking')
    
    args = parser.parse_args()
    
    if args.reset:
        status_file = get_status_file()
        if status_file.exists():
            status_file.unlink()
        print("Progress reset")
        return
    
    if args.mark_complete:
        mark_complete(args.mark_complete)
        return
    
    # Default: show status
    show_status()


if __name__ == '__main__':
    main()
