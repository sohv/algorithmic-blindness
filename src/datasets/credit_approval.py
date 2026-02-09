#!/usr/bin/env python3
"""
Credit Approval Dataset (UCI ML Repository)
============================================

Source: https://archive.ics.uci.edu/dataset/27/credit+approval

Dataset Information:
- 690 instances
- 15 attributes (6 numerical, 9 categorical)
- Binary outcome: credit approved/denied
- Contains missing values (marked as '?')

Variables:
A1: b, a (categorical)
A2: continuous
A3: continuous
A4: u, y, l, t (categorical)
A5: g, p, gg (categorical)
A6: c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff (categorical)
A7: v, h, bb, j, n, z, dd, ff, o (categorical)
A8: continuous
A9: t, f (categorical)
A10: t, f (categorical)
A11: continuous
A12: t, f (categorical)
A13: g, p, s (categorical)
A14: continuous
A15: continuous
A16: +,- (class: approved/denied)

Causal Structure (Domain Knowledge):
Based on credit risk assessment literature:
- Income/Employment → Credit History
- Income/Employment → Credit Limit
- Credit History → Approval
- Debt → Approval
- Age → Credit History
- Employment → Income
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder


def load_credit_approval() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load and prepare Credit Approval dataset for causal discovery.

    Returns:
        data: pandas DataFrame with encoded features
        true_graph: Ground truth DAG adjacency matrix (15x15)
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("Installing ucimlrepo...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'ucimlrepo', '--break-system-packages'])
        from ucimlrepo import fetch_ucirepo

    print("Loading Credit Approval dataset...")

    # Fetch dataset
    credit = fetch_ucirepo(id=27)

    # Get features and targets
    X = credit.data.features
    y = credit.data.targets

    # Combine features and target
    df = pd.concat([X, y], axis=1)

    # Handle missing values (marked as '?')
    df = df.replace('?', np.nan)

    # Encode categorical variables
    le = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            # Fill missing categorical values with a placeholder
            df[col] = df[col].fillna('missing')
            df[col] = le.fit_transform(df[col].astype(str))

    # Fill numerical missing values with median
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())

    # Drop any remaining rows with NaN
    df = df.dropna()

    # Convert all to float64 for numerical stability
    df = df.astype(np.float64)

    print(f"Credit Approval dataset loaded: {df.shape[0]} samples, {df.shape[1]} variables")

    # Define ground truth DAG based on domain knowledge
    # Variables (after encoding):
    # 0-14: A1-A15 (credit attributes)
    # 15: A16 (approval outcome)

    n_vars = df.shape[1]
    true_graph = np.zeros((n_vars, n_vars))

    # Known causal relationships from credit risk literature
    # (These are approximate - real credit scoring is complex)

    # A2 (age) -> A8 (credit history years)
    if n_vars >= 9:
        true_graph[1, 7] = 1

    # A3 (debt) -> A15 (approval)
    if n_vars >= 16:
        true_graph[2, 15] = 1

    # A8 (credit history) -> A15 (approval)
    if n_vars >= 16:
        true_graph[7, 15] = 1

    # A11 (income/months employed) -> A15 (approval)
    if n_vars >= 16:
        true_graph[10, 15] = 1

    # A14 (income) -> A15 (approval)
    if n_vars >= 16:
        true_graph[13, 15] = 1

    # A2 (age) -> A11 (months employed)
    if n_vars >= 12:
        true_graph[1, 10] = 1

    # A11 (months employed) -> A14 (income)
    if n_vars >= 15:
        true_graph[10, 13] = 1

    return df, true_graph


def get_dataset_info() -> dict:
    """Return dataset metadata."""
    return {
        'name': 'credit',
        'full_name': 'Credit Approval',
        'domain': 'Finance (credit risk assessment)',
        'n_nodes': 16,
        'n_samples': 690,
        'n_edges': 7,
        'data_type': 'Mixed (categorical + continuous)',
        'complexity': 'Medium (16 variables)',
        'description': 'Credit card application approval prediction'
    }


if __name__ == "__main__":
    # Test loading
    print("="*80)
    print("CREDIT APPROVAL DATASET TEST")
    print("="*80)

    data, true_graph = load_credit_approval()

    print(f"\nData shape: {data.shape}")
    print(f"\nFirst 5 rows:")
    print(data.head())

    print(f"\nGround truth DAG:")
    print(f"Nodes: {true_graph.shape[0]}")
    print(f"Edges: {int(true_graph.sum())}")

    print(f"\nEdge list:")
    for i in range(true_graph.shape[0]):
        for j in range(true_graph.shape[1]):
            if true_graph[i, j] == 1:
                print(f"  A{i+1} -> A{j+1}")

    print("\n" + "="*80)
    print("Dataset loaded successfully!")
