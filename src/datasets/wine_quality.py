#!/usr/bin/env python3
"""
Wine Quality Dataset (UCI ML Repository)
=========================================

Source: https://archive.ics.uci.edu/ml/datasets/wine+quality

Dataset Information:
- 1599 red wine samples
- 11 physicochemical properties + quality score
- No missing values
- All continuous variables

Variables:
1. fixed acidity (tartaric acid - g/dm^3)
2. volatile acidity (acetic acid - g/dm^3)
3. citric acid (g/dm^3)
4. residual sugar (g/dm^3)
5. chlorides (sodium chloride - g/dm^3)
6. free sulfur dioxide (mg/dm^3)
7. total sulfur dioxide (mg/dm^3)
8. density (g/cm^3)
9. pH
10. sulphates (potassium sulphate - g/dm^3)
11. alcohol (% by volume)
12. quality (score between 0 and 10)

Causal Structure (Chemistry Domain Knowledge):
Based on wine chemistry and sensory science:
- Fixed acidity -> pH (strong chemical relationship)
- Citric acid -> pH (acidification)
- Volatile acidity -> Quality (spoilage indicator)
- Alcohol -> Quality (taste preference)
- pH -> Quality (balance)
- Sulphates -> Free SO2 (preservation)
- Free SO2 -> Total SO2 (cumulative)
- Residual sugar -> Density (mass relationship)
- Alcohol -> Density (dilution effect)
"""

import numpy as np
import pandas as pd
from typing import Tuple


def load_wine_quality() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load and prepare Wine Quality dataset for causal discovery.

    Returns:
        data: pandas DataFrame with wine physicochemical properties
        true_graph: Ground truth DAG adjacency matrix (12x12)
    """
    print("Loading Wine Quality dataset...")

    # Load dataset from UCI repository
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    try:
        df = pd.read_csv(url, sep=';')
    except Exception as e:
        print(f"Failed to download from UCI. Trying alternative...")
        # Alternative: use local copy or sklearn
        try:
            from sklearn.datasets import fetch_openml
            wine = fetch_openml('wine-quality-red', version=1, as_frame=True, parser='auto')
            df = wine.data.copy()
            df['quality'] = wine.target
        except Exception as e2:
            raise RuntimeError(f"Failed to load Wine Quality dataset: {e}, {e2}")

    # Data is already clean (no missing values)
    # All variables are continuous

    # Convert to float64 for numerical stability
    df = df.astype(np.float64)

    print(f"Wine Quality dataset loaded: {df.shape[0]} samples, {df.shape[1]} variables")

    # Define ground truth DAG based on chemistry domain knowledge
    # Variables (0-indexed):
    # 0: fixed acidity
    # 1: volatile acidity
    # 2: citric acid
    # 3: residual sugar
    # 4: chlorides
    # 5: free sulfur dioxide
    # 6: total sulfur dioxide
    # 7: density
    # 8: pH
    # 9: sulphates
    # 10: alcohol
    # 11: quality

    n_vars = 12
    true_graph = np.zeros((n_vars, n_vars))

    # Chemistry-based causal relationships

    # Acidity affects pH
    true_graph[0, 8] = 1  # fixed acidity -> pH
    true_graph[2, 8] = 1  # citric acid -> pH

    # pH affects quality (balance)
    true_graph[8, 11] = 1  # pH -> quality

    # Volatile acidity affects quality (spoilage)
    true_graph[1, 11] = 1  # volatile acidity -> quality

    # Alcohol affects quality (preference) and density
    true_graph[10, 11] = 1  # alcohol -> quality
    true_graph[10, 7] = 1   # alcohol -> density (dilution)

    # Sugar affects density (mass)
    true_graph[3, 7] = 1   # residual sugar -> density

    # Sulphates affect SO2 levels (preservation)
    true_graph[9, 5] = 1   # sulphates -> free SO2

    # Free SO2 is part of total SO2
    true_graph[5, 6] = 1   # free SO2 -> total SO2

    # Chlorides affect density (dissolved salts)
    true_graph[4, 7] = 1   # chlorides -> density

    # Additional quality factors
    true_graph[9, 11] = 1  # sulphates -> quality (preservation)

    print(f"Ground truth DAG: {int(true_graph.sum())} edges")

    return df, true_graph


def get_dataset_info() -> dict:
    """Return dataset metadata."""
    return {
        'name': 'wine',
        'full_name': 'Wine Quality (Red)',
        'domain': 'Chemistry (wine physicochemical properties)',
        'n_nodes': 12,
        'n_samples': 1599,
        'n_edges': 11,
        'data_type': 'Continuous',
        'complexity': 'Medium (12 variables)',
        'description': 'Red wine quality prediction from physicochemical tests'
    }


if __name__ == "__main__":
    # Test loading
    print("="*80)
    print("WINE QUALITY DATASET TEST")
    print("="*80)

    data, true_graph = load_wine_quality()

    print(f"\nData shape: {data.shape}")
    print(f"\nColumn names:")
    for i, col in enumerate(data.columns):
        print(f"  {i}: {col}")

    print(f"\nFirst 5 rows:")
    print(data.head())

    print(f"\nGround truth DAG:")
    print(f"Nodes: {true_graph.shape[0]}")
    print(f"Edges: {int(true_graph.sum())}")

    print(f"\nEdge list:")
    var_names = data.columns.tolist()
    for i in range(true_graph.shape[0]):
        for j in range(true_graph.shape[1]):
            if true_graph[i, j] == 1:
                source = var_names[i] if i < len(var_names) else f"var{i}"
                target = var_names[j] if j < len(var_names) else f"var{j}"
                print(f"  {source} -> {target}")

    print(f"\nData statistics:")
    print(data.describe())

    print("\n" + "="*80)
    print("Dataset loaded successfully!")
