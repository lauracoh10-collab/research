"""
Data loading utilities for SHAP research project

Author: Laura Cohen
Date: March 2026
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Base directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'


def load_mimic_admissions(filename='admissions.csv'):
    """
    Load MIMIC-IV admissions dataset
    
    Parameters:
    -----------
    filename : str
        Name of the admissions file
    
    Returns:
    --------
    pd.DataFrame
        Admissions data
    """
    path = DATA_DIR / 'raw' / 'mimic' / filename
    
    if not path.exists():
        raise FileNotFoundError(
            f"MIMIC data not found at {path}. "
            "Please download from PhysioNet and place in data/raw/mimic/"
        )
    
    return pd.read_csv(path)


def load_censo_data(filename='censo_escolar_2024.csv'):
    """
    Load Brazilian education census data
    
    Parameters:
    -----------
    filename : str
        Name of the census file
    
    Returns:
    --------
    pd.DataFrame
        Census data
    """
    path = DATA_DIR / 'raw' / 'censo' / filename
    
    if not path.exists():
        raise FileNotFoundError(
            f"Census data not found at {path}. "
            "Please download from INEP and place in data/raw/censo/"
        )
    
    return pd.read_csv(path)


def load_synthetic(dataset_name='baseline'):
    """
    Load synthetic dataset
    
    Parameters:
    -----------
    dataset_name : str
        Name of synthetic dataset variant
        Options: 'baseline', 'high_correlation', 'low_noise', 'high_noise'
    
    Returns:
    --------
    pd.DataFrame
        Synthetic data
    """
    path = DATA_DIR / 'synthetic' / f'synthetic_{dataset_name}.csv'
    
    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic data not found at {path}. "
            "Please run 02_synthetic_data_generation.ipynb first."
        )
    
    return pd.read_csv(path)


def load_processed(filename):
    """
    Load processed dataset
    
    Parameters:
    -----------
    filename : str
        Name of processed file
    
    Returns:
    --------
    pd.DataFrame
        Processed data
    """
    path = DATA_DIR / 'processed' / filename
    
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found at {path}")
    
    return pd.read_csv(path)


def save_processed(df, filename):
    """
    Save processed dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to save
    filename : str
        Output filename
    """
    path = DATA_DIR / 'processed' / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✓ Saved processed data to {path}")


def save_synthetic(df, dataset_name='baseline'):
    """
    Save synthetic dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Synthetic data
    dataset_name : str
        Name identifier for the dataset
    """
    path = DATA_DIR / 'synthetic' / f'synthetic_{dataset_name}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✓ Saved synthetic data to {path}")


def generate_synthetic_data(n_samples=1000, n_features=5, noise_level=0.2, seed=42):
    """
    Generate synthetic dataset with controlled feature correlations
    
    Parameters:
    -----------
    n_samples : int
        Number of observations
    n_features : int
        Number of features
    noise_level : float
        Noise level for feature generation
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (DataFrame with features, list of true coefficients)
    """
    np.random.seed(seed)
    
    data = {}
    coefficients = []
    
    # First feature: independent random normal
    data['f1'] = np.random.randn(n_samples)
    
    # Subsequent features: correlated with previous feature
    for i in range(2, n_features + 1):
        alpha_i = np.random.randn()
        data[f'f{i}'] = alpha_i * data[f'f{i-1}'] + noise_level * np.random.randn(n_samples)
        coefficients.append(alpha_i)
    
    df = pd.DataFrame(data)
    return df, coefficients


def create_target_variable(df, coefficients, noise_level=0.1, seed=42):
    """
    Create target variable from features with known coefficients
    
    Parameters:
    -----------
    df : pd.DataFrame
        Feature matrix
    coefficients : list
        True feature coefficients
    noise_level : float
        Noise level for target
    seed : int
        Random seed
    
    Returns:
    --------
    np.array
        Target variable
    """
    np.random.seed(seed)
    
    # Linear combination of features
    y = np.zeros(len(df))
    for i, (col, coef) in enumerate(zip(df.columns, coefficients)):
        y += coef * df[col].values
    
    # Add noise
    y += noise_level * np.random.randn(len(df))
    
    return y


def get_data_summary(df):
    """
    Print summary statistics for a dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to summarize
    """
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values ✓")
    else:
        print(missing[missing > 0])
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Generate and save synthetic data
    df_synthetic, true_coefs = generate_synthetic_data(
        n_samples=1000,
        n_features=5,
        noise_level=0.2
    )
    
    print("Generated synthetic data:")
    get_data_summary(df_synthetic)
    
    print(f"\nTrue coefficients: {true_coefs}")
