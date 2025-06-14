"""
Module for visualizing distributions of real and synthetic data.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any

def visualize_distributions(real_data: pd.DataFrame, synthetic_data: pd.DataFrame,
                            column_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Plots the distributions of a specified column from both real and synthetic datasets
    on the same chart for easy visual comparison.

    Args:
        real_data: The original pandas DataFrame.
        synthetic_data: The generated synthetic pandas DataFrame.
        column_name: The name of the column to visualize.
        metadata (Optional): Metadata dictionary. If provided, helps determine column type.
    """
    if column_name not in real_data.columns or column_name not in synthetic_data.columns:
        print(f"Column '{column_name}' not found in one or both DataFrames.")
        return

    sdtype = None
    if metadata and column_name in metadata.get("columns", {}):
        sdtype = metadata["columns"][column_name].get("sdtype")

    real_col = real_data[column_name].dropna()
    synth_col = synthetic_data[column_name].dropna()

    plt.figure(figsize=(10, 6))

    if sdtype == 'numerical' or pd.api.types.is_numeric_dtype(real_col):
        sns.histplot(real_col, color="blue", label="Real Data", kde=True, stat="density", common_norm=False)
        sns.histplot(synth_col, color="orange", label="Synthetic Data", kde=True, stat="density", common_norm=False)
        plt.title(f"Distribution Comparison for Numerical Column: {column_name}")
    elif sdtype == 'categorical' or sdtype == 'boolean' or real_col.dtype == 'object':
        # For categorical, plot bar charts of value counts
        real_counts = real_col.value_counts(normalize=True).sort_index()
        synth_counts = synth_col.value_counts(normalize=True).sort_index()

        df_plot = pd.DataFrame({'Real': real_counts, 'Synthetic': synth_counts}).fillna(0)
        df_plot.plot(kind='bar', position=0.5, width=0.4)
        plt.title(f"Distribution Comparison for Categorical Column: {column_name}")
        plt.ylabel("Proportion")
    else: # Fallback for other types like datetime (could be more specific)
        sns.histplot(real_col, color="blue", label="Real Data", kde=False)
        sns.histplot(synth_col, color="orange", label="Synthetic Data", kde=False)
        plt.title(f"Distribution Comparison for Column: {column_name} (Type: {real_col.dtype})")

    plt.xlabel(column_name)
    plt.legend()
    plt.tight_layout()
    plt.show()