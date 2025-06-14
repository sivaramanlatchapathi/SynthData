"""
Gaussian Copula Synthesizer.
"""
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from scipy import stats
from scipy.linalg import cholesky

from .base import BaseSynthesizer
from synthdata.metadata import detect_metadata

class GaussianCopulaSynthesizer(BaseSynthesizer):
    """
    Synthesizer based on modeling the multivariate distribution using Gaussian Copulas.

    It handles numerical and categorical data by fitting distributions to numerical
    columns and using frequency-based sampling for categorical ones.

    Args:
        metadata (Dict[str, Any], optional): Metadata describing the data.
            If None, it will be detected from data in `fit`.
        default_numerical_distribution (str): Default distribution for numerical columns,
            e.g., 'norm', 'beta', 'truncnorm', 'uniform'.
            For simplicity, we'll primarily use 'norm' here.
        verbose (bool): Whether to print progress messages.
    """

    def __init__(self, metadata: Dict[str, Any] = None,
                 default_numerical_distribution: str = 'norm', verbose: bool = False):
        super().__init__(metadata, verbose)
        self.default_numerical_distribution = default_numerical_distribution
        self._distributions: Dict[str, Any] = {}
        self._correlation_matrix: Union[np.ndarray, None] = None
        self._column_order: List[str] = []
        self._categorical_frequencies: Dict[str, pd.Series] = {}

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the synthesizer to the real data.

        Args:
            data: The real data as a pandas DataFrame.
        """
        super().fit(data) # Handles metadata detection if needed
        if self.metadata is None: # Should be set by super().fit()
            raise ValueError("Metadata could not be determined.")

        self._column_order = list(data.columns)
        numerical_cols = [col for col, meta in self.metadata['columns'].items()
                          if meta['sdtype'] == 'numerical']
        categorical_cols = [col for col, meta in self.metadata['columns'].items()
                            if meta['sdtype'] == 'categorical']

        # 1. Fit distributions to numerical columns and transform to standard normal
        transformed_numerical_data = pd.DataFrame()
        for col in numerical_cols:
            column_data = data[col].dropna()
            # Simplified: Fit a normal distribution and get CDF values (percentiles)
            # A more robust solution would use libraries like `copulas` or `scipy.stats.fit`
            loc, scale = stats.norm.fit(column_data)
            self._distributions[col] = {'name': 'norm', 'loc': loc, 'scale': scale}
            # Transform to uniform using CDF, then to standard normal using PPF
            uniform_values = stats.norm.cdf(column_data, loc=loc, scale=scale)
            # Clip to avoid inf values at 0 and 1 for PPF
            uniform_values = np.clip(uniform_values, 1e-7, 1 - 1e-7)
            transformed_numerical_data[col] = stats.norm.ppf(uniform_values)

        # 2. Calculate correlation matrix of the transformed numerical data
        if not transformed_numerical_data.empty:
            self._correlation_matrix = transformed_numerical_data.corr().to_numpy()
        else:
            self._correlation_matrix = np.array([[]]) # Handle case with no numerical columns

        # 3. Store frequencies for categorical columns
        for col in categorical_cols:
            self._categorical_frequencies[col] = data[col].value_counts(normalize=True)

        if self.verbose:
            print("GaussianCopulaSynthesizer fitted.")

    def sample(self, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data."""
        if not self._fitted or self.metadata is None or self._correlation_matrix is None:
            raise RuntimeError("Synthesizer is not fitted yet. Call fit() first.")

        synthetic_df = pd.DataFrame()
        num_numerical_cols = len([col for col, meta in self.metadata['columns'].items()
                                  if meta['sdtype'] == 'numerical'])

        if num_numerical_cols > 0 and self._correlation_matrix.size > 0:
            # 1. Sample from multivariate normal using the Cholesky decomposition of the correlation matrix
            mean = np.zeros(num_numerical_cols)
            # Add small epsilon for numerical stability if matrix is not perfectly PSD
            chol_decomp = cholesky(self._correlation_matrix + 1e-6 * np.eye(num_numerical_cols), lower=True)
            normal_samples = np.random.normal(0, 1, size=(num_rows, num_numerical_cols)) @ chol_decomp.T

            # 2. Transform back to original distributions
            idx = 0
            for col, meta_info in self.metadata['columns'].items():
                if meta_info['sdtype'] == 'numerical':
                    # Transform standard normal samples to uniform, then to original scale using PPF
                    uniform_samples = stats.norm.cdf(normal_samples[:, idx])
                    dist_params = self._distributions[col]
                    # Simplified: Assuming normal for inverse transform
                    synthetic_df[col] = stats.norm.ppf(uniform_samples, loc=dist_params['loc'], scale=dist_params['scale'])
                    idx += 1

        # 3. Sample categorical columns based on learned frequencies
        for col, meta_info in self.metadata['columns'].items():
            if meta_info['sdtype'] == 'categorical':
                frequencies = self._categorical_frequencies[col]
                synthetic_df[col] = np.random.choice(frequencies.index, size=num_rows, p=frequencies.values)

        return synthetic_df[self._column_order] # Ensure original column order