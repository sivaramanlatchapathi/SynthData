"""
Base class for all synthesizers.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
from synthdata.metadata import detect_metadata

class BaseSynthesizer(ABC):
    """
    Abstract Base Class for synthesizers.

    Each synthesizer should inherit from this class and implement
    the `fit` and `sample` methods.

    Args:
        metadata (Dict[str, Any]): A dictionary describing the data.
            If None, it will be detected from the data in `fit`.
        verbose (bool): Whether to print progress messages.
    """

    def __init__(self, metadata: Dict[str, Any] = None, verbose: bool = False):
        self.metadata = metadata
        self.verbose = verbose
        self._fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the synthesizer to the real data.

        Args:
            data: The real data as a pandas DataFrame.
        """
        if self.metadata is None:
            if self.verbose:
                print("Metadata not provided. Detecting metadata from data...")
            self.metadata = detect_metadata(data)
        self._fitted = True

    @abstractmethod
    def sample(self, num_rows: int) -> pd.DataFrame:
        """
        Generate synthetic data.

        Args:
            num_rows: The number of rows to generate.
        """
        pass