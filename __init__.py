"""
SynthData: A Python package for generating synthetic tabular data.
"""

__version__ = "0.1.0"

from .metadata import detect_metadata
from .synthesizers import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer
)
from .evaluation import quality_report, visualize_distributions

__all__ = [
    "detect_metadata",
    "GaussianCopulaSynthesizer",
    "CTGANSynthesizer",
    "TVAESynthesizer",
    "quality_report",
    "visualize_distributions",
]