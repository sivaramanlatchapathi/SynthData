# SynthData

**SynthData** is a Python package designed for generating synthetic tabular data. It learns from real data and creates new data that mimics the statistical properties and correlations of the original dataset. This is particularly useful for augmenting datasets, preserving privacy, and testing data-driven applications.

## Features

*   **Multiple Synthesis Models**:
    *   `GaussianCopulaSynthesizer`: Models multivariate distributions using Gaussian Copulas.
    *   `CTGANSynthesizer`: Implements Conditional Tabular GAN for complex data types.
    *   `TVAESynthesizer`: Uses a Tabular Variational Autoencoder.
*   **Intuitive API**: A consistent, object-oriented API for all synthesizers.
*   **Metadata Handling**: Automatic metadata detection with options for manual overrides.
*   **Data Quality Evaluation**: Comprehensive reports and visualizations to compare synthetic data with real data.

## Installation

You can install SynthData using pip:

```bash
pip install synthdata
```

Alternatively, if you have the source code, you can install it directly:

```bash
pip install .
```

## Quick Start

Here's a basic example of how to use SynthData:

```python
import pandas as pd
from synthdata.synthesizers import GaussianCopulaSynthesizer
from synthdata.evaluation import quality_report, visualize_distributions
from synthdata.metadata import detect_metadata

# 1. Load your real data
data = {
    'age': [25, 30, 35, 40, 45, 28, 32, 38, 42, 48],
    'salary': [50000, 60000, 70000, 80000, 90000, 55000, 65000, 75000, 85000, 95000],
    'city': ['New York', 'London', 'Paris', 'Tokyo', 'New York', 'London', 'Paris', 'Tokyo', 'New York', 'London']
}
real_data_df = pd.DataFrame(data)

# 2. (Optional) Detect or define metadata
# If metadata is not provided, it will be detected automatically.
metadata = detect_metadata(real_data_df)
# You can also manually define it:
# metadata = {
#     "columns": {
#         "age": {"sdtype": "numerical", "computer_representation": "Integer"},
#         "salary": {"sdtype": "numerical", "computer_representation": "Integer"},
#         "city": {"sdtype": "categorical"}
#     }
# }

# 3. Initialize and fit the synthesizer
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_data_df)

# 4. Generate synthetic data
synthetic_data_df = synthesizer.sample(num_rows=100)

print("Synthetic Data Sample:")
print(synthetic_data_df.head())

# 5. Evaluate the quality of the synthetic data
report = quality_report(real_data_df, synthetic_data_df, metadata)
print("\nQuality Report:")
print(report)

# 6. Visualize distributions for a specific column
visualize_distributions(real_data_df, synthetic_data_df, column_name='salary')
```

## API Documentation

### Synthesizers

All synthesizers share a common API:

*   `synthdata.synthesizers.<Model>Synthesizer(metadata: dict)`: Initializes the synthesizer.
*   `fit(real_data: pd.DataFrame)`: Trains the model on the real data.
*   `sample(num_rows: int) -> pd.DataFrame`: Generates synthetic data.

### Evaluation

*   `synthdata.evaluation.quality_report(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: dict) -> dict`: Generates a quality report.
*   `synthdata.evaluation.visualize_distributions(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, column_name: str)`: Plots column distributions.

### Metadata

*   `synthdata.metadata.detect_metadata(data: pd.DataFrame) -> dict`: Automatically infers metadata from a DataFrame.

## Contributing

Contributions are welcome! Please refer to the project's issue tracker on GitHub.

## License

This project is licensed under the MIT License.
