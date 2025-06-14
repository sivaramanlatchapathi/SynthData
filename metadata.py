"""
Module for metadata detection and handling.
"""
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np

SDTYPE_MAPPING = {
    'i': 'numerical',  # Integer
    'u': 'numerical',  # Unsigned integer
    'f': 'numerical',  # Float
    'b': 'boolean',    # Boolean
    'O': 'categorical',# Object (typically strings)
    'S': 'categorical',# Byte string
    'U': 'categorical',# Unicode string
    'M': 'datetime',   # Datetime
}

COMPUTER_REPRESENTATION_MAPPING = {
    'i': 'Integer',
    'u': 'Integer',
    'f': 'Float',
    'b': 'Boolean',
    'O': 'String',
    'S': 'String',
    'U': 'String',
    'M': 'Timestamp',
}

def detect_metadata(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Automatically detects metadata from a pandas DataFrame.

    Args:
        data: The input pandas DataFrame.

    Returns:
        A dictionary representing the metadata.
        Example:
        {
            "columns": {
                "column_name_1": {"sdtype": "numerical", "computer_representation": "Float"},
                "column_name_2": {"sdtype": "categorical"}
            },
            "primary_key": "optional_column_id" # Simple heuristic: first unique int column
        }
    """
    metadata: Dict[str, Any] = {"columns": {}}
    primary_key_candidate: Union[str, None] = None

    for column_name in data.columns:
        column_data = data[column_name]
        dtype_kind = column_data.dtype.kind

        sdtype = SDTYPE_MAPPING.get(dtype_kind, 'unknown')
        computer_representation = COMPUTER_REPRESENTATION_MAPPING.get(dtype_kind, 'Unknown')

        if sdtype == 'categorical' and column_data.nunique() / len(column_data) > 0.9:
            # If high cardinality categorical, consider it as an ID if numerical like
            pass # Could be an ID, but let's keep it simple for now

        metadata["columns"][str(column_name)] = {
            "sdtype": sdtype,
            "computer_representation": computer_representation
        }

        # Simple heuristic for primary key: first unique integer column
        if primary_key_candidate is None and sdtype == 'numerical' and \
           computer_representation == 'Integer' and column_data.is_unique and not column_data.isnull().any():
            primary_key_candidate = str(column_name)

    if primary_key_candidate:
        metadata["primary_key"] = primary_key_candidate

    return metadata