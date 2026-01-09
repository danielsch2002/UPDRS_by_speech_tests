"""Load, clean, and normalize the Parkinson's Telemonitoring dataset.

This module processes the raw data by renaming columns for consistency and 
scaling the 16 dysphonia features to the [0, 1] range, as described in the 
Tsanas et al. (2010) methodology.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import Paths
from src.logger import logger_inst

def load_raw_data(input_path: Path) -> pd.DataFrame:
    """Load the raw dataset from CSV.

    Args:
        input_path: Path to the raw .data file.

    Returns:
        pd.DataFrame: The loaded raw data.
    """
    logger_inst.info("Loading raw data from %s...", input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Raw data not found at {input_path}. Run fetch.py first.")

    return pd.read_csv(input_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and normalize features to [0, 1].

    Args:
        df: The raw DataFrame.

    Returns:
        pd.DataFrame: The processed DataFrame with normalized features.
    """

    # 1. Rename columns for consistency
    # Rename 'subject#' to 'subject_id' to follow standard naming conventions
    df = df.rename(columns={"subject#": "subject_id"})

    # 2. Identify columns to normalize
    # Exclude metadata and target variables (UPDRS scores)
    non_feature_cols = [
        "subject_id",
        "age",
        "sex",
        "test_time",
        "motor_UPDRS",
        "total_UPDRS"
    ]

    # Select only the 16 dysphonia measures
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    logger_inst.info("Normalizing %d dysphonia features to range [0, 1]...", len(feature_cols))

    # 3. Apply Min-Max Scaling (0-1)
    # The paper explicitly states: "Following normalization to the range 0-1"
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Return dataframe
    return df


def load_data() -> None:
    """CLI entrypoint for dataset processing."""

    paths = Paths.from_here()
    input_path = paths.data_raw / "parkinsons_updrs.data"
    output_path = paths.data_processed / "parkinsons_normalized.csv"

    try:
        df_raw = load_raw_data(input_path)
        df_clean = preprocess_data(df_raw)

        df_clean.to_csv(output_path, index=False)
        logger_inst.info("Saved processed data to %s", output_path)
        logger_inst.info("Processed dataframe shape: %s", df_clean.shape)

    except Exception as exc:
        logger_inst.error("Failed to process dataset: %s", exc)
        raise RuntimeError(f"Processing failed: {exc}") from exc