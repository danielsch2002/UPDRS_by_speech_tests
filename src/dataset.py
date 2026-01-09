"""
Data Processing and Normalization Module
---------------------------------------
This module handles the transformation of raw Parkinson's telemonitoring data
into a format suitable for analysis. Following the methodology of
Tsanas et al. (2010), it performs column renaming and feature scaling.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import Paths
from src.logger import logger_inst


def load_raw_data(input_path: Path) -> pd.DataFrame:
    """
    Reads the raw dataset from the specified disk location.

    Args:
        input_path (Path): Path to the raw CSV/data file.

    Returns:
        pd.DataFrame: The loaded raw dataset.

    Raises:
        FileNotFoundError: If the raw data file is missing.
    """
    if not input_path.exists():
        logger_inst.error("Raw data file not found: %s", input_path)
        raise FileNotFoundError(f"Missing raw data at {input_path}. Ensure fetch stage is complete.")

    logger_inst.info("Loading raw dataset for processing...")
    return pd.read_csv(input_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans column names and normalizes dysphonia features.

    Normalization is performed using Min-Max scaling to the [0, 1] range
    to ensure feature parity during statistical analysis.

    Args:
        df (pd.DataFrame): The raw input dataframe.

    Returns:
        pd.DataFrame: The processed dataframe with normalized features.
    """
    # 1. Standardize identifier column names
    if "subject#" in df.columns:
        df = df.rename(columns={"subject#": "subject_id"})
        logger_inst.debug("Renamed 'subject#' to 'subject_id'.")

    # 2. Define column groups
    # Metadata and targets are kept in their original scale
    metadata_and_targets = [
        "subject_id", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS"
    ]

    # Identify numerical dysphonia features for scaling
    feature_cols = [c for c in df.columns if c not in metadata_and_targets]

    if not feature_cols:
        logger_inst.error("No features identified for normalization.")
        raise ValueError("Feature selection failed: No dysphonia measures found.")

    logger_inst.info("Normalizing %d dysphonia features using Min-Max scaling [0, 1]...", len(feature_cols))

    # 3. Apply Min-Max Normalization
    # Rescales features such that the minimum value is 0 and maximum is 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df


def load_data() -> None:
    """
    Main entrypoint for the data processing stage.
    Coordinates loading, cleaning, and saving the processed dataset.
    """
    paths = Paths.from_here()
    input_path = paths.data_raw / "parkinsons_updrs.data"
    output_path = paths.data_processed / "parkinsons_normalized.csv"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Pipeline execution
        raw_df = load_raw_data(input_path)
        processed_df = preprocess_data(raw_df)

        # Export results
        processed_df.to_csv(output_path, index=False)

        logger_inst.info("Preprocessing complete. Processed data saved to: %s", output_path)
        logger_inst.info("Final dataset dimensions: %s", processed_df.shape)

    except Exception as exc:
        logger_inst.critical("Data processing pipeline failed: %s", exc)
        raise RuntimeError(f"Processing stage failed: {exc}") from exc


if __name__ == "__main__":
    load_data()