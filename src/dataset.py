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
import numpy as np
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
    Cleans column names, applies Log Transform to skewed features,
    and normalizes dysphonia features using Min-Max scaling.
    """
    # 1. Standardize identifier column names
    if "subject#" in df.columns:
        df = df.rename(columns={"subject#": "subject_id"})

    # 2. Define column groups
    metadata_and_targets = [
        "subject_id", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS"
    ]
    feature_cols = [c for c in df.columns if c not in metadata_and_targets]

    if not feature_cols:
        raise ValueError("Feature selection failed: No dysphonia measures found.")

    # 3. Apply Log Transform to skewed features (|skew| > 1.5)
    # This reduces the impact of outliers and helps MinMaxScaler spread the data better.
    # Using log1p (log(1+x)) to handle any potential zero values safely.
    skew_series = df[feature_cols].skew()
    high_skew_feats = skew_series[abs(skew_series) > 1.5].index

    if not high_skew_feats.empty:
        logger_inst.info("Applying Log Transform to %d skewed features: %s",
                         len(high_skew_feats), list(high_skew_feats))
        df[high_skew_feats] = np.log1p(df[high_skew_feats])

    # 4. Apply Min-Max Normalization [0, 1]
    # Now that skewed data is "compressed", the [0, 1] range will be more informative.
    logger_inst.info("Normalizing %d features using Min-Max scaling...", len(feature_cols))
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