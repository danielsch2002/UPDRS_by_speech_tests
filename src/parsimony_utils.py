"""
Parsimony Utilities
-------------------
Helper functions to prepare cross-stage comparisons for Parkinson's UPDRS analysis.
"""

import pandas as pd
from pathlib import Path
from src.config import Paths


def prepare_comparison_sets(candidate_sets: dict, input_path: Path) -> dict:
    """
    Appends the full 16-feature set to the candidates for final evaluation.
    Ensures that Stage 6 can compare parsimonious sets against the baseline.
    """
    df = pd.read_csv(input_path)

    # Extract all 16 original acoustic features
    full_features = [c for c in df.columns if c not in Paths.SKIP_COLS]

    # Create a copy to avoid modifying the original dictionary in-place
    comparison_sets = candidate_sets.copy()
    comparison_sets["Full (16)"] = full_features

    return comparison_sets