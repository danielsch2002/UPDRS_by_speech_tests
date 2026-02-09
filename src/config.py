"""
Project Configuration
---------------------
This module centralizes project settings, separated into directory paths
and algorithmic constants.
"""

from __future__ import annotations
from pathlib import Path

# Resolve the absolute path to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Paths:
    """
    Static container for project directory paths only.
    Used for locating data and saving reports/outputs.
    """
    data = PROJECT_ROOT / "data"
    reports = PROJECT_ROOT / "reports"

    data_raw = data / "raw"
    data_processed = data / "processed"

    figures = reports / "figures"
    tables = reports / "tables"

    @classmethod
    def from_here(cls) -> type[Paths]:
        """Returns the class to access path members."""
        return cls


class Config:
    """
    Project Constants.
    Includes hyperparameters for regression models and CV settings.
    """

    # --- Dataset Column Logic ---
    FEATURE_COLS = [
        'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
        'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11',
        'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
    ]
    # Columns to be excluded from features during training
    SKIP_COLS = ['subject_id', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS']

    # --- Model Hyperparameters ---

    # LASSO Regression Settings
    LASSO_ALPHA_OPTIMAL = 0.0153
    LASSO_MAX_ITER = 10000

    # IRLS (Huber Regressor) Settings
    IRLS_MAX_ITER = 10000
    IRLS_TOL = 1e-1
    IRLS_ALPHA = 0.1

    # CART (Decision Tree) Settings
    CART_MAX_DEPTH = 6
    CART_MIN_SAMPLES_SPLIT = 40
    CART_MIN_SAMPLES_LEAF = 20

    # --- Global Execution Settings ---
    RANDOM_STATE = 42

    # Cross-Validation Settings
    # Standard 10-fold CV as mentioned in Tsanas et al. (2010)
    N_FOLDS = 10
    # Number of iterations for robust Subject-Wise and Random CV estimates
    N_CV_ITERATIONS = 100

    # --- Paper Reference Values (Tsanas et al. 2010) ---
    # Key benchmarks for result verification
    PAPER_BEST_FEATURES = ['Jitter(Abs)', 'Shimmer', 'NHR', 'HNR', 'DFA', 'PPE']
    PAPER_MAE_MOTOR = 5.95
    PAPER_MAE_TOTAL = 7.52