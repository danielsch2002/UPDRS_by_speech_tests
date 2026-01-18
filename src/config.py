"""
Project Path Configuration
--------------------------
This module centralizes all directory and file path logic. By resolving the 
PROJECT_ROOT dynamically, the pipeline remains functional regardless of the 
environment it is executed in.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

# Resolve the absolute path to the project root (one level up from /src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Paths:
    """
    Static container for project directory paths.

    Provides a clean interface for accessing data, reports, and figure 
    locations without hard coding strings in the logic modules.
    """
    # Main Directories
    data = PROJECT_ROOT / "data"
    reports = PROJECT_ROOT / "reports"

    # Data Sub-directories
    data_raw = data / "raw"
    data_processed = data / "processed"

    # Output Sub-directories
    figures = reports / "figures"
    tables = reports / "tables"

    # these are the 16 voice measures from the dataset
    FEATURE_COLS = [
        'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
        'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11',
        'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
    ]

    # columns we dont want as features
    SKIP_COLS = ['subject_id', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS']

    # LASSO settings
    # alpha range: from 0.0001 to 10, logarithmic scale - we try 500 different values
    ALPHA_MIN = -4   # 10^-4 = 0.0001
    ALPHA_MAX = 1    # 10^1 = 10
    N_ALPHAS = 500
    LASSO_ALPHA_OPTIMAL = 0.0153 # Optimal alpha found during feature selection (yields ~6 features)

    @classmethod
    def get_alpha_range(cls):
        """
        Returns the log-spaced alpha range for LASSO.
        Using cls allows the method to access class attributes safely.
        """
        return np.logspace(cls.ALPHA_MIN, cls.ALPHA_MAX, cls.N_ALPHAS)

    # === CART settings ===
    # these control how big the tree can grow (pruning basically)
    CART_MAX_DEPTH = 10
    CART_MIN_SPLIT = 10
    CART_MIN_LEAF = 5
    RANDOM_STATE = 42

    # === Cross validation ===
    N_FOLDS = 10  # 10-fold like in the paper
    N_CV_ITERATIONS = 1000   # Number of CV repetitions for robust estimates

    # === Expected results ===
    # the paper found these 6 features to be optimal
    PAPER_BEST_FEATURES = [
        'Jitter(Abs)',
        'Shimmer',
        'NHR',
        'HNR',
        'DFA',
        'PPE'
    ]

    # paper's MAE results (for reference)
    PAPER_MAE_MOTOR = 5.95
    PAPER_MAE_TOTAL = 7.52

    @classmethod
    def from_here(cls) -> type[Paths]:
        """
        Entry point to access the path configuration.

        Returns:
            The Paths class with resolved directory members.
        """
        return cls