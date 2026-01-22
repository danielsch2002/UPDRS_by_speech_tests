"""
LASSO Feature Selection Module
------------------------------
Identifies all BIC optima (local and global) for parsimonious modeling.
Reproduces the methodology of Tsanas et al. (2010).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LassoLarsIC, lars_path

from src.config import Paths
from src.logger import logger_inst


def get_features_at_step(X: pd.DataFrame, y: pd.Series, step: int) -> list[str]:
    """
    Extracts feature names for a specific step in the LARS path.
    Uses lars_path functional interface to avoid attribute errors.
    """
    if step == 0:
        return []
    # Compute the full regularization path to extract coefficients at 'step'
    _, _, coefs = lars_path(X.values, y.values, method='lasso')
    current_coefs = coefs[:, step]
    return [X.columns[i] for i in range(len(current_coefs)) if current_coefs[i] != 0]


def run_lasso_pipeline(input_path: Path, output_tables: Path, output_figures: Path) -> dict[int, list[str]]:
    """
    Executes selection and returns a dictionary of all optimal feature sets {step: [features]}.
    Features aligned printing blocks to prevent logger-induced misalignment.
    """
    logger_inst.info("=== Starting Stage 5: Multi-Optima Feature Selection (LassoLars) ===")

    if not input_path.exists():
        logger_inst.error("Input path missing: %s", input_path)
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)
    feature_names = [c for c in df.columns if c not in Paths.SKIP_COLS]
    X, y = df[feature_names], df["total_UPDRS"]

    # 1. Fit LassoLarsIC
    model_bic = LassoLarsIC(criterion='bic', normalize=False, max_iter=2000)
    model_bic.fit(X, y)
    bic_values = model_bic.criterion_

    # 2. Automated Optima Detection (Local and Global)
    optima_steps = []
    for i in range(len(bic_values)):
        is_min = True
        if i > 0 and bic_values[i] >= bic_values[i-1]: is_min = False
        if i < len(bic_values)-1 and bic_values[i] >= bic_values[i+1]: is_min = False
        if is_min: optima_steps.append(i)

    # 3. Aligned Summary Table (Built as a single block)
    w_step, w_bic, w_status = 8, 15, 20
    header = f"{'Step':<{w_step}} | {'BIC Score':<{w_bic}} | {'Status':<{w_status}}"
    rule = "-" * len(header)

    table_lines = [rule, header, rule]
    global_min_idx = np.argmin(bic_values)
    for step, val in enumerate(bic_values):
        status = "GLOBAL MINIMUM" if step == global_min_idx else ("LOCAL MINIMUM" if step in optima_steps else "")
        table_lines.append(f"{step:<{w_step}} | {val:<{w_bic}.2f} | {status:<{w_status}}")
    table_lines.append(rule)

    # Print the entire table block at once to maintain alignment
    logger_inst.info("\n" + "\n".join(table_lines))

    # 4. Extract and Log Features for each candidate (Built as single blocks)
    all_optima_sets = {}
    for step in optima_steps:
        feats = get_features_at_step(X, y, step)
        all_optima_sets[step] = feats

        label = "GLOBAL" if step == global_min_idx else "LOCAL"
        title = f"{label} OPTIMUM AT STEP {step} ({len(feats)} Features)"

        feat_block = ["="*55, f"{title:^55}", "-"*55]
        for i, f in enumerate(feats, 1):
            feat_block.append(f"{i:>2}. {f:<30}")
        feat_block.append("="*55)

        # Print the entire feature set block at once
        logger_inst.info("\n" + "\n".join(feat_block))

    # 5. Save Global Minimum as default CSV for Stage 6
    best_features = all_optima_sets[global_min_idx]
    required_cols = best_features + ["subject_id", "motor_UPDRS", "total_UPDRS"]
    output_csv = Paths.data_processed / "best_features_parkinsons_normalized.csv"
    df[required_cols].to_csv(output_csv, index=False)

    logger_inst.info(f"Lasso analysis complete. Found {len(optima_steps)} candidate optima.")

    # Return the dictionary to main.py for comparative modeling in Stage 6
    return all_optima_sets