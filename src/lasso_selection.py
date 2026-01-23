"""
LASSO Feature Selection Module
------------------------------
Reproduces the feature selection protocol by Tsanas et al. (2010).
Uses the Least Angle Regression (LARS) algorithm and BIC Calculation
to identify a parsimonious feature subset for UPDRS prediction.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import lars_path
from src.config import Config
from src.logger import logger_inst


def run_lasso_pipeline(input_path: Path, output_tables: Path, output_figures: Path) -> dict[str, list[str]]:
    """
    Executes the LASSO selection pipeline and logs each step of the BIC path.
    """
    logger_inst.info("=== Stage 5: LASSO Selection ===")

    # Load preprocessed data
    df = pd.read_csv(input_path)
    feature_names = [c for c in df.columns if c not in Config.SKIP_COLS]
    X, y = df[feature_names].values, df["total_UPDRS"].values

    # Step 1: Generate the LARS path
    alphas, active, coefs = lars_path(X, y, method='lasso')

    # Step 2: Path Evaluation via BIC
    n_samples = X.shape[0]
    p_total_features = len(feature_names)

    # Panelty parameter
    gamma = 1.0

    bic_values = []
    path_details = []

    # Log header clearly
    logger_inst.info(f"{'Step':<5} | {'Alpha':<10} | {'k':<3} | {'BIC Score':<12}")
    logger_inst.info("-" * 45)

    for i in range(coefs.shape[1]):
        y_pred = X @ coefs[:, i]
        mse = np.mean((y - y_pred) ** 2)

        current_indices = np.where(coefs[:, i] != 0)[0]
        current_features = [feature_names[idx] for idx in current_indices]
        k = len(current_features)
        current_alpha = alphas[i] if i < len(alphas) else 0.0

        if mse > 0:
            # BIC Formula to balance fit and complexity
            standard_bic = n_samples * np.log(mse) + k * np.log(n_samples)
            ext_penalty = 2 * gamma * k * np.log(p_total_features)
            bic_val = standard_bic + ext_penalty
        else:
            bic_val = np.inf

        bic_values.append(bic_val)

        # Log each step individually to ensure visibility in terminal
        logger_inst.info(f"{i:<5} | {current_alpha:<10.6f} | {k:<3} | {bic_val:<12.2f}")

        path_details.append({
            "Step": i,
            "Alpha": current_alpha,
            "k": k,
            "BIC": bic_val,
            "Features": current_features
        })

    # Step 3: Optimization Result Extraction
    best_idx = np.argmin(bic_values)
    optimal_step_data = path_details[best_idx]
    best_features = optimal_step_data["Features"]
    best_alpha = optimal_step_data["Alpha"]

    logger_inst.info("-" * 45)
    logger_inst.info(f"Optimal Alpha (min BIC): {best_alpha:.6f}")
    logger_inst.info(f"Selected {len(best_features)} features at Step {best_idx}.")
    logger_inst.info(f"Final Subset: {best_features}")

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(bic_values)), bic_values, marker='o', color='purple', label=f'BIC (gamma={gamma})')
    plt.axvline(best_idx, color='black', linestyle='--', label=f'Optimal k={len(best_features)}')
    plt.xlabel('Model Complexity (LARS Steps)')
    plt.ylabel('BIC Value')
    plt.title(f'Feature Selection: BIC Optimization (gamma={gamma})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_figures / "lasso_ic_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    return {"BIC_Optimal": best_features}