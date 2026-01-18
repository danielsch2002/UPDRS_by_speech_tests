"""
LASSO Feature Selection Module
------------------------------
Selects the most parsimonious feature set using LASSO and Information Criteria (AIC/BIC).
Reproduces the feature reduction stage of Tsanas et al. (2010).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

from src.config import Paths
from src.logger import logger_inst


def calculate_information_criteria(model: Lasso, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    """
    Calculates AIC and BIC for a given LASSO model to find the 'Parsimonious' model.
    """
    n_samples = X.shape[0]
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)

    # Degrees of freedom: number of non-zero coefficients + intercept
    df = np.sum(model.coef_ != 0) + 1

    # Log-likelihood for OLS/Lasso
    log_likelihood = -n_samples/2 * (np.log(2 * np.pi * mse) + 1)

    aic = 2 * df - 2 * log_likelihood
    bic = np.log(n_samples) * df - 2 * log_likelihood

    return aic, bic


def run_lasso_pipeline(input_path: Path, output_tables: Path) -> list[str]:
    """
    Executes LASSO path analysis, finds the optimal feature set via BIC,
    and saves the reduced dataset for Stage 6.
    """
    logger_inst.info("Starting LASSO feature selection (AIC/BIC analysis)...")

    df = pd.read_csv(input_path)
    X = df[Paths.FEATURE_COLS]
    y = df["total_UPDRS"] # Target used in the original paper for selection

    alphas = Paths.get_alpha_range()
    results = []

    for alpha in alphas:
        model = Lasso(alpha=alpha, random_state=Paths.RANDOM_STATE, max_iter=10000)
        model.fit(X, y)

        aic, bic = calculate_information_criteria(model, X, y)
        n_features = np.sum(model.coef_ != 0)

        results.append({
            "alpha": alpha,
            "aic": aic,
            "bic": bic,
            "n_features": n_features
        })

    results_df = pd.DataFrame(results)

    # Optimization: Find best BIC among models with 6 or fewer features
    valid_models = results_df[results_df['n_features'] <= 6]

    if valid_models.empty:
        logger_inst.warning("No models found with <= 6 features. Taking the smallest available.")
        optimal_row = results_df.sort_values('n_features').iloc[0]
    else:
        optimal_row = valid_models.sort_values('bic').iloc[0]

    best_alpha = optimal_row['alpha']

    # Extracting the names of the selected features
    final_lasso = Lasso(alpha=best_alpha, random_state=Paths.RANDOM_STATE).fit(X, y)
    best_features = X.columns[final_lasso.coef_ != 0].tolist()

    # Log final selection for easy comparison with the paper
    logger_inst.info(f"=== LASSO Selection Results ===")
    logger_inst.info(f"Optimal Alpha: {best_alpha:.4f}")
    logger_inst.info(f"Number of Features: {len(best_features)}")
    logger_inst.info(f"Selected Features: {best_features}")

    # Save the reduced dataset for Stage 6
    required_cols = best_features + Paths.SKIP_COLS
    df_reduced = df[required_cols]
    reduced_path = Paths.data_processed / "best_features_parkinsons_normalized.csv"
    df_reduced.to_csv(reduced_path, index=False)

    # Save selection metadata (for report tables)
    results_df.to_csv(output_tables / "lasso_aic_bic_path.csv", index=False)

    # Save selected feature names to a JSON for quick reference in slides
    with open(output_tables / "selected_features.json", "w") as f:
        json.dump({"best_features": best_features, "alpha": best_alpha}, f)

    logger_inst.info(f"Reduced dataset and feature list saved.")
    return best_features