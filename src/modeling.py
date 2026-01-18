"""
Predictive Modeling Module
--------------------------
Implements regression models as per Tsanas et al. (2010).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_validate, GroupShuffleSplit, RepeatedKFold
from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import Paths

from src.logger import logger_inst


def get_model_definitions(model_names: list[str]) -> list[tuple[object, str]]:
    """
    Returns only the models requested by the pipeline.

    Args:
        model_names: List of strings like ["LS", "IRLS", "LASSO", "CART"]
    """
    # Mapping string names to actual model objects and display names
    all_models = {
        "LS": (LinearRegression(), "Least Squares (LS)"),
        "IRLS": (HuberRegressor(max_iter=10000, tol=1e-1, alpha=0.1, warm_start=True), "IRLS (Robust)"),
        "LASSO": (Lasso(alpha=Paths.LASSO_ALPHA_OPTIMAL, max_iter=10000), "LASSO"),
        "CART": (DecisionTreeRegressor(random_state=42, ccp_alpha=0.01), "CART (Non-linear)")
    }

    # Filter and return only requested models in the order they appear in model_names
    return [all_models[name] for name in model_names if name in all_models]


def visualize_cart_tree(model: DecisionTreeRegressor, features: list[str], output_path: Path) -> None:
    """Generates a high-resolution visualization of the Decision Tree."""
    logger_inst.info(f"Generating CART visualization at {output_path}...")
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=features,
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=4
    )
    plt.title("CART Decision Tree Structure")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _generate_final_visualization(df: pd.DataFrame, features: list[str], output_dir: Path) -> None:
    """Helper to train and visualize the final CART model on the full dataset."""
    final_cart = DecisionTreeRegressor(random_state=42, ccp_alpha=0.01)
    final_cart.fit(df[features], df["total_UPDRS"])

    visualize_cart_tree(
        final_cart,
        features,
        output_dir / "cart_final_structure.png"
    )


def run_cv_logic(model_obj, X, y, cv_generator, groups=None) -> dict:
    """Executes cross-validation for the given split method."""
    cv_results = cross_validate(
        model_obj, X, y,
        groups=groups,
        cv=cv_generator,
        scoring='neg_mean_absolute_error',
        return_train_score=True,
        n_jobs=2
    )
    return {
        "Train MAE": round(-np.mean(cv_results['train_score']), 2),
        "Test MAE": round(-np.mean(cv_results['test_score']), 2),
        "Test SD": round(np.std(-cv_results['test_score']), 3)
    }


def run_modeling_pipeline(models: list[str], input_path: Path, output_tables: Path, output_figures: Path) -> None:
    """
    Coordinates the 1,000-iteration Subject-Wise and Random CV pipeline.
    """
    logger_inst.info(f"Starting Modeling Pipeline for: {models}")

    df = pd.read_csv(input_path)

    # Use features not in SKIP_COLS logic
    features = [c for c in df.columns if c not in Paths.SKIP_COLS]
    targets = ["motor_UPDRS", "total_UPDRS"]
    groups = df["subject_id"]

    cv_methods = [
        (RepeatedKFold(n_splits=10, n_repeats=1000, random_state=42), "Random", None),
        (GroupShuffleSplit(n_splits=1000, test_size=0.1, random_state=42), "Subject-Wise", groups)
    ]

    all_results = []

    for target_col in targets:
        logger_inst.info(f"Processing target: {target_col}")
        X, y = df[features], df[target_col]

        for cv_gen, method_name, grp in cv_methods:
            logger_inst.info(f"Evaluating {method_name} splitting for {target_col}...")

            # Pass the list of models from the pipeline call to get definitions
            for model_obj, model_name in get_model_definitions(models):
                logger_inst.info(f"Running iterations for {model_name}...")
                metrics = run_cv_logic(model_obj, X, y, cv_gen, groups=grp)
                all_results.append({
                    "Method": method_name,
                    "Target": target_col,
                    "Model": model_name,
                    **metrics
                })

    # Saving results
    results_df = pd.DataFrame(all_results)
    suffix = "_".join(models).lower()
    results_df.to_csv(output_tables / f"modeling_results_{suffix}.csv", index=False)

    # Visualization (only if CART was requested)
    if "CART" in models:
        _generate_final_visualization(df, features, output_figures)

    logger_inst.info("Pipeline completed successfully.")