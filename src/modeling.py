"""
Predictive Modeling Module
--------------------------
Implements regression models as per Tsanas et al. (2010):
I.   Least Squares (LS) - Standard Linear Regression.
II.  Iteratively Reweighted LS (IRLS) - Robust Regression (Huber Regressor).
IV.  Non-Linear: Classification And Regression Trees (CART).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_validate, GroupShuffleSplit
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.logger import logger_inst


def get_model_definitions() -> list[tuple[object, str]]:
    """Returns the standardized models used in the reproduction study."""
    return [
        (LinearRegression(), "Least Squares (LS)"),
        # Maximum iterations to allow enough time
        # High tolerance: Stops even if precision is loose
        # Stronger regularization to stabilize multicollinearity
        (HuberRegressor(max_iter=10000, tol=1e-1, alpha=0.1, warm_start=True), "IRLS (Robust)"),
        (DecisionTreeRegressor(random_state=42, max_depth=5), "CART (Non-linear)")
    ]


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
        max_depth=5
    )
    plt.title("CART Decision Tree Structure")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _generate_final_visualization(df: pd.DataFrame, features: list[str], output_dir: Path) -> None:
    """Helper to train and visualize the final CART model on the full dataset."""
    final_cart = DecisionTreeRegressor(random_state=42, max_depth=5)
    final_cart.fit(df[features], df["total_UPDRS"])

    visualize_cart_tree(
        final_cart,
        features,
        output_dir / "cart_final_structure.png"
    )


def run_cv_for_model(model_obj, X, y, groups, gss):
    """
    Executes subject-wise cross-validation.
    Groups ensure that all samples from a single patient stay in the same fold.
    """
    cv_results = cross_validate(
        model_obj, X, y,
        groups=groups,
        cv=gss,
        scoring='neg_mean_absolute_error',
        return_train_score=True,
        n_jobs=2
    )
    return {
        "Train MAE": round(-np.mean(cv_results['train_score']), 2),
        "Test MAE": round(-np.mean(cv_results['test_score']), 2),
        "Test SD": round(np.std(-cv_results['test_score']), 3)
    }


def run_modeling_pipeline(input_path: Path, output_tables: Path, output_figures: Path) -> None:
    """
    Coordinates the 1,000-iteration Subject-Wise CV pipeline.
    Uses GroupShuffleSplit to maintain patient independence between Train/Test.
    """
    logger_inst.info("Starting Subject-Wise pipeline with 1,000 iterations.")

    df = pd.read_csv(input_path)
    features = [c for c in df.columns if
                c not in ["subject_id", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS"]]
    targets = ["motor_UPDRS", "total_UPDRS"]

    # Identify patient groups to prevent data leakage
    groups = df["subject_id"]

    # GroupShuffleSplit ensures 10% of subjects (not rows) are held out per iteration
    gss = GroupShuffleSplit(n_splits=1000, test_size=0.1, random_state=42)

    all_results = []

    for target_col in targets:
        logger_inst.info(f"Processing target: {target_col}")
        X, y = df[features], df[target_col]

        for model_obj, model_name in get_model_definitions():
            logger_inst.info(f"Running 1,000 subject-wise iterations for {model_name}...")
            metrics = run_cv_for_model(model_obj, X, y, groups, gss)
            all_results.append({"Target": target_col, "Model": model_name, **metrics})

    # Saving Results
    results_df = pd.DataFrame(all_results)
    print(f"\n{'=' * 75}\nSUBJECT-WISE REPRODUCTION TABLE: 1,000 ITERATIONS\n{'=' * 75}")
    print(results_df.to_string(index=False))

    results_df.to_csv(output_tables / "table_III_subject_wise_cv.csv", index=False)

    # Visualization
    _generate_final_visualization(df, features, output_figures)
    logger_inst.info("Pipeline completed successfully.")