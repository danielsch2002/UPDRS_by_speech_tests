"""
Predictive Modeling Module
--------------------------
Implements regression models as per Tsanas et al. (2010).
Supports baseline evaluations and comparative parsimony analysis.
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
import seaborn as sns

from src.config import Config
from src.logger import logger_inst


def get_model_definitions(model_names: list[str]) -> list[tuple[object, str]]:
    """Returns requested model objects initialized with Config settings."""
    all_models = {
        "LS": (LinearRegression(), "LS"),
        "IRLS": (HuberRegressor(
            max_iter=Config.IRLS_MAX_ITER,
            tol=Config.IRLS_TOL,
            alpha=Config.IRLS_ALPHA,
            warm_start=True), "IRLS"),
        "LASSO": (Lasso(
            alpha=Config.LASSO_ALPHA_OPTIMAL,
            max_iter=Config.LASSO_MAX_ITER), "LASSO"),
        "CART": (DecisionTreeRegressor(
            random_state=Config.RANDOM_STATE,
            max_depth=Config.CART_MAX_DEPTH,
            min_samples_split=Config.CART_MIN_SAMPLES_SPLIT,
            min_samples_leaf=Config.CART_MIN_SAMPLES_LEAF), "CART")
    }
    return [all_models[name] for name in model_names if name in all_models]


def _plot_cart_tree(df: pd.DataFrame, features: list[str], output_figures: Path, subset_name: str):
    """Generates a visual representation of the Decision Tree (CART)."""
    model, _ = get_model_definitions(["CART"])[0]
    # Fit on all data for visualization purposes
    model.fit(df[features], df["total_UPDRS"])

    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=features, filled=True, rounded=True, fontsize=10)
    plt.title(f"Decision Tree Structure - {subset_name} Set")

    filename = f"cart_tree_{subset_name.lower()}.png"
    plt.savefig(output_figures / filename, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_results(results_df: pd.DataFrame, output_figures: Path, stage_label: str, subset_name: str):
    """Generates side-by-side comparison of CV methods."""
    sns.set_theme(style="whitegrid")
    plot_df = results_df[results_df['Target'] == 'total_UPDRS'].copy()
    if plot_df.empty: return

    plt.figure(figsize=(12, 6))
    model_order = ["LS", "IRLS", "LASSO", "CART"] if "LS" in plot_df['Model'].values else ["IRLS", "CART"]

    ax = sns.barplot(data=plot_df, x="Model", y="Test MAE", hue="Method", palette="coolwarm", order=model_order)
    plt.title(f"{stage_label}: {subset_name} Set Performance")
    plt.ylim(0, plot_df['Test MAE'].max() * 1.3)

    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontweight='bold')

    filename = f"{stage_label.lower().replace(' ', '_')}_{subset_name.lower()}_comparison.png"
    plt.savefig(output_figures / filename, dpi=300, bbox_inches='tight')
    plt.close()


def run_cv_logic(model_obj, X, y, cv_generator, groups=None) -> dict:
    """Executes CV and returns performance metrics."""
    cv_results = cross_validate(model_obj, X, y, groups=groups, cv=cv_generator,
                                scoring='neg_mean_absolute_error', return_train_score=True, n_jobs=2)
    return {
        "Train MAE": round(-np.mean(cv_results['train_score']), 2),
        "Test MAE": round(-np.mean(cv_results['test_score']), 2),
        "Test SD": round(np.std(-cv_results['test_score']), 3)
    }


def run_modeling_pipeline(models: list[str], input_path: Path, output_tables: Path,
                          output_figures: Path, candidate_sets: dict = None,
                          stage_label: str = "Stage 4") -> pd.DataFrame:
    """
    Coordinates modeling. If Stage 6, it skips re-running the full set if possible.
    Returns the results DataFrame for potential re-use.
    """
    logger_inst.info(f"--- Starting {stage_label} Pipeline ---")
    df = pd.read_csv(input_path)
    groups = df["subject_id"]
    targets = ["motor_UPDRS", "total_UPDRS"]
    cv_methods = [
        (RepeatedKFold(n_splits=Config.N_FOLDS, n_repeats=Config.N_CV_ITERATIONS, random_state=Config.RANDOM_STATE),
         "Random", None),
        (GroupShuffleSplit(n_splits=Config.N_CV_ITERATIONS, test_size=0.1, random_state=Config.RANDOM_STATE),
         "Subject-Wise", groups)
    ]

    # --- Case 1: Stage 4 (Baseline on Full Set) ---
    if stage_label == "Stage 4":
        features = [c for c in df.columns if c not in Config.SKIP_COLS]
        all_results = []
        for target in targets:
            for cv_gen, method, grp in cv_methods:
                for model_obj, name in get_model_definitions(models):
                    metrics = run_cv_logic(model_obj, df[features], df[target], cv_gen, groups=grp)
                    all_results.append({"Method": method, "Target": target, "Model": name, **metrics})

        res_df = pd.DataFrame(all_results)
        _plot_results(res_df, output_figures, stage_label, "Full")
        _plot_cart_tree(df, features, output_figures, "Full")  # Restore CART Plot
        res_df.to_csv(output_tables / "stage4_full_results.csv", index=False)
        return res_df

    # --- Case 2: Stage 6 (Focused on BIC Subset) ---
    if stage_label == "Stage 6" and candidate_sets:
        bic_feats = candidate_sets.get("Our BIC Subset (7)")
        if bic_feats:
            logger_inst.info(f"Running Stage 6 evaluation ONLY on BIC subset ({len(bic_feats)} features)...")
            bic_results = []
            for target in targets:
                for cv_gen, method, grp in cv_methods:
                    for model_obj, name in get_model_definitions(models):
                        metrics = run_cv_logic(model_obj, df[bic_feats], df[target], cv_gen, groups=grp)
                        bic_results.append({"Method": method, "Target": target, "Model": name, **metrics})

            res_df = pd.DataFrame(bic_results)
            _plot_results(res_df, output_figures, stage_label, "BIC")
            _plot_cart_tree(df, bic_feats, output_figures, "BIC")  # CART Plot for subset
            res_df.to_csv(output_tables / "stage6_bic_results.csv", index=False)
            return res_df
    logger_inst.info(f"{stage_label} completed successfully.")