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
import seaborn as sns
from src.config import Paths

from src.logger import logger_inst


def get_model_definitions(model_names: list[str]) -> list[tuple[object, str]]:
    """Returns requested model objects with their descriptive names."""
    all_models = {
        "LS": (LinearRegression(), "LS"),
        "IRLS": (HuberRegressor(max_iter=10000, tol=1e-1, alpha=0.1, warm_start=True), "IRLS"),
        "LASSO": (Lasso(alpha=Paths.LASSO_ALPHA_OPTIMAL, max_iter=10000), "LASSO"),
        "CART": (DecisionTreeRegressor(random_state=42, max_depth=4), "CART")
    }
    return [all_models[name] for name in model_names if name in all_models]


def plot_optima_comparison(comparison_results: list[dict], output_figures: Path):
    """
    Generates a bar chart comparing Test MAE across different identified optima.
    """
    if not comparison_results:
        return

    df_comp = pd.DataFrame(comparison_results)
    plt.figure(figsize=(10, 6))
    df_comp = df_comp.sort_values('Features')

    labels = [f"Step {row['Step']}\n({row['Features']} Feats)" for _, row in df_comp.iterrows()]

    bars = plt.bar(labels, df_comp['Test MAE'], color='#e74c3c', width=0.6)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.3f}',
                 ha='center', va='bottom', fontweight='bold')

    plt.title("Performance Comparison: Identified BIC Optima (Total UPDRS)")
    plt.ylabel("Test MAE")
    plt.grid(axis='y', alpha=0.3)

    save_path = output_figures / "optima_performance_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger_inst.info(f"Optima comparison graph saved to {save_path}")


def visualize_cart_tree(model: DecisionTreeRegressor, features: list[str], output_path: Path) -> None:
    """Generates a high-resolution visualization of the Decision Tree."""
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=features, filled=True, rounded=True, fontsize=10, max_depth=4)
    plt.title("CART Decision Tree Structure")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _generate_final_visualization(df: pd.DataFrame, features: list[str], output_dir: Path) -> None:
    """Helper to train and visualize the final CART model on the full dataset."""
    logger_inst.info("Generating final CART decision tree visualization...")
    final_cart = DecisionTreeRegressor(random_state=42, max_depth=4)
    final_cart.fit(df[features], df["total_UPDRS"])
    visualize_cart_tree(final_cart, features, output_dir / "cart_final_structure.png")


def _plot_results(results_df: pd.DataFrame, output_figures: Path):
    """
    Generates performance comparison plots.
    Updated Stage 4 graph to show ALL models for Subject-Wise Total UPDRS.
    """
    sns.set_theme(style="whitegrid")

    # 1. Overfitting Analysis (Train vs Test)
    metrics_melted = results_df.melt(
        id_vars=['Method', 'Target', 'Model'],
        value_vars=['Train MAE', 'Test MAE'],
        var_name='Metric', value_name='MAE'
    )

    for target in results_df['Target'].unique():
        plt.figure(figsize=(12, 6))
        subset = metrics_melted[metrics_melted['Target'] == target]
        sns.barplot(data=subset, x='Model', y='MAE', hue='Metric')
        plt.title(f"Model Overfitting Analysis: Train vs Test MAE ({target})")
        plt.savefig(output_figures / f"overfitting_analysis_{target}.png")
        plt.close()

    # 2. Stage 4 Baseline Comparison: Focused on Subject-Wise Total UPDRS
    # This prevents misleading averaging and shows all 4 models side-by-side
    sw_total_df = results_df[(results_df['Target'] == 'total_UPDRS') &
                             (results_df['Method'] == 'Subject-Wise')]

    if not sw_total_df.empty:
        plt.figure(figsize=(10, 6))
        sw_total_df = sw_total_df.sort_values('Model')

        bars = plt.bar(sw_total_df['Model'], sw_total_df['Test MAE'], color='#3498db', width=0.6)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}',
                     ha='center', va='bottom', fontweight='bold')

        plt.title("Stage 4 Baseline: Subject-Wise Performance Comparison")
        plt.ylabel("Test MAE (Total UPDRS)")
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(output_figures / "stage4_model_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()


def run_cv_logic(model_obj, X, y, cv_generator, groups=None) -> dict:
    """Executes cross-validation and returns metrics."""
    cv_results = cross_validate(
        model_obj, X, y, groups=groups, cv=cv_generator,
        scoring='neg_mean_absolute_error', return_train_score=True, n_jobs=2
    )
    return {
        "Train MAE": round(-np.mean(cv_results['train_score']), 2),
        "Test MAE": round(-np.mean(cv_results['test_score']), 2),
        "Test SD": round(np.std(-cv_results['test_score']), 3)
    }


def run_modeling_pipeline(models: list[str], input_path: Path, output_tables: Path,
                          output_figures: Path, candidate_sets: dict = None) -> None:
    """
    Coordinates the modeling pipeline with detailed logging for each step.
    """
    logger_inst.info(f"Starting Modeling Pipeline for: {models}")
    df = pd.read_csv(input_path)
    targets = ["motor_UPDRS", "total_UPDRS"]
    groups = df["subject_id"]

    # Case A: Evaluated candidates from Stage 6
    if candidate_sets:
        logger_inst.info("--- Evaluating Parsimonious Candidate Sets (Stage 6) ---")
        comparison_data = []
        for step, feats in candidate_sets.items():
            if not feats: continue
            logger_inst.info(f"Running IRLS Evaluation for {step} ({len(feats)} features)...")
            X, y = df[feats], df["total_UPDRS"]
            model_obj = HuberRegressor(max_iter=10000, tol=1e-1)
            cv_gen = GroupShuffleSplit(n_splits=Paths.N_CV_ITERATIONS, test_size=0.1, random_state=42)
            metrics = run_cv_logic(model_obj, X, y, cv_gen, groups=groups)
            comparison_data.append({"Step": step, "Features": len(feats), "Test MAE": metrics["Test MAE"]})

        plot_optima_comparison(comparison_data, output_figures)

    # Case B: Standard Baseline Pipeline (Stage 4)
    logger_inst.info("--- Running Baseline Model Evaluations ---")
    features = [c for c in df.columns if c not in Paths.SKIP_COLS]
    cv_methods = [
        (RepeatedKFold(n_splits=10, n_repeats=Paths.N_CV_ITERATIONS, random_state=42), "Random", None),
        (GroupShuffleSplit(n_splits=Paths.N_CV_ITERATIONS, test_size=0.1, random_state=42), "Subject-Wise", groups)
    ]

    all_results = []
    for target_col in targets:
        logger_inst.info(f"Processing Target: {target_col}")
        X, y = df[features], df[target_col]

        for cv_gen, method_name, grp in cv_methods:
            logger_inst.info(f"  Applying {method_name} Cross-Validation...")

            for model_obj, model_name in get_model_definitions(models):
                logger_inst.info(f"    Training {model_name}...")
                metrics = run_cv_logic(model_obj, X, y, cv_gen, groups=grp)
                all_results.append({
                    "Method": method_name,
                    "Target": target_col,
                    "Model": model_name,
                    **metrics
                })

    results_df = pd.DataFrame(all_results)

    # Print summary table
    logger_inst.info("\n" + "="*90 + "\n" + results_df.to_string(index=False) + "\n" + "="*90)

    _plot_results(results_df, output_figures)
    results_df.to_csv(output_tables / f"modeling_results_final.csv", index=False)

    if "CART" in models:
        _generate_final_visualization(df, features, output_figures)

    logger_inst.info("Modeling pipeline completed successfully.")