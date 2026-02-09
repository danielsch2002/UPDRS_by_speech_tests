"""
Exploratory Data Analysis (EDA) and Correlation Analysis
-------------------------------------------------------
This module reproduces the statistical analysis and visualizations described
in Section III.A of Tsanas et al. (2010).

It generates:
1. Summary tables for data integrity and descriptive statistics.
2. Visualizations for target distribution and subject balance.
3. Non-parametric Spearman rank correlation analysis (Tables I & II).
4. Probability density estimations and scatter plots for feature analysis.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config import Paths
from src.logger import logger_inst

sns.set_theme(style="whitegrid")


def load_processed_data(filepath: Path) -> pd.DataFrame:
    """
    Loads and validates the processed dataset.

    Args:
        filepath (Path): Path to the normalized CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if not filepath.exists():
        logger_inst.error("Processed data file missing at: %s", filepath)
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    logger_inst.info("Successfully loaded processed data. Shape: %s", df.shape)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Returns a list of the 16 dysphonia features, excluding metadata and targets."""
    excluded = {"subject_id", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS"}
    return [c for c in df.columns if c not in excluded]


def check_data_health(df: pd.DataFrame, output_path: Path) -> None:
    """Checks for missing values and saves descriptive statistics."""
    logger_inst.info("Initiating data health check and descriptive stats...")

    null_counts = df.isnull().sum()
    if null_counts.any():
        logger_inst.warning("Null values detected in dataset:\n%s", null_counts[null_counts > 0])

    stats_summary = df.describe().transpose()
    stats_summary.to_csv(output_path)
    logger_inst.info("Descriptive statistics saved to %s", output_path)


def plot_basic_descriptives(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates histograms for targets and subject recording counts."""
    logger_inst.info("Generating target and subject distribution plots...")

    # Target distribution plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["motor_UPDRS"], kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title("Motor UPDRS Distribution")

    sns.histplot(df["total_UPDRS"], kde=True, ax=axes[1], color="salmon")
    axes[1].set_title("Total UPDRS Distribution")

    plt.tight_layout()
    plt.savefig(output_dir / "target_distributions.png", dpi=300)
    plt.close()

    # Subject recording balance
    plt.figure(figsize=(12, 5))
    counts = df["subject_id"].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, color="teal")
    plt.title("Number of Recordings per Subject")
    plt.xlabel("Subject ID")
    plt.ylabel("Count")

    plt.savefig(output_dir / "subject_balance.png", dpi=300)
    logger_inst.info("Basic descriptive plots saved to %s", output_dir)
    plt.close()


def generate_correlation_tables(df: pd.DataFrame, features: list[str], output_dir: Path) -> None:
    """
    Computes Spearman correlations for feature relevance and redundancy.
    Reproduces Tables I & II logic from the reference paper.
    """
    logger_inst.info("Computing Spearman rank correlations (Table I & II)...")

    # Table I: Feature-Target Relevance
    relevance = []
    for f in features:
        rho_m, _ = stats.spearmanr(df[f], df["motor_UPDRS"])
        rho_t, _ = stats.spearmanr(df[f], df["total_UPDRS"])
        relevance.append({
            "Feature": f,
            "Motor_Rho": round(rho_m, 2),
            "Total_Rho": round(rho_t, 2)
        })

    pd.DataFrame(relevance).to_csv(output_dir / "table_I_relevance.csv", index=False)

    # Table II: Inter-feature Correlation Matrix
    corr_matrix = df[features].corr(method="spearman").round(2)
    corr_matrix.to_csv(output_dir / "table_II_full_matrix.csv")

    # Filter for high redundancy (|Rho| >= 0.95)
    pairs = corr_matrix.stack().reset_index()
    pairs.columns = ["F1", "F2", "Rho"]
    high_corr = pairs[(pairs["F1"] < pairs["F2"]) & (pairs["Rho"].abs() >= 0.95)]
    high_corr.to_csv(output_dir / "high_redundancy_list.csv", index=False)

    logger_inst.info("Correlation tables successfully exported.")


def plot_distributions(df: pd.DataFrame, features: list[str], output_dir: Path) -> None:
    """Generates Figure 1a: Gaussian KDE plots for all features."""
    logger_inst.info("Generating Figure 1a (Feature Density Estimates)...")

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        sns.kdeplot(data=df, x=feat, ax=axes[i], fill=True, color="black", linewidth=1.2)
        axes[i].set_title(feat, fontsize=10)
        axes[i].set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_dir / "Fig_1a_densities.png", dpi=300)
    logger_inst.info("Figure 1a saved to %s", output_dir)
    plt.close()


def plot_scatter_grids(df: pd.DataFrame, features: list[str], target: str, output_dir: Path) -> None:
    """Generates scatter plots with regression trends for a given target."""
    logger_inst.info("Generating scatter grid for target: %s", target)

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        sns.regplot(
            data=df, x=feat, y=target, ax=axes[i],
            scatter_kws={'alpha': 0.2, 's': 3, 'color': 'navy'},
            line_kws={'color': 'red', 'lw': 1}
        )
        axes[i].set_title(feat, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f"scatter_grid_{target}.png", dpi=300)
    logger_inst.info("Scatter grid for %s saved.", target)
    plt.close()


def generate_exploration() -> None:
    """Main execution entry point for EDA."""
    paths = Paths.from_here()
    input_csv = paths.data_processed / "parkinsons_normalized.csv"

    # Setup output paths
    tables_path = paths.reports / "tables"
    figures_path = paths.figures
    for p in [tables_path, figures_path]:
        p.mkdir(parents=True, exist_ok=True)

    try:
        df = load_processed_data(input_csv)
        features = get_feature_columns(df)

        # Run Analysis Pipeline
        check_data_health(df, tables_path / "descriptive_stats.csv")
        plot_basic_descriptives(df, figures_path)
        generate_correlation_tables(df, features, tables_path)
        plot_distributions(df, features, figures_path)
        plot_scatter_grids(df, features, "total_UPDRS", figures_path)

        logger_inst.info("Full Exploratory Data Analysis completed successfully.")

    except Exception as err:
        logger_inst.error("Exploration failed: %s", err)
        raise


if __name__ == "__main__":
    generate_exploration()