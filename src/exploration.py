"""Exploratory Data Analysis (EDA) and Correlation Analysis.

This module generates the statistical tables and visualization figures 
described in Section III.A of Tsanas et al. (2010), along with standard 
descriptive visualizations for targets and subject demographics.

It performs the following:
1. Generates descriptive statistics and checks for data integrity.
2. Visualizes target variable distributions and subject recording counts.
3. Computes Spearman rank correlations (non-parametric) for Tables I & II.
4. Visualizes feature probability densities using Gaussian KDE (Fig. 1a).
5. Visualizes feature-target relationships with scatter plots (Fig. 1b, 1c).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config import Paths
from src.logger import logger_inst

def load_processed_data(filepath: Path) -> pd.DataFrame:
    """Load the normalized dataset from the processed directory.

    Args:
        filepath: Path to the processed CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data not found at {filepath}")
    
    df = pd.read_csv(filepath)
    logger_inst.info("Loaded processed data with shape %s", df.shape)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Identify the 16 dysphonia feature columns, excluding metadata and targets."""
    exclude_cols = {
        "subject_id", 
        "age", 
        "sex", 
        "test_time", 
        "motor_UPDRS", 
        "total_UPDRS"
    }
    return [c for c in df.columns if c not in exclude_cols]


def check_data_health(df: pd.DataFrame, output_path: Path) -> None:
    """Generate basic descriptive statistics and check for missing values.

    Args:
        df: The dataframe to analyze.
        output_path: Path to save the summary CSV.
    """
    logger_inst.info("Checking data health...")
    
    # Check for missing values
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger_inst.warning("Missing values detected:\n%s", null_counts[null_counts > 0])
    else:
        logger_inst.info("No missing values detected.")

    # Generate descriptive statistics
    desc_stats = df.describe().transpose()
    desc_stats.to_csv(output_path)
    logger_inst.info("Saved descriptive statistics to %s", output_path)


def plot_basic_descriptives(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate basic histograms for targets and subject recording counts.

    This visualizes the distribution of the target variables (UPDRS scores)
    and checks the balance of the dataset across different subjects.

    Args:
        df: The dataframe containing targets and subject IDs.
        output_dir: Directory to save the resulting figures.
    """
    logger_inst.info("Generating basic descriptive plots (Targets & Subjects)...")
    
    # 1. Target Distributions (Motor UPDRS & Total UPDRS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.histplot(data=df, x="motor_UPDRS", kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title("Distribution of Motor UPDRS Scores")
    
    sns.histplot(data=df, x="total_UPDRS", kde=True, ax=axes[1], color="salmon")
    axes[1].set_title("Distribution of Total UPDRS Scores")
    
    plt.tight_layout()
    target_path = output_dir / "Fig_Basic_Target_Distributions.png"
    plt.savefig(target_path, dpi=300)
    logger_inst.info("Saved Target Distributions to %s", target_path)
    plt.close()

    # 2. Subject Recording Counts (Check for class imbalance)
    plt.figure(figsize=(14, 6))
    subject_counts = df["subject_id"].value_counts().sort_index()
    
    sns.barplot(x=subject_counts.index, y=subject_counts.values, color="gray")
    plt.title("Number of Recordings per Subject")
    plt.xlabel("Subject ID")
    plt.ylabel("Count")
    plt.xticks(rotation=90, fontsize=8) # Rotate labels if many subjects
    
    plt.tight_layout()
    subject_path = output_dir / "Fig_Basic_Subject_Counts.png"
    plt.savefig(subject_path, dpi=300)
    logger_inst.info("Saved Subject Counts to %s", subject_path)
    plt.close()


def generate_correlation_tables(df: pd.DataFrame, features: list[str], output_dir: Path) -> None:
    """Calculate Spearman rank correlations for Tables I and II.
    
    Correlations are rounded to 2 decimal places to match the paper's methodology.
    
    Args:
        df: The dataframe containing features and targets.
        features: List of feature column names.
        output_dir: Directory to save the resulting CSV tables.
    """
    logger_inst.info("Calculating Spearman correlations (Non-parametric)...")

    # --- Table I: Feature-Target Correlations ---
    table_i_data = []
    for feat in features:
        # Spearmanr returns (correlation_coefficient, p_value)
        rho_motor, p_motor = stats.spearmanr(df[feat], df["motor_UPDRS"])
        rho_total, p_total = stats.spearmanr(df[feat], df["total_UPDRS"])
        
        table_i_data.append({
            "Dysphonia Measure": feat,
            # Round to 2 decimals as per paper methodology
            "Motor UPDRS Correlation": round(rho_motor, 2),
            "Motor p-value": p_motor,
            "Total UPDRS Correlation": round(rho_total, 2),
            "Total p-value": p_total
        })
    
    df_table_i = pd.DataFrame(table_i_data)
    table_i_path = output_dir / "Table_I_Correlations.csv"
    df_table_i.to_csv(table_i_path, index=False)
    
    print("\n--- Table I: Correlations with UPDRS (First 5 rows) ---")
    print(df_table_i.head().to_string(index=False))
    logger_inst.info("Saved Table I to %s", table_i_path)

    # --- Table II: Feature-Feature Correlations ---
    # 1. Calculate the correlation matrix and round to 2 decimals IMMEDIATELY
    # This ensures that subsequent filtering (>= 0.95) captures values like 0.946
    corr_matrix = df[features].corr(method="spearman").round(2)
    
    # Save the rounded matrix
    table_ii_full_path = output_dir / "Table_II_All_Correlations.csv"
    corr_matrix.to_csv(table_ii_full_path)
    logger_inst.info("Saved Table II (Full Matrix, rounded) to %s", table_ii_full_path)

    # 2. Identify high correlations (>= 0.95) using the ROUNDED values
    corr_pairs = corr_matrix.stack().reset_index()
    corr_pairs.columns = ["Measure A", "Measure B", "Rho"]
    
    # Filter out self-correlations and duplicates (keep A-B, drop B-A)
    corr_pairs = corr_pairs[corr_pairs["Measure A"] < corr_pairs["Measure B"]]
    high_corr = corr_pairs[corr_pairs["Rho"].abs() >= 0.95].sort_values(by="Rho", ascending=False)
    
    # Save this highlight list as a helper file
    table_ii_high_path = output_dir / "Table_II_High_Correlations_List.csv"
    high_corr.to_csv(table_ii_high_path, index=False)
    
    print("\n--- Table II: Highly Correlated Pairs (|Rho| >= 0.95) ---")
    print("(Based on rounding to 2 decimal places first)")
    print(high_corr.to_string(index=False))


def plot_distributions(df: pd.DataFrame, features: list[str], output_dir: Path) -> None:
    """Generate Kernel Density Estimation plots for all features (Fig 1a)."""
    logger_inst.info("Generating distribution plots (Gaussian KDE)...")
    
    n_cols = 4
    n_rows = (len(features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        # The paper uses Gaussian kernels for probability density
        sns.kdeplot(data=df, x=feat, ax=axes[i], fill=True, color="black", linewidth=1.5)
        axes[i].set_title(feat, fontsize=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Density" if i % n_cols == 0 else "")
        axes[i].tick_params(labelsize=8)
    
    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    save_path = output_dir / "Fig_1a_Distributions.png"
    plt.savefig(save_path, dpi=300)
    logger_inst.info("Saved Figure 1a to %s", save_path)
    plt.close()


def plot_scatter_vs_target(df: pd.DataFrame, features: list[str], target: str, filename: str, output_dir: Path) -> None:
    """Generate scatter plots of Features vs Target with regression lines (Fig 1b/1c)."""
    logger_inst.info("Generating scatter plots for target: %s...", target)
    
    n_cols = 4
    n_rows = (len(features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        # Scatter plot with linear fit (regplot)
        sns.regplot(
            data=df, 
            x=feat, 
            y=target, 
            ax=axes[i], 
            scatter_kws={"s": 2, "alpha": 0.2, "color": "black"}, 
            line_kws={"color": "gray", "linewidth": 1.5}
        )
        
        # Calculate correlation for annotation
        rho, _ = stats.spearmanr(df[feat], df[target])
        
        axes[i].set_title(f"R={rho:.2f}", fontsize=10, loc='right')
        axes[i].set_xlabel(feat, fontsize=9)
        axes[i].set_ylabel(target if i % n_cols == 0 else "")
        axes[i].tick_params(labelsize=8)
    
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300)
    logger_inst.info("Saved scatter plot grid to %s", save_path)
    plt.close()


def generate_exploration() -> None:
    """CLI entrypoint."""
    paths = Paths.from_here()
    input_file = paths.data_processed / "parkinsons_normalized.csv"
    
    # Define output directories
    tables_dir = paths.reports / "tables"
    figures_dir = paths.figures
    
    # Ensure directories exist
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_processed_data(input_file)
        features = get_feature_columns(df)
        
        # 1. Data Health Check & Descriptive Stats
        check_data_health(df, tables_dir / "Descriptive_Statistics.csv")

        # 2. Basic Visual Descriptives (Targets & Subjects)
        plot_basic_descriptives(df, figures_dir)
        
        # 3. Correlation Tables (Tables I & II)
        generate_correlation_tables(df, features, tables_dir)
        
        # 4. Distribution Plots (Fig 1a)
        plot_distributions(df, features, figures_dir)
        
        # 5. Scatter Plots (Fig 1b & 1c)
        plot_scatter_vs_target(df, features, "motor_UPDRS", "Fig_1b_Motor_Scatter.png", figures_dir)
        plot_scatter_vs_target(df, features, "total_UPDRS", "Fig_1c_Total_Scatter.png", figures_dir)
        
        logger_inst.info("Exploration analysis complete.")

    except Exception as exc:
        logger_inst.error("Exploration analysis failed: %s", exc)
        raise RuntimeError(f"Analysis failed: {exc}") from exc