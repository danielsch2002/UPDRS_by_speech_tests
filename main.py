"""
Main Execution Entry Point
--------------------------
This script is responsible for the entire Parkinson's Telemonitoring analysis pipeline.
"""

import pandas as pd
from src import dataset, fetch, exploration, modeling, lasso_selection
from src.config import Paths, Config
from src.logger import logger_inst

paths = Paths.from_here()


def setup_environment() -> None:
    """Ensures directories exist and clears old results safely."""

    required_folders = [paths.data_processed, paths.figures, paths.tables]
    for folder in required_folders:
        folder.mkdir(parents=True, exist_ok=True)

        # Clear old files to ensure a clean run
        for file in folder.glob('*'):
            if file.is_file():
                file.unlink()
    logger_inst.info("Environment ready and cleaned.")


def main() -> None:
    """Executes the Tsanas et al. (2010) pipeline stages."""

    logger_inst.info("=== Starting UPDRS Analysis Pipeline ===")

    try:
        # Stage 0: Infrastructure Setup
        setup_environment()

        # Stage 1: Data Fetching
        logger_inst.info("Stage 1: Executing Data Fetching...")
        fetch.fetch_data()

        # Stage 2: Preprocessing & Normalization
        logger_inst.info("Stage 2: Executing Preprocessing...")
        dataset.load_data()

        # Stage 3: Exploratory Data Analysis (EDA)
        logger_inst.info("Stage 3: Executing Statistical Exploration...")
        exploration.generate_exploration()

        # Stage 4: Initial Modeling (Baseline with all 16 features)
        full_data_path = paths.data_processed / "parkinsons_normalized.csv"
        logger_inst.info("Stage 4: Modeling with all features (Baseline)...")
        modeling.run_modeling_pipeline(
            models=["LS", "IRLS", "LASSO", "CART"],
            input_path=full_data_path,
            output_tables=paths.tables,
            output_figures=paths.figures,
            stage_label="Stage 4"
        )

        # Stage 5: LASSO Feature Selection
        logger_inst.info("Stage 5: Executing LASSO Feature Selection...")
        candidate_sets = lasso_selection.run_lasso_pipeline(
            input_path=full_data_path,
            output_tables=paths.tables,
            output_figures=paths.figures
        )

        # Stage 6: Comparative Modeling & Parsimony Analysis
        logger_inst.info("Stage 6: Evaluating candidate features sets for comparison...")
        df_full = pd.read_csv(full_data_path)

        # 1. Full Set (16 features)
        all_features = [c for c in df_full.columns if c not in Config.SKIP_COLS]

        # 2. Paper's Original Set (from Config)
        paper_features = Config.PAPER_BEST_FEATURES

        # 3. Our BIC-Optimal Set (from Stage 5)
        final_comparison_dict = {
            "Full Set (16)": all_features,
            "Paper Subset (6)": paper_features,
            "Our BIC Subset (7)": candidate_sets["BIC_Optimal"]
        }

        modeling.run_modeling_pipeline(
            models=["IRLS", "CART"],
            input_path=full_data_path,
            output_tables=paths.tables,
            output_figures=paths.figures,
            candidate_sets=final_comparison_dict,
            stage_label="Stage 6"
        )

        logger_inst.info("=== Pipeline Completed Successfully ===")

    except Exception as error:
        logger_inst.critical(f"Pipeline failed: {error}")
        raise


if __name__ == "__main__":
    main()