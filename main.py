"""
Main Execution Entry Point
--------------------------
This script is responsible for the entire Parkinson's Telemonitoring analysis pipeline.
It handles environment setup, data fetching, preprocessing, exploratory
analysis, and predictive modeling (LS, IRLS, LASSO, and CART).
"""

import pandas as pd
from src import dataset, fetch, exploration, modeling, lasso_selection, parsimony_utils
from src.config import Paths
from src.logger import logger_inst

paths = Paths.from_here()


def setup_environment() -> None:
    """
    Ensures that all project directories exist before execution.
    """
    required_folders = [
        paths.data_raw,
        paths.data_processed,
        paths.figures,
        paths.tables
    ]
    for folder in required_folders:
        folder.mkdir(parents=True, exist_ok=True)

    logger_inst.info("Project directory structure exists and ready.")


def main() -> None:
    """
    Executes the Tsanas et al. (2010) pipeline stages.
    """
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
        logger_inst.info("Stage 4: Executing Statistical Exploration...")
        exploration.generate_exploration()

        # Stage 4: Initial Modeling (Baseline with all 16 features)
        logger_inst.info("Stage 4: Modeling with all features (Baseline)...")
        modeling.run_modeling_pipeline(
            models=["LS", "IRLS", "LASSO", "CART"],
            input_path=paths.data_processed / "parkinsons_normalized.csv",
            output_tables=paths.tables,
            output_figures=paths.figures
        )

        # Stage 5: Multi-Optima LASSO Feature Selection
        logger_inst.info("Stage 5: Executing LASSO Feature Selection...")
        candidate_sets = lasso_selection.run_lasso_pipeline(
            input_path=paths.data_processed / "parkinsons_normalized.csv",
            output_tables=paths.tables,
            output_figures=paths.figures
        )

        # Stage 6: Comparative Modeling & Parsimony Analysis
        logger_inst.info("Stage 6: Evaluating candidate optima for comparison...")

        # Inject the full 16-feature set into the comparison candidates
        final_comparison_dict = parsimony_utils.prepare_comparison_sets(
            candidate_sets=candidate_sets,
            input_path=paths.data_processed / "parkinsons_normalized.csv"
        )

        # Runs the final comparison between parsimonious sets and baseline
        modeling.run_modeling_pipeline(
            models=["IRLS", "CART"],
            input_path=paths.data_processed / "parkinsons_normalized.csv",
            output_tables=paths.tables,
            output_figures=paths.figures,
            candidate_sets=final_comparison_dict
        )

        logger_inst.info("=== Pipeline Completed Successfully ===")

    except Exception as error:
        logger_inst.critical(f"Pipeline failed: {error}")
        raise


if __name__ == "__main__":
    main()