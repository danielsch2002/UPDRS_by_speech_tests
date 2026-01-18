"""
Main Execution Entry Point
--------------------------
This script responsible for the entire Parkinson's Telemonitoring analysis pipeline.
It handles environment setup, data fetching, preprocessing, exploratory
analysis, and predictive modeling (LS, IRLS, LASSO, and CART).
"""

from src import dataset, fetch, exploration, modeling, lasso_selection
from src.config import Paths
from src.logger import logger_inst

paths = Paths.from_here()


def setup_environment() -> None:
    """
    Ensures that all necessary project directories exist before execution.
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
    The Tsanas et al. (2010) pipeline stages.
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
        logger_inst.info("Stage 3: Executing Statistical Exploration...")
        exploration.generate_exploration()

        # Stage 4: Modeling (All features)
        # We run all 4 models on the full feature set (16 metrics)
        logger_inst.info("Stage 4: Modeling with all features...")
        modeling.run_modeling_pipeline(
            models=["LS", "IRLS", "LASSO", "CART"],
            input_path=paths.data_processed / "parkinsons_normalized.csv",
            output_tables=paths.tables,
            output_figures=paths.figures
        )

        # Stage 5: LASSO Feature Selection (AIC vs BIC)
        # This stage produces 'best_features_parkinsons_normalized.csv'
        logger_inst.info("Stage 5: Executing LASSO Feature Selection...")
        lasso_selection.run_lasso_pipeline(
            input_path=paths.data_processed / "parkinsons_normalized.csv",
            output_tables=paths.tables
        )

        # Stage 6: IRLS And CART Modeling on best features extracted list.
        # As per the paper, we evaluate the 'Parsimonious' models here.
        logger_inst.info("Stage 6: IRLS And CART Modeling on reduced feature set...")
        modeling.run_modeling_pipeline(
            models=["IRLS", "CART"],
            input_path=paths.data_processed / "best_features_parkinsons_normalized.csv",
            output_tables=paths.tables,
            output_figures=paths.figures
        )

        logger_inst.info("=== Pipeline Completed Successfully ===")

    except Exception as error:
        logger_inst.critical(f"Pipeline failed: {error}")
        raise


if __name__ == "__main__":
    main()