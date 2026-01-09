"""
Main Execution Entry Point
--------------------------
This script responsible for the entire Parkinson's Telemonitoring analysis pipeline.
It handles environment setup, data acquisition, preprocessing, exploratory
analysis, and predictive modeling (LS, IRLS, and CART).
"""

from sklearn.model_selection import train_test_split
from src import dataset, fetch, exploration, modeling
from src.config import Paths
from src.logger import logger_inst

paths = Paths.from_here()

def setup_environment() -> None:
    """
    Ensures that all necessary project directories exist before execution.
    """

    # List of all directories needed for the pipeline
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
        # Downloads the ZIP from UCI and extracts the raw data file
        logger_inst.info("Stage 1: Executing Data Fetching...")
        fetch.fetch_data()

        # Stage 2: Preprocessing & Normalization
        # Cleans columns and applies Min-Max scaling [0,1]
        logger_inst.info("Stage 2: Executing Preprocessing...")
        dataset.load_data()

        # Stage 3: Exploratory Data Analysis (EDA)
        # Generates correlation tables and Fig 1a, 1b, 1c from the paper
        logger_inst.info("Stage 3: Executing Statistical Exploration...")
        exploration.generate_exploration()

        # Stage 4: Modeling
        logger_inst.info("Stage 4: Modeling...")
        modeling.run_modeling_pipeline(
            input_path=paths.data_processed / "parkinsons_normalized.csv",
            output_tables=paths.tables,
            output_figures=paths.figures
        )

        logger_inst.info("=== Pipeline Completed Successfully ===")

    except Exception as error:
        logger_inst.critical(f"Pipeline failed: {error}")
        raise


if __name__ == "__main__":
    main()