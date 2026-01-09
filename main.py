"""
Main Execution Entry Point
--------------------------
This script responsible for the entire Parkinson's Telemonitoring analysis pipeline.
It ensures the environment is ready, fetches raw data, processes it,
and generates exploratory statistical reports and visualizations.
"""

from src import dataset, fetch, exploration
from src.config import Paths
from src.logger import logger_inst


def setup_environment() -> None:
    """
    Ensures that all necessary project directories exist before execution.
    """
    paths = Paths.from_here()

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
    The data science pipeline stages.
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

        logger_inst.info("=== Pipeline Completed Successfully ===")

    except Exception as error:
        logger_inst.critical("Pipeline failed during execution!")
        logger_inst.error("Error details: %s", str(error))
        # Re-raising the error ensures we know exactly where it crashed
        raise


if __name__ == "__main__":
    main()