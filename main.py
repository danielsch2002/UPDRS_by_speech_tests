import os
from src import dataset, fetch, exploration
from src.logger import logger_inst

def setup_env():

    # Create needed folders
    for folder in ['data/raw', 'data/processed', 'results']:
        os.makedirs(folder, exist_ok=True)
    logger_inst.info("Project structure is ready.")


def main():
    logger_inst.info("Starting UPDRS Analysis Pipeline")

    # Set up environment
    setup_env()

    # Import data
    fetch.fetch_data()

    # Preprocess
    dataset.load_data()

    # Statistics and graphs
    exploration.generate_exploration()

    logger_inst.info("Pipeline finished successfully")

if __name__ == "__main__":
    main()