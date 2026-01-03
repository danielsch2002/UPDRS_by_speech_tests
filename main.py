import logging
import os
from src import data_import

# Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_env():

    # Create needed folders
    for folder in ['data/raw', 'data/processed', 'results']:
        os.makedirs(folder, exist_ok=True)
    logger.info("Project structure is ready.")


def main():
    logger.info("Starting UPDRS Analysis Pipeline")

    # Set up environment
    setup_env()

    # Import data
    data = data_import.load_data()

    logger.info("Pipeline finished successfully")

if __name__ == "__main__":
    main()