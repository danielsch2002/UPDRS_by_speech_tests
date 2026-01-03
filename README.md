# UPDRS by Speech Tests Analysis

## Project Description
This project aims to analyze and predict **UPDRS (Unified Parkinson's Disease Rating Scale)** scores using various speech features. 
**Hypothesis:** Specific vocal impairments (jitter, shimmer, etc.) correlate significantly with the progression of Parkinson's disease symptoms.

## Folder Structure
- `src/`: Core logic and modules.
  - 
- `data/`:
  - `raw/`: Original dataset (not included in Git).
  - `processed/`: Cleaned data files.
- `tests/`: Unit tests for each stage.
- `main.py`: Entry point for execution.
- `pyproject.toml`: Project metadata and dependencies.

## Key Stages
1. **Data Import**: Loading CSV files and verifying data integrity.
2. **Processing**: Handling missing values, normalizing speech features, and outlier detection.
3. **Modeling**: Running regression models to predict UPDRS scores.
4. **Analysis & Visualization**: Generating correlation matrices and trend graphs.

## Paper Link
The paper reproduced in this project can be found here: [https://ieeexplore.ieee.org/document/5339170].

## Data Source
The dataset used in this project can be found here: [https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring].

## Instructions for Running the Project
1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd UPDRS_by_speech_tests
   
2. **Setup Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies:**
    ```bash
    pip install .
   
4. **Run The Project:**
    ```bash
    py main.py
   
5. **Run Tests:**
    ```bash
    pytest tests/
