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
1. **Data Acquisition**: Automated fetching from the UCI Machine Learning Repository.
2. **Preprocessing**: Min-Max scaling and outlier detection.
3. **Exploration**: Statistical analysis and correlation mapping.
4. **Modeling**: Comparison of LS, IRLS, LASSO, and CART models.
5. **Feature Selection**: Identifying the 6 most parsimonious features using LASSO and AIC/BIC criteria.
6. **Model Validation**: Re-evaluating models on the reduced 6-feature set to confirm minimal loss in predictive accuracy.

## Instructions for Running
1. **Clone the repository:**

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
