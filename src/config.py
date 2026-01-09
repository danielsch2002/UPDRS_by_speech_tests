from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Paths:
    data = PROJECT_ROOT / "data"
    data_raw = data / "raw"
    data_processed = data / "processed"
    reports = PROJECT_ROOT / "reports"
    figures = reports / "figures"

    @classmethod
    def from_here(cls):
        return cls