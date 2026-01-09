"""
Project Path Configuration
--------------------------
This module centralizes all directory and file path logic. By resolving the 
PROJECT_ROOT dynamically, the pipeline remains functional regardless of the 
environment it is executed in.
"""

from __future__ import annotations
from pathlib import Path

# Resolve the absolute path to the project root (one level up from /src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Paths:
    """
    Static container for project directory paths.

    Provides a clean interface for accessing data, reports, and figure 
    locations without hard coding strings in the logic modules.
    """
    # Main Directories
    data = PROJECT_ROOT / "data"
    reports = PROJECT_ROOT / "reports"

    # Data Sub-directories
    data_raw = data / "raw"
    data_processed = data / "processed"

    # Output Sub-directories
    figures = reports / "figures"
    tables = reports / "tables"

    @classmethod
    def from_here(cls) -> type[Paths]:
        """
        Entry point to access the path configuration.

        Returns:
            The Paths class with resolved directory members.
        """
        return cls