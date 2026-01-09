"""Fetch the Parkinson's Telemonitoring dataset from UCI.

We retrieve the official ZIP archive from the UCI repository, extract the 
raw CSV data file, and save it to disk. This ensures the pipeline is 
reproducible starting from the raw source.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

from src.config import Paths
from src.logger import logger_inst

# The official download link from the dataset landing page
ZIP_URL = "https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip"
# The specific filename inside the ZIP that we want
TARGET_FILENAME = "parkinsons_updrs.data"


def _http_get_bytes(url: str, timeout_s: int = 30) -> bytes:
    """GET a URL and return raw bytes.

    Args:
        url: Full URL to request.
        timeout_s: Request timeout in seconds.

    Returns:
        Raw bytes of the response body.

    Raises:
        RuntimeError: If HTTP status is not 200 or request fails.
    """
    logger_inst.debug("GET %s", url)
    # Set a User-Agent to properly identify the client to the server
    req = urllib.request.Request(url, headers={"User-Agent": "parkinsons-repro-lab/1.0"})
    
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            if status != 200:
                raise RuntimeError(f"HTTP {status}")
            return resp.read()
    except Exception as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc


def fetch_and_extract(url: str, target_file: str, out_path: Path) -> None:
    """Fetch a ZIP archive and extract a specific file to disk.

    Args:
        url: URL of the ZIP file.
        target_file: Filename inside the ZIP to extract.
        out_path: Local path where the file should be saved.

    Raises:
        FileNotFoundError: If the target file is not in the ZIP.
    """
    # Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger_inst.info("Downloading ZIP from %s...", url)
    zip_bytes = _http_get_bytes(url)

    logger_inst.info("Extracting '%s'...", target_file)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        if target_file not in zf.namelist():
            raise FileNotFoundError(f"'{target_file}' not found in ZIP archive.")
        
        data_content = zf.read(target_file)
        
        with out_path.open("wb") as f:
            f.write(data_content)
            
    logger_inst.info("Cached raw data at %s", out_path)


def fetch_data() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Fetch and unzip Parkinson's data.")
    parser.add_argument("--url", default=ZIP_URL, help="URL of the dataset ZIP.")
    args = parser.parse_args()

    paths = Paths.from_here()
    raw_file_path = paths.data_raw / TARGET_FILENAME

    fetch_and_extract(url=args.url, target_file=TARGET_FILENAME, out_path=raw_file_path)