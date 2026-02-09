"""
Dataset Fetching Module
--------------------------
Automates the fetching of the Parkinson's Telemonitoring dataset from the
UCI Machine Learning Repository. This ensures that the raw data is fetched
directly from the source, promoting reproducibility.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

from src.config import Paths
from src.logger import logger_inst

# Dataset source constants
ZIP_URL = "https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip"
TARGET_FILENAME = "parkinsons_updrs.data"


def _http_get_bytes(url: str, timeout_s: int = 30) -> bytes:
    """
    Performs an HTTP GET request to retrieve binary data.

    Args:
        url (str): The URL to download.
        timeout_s (int): Timeout in seconds.

    Returns:
        bytes: The raw content of the response.

    Raises:
        RuntimeError: If the server returns an error or the connection fails.
    """
    logger_inst.debug("Initiating GET request: %s", url)
    headers = {"User-Agent": "Parkinsons-Research-Pipeline/1.0"}
    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                logger_inst.error("HTTP error occurred: %s", status)
                raise RuntimeError(f"Server returned HTTP {status}")
            return response.read()
    except Exception as exc:
        logger_inst.error("Failed to download data: %s", exc)
        raise RuntimeError(f"HTTP request failed: {exc}") from exc


def fetch_and_extract(url: str, target_file: str, out_path: Path) -> None:
    """
    Downloads a ZIP archive from a URL and extracts a specific file.

    Args:
        url (str): Source ZIP URL.
        target_file (str): Filename to extract from the archive.
        out_path (Path): Destination path for the extracted file.

    Raises:
        FileNotFoundError: If the target_file is not found inside the ZIP.
    """
    # Ensure the directory structure exists before writing
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger_inst.info("Connecting to UCI Repository to fetch dataset...")
    zip_bytes = _http_get_bytes(url)

    logger_inst.info("Processing archive and extracting '%s'...", target_file)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        if target_file not in zf.namelist():
            logger_inst.error("Extraction failed: '%s' not found in ZIP.", target_file)
            raise FileNotFoundError(f"'{target_file}' is missing from the archive.")

        data_content = zf.read(target_file)

        with out_path.open("wb") as f:
            f.write(data_content)

    logger_inst.info("Raw data successfully cached at: %s", out_path)


def fetch_data() -> None:
    """
    Main entrypoint for the data fetching process.
    Resolves local paths and triggers the download/extract sequence.
    """
    parser = argparse.ArgumentParser(description="UCI Dataset Downloader")
    parser.add_argument("--url", default=ZIP_URL, help="Override the default dataset URL.")
    args = parser.parse_args()

    # Use centralized path management
    paths = Paths.from_here()
    raw_file_path = paths.data_raw / TARGET_FILENAME

    try:
        fetch_and_extract(
            url=args.url,
            target_file=TARGET_FILENAME,
            out_path=raw_file_path
        )
    except Exception as err:
        logger_inst.critical("Data fetching pipeline aborted: %s", err)
        raise


if __name__ == "__main__":
    fetch_data()