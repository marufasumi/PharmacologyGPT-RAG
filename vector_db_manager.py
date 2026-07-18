"""
Download and prepare the persisted Chroma vector database.
"""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_DIRECTORY = PROJECT_ROOT / "vector"
CHROMA_DATABASE_FILE = VECTOR_DIRECTORY / "chroma.sqlite3"

VECTOR_ARCHIVE_URL_ENV = "VECTOR_ARCHIVE_URL"


def vector_database_exists() -> bool:
    """Return True if the persisted Chroma database already exists."""
    return (
        CHROMA_DATABASE_FILE.is_file()
        and any(VECTOR_DIRECTORY.glob("*/data_level0.bin"))
    )


def _safe_extract_archive(
    archive: tarfile.TarFile,
    destination: Path,
) -> None:
    """Safely extract a tar archive."""

    destination = destination.resolve()

    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()

        if destination not in member_path.parents and member_path != destination:
            raise ValueError(f"Unsafe archive member: {member.name}")

    archive.extractall(destination)


def download_vector_database() -> Path:
    """
    Download and extract the vector database if it does not already exist.
    """

    if vector_database_exists():
        return VECTOR_DIRECTORY

    archive_url = os.getenv(VECTOR_ARCHIVE_URL_ENV)

    if not archive_url:
        raise RuntimeError(
            "VECTOR_ARCHIVE_URL environment variable is not configured."
        )

    with tempfile.NamedTemporaryFile(
        suffix=".tar.gz",
        delete=False,
    ) as temporary_file:
        archive_path = Path(temporary_file.name)

    try:
        with requests.get(
            archive_url,
            stream=True,
            timeout=(15, 300),
        ) as response:
            response.raise_for_status()

            with archive_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)

        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract_archive(archive, PROJECT_ROOT)

        if not vector_database_exists():
            raise RuntimeError("Vector database extraction failed.")

        return VECTOR_DIRECTORY

    finally:
        archive_path.unlink(missing_ok=True)