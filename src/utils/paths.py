"""
Path resolution utilities for the sign language recognition project.

Provides robust path resolution that works regardless of where scripts are executed.
"""

from pathlib import Path
import os


def get_project_root() -> Path:
    """
    Get the project root directory by searching for .git directory.

    Returns:
        Path: Absolute path to project root

    Raises:
        RuntimeError: If project root cannot be found
    """
    # Start from current file's directory
    current = Path(__file__).resolve().parent

    # Walk up the directory tree looking for .git
    for parent in [current] + list(current.parents):
        if (parent / '.git').exists():
            return parent

    # If no .git found, assume we're 2 levels deep (src/utils/)
    # and go up to project root
    return Path(__file__).resolve().parent.parent.parent


def get_data_root() -> Path:
    """Get the data directory relative to project root."""
    return get_project_root() / "data"


def get_raw_data_root() -> Path:
    """Get the raw data directory."""
    return get_data_root() / "raw_data"


def get_processed_data_root() -> Path:
    """Get the processed data directory."""
    return get_data_root() / "processed_holistic"


def get_annotations_dir() -> Path:
    """Get the annotations directory for PHOENIX dataset."""
    return get_raw_data_root() / "phoenix-2014-signerindependent-SI5" / "annotations" / "manual"


def resolve_path(path: str) -> Path:
    """
    Resolve a path string to an absolute path.

    If the path is relative, it's resolved relative to project root.
    If the path is absolute, it's returned as-is.

    Args:
        path: Path string (relative or absolute)

    Returns:
        Path: Absolute Path object
    """
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj
    else:
        return get_project_root() / path_obj


def validate_path(path: Path, must_exist: bool = True, description: str = "Path") -> Path:
    """
    Validate that a path exists and is accessible.

    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        description: Description for error messages

    Returns:
        Path: The validated path

    Raises:
        FileNotFoundError: If must_exist=True and path doesn't exist
    """
    if must_exist and not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}\n"
            f"Please ensure the dataset is properly installed."
        )
    return path


# Initialize project root on module load for validation
PROJECT_ROOT = get_project_root()

# Only print once (not on every multiprocessing worker import)
import sys
if not hasattr(sys, '_paths_initialized'):
    # print(f"[paths.py] Detected project root: {PROJECT_ROOT}")
    sys._paths_initialized = True
