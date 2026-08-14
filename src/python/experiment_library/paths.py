"""Project-relative paths used by the experimental-material tooling."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LIBRARY_ROOT = PROJECT_ROOT / "experiment-library"
ENGLISH_ROOT = LIBRARY_ROOT / "en"
