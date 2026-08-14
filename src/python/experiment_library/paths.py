"""Project-root paths used by the experimental-material tooling."""

from pathlib import Path


PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "pyproject.toml").is_file():
    raise RuntimeError(
        "Run experiment-library tooling from the project root (the directory containing pyproject.toml)."
    )
LIBRARY_ROOT = PROJECT_ROOT / "experiment-library"
ENGLISH_ROOT = LIBRARY_ROOT / "en"
