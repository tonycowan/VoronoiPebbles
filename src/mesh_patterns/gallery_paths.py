"""
Project gallery layout and versioned creation naming.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GALLERY_DIR = PROJECT_ROOT / "gallery"
SHAPES_DIR = GALLERY_DIR / "shapes"
PATTERNS_DIR = GALLERY_DIR / "patterns"
CREATIONS_DIR = GALLERY_DIR / "creations"


def ensure_gallery_dirs() -> None:
    for directory in (SHAPES_DIR, PATTERNS_DIR, CREATIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def next_run_number(shape: str, pattern: str) -> int:
    """
    Return the next run number for a shape/pattern pair.
    """

    ensure_gallery_dirs()
    prefix = f"{shape}.{pattern}."
    highest = 0

    for path in CREATIONS_DIR.iterdir():
        if not path.name.startswith(prefix):
            continue

        parts = path.stem.split(".")
        if len(parts) != 3:
            continue

        try:
            run_number = int(parts[2])
        except ValueError:
            continue

        highest = max(highest, run_number)

    return highest + 1


def creation_basename(
    shape: str,
    pattern: str,
    run_number: int | None = None,
) -> str:
    """
    Build the versioned creation filename stem.
    """

    if run_number is None:
        run_number = next_run_number(shape, pattern)

    return f"{shape}.{pattern}.{run_number:03d}"


def creation_path(
    shape: str,
    pattern: str,
    *,
    run_number: int | None = None,
) -> Path:
    """
    Build a versioned output path in gallery/creations.
    """

    basename = creation_basename(shape, pattern, run_number=run_number)
    return CREATIONS_DIR / f"{basename}.stl"
