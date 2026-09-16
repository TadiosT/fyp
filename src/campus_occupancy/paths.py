"""Filesystem anchors. Data, assets and models live at the project root, next
to `src/`; workers chdir to PROJECT_ROOT so the relative paths in
`campus_occupancy.config` resolve regardless of where they were launched from."""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]          # src/campus_occupancy → project root
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
PAGES_DIR = PACKAGE_DIR / "pages"
