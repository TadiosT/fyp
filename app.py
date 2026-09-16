"""Streamlit entry point. Run from the project root:  streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))   # works even without `pip install -e .`

from campus_occupancy.app import main  # noqa: E402

main()
