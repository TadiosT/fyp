"""Process orchestrator entry point. Run from the project root:  python launcher.py

Works with any Python (e.g. conda's base): only the standard library is needed
here; the launcher then picks the project venv for every child process.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from campus_occupancy.workers.launcher import main  # noqa: E402

if __name__ == "__main__":
    main()
