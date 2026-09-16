"""Boots the background workers and the Streamlit frontend, and shuts them
down together. Invoked via the thin `launcher.py` at the project root."""
from __future__ import annotations

import os
import subprocess
import sys
import time

from campus_occupancy.paths import PROJECT_ROOT

ROOT = str(PROJECT_ROOT)


def pick_interpreter() -> str:
    """Prefer the project venv even if the launcher was started with another
    Python (e.g. conda's base); every worker needs the venv's packages."""
    venv_py = os.path.join(ROOT, "venv", "bin", "python")
    return venv_py if os.path.exists(venv_py) else sys.executable


def preflight(py: str) -> None:
    check = "import pandas, sqlalchemy, sqlmodel, ultralytics, streamlit, campus_occupancy"
    r = subprocess.run([py, "-c", check], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"Interpreter {py} is missing required packages:\n{r.stderr.strip()}")
        print('Install the project into the venv:  venv/bin/pip install -e ".[dev]"')
        sys.exit(1)


def main() -> None:
    os.chdir(ROOT)
    py = pick_interpreter()
    print("Starting Campus Occupancy Dashboard System...")
    print(f"   interpreter: {py}\n")
    preflight(py)

    print("-> Booting up Huxley Simulator (desks, wall-clock synced)...")
    simulator_process = subprocess.Popen([py, "-m", "campus_occupancy.workers.lab_simulator", "--lab", "huxley"])
    time.sleep(2)

    print("-> Booting up Sensor Worker (Netatmo poll -> SQLite)...")
    sensor_process = subprocess.Popen([py, "-m", "campus_occupancy.workers.sensor_worker"])

    print("-> Booting up White City Simulator (desks + virtual sensors)...")
    wc_process = subprocess.Popen([py, "-m", "campus_occupancy.workers.lab_simulator", "--lab", "wc"])

    print("-> Booting up Streamlit App (Frontend)...")
    streamlit_process = subprocess.Popen([py, "-m", "streamlit", "run", "app.py"])

    print("\n System is fully online. Press Ctrl+C in this terminal to shut everything down cleanly.")

    children = [
        ("Huxley Simulator", simulator_process),
        ("Sensor Worker", sensor_process),
        ("WC Simulator", wc_process),
        ("Streamlit", streamlit_process),
    ]

    try:
        while True:
            time.sleep(1)
            dead = [name for name, p in children if p.poll() is not None]
            if dead:
                codes = [p.returncode for _, p in children if p.poll() is not None]
                print(f"\n{', '.join(dead)} stopped unexpectedly (exit code(s): {codes}). Shutting down...")
                break
    except KeyboardInterrupt:
        print("\n\nShutdown signal received. Stopping services...")
    finally:
        for name, proc in children:
            if proc.poll() is not None:
                continue
            print(f"-> Terminating {name}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"   {name} did not exit in 5s; killing.")
                proc.kill()
                proc.wait()
        print("All systems offline. Goodbye!")


if __name__ == "__main__":
    main()
