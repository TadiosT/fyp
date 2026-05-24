import subprocess
import sys
import time


def main():
    print("🚀 Starting Huxley Labs Dashboard System...\n")

    # Start the Background Simulator
    print("-> Booting up Simulator (Background Worker)...")
    # sys.executable ensures it uses the exact same Python virtual environment (venv)
    simulator_process = subprocess.Popen([sys.executable, "simulator.py"])

    # Give the simulator 2 seconds to load the CSV and get comfortable
    time.sleep(2)

    # Start the Streamlit Dashboard
    print("-> Booting up Streamlit App (Frontend)...")
    # This translates to: python -m streamlit run app.py
    streamlit_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py"])

    print("\n System is fully online. Press Ctrl+C in this terminal to shut everything down cleanly.")

    children = [("Simulator", simulator_process), ("Streamlit", streamlit_process)]

    try:
        while True:
            time.sleep(1)
            dead = [name for name, p in children if p.poll() is not None]
            if dead:
                print(f"\n{', '.join(dead)} stopped unexpectedly (exit code(s): "
                      f"{[p.returncode for _, p in children if p.poll() is not None]}). Shutting down...")
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