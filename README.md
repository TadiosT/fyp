# Campus Occupancy Dashboard

Streamlit dashboard showing live occupancy for the Huxley computing labs, Lecture
Hall 144 (YOLO person detection on a video feed) and the White City computing
labs (with virtual air-quality sensors), plus real Netatmo Home Coach readings
with an offline failover.

## Layout

```
.
├── app.py                     Streamlit entry point        →  streamlit run app.py
├── launcher.py                starts workers + frontend   →  python launcher.py
├── pyproject.toml             package metadata, dependencies, pytest + coverage config
├── src/campus_occupancy/      the application package
│   ├── app.py                 home page + navigation
│   ├── config.py              LabConfig / SimConfig and the per-lab settings
│   ├── paths.py               project-root anchors
│   ├── pages/                 Streamlit pages (Huxley, Lecture Hall, White City)
│   ├── simulation/            desk-occupancy engine, session simulator, seat snapping, zones
│   ├── data_io/               SQLite store, Netatmo client + failover, boot-time preloader
│   ├── cv/                    lecture-hall video / homography / YOLO helpers
│   ├── dashboard/             shared lab dashboard renderer and Plotly floor map
│   ├── workers/               launcher, lab simulators, Netatmo sensor worker
│   └── tools/                 calibration UIs and the seed generator
├── tests/                     pytest suite (see tests/README.md)
├── data/                      logs, coordinates, sensor DB (mostly generated; see .gitignore)
├── assets/                    floor plans
├── models/                    yolov8s.pt (not committed)
└── experimental_notebooks/  exploratory Colab work (occupancy heatmaps, mock-log prototyping)
```

## Setup

```bash
python -m venv venv
venv/bin/pip install -e ".[dev]"
venv/bin/python -m campus_occupancy.tools.generate_lab_seed --lab huxley
venv/bin/python -m campus_occupancy.tools.generate_lab_seed --lab wc
```

The per-minute occupancy logs and hourly history are generated rather than committed, so
seed both labs before the first run. Everything else works from that point on.

Two features need files this repository does not distribute:

| Feature | Needs |
|---|---|
| Lecture Hall 144 page | `data/lecture_video.mp4`, `models/yolov8s.pt` (pre-trained YOLOv8s from Ultralytics), and `data/homography_lecture.npz` produced by the homography calibrator |
| Live Netatmo readings | `.streamlit/secrets.toml` containing a `[netatmo]` block with `access_token` and `device_id` |

Without them the rest of the dashboard still runs, and the detector-dependent tests skip
themselves.

## Running

| Task | Command (from the project root) |
|---|---|
| Everything (workers + UI) | `python launcher.py` |
| UI only | `venv/bin/streamlit run app.py` |
| Seed a lab's mock data | `venv/bin/python -m campus_occupancy.tools.generate_lab_seed --lab wc --reset` |
| One simulator on its own | `venv/bin/python -m campus_occupancy.workers.lab_simulator --lab huxley` |
| Calibrate desks / seats / sensors / homography | `venv/bin/streamlit run src/campus_occupancy/tools/calibrate_<tool>.py` |
| Tests | `venv/bin/pytest` |
