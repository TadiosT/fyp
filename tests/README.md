# Tests

```
tests/
├── conftest.py            shared fixtures: scratch DB, fake secrets, synthetic lab, patched configs
├── support/helpers.py     synthetic data builders (desks, sensors, logs, images, videos)
├── unit/                  pure Python, no Streamlit UI, fast (~5 s)
│   ├── simulation/        wc_sim engine, session simulator, seat snapping, seating zones
│   ├── data_io/           SQLite store, Netatmo client + failover, preloader, seed generator
│   │                      (not named `data/`: pytest's norecursedirs skips any dir with that name)
│   ├── cv/                lecture-hall homography / cropping / detection, Plotly floor map
│   ├── workers/           lab simulator loop, sensor worker loop, launcher
│   └── dashboard/         lab_dashboard helpers
└── integration/           Streamlit AppTest renders of pages and calibration tools
                           (marker `integration` is applied automatically by directory)
```

All tests are hermetic: a session fixture seeds a small synthetic lab into a temp
directory and the app's `LabConfig`s are re-pointed at it, so nothing under `data/`
is read or written and the suite passes on a clone with no data or model files.

## Running

| Command | What runs |
|---|---|
| `venv/bin/pytest` | everything, with a coverage table (default) |
| `venv/bin/pytest tests/unit` | unit tests only |
| `venv/bin/pytest -m "not integration"` | same as above, by marker |
| `venv/bin/pytest -m "not yolo"` | CI-safe: skips the three tests that need `models/yolov8s.pt` + `data/lecture_video.mp4` (they also self-skip when the files are missing) |
| `venv/bin/pytest --no-cov -q` | quickest feedback loop |

Config: `pytest.ini` (markers, warning filters, default options) and `.coveragerc`
(coverage sources; calibrator scripts and shims are excluded).
