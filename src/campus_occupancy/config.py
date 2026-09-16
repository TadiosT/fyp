"""Per-lab configuration.

* ``LabConfig`` instances drive the dashboard pages and the boot-time preloader.
* ``SimConfig`` instances give the seed tool and the live simulator their paths.

Paths are relative to the project root (see ``campus_occupancy.paths``); the
entry points and workers make sure the working directory is the project root.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LabConfig:
    title: str
    data_file: str
    coords_file: str
    floor_plan: str
    calibrate_hint: str
    map_height: int = 650
    sensor_positions: str | None = None
    history_file: str | None = None
    missing_data_hint: str = "Ensure the simulator has been started via launcher.py."


@dataclass(frozen=True)
class SimConfig:
    name: str
    desks_csv: str            # pc_id, x_px, y_px[, zone]
    log_csv: str              # per-desk per-minute rows (rolling 48 h)
    hourly_csv: str           # hourly state counts (90 days)
    sensors_csv: str | None = None   # virtual sensors, optional


SEED_CMD = "python -m campus_occupancy.tools.generate_lab_seed"
CALIBRATE_CMD = "streamlit run src/campus_occupancy/tools"

WC_CFG = LabConfig(
    title="White City Computing Labs - Occupancy Dashboard",
    data_file="data/wc_mock_access_logs.csv",
    coords_file="data/wc_workstation_coordinates.csv",
    floor_plan="assets/wc_lab.png",
    calibrate_hint=f"{CALIBRATE_CMD}/calibrate_wc_lab.py",
    map_height=1100,
    sensor_positions="data/wc_sensor_positions.csv",
    history_file="data/wc_occupancy_hourly.csv",
    missing_data_hint=(f"Calibrate desks with `{CALIBRATE_CMD}/calibrate_wc_lab.py`, "
                       f"then seed data with `{SEED_CMD} --lab wc --reset`."),
)

HUXLEY_CFG = LabConfig(
    title="Huxley Computing Labs - Occupancy Dashboard",
    data_file="data/huxley_mock_access_logs_with_users.csv",
    coords_file="data/workstation_coordinates.csv",
    floor_plan="assets/huxley_lab_floor_plan.jpg",
    calibrate_hint=f"{CALIBRATE_CMD}/calibrate_coordinates.py",
    history_file="data/huxley_occupancy_hourly.csv",
    missing_data_hint=f"Seed data with `{SEED_CMD} --lab huxley --reset`.",
)

PAGE_CONFIGS = {"wc": WC_CFG, "huxley": HUXLEY_CFG}

SIM_CONFIGS = {
    "wc": SimConfig(
        name="wc",
        desks_csv="data/wc_workstation_coordinates.csv",
        log_csv="data/wc_mock_access_logs.csv",
        hourly_csv="data/wc_occupancy_hourly.csv",
        sensors_csv="data/wc_sensor_positions.csv",
    ),
    "huxley": SimConfig(
        name="huxley",
        desks_csv="data/workstation_coordinates.csv",
        log_csv="data/huxley_mock_access_logs_with_users.csv",
        hourly_csv="data/huxley_occupancy_hourly.csv",
    ),
}
