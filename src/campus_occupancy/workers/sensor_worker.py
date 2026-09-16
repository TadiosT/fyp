"""Background poller that fetches one Netatmo Home Coach snapshot every 5 min,
writes it to the local SQLite store, and prunes rows older than 3 months.

Run:  python -m campus_occupancy.workers.sensor_worker   (the launcher does this).

The home page / lab pages read the same DB through
``campus_occupancy.data_io.netatmo.get_air_quality_data`` as an offline
failover when the live Netatmo call can't get through.
"""
from __future__ import annotations

import logging
import os
import time

from campus_occupancy.data_io.database import init_db, prune_older_than, save_reading
from campus_occupancy.data_io.netatmo import fetch_homecoach_snapshot
from campus_occupancy.paths import PROJECT_ROOT

POLL_INTERVAL_S = 300   # 5 min
RETENTION_DAYS = 90     # 3 months
LOCATION_FALLBACK = "Unknown"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] sensor_worker %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sensor_worker")


def resolve_location_name(snap: dict) -> str:
    """Prefer an explicit secrets-side override, then the device's station_name,
    then a hardcoded fallback so the row still has *something* to filter on."""
    try:
        import streamlit as st  # local import, only reached inside the loop
        configured = st.secrets.get("netatmo", {}).get("location_name", "")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
    except Exception:
        pass  # secrets unreadable in this context, fall through

    return snap.get("device_name") or LOCATION_FALLBACK


def tick() -> None:
    """One poll iteration: fetch -> save -> prune."""
    # Bypass the @st.cache_data wrapper on fetch_homecoach_snapshot; in a
    # plain Python subprocess we want a fresh hit every 5 min, not Streamlit's
    # session-scoped cache.
    raw_fetch = getattr(fetch_homecoach_snapshot, "__wrapped__", fetch_homecoach_snapshot)
    snap = raw_fetch()

    if snap["ok"]:
        location = resolve_location_name(snap)
        save_reading(snap, location_name=location)
        log.info(
            "saved %s @ %s (T=%s°C, H=%s%%, CO2=%s ppm, N=%s dB)",
            location, snap["time_utc"],
            snap["temperature"], snap["humidity"], snap["co2"], snap["noise"],
        )
    else:
        log.warning("netatmo unhealthy: %s", snap["error"])

    removed = prune_older_than(days=RETENTION_DAYS)
    if removed:
        log.info("pruned %d rows older than %d days", removed, RETENTION_DAYS)


def main() -> None:
    # st.secrets resolves .streamlit/secrets.toml relative to the working
    # directory, so anchor to the project root whatever launched us.
    os.chdir(PROJECT_ROOT)
    log.info("starting; poll every %ds, retain %dd", POLL_INTERVAL_S, RETENTION_DAYS)
    init_db()
    while True:
        try:
            tick()
        except KeyboardInterrupt:
            log.info("interrupted; shutting down")
            break
        except Exception:
            log.exception("tick failed; continuing")
        try:
            time.sleep(POLL_INTERVAL_S)
        except KeyboardInterrupt:
            log.info("interrupted during sleep; shutting down")
            break


if __name__ == "__main__":
    main()
