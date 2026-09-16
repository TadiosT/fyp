"""Netatmo Smart Indoor Air Quality Monitor (Home Coach) client.

One public function, ``fetch_homecoach_snapshot()``, returns a normalised
reading dict (temperature, humidity, CO₂, noise, timestamp) suitable for the
home page tile, with all error states caught and surfaced as ``ok=False``
plus a human-readable ``error`` string.

Credentials live in ``.streamlit/secrets.toml`` under ``[netatmo]``:

    [netatmo]
    access_token = "..."
    device_id    = ""   # optional; empty ⇒ first device returned

The Netatmo OAuth2 access token expires (default ~3 h). When that happens the
API returns 401 and this client surfaces a clear warning; the caller decides
whether to refresh. No refresh-token flow is implemented yet.

Note on cache invalidation: ``@st.cache_data(ttl=300)`` keeps one shared
snapshot per session for 5 minutes. If you swap the token in
``secrets.toml`` mid-session, either rerun the app or click "Clear cache"
in Streamlit's developer menu; otherwise the stale failure is sticky for
up to 5 minutes.
"""
from __future__ import annotations

import time

import requests
import streamlit as st

NETATMO_URL = "https://api.netatmo.com/api/gethomecoachsdata"
REQUEST_TIMEOUT_S = 10


def _empty(error: str) -> dict:
    return {
        "ok": False,
        "error": error,
        "device_name": None,
        "temperature": None,
        "humidity": None,
        "co2": None,
        "noise": None,
        "time_utc": None,
    }


def _pick_device(devices: list[dict], device_id: str) -> dict | None:
    """Return the device matching ``device_id`` (if given) or the first device."""
    if device_id:
        for d in devices:
            if d.get("_id") == device_id:
                return d
        return None
    return devices[0] if devices else None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_homecoach_snapshot() -> dict:
    """One Netatmo Home Coach reading, cached for 5 minutes.

    Returns a normalised dict regardless of success::

        {
            "ok":          bool,
            "error":       str | None,
            "device_name": str | None,
            "temperature": float | None,  # °C
            "humidity":    int   | None,  # %
            "co2":         int   | None,  # ppm
            "noise":       int   | None,  # dB
            "time_utc":    int   | None,  # epoch seconds of reading
        }
    """
    netatmo_cfg = {}
    try:
        netatmo_cfg = st.secrets.get("netatmo", {})
    except (FileNotFoundError, KeyError, AttributeError):
        # No secrets.toml on disk, or no [netatmo] section.
        netatmo_cfg = {}

    token = (netatmo_cfg.get("access_token") or "").strip()
    device_id = (netatmo_cfg.get("device_id") or "").strip()

    if not token or token.startswith("REPLACE_"):
        return _empty("Netatmo access token not configured.")

    try:
        response = requests.get(
            NETATMO_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=REQUEST_TIMEOUT_S,
        )
    except (requests.ConnectionError, requests.Timeout):
        return _empty("Netatmo API unreachable.")
    except requests.RequestException as e:
        return _empty(f"Netatmo request failed: {e.__class__.__name__}.")

    if response.status_code in (401, 403):
        return _empty("Netatmo token expired or unauthorised.")
    if response.status_code != 200:
        return _empty(f"Netatmo API returned {response.status_code}.")

    try:
        payload = response.json()
    except ValueError:
        return _empty("Netatmo API returned malformed JSON.")

    devices = (payload.get("body") or {}).get("devices") or []
    if not devices:
        return _empty("No Home Coach devices on this account.")

    device = _pick_device(devices, device_id)
    if device is None:
        return _empty(f"Device id '{device_id}' not found on account.")

    dash = device.get("dashboard_data") or {}
    return {
        "ok": True,
        "error": None,
        "device_name": device.get("station_name") or device.get("module_name"),
        "temperature": dash.get("Temperature"),
        "humidity": dash.get("Humidity"),
        "co2": dash.get("CO2"),
        "noise": dash.get("Noise"),
        "time_utc": dash.get("time_utc"),
    }


def get_air_quality_data(location_name: str | None = None) -> dict:
    """Live-first read with offline failover to the local SQLite store.

    Returns the same shape as :func:`fetch_homecoach_snapshot` plus two extra
    keys for the UI to drive its "live vs cache" badge:

    - ``source``:    ``"live"`` / ``"cache"`` / ``"none"``
    - ``cache_age``: seconds since the cached row's sensor timestamp, or
                      ``None`` when no cache fallback was used.

    Failover only kicks in when the live snapshot is not ``ok`` *and* a row
    exists in ``data/sensor_readings.db`` (written by ``sensor_worker.py``).
    The original ``error`` from the live call is preserved on cache results
    so the UI can explain *why* it's showing offline data.
    """
    # Local import keeps lib/netatmo.py importable in environments that don't
    # have sqlmodel yet (e.g. fresh checkouts before pip install). The live
    # path stays usable on its own.
    from datetime import datetime, timezone

    snap = fetch_homecoach_snapshot()
    if snap["ok"]:
        return {**snap, "source": "live", "cache_age": None}

    try:
        from campus_occupancy.data_io.database import latest_reading
        row = latest_reading(location_name)
    except Exception:
        # If the DB module can't load (missing dep, bad path, …) we still want
        # to surface the live error rather than crash. Treat as "no cache".
        row = None

    if row is None:
        return {**snap, "source": "none", "cache_age": None}

    cached_ts_utc = row.timestamp.replace(tzinfo=timezone.utc) \
        if row.timestamp.tzinfo is None else row.timestamp
    age_s = int((datetime.now(timezone.utc) - cached_ts_utc).total_seconds())

    return {
        "ok": True,
        "error": snap["error"],
        "device_name": row.device_name,
        "temperature": row.temperature,
        "humidity": row.humidity,
        "co2": row.co2,
        "noise": row.noise,
        "time_utc": int(cached_ts_utc.timestamp()),
        "source": "cache",
        "cache_age": age_s,
    }


if __name__ == "__main__":
    # Quick smoke test:  venv/bin/python lib/netatmo.py
    # st.secrets resolves .streamlit/secrets.toml relative to the *current*
    # working directory. PyCharm and similar IDEs often run from ~ or lib/,
    # which would silently fall back to "access token not configured." Force
    # CWD to the repo root so this works from any launcher.
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).resolve().parent.parent)

    snap = fetch_homecoach_snapshot()
    print(f"[{time.strftime('%H:%M:%S')}] netatmo snapshot:")
    for k, v in snap.items():
        print(f"  {k:>12}: {v}")
