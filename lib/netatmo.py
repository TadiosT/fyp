"""Netatmo Smart Indoor Air Quality Monitor (Home Coach) client.

One public function: ``fetch_homecoach_snapshot()`` — returns a normalised
reading dict (temperature, humidity, CO₂, noise, timestamp) suitable for the
home page tile, with all error states caught and surfaced as ``ok=False``
plus a human-readable ``error`` string.

Credentials live in ``.streamlit/secrets.toml`` under ``[netatmo]``:

    [netatmo]
    access_token = "..."
    device_id    = ""   # optional; empty ⇒ first device returned

The Netatmo OAuth2 access token expires (default ~3 h). When that happens the
API returns 401 and this client surfaces a clear warning — caller decides
whether to refresh. No refresh-token flow is implemented yet.

Note on cache invalidation: ``@st.cache_data(ttl=300)`` keeps one shared
snapshot per session for 5 minutes. If you swap the token in
``secrets.toml`` mid-session, either rerun the app or click "Clear cache"
in Streamlit's developer menu — otherwise the stale failure is sticky for
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


if __name__ == "__main__":
    # Quick smoke test:  venv/bin/python lib/netatmo.py
    # Reads secrets.toml via Streamlit's secrets loader (works outside a running
    # Streamlit server too, as long as .streamlit/secrets.toml exists).
    snap = fetch_homecoach_snapshot()
    print(f"[{time.strftime('%H:%M:%S')}] netatmo snapshot:")
    for k, v in snap.items():
        print(f"  {k:>12}: {v}")
