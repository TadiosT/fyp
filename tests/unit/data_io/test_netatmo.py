from datetime import datetime, timedelta, timezone

import pytest
import requests

import campus_occupancy.data_io.netatmo as netatmo
from campus_occupancy.data_io.database import save_reading


def raw_fetch():
    fn = netatmo.fetch_homecoach_snapshot
    return getattr(fn, "__wrapped__", fn)


class FakeResponse:
    def __init__(self, status=200, payload=None, bad_json=False):
        self.status_code = status
        self._payload = payload
        self._bad = bad_json

    def json(self):
        if self._bad:
            raise ValueError("no json")
        return self._payload


@pytest.fixture
def http(monkeypatch):
    """Install a fake requests.get; returns a recorder with .calls and .response."""
    rec = type("Rec", (), {})()
    rec.calls = []
    rec.response = FakeResponse(200, {"body": {"devices": []}})
    rec.raise_exc = None

    def fake_get(url, headers=None, timeout=None):
        rec.calls.append({"url": url, "headers": headers, "timeout": timeout})
        if rec.raise_exc:
            raise rec.raise_exc
        return rec.response

    monkeypatch.setattr(netatmo.requests, "get", fake_get)
    return rec


DEVICE = {"_id": "70:ee:50:aa", "station_name": "Bedroom",
          "dashboard_data": {"Temperature": 20.7, "Humidity": 63, "CO2": 651, "Noise": 39, "time_utc": 1780463254}}


@pytest.mark.parametrize("token", ["REPLACE_WITH_REAL_ACCESS_TOKEN", "", "   ", None])
def test_placeholder_or_missing_token_makes_no_call(http, fake_secrets, token):
    fake_secrets({"access_token": token} if token is not None else {})
    out = raw_fetch()()
    assert out["ok"] is False and out["error"] == "Netatmo access token not configured."
    assert http.calls == []


def test_no_secrets_section(http, fake_secrets):
    fake_secrets(None)
    assert raw_fetch()()["error"] == "Netatmo access token not configured."


@pytest.mark.parametrize("exc,msg", [
    (requests.ConnectionError(), "Netatmo API unreachable."),
    (requests.Timeout(), "Netatmo API unreachable."),
    (requests.TooManyRedirects(), "Netatmo request failed: TooManyRedirects."),
])
def test_network_errors(http, fake_secrets, exc, msg):
    fake_secrets({"access_token": "tok"})
    http.raise_exc = exc
    assert raw_fetch()()["error"] == msg


@pytest.mark.parametrize("status,msg", [(401, "Netatmo token expired or unauthorised."),
                                        (403, "Netatmo token expired or unauthorised."),
                                        (500, "Netatmo API returned 500.")])
def test_http_status_errors(http, fake_secrets, status, msg):
    fake_secrets({"access_token": "tok"})
    http.response = FakeResponse(status)
    out = raw_fetch()()
    assert out["error"] == msg and out["temperature"] is None


def test_malformed_json_and_no_devices(http, fake_secrets):
    fake_secrets({"access_token": "tok"})
    http.response = FakeResponse(200, bad_json=True)
    assert raw_fetch()()["error"] == "Netatmo API returned malformed JSON."
    http.response = FakeResponse(200, {"body": {}})
    assert raw_fetch()()["error"] == "No Home Coach devices on this account."


def test_happy_path_maps_fields_and_sends_bearer(http, fake_secrets):
    fake_secrets({"access_token": "tok123"})
    http.response = FakeResponse(200, {"body": {"devices": [DEVICE]}})
    out = raw_fetch()()
    assert out == {"ok": True, "error": None, "device_name": "Bedroom", "temperature": 20.7,
                   "humidity": 63, "co2": 651, "noise": 39, "time_utc": 1780463254}
    call = http.calls[0]
    assert call["url"] == netatmo.NETATMO_URL and call["headers"]["Authorization"] == "Bearer tok123"
    assert call["timeout"] == netatmo.REQUEST_TIMEOUT_S


def test_device_id_selection(http, fake_secrets):
    other = {**DEVICE, "_id": "zz", "station_name": "Office"}
    http.response = FakeResponse(200, {"body": {"devices": [DEVICE, other]}})
    fake_secrets({"access_token": "tok", "device_id": "zz"})
    assert raw_fetch()()["device_name"] == "Office"
    fake_secrets({"access_token": "tok", "device_id": "missing"})
    assert raw_fetch()()["error"] == "Device id 'missing' not found on account."
    fake_secrets({"access_token": "tok"})
    assert raw_fetch()()["device_name"] == "Bedroom"


def test_missing_dashboard_fields_default_to_none(http, fake_secrets):
    fake_secrets({"access_token": "tok"})
    http.response = FakeResponse(200, {"body": {"devices": [{"module_name": "M", "dashboard_data": {"CO2": 500}}]}})
    out = raw_fetch()()
    assert out["ok"] and out["device_name"] == "M" and out["co2"] == 500 and out["temperature"] is None


# ─────────────────────── failover wrapper ───────────────────────

def _live(ok, **over):
    base = {"ok": ok, "error": None if ok else "Netatmo token expired or unauthorised.",
            "device_name": "Bedroom", "temperature": 20.0, "humidity": 60, "co2": 600, "noise": 35,
            "time_utc": int(datetime.now(timezone.utc).timestamp())}
    base.update(over)
    return base


def test_live_path(monkeypatch, scratch_db):
    monkeypatch.setattr(netatmo, "fetch_homecoach_snapshot", lambda: _live(True))
    out = netatmo.get_air_quality_data("Bedroom")
    assert out["source"] == "live" and out["cache_age"] is None and out["ok"]


def test_cache_path_uses_latest_row(monkeypatch, scratch_db):
    monkeypatch.setattr(netatmo, "fetch_homecoach_snapshot", lambda: _live(False))
    when = datetime.now(timezone.utc) - timedelta(minutes=7)
    save_reading(_live(True, co2=999, time_utc=int(when.timestamp())), "Bedroom")
    out = netatmo.get_air_quality_data("Bedroom")
    assert out["source"] == "cache" and out["ok"] and out["co2"] == 999
    assert out["error"] == "Netatmo token expired or unauthorised."
    assert 6 * 60 <= out["cache_age"] <= 9 * 60
    assert out["time_utc"] == int(when.replace(microsecond=0).timestamp())


def test_none_path_when_no_cached_row(monkeypatch, scratch_db):
    monkeypatch.setattr(netatmo, "fetch_homecoach_snapshot", lambda: _live(False))
    out = netatmo.get_air_quality_data("Bedroom")
    assert out["source"] == "none" and not out["ok"] and out["cache_age"] is None


def test_cache_path_survives_db_failure(monkeypatch):
    monkeypatch.setattr(netatmo, "fetch_homecoach_snapshot", lambda: _live(False))
    import campus_occupancy.data_io.database as db
    monkeypatch.setattr(db, "latest_reading", lambda loc=None: (_ for _ in ()).throw(RuntimeError("db down")))
    out = netatmo.get_air_quality_data()
    assert out["source"] == "none"
