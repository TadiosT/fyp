"""Local SQLite store for sensor readings (real Netatmo + virtual White City sensors).

One module owns the engine, the model, and the read/write helpers used by
``workers/sensor_worker.py``, ``workers/lab_simulator.py``, ``tools/generate_lab_seed.py``,
``lib/netatmo.py`` and ``lib/lab_dashboard.py``.

Timestamps are stored as naive **UTC**. Use :func:`utc` to make a row's
timestamp tz-aware before doing arithmetic with it.

Journal mode is the classic rollback journal (``DELETE``), not WAL: WAL's
shared-memory file produced ``disk I/O error`` on this machine's synced
folder. Set ``FYP_DB_PATH`` to relocate the database if that ever recurs.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, TypeVar

import pandas as pd
from sqlalchemy import Engine, event
from sqlalchemy.exc import OperationalError
from sqlmodel import Field, Session, SQLModel, create_engine, select

DB_PATH = os.environ.get("FYP_DB_PATH", "data/sensor_readings.db")

T = TypeVar("T")


class SensorReading(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    device_name: Optional[str] = Field(default=None, index=True)
    location_name: str = Field(index=True)
    temperature: Optional[float] = None
    humidity: Optional[int] = None
    co2: Optional[int] = None
    noise: Optional[int] = None
    timestamp: datetime = Field(index=True)  # naive UTC


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        d = os.path.dirname(DB_PATH)
        if d:
            os.makedirs(d, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False}
        )

        @event.listens_for(_engine, "connect")
        def _pragmas(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA busy_timeout=5000")
            try:
                cur.execute("PRAGMA journal_mode=DELETE")
            except Exception:
                pass  # another process may hold the DB in WAL; harmless
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

    return _engine


def _retry(fn: Callable[[], T], attempts: int = 3, backoff: float = 0.5) -> T:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except OperationalError as e:
            last = e
            time.sleep(backoff * (i + 1))
    assert last is not None
    raise last


def utc(dt: datetime) -> datetime:
    """Return an aware UTC datetime: naive input is interpreted as UTC (the
    storage convention); aware input in any zone is converted to UTC."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def init_db() -> None:
    _retry(lambda: SQLModel.metadata.create_all(get_engine()))


def _row_from_snap(snap: dict, location_name: str) -> SensorReading:
    return SensorReading(
        device_name=snap.get("device_name"),
        location_name=location_name,
        temperature=snap.get("temperature"),
        humidity=snap.get("humidity"),
        co2=snap.get("co2"),
        noise=snap.get("noise"),
        timestamp=datetime.fromtimestamp(snap["time_utc"], tz=timezone.utc),
    )


def save_reading(snap: dict, location_name: str) -> None:
    """Insert one row from a Netatmo-shaped snapshot. No-op if not ok."""
    if not snap.get("ok") or snap.get("time_utc") is None:
        return

    def _do():
        with Session(get_engine()) as s:
            s.add(_row_from_snap(snap, location_name))
            s.commit()

    _retry(_do)


def save_readings_bulk(snaps: list[dict]) -> int:
    """Insert many snapshots in one transaction (location_name = device_name)."""
    rows = [_row_from_snap(sn, sn["device_name"]) for sn in snaps
            if sn.get("ok") and sn.get("time_utc") is not None]
    if not rows:
        return 0

    def _do():
        with Session(get_engine()) as s:
            s.add_all(rows)
            s.commit()

    _retry(_do)
    return len(rows)


def latest_reading(location_name: str | None = None) -> Optional[SensorReading]:
    def _do():
        with Session(get_engine()) as s:
            stmt = select(SensorReading)
            if location_name:
                stmt = stmt.where(SensorReading.location_name == location_name)
            stmt = stmt.order_by(SensorReading.timestamp.desc()).limit(1)
            return s.exec(stmt).first()

    return _retry(_do)


def readings_between(location_name: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Rows for one location in [start, end] (aware or naive-UTC datetimes)."""
    s_utc = utc(start).replace(tzinfo=None)
    e_utc = utc(end).replace(tzinfo=None)

    def _do():
        with Session(get_engine()) as s:
            stmt = (select(SensorReading)
                    .where(SensorReading.location_name == location_name)
                    .where(SensorReading.timestamp >= s_utc)
                    .where(SensorReading.timestamp <= e_utc)
                    .order_by(SensorReading.timestamp))
            rows = s.exec(stmt).all()
        return pd.DataFrame(
            [{"timestamp": utc(r.timestamp), "temperature": r.temperature,
              "humidity": r.humidity, "co2": r.co2, "noise": r.noise} for r in rows]
        )

    return _retry(_do)


def delete_readings(location_names: list[str], since: datetime | None = None) -> int:
    """Delete rows for the given locations (optionally only those at/after `since`)."""
    def _do():
        with Session(get_engine()) as s:
            stmt = select(SensorReading).where(SensorReading.location_name.in_(location_names))
            if since is not None:
                stmt = stmt.where(SensorReading.timestamp >= utc(since).replace(tzinfo=None))
            rows = s.exec(stmt).all()
            for r in rows:
                s.delete(r)
            s.commit()
            return len(rows)

    return _retry(_do)


def prune_older_than(days: int = 90) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).replace(tzinfo=None)

    def _do():
        with Session(get_engine()) as s:
            rows = s.exec(select(SensorReading).where(SensorReading.timestamp < cutoff)).all()
            for r in rows:
                s.delete(r)
            s.commit()
            return len(rows)

    return _retry(_do)
