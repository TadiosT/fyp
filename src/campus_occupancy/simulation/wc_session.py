"""Lab-session simulator for the White City labs.

Given the current desk state, live sensor readings per zone, and a session
description (attendance, duration), decide which zone hosts the session, which
is reserved for quiet study, and in what order the others open as overflow,
then play the session minute by minute so students are directed to an
additional lab only once the active one reaches the capacity threshold.

Sensor data enters the decision in three places:
* zone **roles** (best-air zone teaches, quietest low-occupancy zone is quiet study),
* the **overflow order** (better air first),
* **early overflow** when the measured CO₂ plus the predicted rise from the
  students already directed there would exceed ``CO2_LIMIT``.

Pure Python + pandas; no Streamlit, no I/O.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd

CO2_LIMIT = 1200
CO2_PER_FULL = 900          # ppm rise from empty → full (matches wc_sim)
DEFAULT_READING = {"co2": 420, "noise": 32, "temperature": 21.0, "humidity": 45}

LABELS = ["Teaching lab", "Overflow", "Quiet study", "Already in use", "Empty"]


@dataclass
class SessionResult:
    zone_table: pd.DataFrame          # zone, capacity, in_use_now, free_now, co2, noise, temperature, air_score, role
    roles: dict                       # {"teaching": z, "quiet": z, "overflow_order": [..]}
    timeline: pd.DataFrame            # index minute; columns zone → occupancy fraction (existing + session)
    session_counts: pd.DataFrame      # index minute; columns zone → session students seated
    events: list[dict]                # {"minute", "message", "level"}
    recommendations: list[str]
    unseated: int
    desks: pd.DataFrame = field(repr=False)   # pc_id, zone, x_px, y_px, in_use_now


def air_score(r: dict) -> float:
    return (0.5 * max(0.0, r["co2"] - 420) / CO2_PER_FULL
            + 0.3 * max(0.0, r["noise"] - 32) / 25
            + 0.2 * abs(r["temperature"] - 21) / 3)


def build_zone_table(desks: pd.DataFrame, zone_readings: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for zone, g in desks.groupby("zone"):
        r = {**DEFAULT_READING, **zone_readings.get(zone, {})}
        cap = len(g)
        in_use = int(g["in_use_now"].sum())
        rows.append({"zone": zone, "capacity": cap, "in_use_now": in_use, "free_now": cap - in_use,
                     "co2": r["co2"], "noise": r["noise"], "temperature": r["temperature"],
                     "air_score": round(air_score(r), 3)})
    return pd.DataFrame(rows).sort_values("zone").reset_index(drop=True)


def assign_roles(zt: pd.DataFrame) -> dict:
    if len(zt) == 1:
        z = zt.iloc[0]["zone"]
        return {"teaching": z, "quiet": None, "overflow_order": []}
    median = zt["air_score"].median()
    good = zt[zt["air_score"] <= median]
    teaching = good.sort_values(["free_now", "air_score"], ascending=[False, True]).iloc[0]["zone"]
    rest = zt[zt["zone"] != teaching]
    quiet = rest.sort_values(["noise", "in_use_now"]).iloc[0]["zone"]
    overflow = rest[rest["zone"] != quiet].sort_values("air_score")["zone"].tolist()
    return {"teaching": teaching, "quiet": quiet, "overflow_order": overflow + [quiet]}


def arrival_profile(n: int, window: int, rng: random.Random) -> list[int]:
    """Front-loaded arrivals: 60 % in the first third of the window, 30 % in the
    rest of it, 10 % trickling in over the following window."""
    a, b = max(1, window // 3), max(1, window)
    total = b + window
    per_min = [0] * (total + 1)
    buckets = [(0, a, 0.6), (a, b, 0.3), (b, total, 0.1)]
    remaining = n
    for start, end, share in buckets:
        count = int(round(n * share)) if (start, end) != buckets[-1][:2] else remaining
        count = min(count, remaining)
        remaining -= count
        span = max(1, end - start)
        for _ in range(count):
            per_min[start + rng.randrange(span)] += 1
    return per_min


def simulate_session(
    desks: pd.DataFrame,
    zone_readings: dict[str, dict],
    attendance: int = 250,
    duration_min: int = 120,
    threshold: float = 0.75,
    arrival_window_min: int = 15,
    seed: int = 1,
) -> SessionResult:
    rng = random.Random(seed)
    zt = build_zone_table(desks, zone_readings)
    roles = assign_roles(zt)
    order = [roles["teaching"]] + roles["overflow_order"]
    cap = dict(zip(zt["zone"], zt["capacity"]))
    base = dict(zip(zt["zone"], zt["in_use_now"]))
    co2_now = dict(zip(zt["zone"], zt["co2"]))

    session = {z: 0 for z in order}
    open_idx = 0
    events: list[dict] = [{"minute": 0, "level": "info",
                           "message": f"Session opens in Zone {roles['teaching']} "
                                      f"(teaching lab; {cap[roles['teaching']] - base[roles['teaching']]} desks free)."}]
    arrivals = arrival_profile(attendance, arrival_window_min, rng)
    unseated = 0
    tl_rows, sc_rows = [], []

    def fill(z):  # occupancy fraction incl. existing users
        return (base[z] + session[z]) / cap[z]

    def predicted_co2(z):
        return co2_now[z] + CO2_PER_FULL * (session[z] / cap[z])

    def can_take(z):
        return fill(z) < threshold and predicted_co2(z) <= CO2_LIMIT and base[z] + session[z] < cap[z]

    for m in range(duration_min + 1):
        for _ in range(arrivals[m] if m < len(arrivals) else 0):
            placed = False
            while not placed:
                z = order[open_idx]
                if can_take(z):
                    session[z] += 1
                    placed = True
                elif open_idx + 1 < len(order):
                    nxt = order[open_idx + 1]
                    reason = (f"predicted CO₂ {predicted_co2(z):.0f} ppm would exceed {CO2_LIMIT}"
                              if fill(z) < threshold else f"reached {fill(z):.0%}")
                    level = "warning" if nxt == roles["quiet"] else "info"
                    tag = " ⚠️ quiet-study zone opened" if nxt == roles["quiet"] else ""
                    events.append({"minute": m, "level": level,
                                   "message": f"Zone {z} {reason} → opened Zone {nxt}{tag}."})
                    open_idx += 1
                else:
                    # everything at threshold: seat in the zone with most physical space
                    z = max(order, key=lambda q: cap[q] - base[q] - session[q])
                    if base[z] + session[z] < cap[z]:
                        session[z] += 1
                    else:
                        unseated += 1
                    placed = True
        if m == duration_min:
            for z in session:
                session[z] = 0
            events.append({"minute": m, "level": "info", "message": "Session ends; all seats released."})
        elif m > int(duration_min * 0.75):
            for z in session:
                leaving = sum(1 for _ in range(session[z]) if rng.random() < 0.02)
                session[z] -= leaving
        tl_rows.append({"minute": m, **{z: round(fill(z), 4) for z in order}})
        sc_rows.append({"minute": m, **{z: session[z] for z in order}})

    timeline = pd.DataFrame(tl_rows).set_index("minute")
    counts = pd.DataFrame(sc_rows).set_index("minute")
    zt["role"] = zt["zone"].map(lambda z: "Teaching lab" if z == roles["teaching"]
                                else "Quiet study" if z == roles["quiet"] else "Overflow")

    t, q = roles["teaching"], roles["quiet"]
    tr = zt.set_index("zone")
    recs = [f"**Busy lab → Zone {t}**: {tr.loc[t, 'free_now']} of {tr.loc[t, 'capacity']} desks free, "
            f"CO₂ {tr.loc[t, 'co2']} ppm, noise {tr.loc[t, 'noise']} dB (best air among the large zones)."]
    if q is not None:
        recs.append(f"**Quiet study → Zone {q}**: quietest at {tr.loc[q, 'noise']} dB with "
                    f"{tr.loc[q, 'free_now']} desks free; only used for overflow as a last resort.")
    if roles["overflow_order"][:-1] if q else roles["overflow_order"]:
        chain = " → ".join(f"Zone {z}" for z in (roles["overflow_order"][:-1] if q else roles["overflow_order"]))
        recs.append(f"**Overflow order** (by air quality): {chain}.")
    peak = counts.sum(axis=1).max()
    used = [z for z in order if counts[z].max() > 0]
    recs.append(f"Peak {int(peak)} students seated across {len(used)} zone(s): "
                + ", ".join(f"Zone {z}" for z in used) + (f"; {unseated} could not be seated." if unseated else "."))

    return SessionResult(zone_table=zt, roles=roles, timeline=timeline, session_counts=counts,
                         events=events, recommendations=recs, unseated=unseated, desks=desks)


def desk_states(result: SessionResult, minute: int) -> pd.Series:
    """pc_id → label at a given minute (session desks filled in pc_id order per zone)."""
    m = min(max(0, minute), result.session_counts.index.max())
    labels = {}
    role_of = dict(zip(result.zone_table["zone"], result.zone_table["role"]))
    for zone, g in result.desks.sort_values("pc_id").groupby("zone"):
        n = int(result.session_counts.loc[m, zone]) if zone in result.session_counts.columns else 0
        free = g[~g["in_use_now"]]
        for pc in g[g["in_use_now"]]["pc_id"]:
            labels[pc] = "Already in use"
        for i, pc in enumerate(free["pc_id"]):
            labels[pc] = role_of.get(zone, "Overflow") if i < n else "Empty"
    return pd.Series(labels, name="label")
