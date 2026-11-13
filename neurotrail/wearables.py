"""
Import real data from your phone or watch.

Supported:
  * Apple Health export (export.zip from the Health app, or the export.xml inside it).
    Daily totals for steps, sleep, resting heart rate, heart-rate variability and walking speed.
    iPhone and Apple Watch both record steps; to avoid double counting we take, for each day,
    the source with the most steps (Apple's own approach is similar de-duplication).
  * A simple CSV from any app (Fitbit, Google Fit, Garmin exports): columns `date` plus any of
    `steps`, `sleep_hours`, `resting_hr`, `hrv_ms`, `walking_speed`.
Files are parsed in memory; only the daily numbers are stored.
"""
from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from xml.etree.ElementTree import iterparse

FIELDS = ("steps", "sleep_hours", "resting_hr", "hrv_ms", "walking_speed")
# Hourly series kept per day for the rhythm features (neurotrail/rhythm.py):
#   steps_h  : 24 hourly step counts (from the same source chosen for the daily total)
#   asleep_h : 24 values, 1 = asleep for >= 30 min of that clock hour, 0 = awake, None = unknown
HOURLY = ("steps_h", "asleep_h")
ASLEEP = {"HKCategoryValueSleepAnalysisAsleep", "HKCategoryValueSleepAnalysisAsleepCore",
          "HKCategoryValueSleepAnalysisAsleepDeep", "HKCategoryValueSleepAnalysisAsleepREM",
          "HKCategoryValueSleepAnalysisAsleepUnspecified"}


def _dt(s: str) -> datetime:
    return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")


def parse_apple(xml_stream) -> dict[str, dict]:
    steps = defaultdict(lambda: defaultdict(float))       # day -> source -> steps
    steps_h = defaultdict(lambda: defaultdict(lambda: [0.0] * 24))   # day -> source -> hourly
    intervals = defaultdict(list)                         # source -> [(start, end)] asleep
    sleep = defaultdict(lambda: defaultdict(float))       # night (end date) -> source -> hours
    avg = {k: defaultdict(list) for k in ("resting_hr", "hrv_ms", "walking_speed")}
    typemap = {"HKQuantityTypeIdentifierRestingHeartRate": "resting_hr",
               "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv_ms",
               "HKQuantityTypeIdentifierWalkingSpeed": "walking_speed"}
    for _, el in iterparse(xml_stream, events=("end",)):
        if el.tag != "Record":
            el.clear()
            continue
        t, a = el.get("type"), el.attrib
        try:
            if t == "HKQuantityTypeIdentifierStepCount":
                v, src = float(a["value"]), a.get("sourceName", "?")
                steps[a["startDate"][:10]][src] += v
                steps_h[a["startDate"][:10]][src][int(a["startDate"][11:13])] += v
            elif t == "HKCategoryTypeIdentifierSleepAnalysis" and a.get("value") in ASLEEP:
                s, e = _dt(a["startDate"]), _dt(a["endDate"])
                sleep[e.date().isoformat()][a.get("sourceName", "?")] += (e - s).total_seconds() / 3600
                intervals[a.get("sourceName", "?")].append((s, e))
            elif t in typemap:
                v = float(a["value"])
                if t.endswith("WalkingSpeed") and a.get("unit") == "km/hr":
                    v /= 3.6            # store m/s
                avg[typemap[t]][a["startDate"][:10]].append(v)
        except (KeyError, ValueError):
            pass
        el.clear()
    days: dict[str, dict] = defaultdict(dict)
    for d, src in steps.items():
        best = max(src, key=src.get)
        days[d]["steps"] = round(src[best])
        days[d]["steps_h"] = [round(x) for x in steps_h[d][best]]
    if intervals:
        best = max(intervals, key=lambda k: sum((e - s).total_seconds() for s, e in intervals[k]))
        for d, hours in asleep_hours(intervals[best]).items():
            days.setdefault(d, {})["asleep_h"] = hours
    for d, src in sleep.items():
        h = max(src.values())
        if 0.5 <= h <= 16:
            days[d]["sleep_hours"] = round(h, 2)
    for k, by_day in avg.items():
        for d, vals in by_day.items():
            days[d][k] = round(sum(vals) / len(vals), 2)
    return dict(days)


def asleep_hours(spans: list[tuple[datetime, datetime]]) -> dict[str, list]:
    """Asleep intervals -> per-day 24-hour states. A clock hour with >= 30 min asleep is 1.
    Days between the first and last tracked night count as tracked (awake = 0); outside
    that range nothing is known."""
    if not spans:
        return {}
    mins = defaultdict(float)                     # (day, hour) -> asleep minutes
    for s, e in spans:
        t = s
        while t < e:
            nxt = min(e, (t.replace(minute=0, second=0) + timedelta(hours=1)))
            mins[(t.date().isoformat(), t.hour)] += (nxt - t).total_seconds() / 60
            t = nxt
    first = min(s for s, _ in spans).date()
    last = max(e for _, e in spans).date()
    out, d = {}, first
    while d <= last:
        k = d.isoformat()
        out[k] = [1 if mins.get((k, h), 0) >= 30 else 0 for h in range(24)]
        d += timedelta(days=1)
    return out


def _hm(day: str, hm: str) -> datetime:
    return datetime.strptime(f"{day} {hm.strip()[:5]}", "%Y-%m-%d %H:%M")


def parse_csv(text: str) -> dict[str, dict]:
    rows = csv.DictReader(io.StringIO(text))
    cols = {c.lower().strip(): c for c in (rows.fieldnames or [])}
    if "date" not in cols:
        raise ValueError("CSV needs a 'date' column (YYYY-MM-DD) plus steps / sleep_hours / resting_hr / hrv_ms / walking_speed "
                         "(optional: bedtime and waketime as HH:MM, for sleep regularity).")
    days, spans = {}, []
    for r in rows:
        d = (r[cols["date"]] or "").strip()[:10]
        try:
            datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            continue
        rec = {}
        for f in FIELDS:
            if f in cols and (r[cols[f]] or "").strip():
                try:
                    rec[f] = float(r[cols[f]])
                except ValueError:
                    pass
        hs = (r.get(cols.get("hourly_steps", ""), "") or "").strip()
        if hs:
            try:
                vals = [float(x) for x in hs.split(";")]
                if len(vals) == 24:
                    rec["steps_h"] = vals
                    rec.setdefault("steps", round(sum(vals)))
            except ValueError:
                pass
        bt, wt = (r.get(cols.get("bedtime", ""), "") or "").strip(), (r.get(cols.get("waketime", ""), "") or "").strip()
        if bt and wt:
            # the row's date is the morning you woke up; bedtime after noon belongs to the evening before
            try:
                wake = _hm(d, wt)
                bed = _hm(d, bt)
                if bed >= wake:
                    bed -= timedelta(days=1)
                if timedelta(hours=1) <= wake - bed <= timedelta(hours=16):
                    spans.append((bed, wake))
                    rec.setdefault("sleep_hours", round((wake - bed).total_seconds() / 3600, 2))
            except ValueError:
                pass
        if rec:
            days[d] = rec
    for d, hours in asleep_hours(spans).items():
        days.setdefault(d, {})["asleep_h"] = hours
    return days


def parse_upload(data: bytes, filename: str) -> dict[str, dict]:
    name = filename.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = next((n for n in z.namelist() if n.endswith("export.xml")), None)
            if xml is None:
                raise ValueError("No export.xml inside that zip. Export from the Health app: profile → Export All Health Data.")
            with z.open(xml) as f:
                return parse_apple(f)
    if name.endswith(".xml"):
        return parse_apple(io.BytesIO(data))
    if name.endswith(".csv"):
        return parse_csv(data.decode("utf-8", errors="replace"))
    raise ValueError("Upload Apple Health export.zip / export.xml, or a CSV.")


def summary(days: dict[str, dict], window: int = 14) -> dict:
    """Averages over the most recent `window` days that have data."""
    if not days:
        return {}
    last = max(days)
    cut = (datetime.fromisoformat(last) - timedelta(days=window - 1)).date().isoformat()
    recent = {d: v for d, v in days.items() if d >= cut}
    out = {"days_with_data": len(days), "from": min(days), "to": last, "window_days": window}
    for f in FIELDS:
        vals = [v[f] for v in recent.values() if f in v]
        if vals:
            out[f] = round(sum(vals) / len(vals), 2)
    return out
