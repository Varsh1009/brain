"""
"Your week, your brain": turns imported phone/watch data into brain-health feedback.

Three honest layers, each labelled in the app:

1. WHAT 2,534 OLDER AMERICANS SHOW (NHANES 2011-2014 wrist monitors, models/train_device.py).
   Device-measured sleep regularity, daily movement and sleep length vs. thinking speed (DSST),
   adjusted for age, sex, education, income AND health status. Association, not proof of cause.

2. WHERE YOU SIT. The same rhythm code (neurotrail/rhythm.py) runs on your watch's hourly data.
   Personal "points" are shown only for sleep regularity, because it is built the same way on
   both devices (asleep/awake per clock hour). Movement and sleep length are compared with
   evidence-based targets instead (UK Biobank steps; 7-9 h sleep), because a wrist
   accelerometer's MIMS units and "asleep" minutes don't convert cleanly to a phone's numbers.

3. YOUR OWN TREND. Speed Sprint scores after nights of 7-9 h sleep vs. other nights, from your
   own plays, with a bootstrap range, and "still learning" until there is enough data.
   Your first 3 plays are skipped (people improve quickly at first just from practice).
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

from neurotrail import rhythm

ART = Path(__file__).resolve().parents[1] / "models" / "artifacts"
WINDOW = 14
MIN_NIGHTS = 5
STEP_TARGETS = (3826, 9826)  # UK Biobank, Del Pozo Cruz et al. JAMA Neurol 2022 (PubMed 36036987)


@lru_cache(maxsize=1)
def assoc() -> dict:
    return json.loads((ART / "device_associations.json").read_text())


def _band(age: float | None) -> str:
    if age is None or age < 60:
        return "all"
    return "60-69" if age < 70 else "70-79" if age < 80 else "80+"


def percentile(feature: str, value: float, age: float | None) -> int | None:
    """Rough percentile among NHANES adults 60+ (by age band), linear between stored points."""
    ref = assoc()["reference"]
    vals = ref["values"].get(feature, {}).get(_band(age)) or ref["values"][feature]["all"]
    ps = ref["percentiles"]
    if value is None or not math.isfinite(value):
        return None
    if value <= vals[0]:
        return ps[0]
    if value >= vals[-1]:
        return ps[-1]
    return int(round(float(np.interp(value, vals, ps))))


def _series(days: dict, keys: list[str], field: str) -> np.ndarray:
    out = []
    for k in keys:
        v = days.get(k, {}).get(field)
        out.extend([np.nan] * 24 if not v or len(v) != 24 else
                   [np.nan if x is None else float(x) for x in v])
    return np.array(out, float)


def _keys(end: str, n: int) -> list[str]:
    e = date.fromisoformat(end)
    return [(e - timedelta(days=n - 1 - i)).isoformat() for i in range(n)]


def rhythm_window(days: dict, end: str, n: int = WINDOW) -> dict:
    keys = _keys(end, n)
    asleep = _series(days, keys, "asleep_h")
    steps = _series(days, keys, "steps_h")
    hours = np.tile(np.arange(24), n)
    f = rhythm.features(steps, asleep, hours)
    nights = int(sum(1 for k in keys if days.get(k, {}).get("asleep_h")))
    step_days = [days[k]["steps"] for k in keys if "steps" in days.get(k, {})]
    sleep_h = [days[k]["sleep_hours"] for k in keys if "sleep_hours" in days.get(k, {})]
    return {"from": keys[0], "to": keys[-1], "nights": nights, "step_days": len(step_days),
            "sri": f["sri"] if nights >= MIN_NIGHTS else math.nan,
            "is": f["is"], "iv": f["iv"], "sleep_mid": f["sleep_mid"],
            "steps": float(np.mean(step_days)) if step_days else math.nan,
            "sleep_hours": float(np.mean(sleep_h)) if sleep_h else math.nan}


def _clean(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(x, 2)


def _finding(f: str) -> dict:
    r = assoc()["features"][f]
    a, h = r["alone"], r["health_adjusted"]
    return {"label": r["label"], "points": round(a["points"], 1), "ci": [round(x, 1) for x in a["ci"]],
            "p": a["p_holm"], "health_points": round(h["points"], 1),
            "health_ci": [round(x, 1) for x in h["ci"]], "health_p": h["p"], "n": a["n"],
            "from": a["from"], "to": a["to"]}


def sprint_vs_sleep(days: dict, rounds: list[dict]) -> dict:
    """Your Speed Sprint after a 7-9 h night vs. other nights (skip first 3 plays)."""
    good, other = [], []
    for r in rounds[3:]:
        sh = days.get(r["date"], {}).get("sleep_hours")
        if sh is None:
            continue
        (good if 7 <= sh <= 9 else other).append(r["correct"])
    out = {"n_good": len(good), "n_other": len(other), "plays_used": len(good) + len(other)}
    if len(good) < 3 or len(other) < 3:
        out["status"] = "learning"
        return out
    rng = np.random.default_rng(0)
    g, o = np.array(good, float), np.array(other, float)
    boots = [rng.choice(g, len(g)).mean() - rng.choice(o, len(o)).mean() for _ in range(4000)]
    out |= {"status": "ok", "mean_good": round(float(g.mean()), 1), "mean_other": round(float(o.mean()), 1),
            "diff": round(float(g.mean() - o.mean()), 1),
            "ci": [round(float(np.percentile(boots, 2.5)), 1), round(float(np.percentile(boots, 97.5)), 1)]}
    return out


def weekly(days: dict, rounds: list[dict], end: str, weeks: int = 8) -> list[dict]:
    out = []
    for w in range(weeks - 1, -1, -1):
        stop = (date.fromisoformat(end) - timedelta(days=7 * w)).isoformat()
        r = rhythm_window(days, stop, 7)
        keys = set(_keys(stop, 7))
        sp = [x["correct"] for x in rounds if x["date"] in keys]
        if not (r["step_days"] or r["nights"] or sp):
            continue
        out.append({"week_end": stop, "steps": _clean(r["steps"]),
                    "sleep_hours": _clean(r["sleep_hours"]),
                    "sri": _clean(r["sri"]) if r["nights"] >= 4 else None,
                    "sprint": round(float(np.mean(sp)), 1) if sp else None})
    return out


# Extra watch signals. Each says where its evidence comes from; none gets personal "points".
SIGNALS = {
    "walking_speed": {
        "title": "Walking speed", "unit": "m/s", "better": "higher",
        "why": "Walking speed is one of the earliest everyday signs that thinking may be changing. "
               "In 8,699 adults over 60 (6 cohorts), people whose walking slowed over time had about "
               "2.2 times the dementia risk; slowing walk plus slipping memory, about 6.3 times.",
        "cite": "Tian et al., JAMA Netw Open 2020", "url": "https://pubmed.ncbi.nlm.nih.gov/32083691/",
        "watch": "A steady drop over months is worth mentioning to your doctor."},
    "hrv_ms": {
        "title": "Heart-rate variability", "unit": "ms", "better": "higher",
        "why": "Higher heart-rate variability (a calmer, more flexible nervous system) went with better "
               "attention, memory and thinking speed across 20 studies of 19,431 people without dementia.",
        "cite": "Forte et al., Front Neurosci 2019 (systematic review)",
        "url": "https://doi.org/10.3389/fnins.2019.00710",
        "watch": "It moves with sleep, stress, alcohol and illness, so look at weeks, not days."},
    "resting_hr": {
        "title": "Resting heart rate", "unit": "bpm", "better": "lower",
        "why": None, "cite": "Tested in NHANES 2011-2014 (seated exam pulse, same 2,534 people)",
        "url": None, "watch": "Good for heart fitness. We tested it for thinking speed and found no clear link."},
}


def signals(days: dict, end: str) -> list[dict]:
    """Recent 4 weeks vs the 4 weeks before, per extra watch signal (your own trend only)."""
    keys_now, keys_before = _keys(end, 28), _keys((date.fromisoformat(end) - timedelta(days=28)).isoformat(), 28)
    out = []
    for k, meta in SIGNALS.items():
        now = [days[d][k] for d in keys_now if k in days.get(d, {})]
        before = [days[d][k] for d in keys_before if k in days.get(d, {})]
        if len(now) < 5:
            continue
        dp = 2 if k == "walking_speed" else 1
        item = {"id": k, **meta, "now": round(float(np.mean(now)), dp), "days": len(now)}
        if len(before) >= 5:
            item["before"] = round(float(np.mean(before)), dp)
            item["change"] = round(item["now"] - item["before"], dp)
        out.append(item)
    if any(s["id"] == "resting_hr" for s in out):
        f = assoc()["features"].get("rhr")
        if f:
            for s in out:
                if s["id"] == "resting_hr":
                    a = f["health_adjusted"]
                    s["why"] = (f"In 2,534 older Americans, resting pulse showed no clear link with thinking speed "
                                f"({a['points']:+.1f} points, range {a['ci'][0]:+.1f} to {a['ci'][1]:+.1f}). "
                                "We show it anyway because it reflects heart fitness, which matters for the brain's blood supply.")
    return out


def build(days: dict, rounds: list[dict], age: float | None) -> dict:
    if not days:
        return {"has_data": False}
    end = max(days)
    r = rhythm_window(days, end)
    sri_f, mov_f, len_f = _finding("sri"), _finding("mims_day"), _finding("sleep_dev")
    a = assoc()["features"]["sri"]
    target = a["p75"]
    slope = a["health_adjusted"]["points"] / (a["health_adjusted"]["to"] - a["health_adjusted"]["from"])
    sri = r["sri"]
    sri_item = {"id": "sri", "title": "Sleep regularity", "value": _clean(sri), "target": round(target),
                "unit": "out of 100", "finding": sri_f,
                "percentile": percentile("sri", sri, age) if math.isfinite(sri) else None,
                "nights": r["nights"]}
    if math.isfinite(sri):
        gap = max(0.0, target - sri)
        sri_item["status"] = "good" if gap == 0 else "room"
        sri_item["your_points"] = round(slope * gap, 1)
    else:
        sri_item["status"] = "unknown"

    steps = r["steps"]
    mov_item = {"id": "steps", "title": "Daily movement", "value": _clean(steps), "unit": "steps a day",
                "targets": list(STEP_TARGETS), "finding": mov_f, "days": r["step_days"],
                "status": "unknown" if not math.isfinite(steps) else
                          "good" if steps >= STEP_TARGETS[1] else "room"}
    sh = r["sleep_hours"]
    len_item = {"id": "sleep_len", "title": "Sleep length", "value": _clean(sh), "unit": "hours a night",
                "targets": [7, 9], "finding": len_f,
                "status": "unknown" if not math.isfinite(sh) else "good" if 7 <= sh <= 9 else "room"}
    items = [sri_item, mov_item, len_item]
    next_up = next((i for i in items if i["status"] == "room"), None)
    return {
        "has_data": True, "window": {"from": r["from"], "to": r["to"], "days": WINDOW},
        "items": items, "next": next_up["id"] if next_up else None,
        "rhythm": {"is": _clean(r["is"]), "iv": _clean(r["iv"]), "sleep_mid": _clean(r["sleep_mid"])},
        "personal": sprint_vs_sleep(days, rounds),
        "weekly": weekly(days, rounds, end),
        "signals": signals(days, end),
        "nhanes_n": assoc()["n_device"],
    }
