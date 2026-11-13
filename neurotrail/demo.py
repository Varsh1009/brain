"""
Sample account for demos: 60 days of SYNTHETIC watch data and Speed Sprint plays.

Clearly labelled as sample data in the app (the player name starts with "Sample"). It exists so a
judge can see the "Your week, your brain" screen in seconds without wearing a watch for weeks.

Honesty rule: the synthetic Sprint scores follow a practice curve plus random noise and do NOT
depend on sleep. So whatever "your Sprint after good sleep" shows for the sample account is noise,
which is exactly what its uncertainty range should show. The NHANES findings on the same screen
are real (models/train_device.py); only this sample person is made up.

Story: weeks 1-4 irregular sleep and ~4,500 steps a day; weeks 5-8 after starting NeuroTrail,
a steadier bedtime and steps rising toward ~7,500.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np

from neurotrail import db, wearables

DAYS = 60


def sample_days(end: date | None = None, seed: int = 7) -> tuple[dict, list[tuple[str, int]]]:
    rng = np.random.default_rng(seed)
    end = end or date.today()
    spans, days, plays = [], {}, []
    for i in range(DAYS):
        d = end - timedelta(days=DAYS - 1 - i)
        later = i >= DAYS // 2
        spread = 25 if later else 160                     # minutes of bedtime wobble
        bed = datetime.combine(d - timedelta(days=1), datetime.min.time()) + timedelta(
            hours=23 if later else 23.6, minutes=float(rng.normal(0, spread)))
        sleep_len = float(rng.normal(7.4 if later else 6.6, 0.4 if later else 1.3))
        wake = bed + timedelta(hours=min(max(sleep_len, 4.0), 10.0))
        spans.append((bed, wake))
        base = 4500 + (3000 * min(1.0, (i - DAYS // 2) / 20) if later else 0)
        total = max(800.0, float(rng.normal(base, 900)))
        prof = np.zeros(24)
        awake = list(range(min(wake.hour + 1, 12), 22))
        prof[awake] = rng.gamma(2.0, 1.0, len(awake))
        if later:                                          # a morning walk at a steady time
            prof[9] += 4
        prof = prof / prof.sum() * total
        days[d.isoformat()] = {"steps": round(total), "steps_h": [round(x) for x in prof],
                               "walking_speed": round(float(rng.normal(1.08 + (0.04 if later else 0), 0.05)), 2),
                               "resting_hr": round(float(rng.normal(67 - (2 if later else 0), 2.5)), 1),
                               "hrv_ms": round(float(rng.normal(31 + (3 if later else 0), 6)), 1)}
        if rng.random() < 0.45:                            # plays Speed Sprint ~3x a week
            n = len(plays)
            practice = 38 + 8 * (1 - np.exp(-n / 3))       # quick early practice gains, then flat
            plays.append((d.isoformat(), int(round(practice + rng.normal(0, 3)))))
    for d, hours in wearables.asleep_hours(spans).items():
        if d in days:
            days[d]["asleep_h"] = hours
    for bed, wake in spans:
        k = wake.date().isoformat()
        if k in days:
            days[k]["sleep_hours"] = round((wake - bed).total_seconds() / 3600, 2)
    return days, plays


def seed_player(name: str) -> None:
    """Fill an existing (new) player with the sample data."""
    days, plays = sample_days()
    db.put_days(name, days, "sample data (synthetic)")
    for d, score in plays:
        db.add_round(name, d, score, int(abs(np.random.default_rng(len(d)).normal(1, 1))), 5)
