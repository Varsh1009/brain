"""
Per-person wrist-monitor features for NHANES 2011-2014 adults 60+ who took the DSST.

Source files (CDC, public): PAXHD (device header: start time), PAXDAY (daily summaries),
PAXHR (hourly summaries). Activity is in MIMS units (Monitor-Independent Movement Summary,
John et al. 2019). Sleep is the NHANES algorithm's "sleep-wear" flag, not lab-measured sleep.

Valid-day rule (stated up front, applied before looking at any cognition result):
  * full days only (day 2..8; day 1 and 9 are partial),
  * at most 4 h of non-wear (>= 20 h worn),
  * at most 60 min of quality-flagged data,
  * a person needs >= 4 valid days.
Hourly features use the valid days only; an hour with >= 30 min of sleep-wear counts as asleep.

Output: models/artifacts/nhanes_device_features.csv (one row per person, no identifiers
beyond the public NHANES SEQN, which is already in the public files).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurotrail.nhanes import RAW, build_table  # noqa: E402
from neurotrail.rhythm import features  # noqa: E402

ART = ROOT / "models" / "artifacts"
log = logging.getLogger("device_features")
MAX_NONWEAR, MAX_QF, MIN_DAYS = 240, 60, 4


def _num(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.str.decode("utf-8").str.strip()
    return pd.to_numeric(s, errors="coerce")


def _clock(s: pd.Series) -> pd.Series:
    """b'12:30:00' -> seconds after midnight."""
    t = s.str.decode("utf-8").str.strip().str.split(":", expand=True).astype(float)
    return t[0] * 3600 + t[1] * 60 + t[2]


def cycle(sfx: str, keep: set[float]) -> pd.DataFrame:
    hd = pd.read_sas(RAW / f"PAXHD_{sfx}.xpt")
    hd = hd[hd["SEQN"].isin(keep) & (hd["PAXSTS"] == 1)]
    start = pd.Series(_clock(hd["PAXFTIME"]).to_numpy(), index=hd["SEQN"])

    day = pd.read_sas(RAW / f"PAXDAY_{sfx}.xpt")
    day = day[day["SEQN"].isin(start.index)].copy()
    day["d"] = _num(day["PAXDAYD"])
    day["valid"] = (day["d"].between(2, 8) & (day["PAXNWMD"] <= MAX_NONWEAR)
                    & (day["PAXQFD"] <= MAX_QF) & (day["PAXTMD"] >= 1440))
    v = day[day["valid"]]
    per = v.groupby("SEQN").agg(valid_days=("d", "size"), mims_day=("PAXMTSD", "mean"),
                                sleep_wear_h=("PAXSWMD", lambda x: x.mean() / 60),
                                wake_wear_h=("PAXWWMD", lambda x: x.mean() / 60))
    per = per[per["valid_days"] >= MIN_DAYS]
    valid_days = v[v["SEQN"].isin(per.index)].groupby("SEQN")["d"].apply(set).to_dict()

    rows = []
    for chunk in pd.read_sas(RAW / f"PAXHR_{sfx}.xpt", chunksize=500_000):
        chunk = chunk[chunk["SEQN"].isin(per.index)]
        if len(chunk):
            rows.append(chunk[["SEQN", "PAXDAYH", "PAXSSNHP", "PAXTMH", "PAXMTSH", "PAXSWMH",
                               "PAXNWMH", "PAXQFH"]])
    hr = pd.concat(rows, ignore_index=True)
    hr["d"] = _num(hr["PAXDAYH"])
    hr["hour"] = (((start.reindex(hr["SEQN"]).to_numpy() + hr["PAXSSNHP"] / 80) // 3600) % 24).astype(int)
    out = []
    for seqn, g in hr.groupby("SEQN", sort=False):
        g = g.sort_values("PAXSSNHP")
        ok = g["d"].isin(valid_days[seqn]) & (g["PAXTMH"] >= 60)
        bad = ~ok | (g["PAXNWMH"] > 30) | (g["PAXQFH"] > 30)
        act = g["PAXMTSH"].where(~bad).to_numpy(float)
        asleep = (g["PAXSWMH"] >= 30).astype(float).where(~bad).to_numpy(float)
        f = features(act, asleep, g["hour"].to_numpy())
        f["SEQN"] = seqn
        out.append(f)
    feats = pd.DataFrame(out).set_index("SEQN")
    res = per.join(feats, how="inner").reset_index()
    log.info("cycle %s: %d with device data, %d with >= %d valid days", sfx, len(start),
             len(res), MIN_DAYS)
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t = build_table()
    keep = set(t["SEQN"])
    df = pd.concat([cycle(s, keep) for s in ("G", "H")], ignore_index=True)
    df.to_csv(ART / "nhanes_device_features.csv", index=False)
    log.info("total %d of %d DSST takers 60+", len(df), len(t))
    log.info("\n%s", df.describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
