"""
Daily-rhythm features from hourly data. The SAME code runs on two sources:

  * NHANES 2011-2014 wrist ActiGraph (hourly MIMS activity + minutes flagged as sleep-wear), and
  * your own phone/watch export (hourly steps + hours marked asleep).

Why these features? They are the ones research links to brain health and that do not depend
on a device's units, so they can be compared across devices more fairly than raw "activity":

  * Sleep Regularity Index (SRI): how often your asleep/awake state at a given clock hour matches
    the same hour 24 h later. -100 (random) .. 100 (identical every day). Lower SRI was linked to
    higher dementia risk in UK Biobank (Yiallourou et al., Neurology 2024, PubMed 38165323).
    We compute it at hourly resolution (the original uses 30 s epochs), so values run higher than
    published epoch-level SRI; we only compare hourly SRI with hourly SRI.
  * Interdaily stability (IS, 0..1): how similar your 24 h activity pattern is from day to day.
  * Intradaily variability (IV, ~0..2): how fragmented activity is within the day (higher = more
    switching between rest and activity). Fragmentation tracks cognitive decline
    (Lim et al., Sleep 2012, PMC3321422).
  * Sleep midpoint (clock hour) and sleep length (hours).

IS / IV are ratios of variances, so they are unit-free; still, a wrist accelerometer and a phone's
step counter see movement differently, and the app says so when it compares you with NHANES.
"""
from __future__ import annotations

import math

import numpy as np


def interdaily_stability(x: np.ndarray, hour: np.ndarray) -> float:
    """IS = variance of the mean 24 h profile / total variance (Witting et al. 1990)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, hour = x[ok], np.asarray(hour)[ok]
    if len(x) < 72 or np.var(x) == 0:
        return math.nan
    prof = np.array([x[hour == h].mean() if np.any(hour == h) else np.nan for h in range(24)])
    if np.isnan(prof).any():
        return math.nan
    return float(np.sum((prof - x.mean()) ** 2) / 24 / np.var(x))


def intradaily_variability(x: np.ndarray) -> float:
    """IV = mean squared hour-to-hour change / variance (needs a continuous hourly series)."""
    x = np.asarray(x, float)
    d = np.diff(x)
    d = d[np.isfinite(d)]
    xv = x[np.isfinite(x)]
    if len(d) < 48 or np.var(xv) == 0:
        return math.nan
    return float(np.mean(d ** 2) / np.var(xv))


def sleep_regularity(asleep: np.ndarray) -> float:
    """Hourly SRI: 200 * P(state at t == state at t+24h) - 100, over pairs where both are known."""
    s = np.asarray(asleep, float)
    if len(s) < 48:
        return math.nan
    a, b = s[:-24], s[24:]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 48:
        return math.nan
    return float(200 * np.mean(a[ok] == b[ok]) - 100)


def sleep_midpoint(asleep: np.ndarray, hour: np.ndarray) -> float:
    """Circular mean clock hour of asleep hours (e.g. 3.5 = 3:30 AM)."""
    s = np.asarray(asleep, float)
    h = np.asarray(hour, float)[s == 1] + 0.5
    if len(h) < 3:
        return math.nan
    ang = h / 24 * 2 * math.pi
    m = math.atan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * math.pi) * 24
    return float(m % 24)


def features(activity: np.ndarray, asleep: np.ndarray, hour: np.ndarray) -> dict:
    """All rhythm features from aligned hourly arrays (NaN where unknown / not worn)."""
    return {
        "sri": sleep_regularity(asleep),
        "is": interdaily_stability(activity, hour),
        "iv": intradaily_variability(activity),
        "sleep_mid": sleep_midpoint(asleep, hour),
    }
