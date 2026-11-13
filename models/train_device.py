"""
How do device-measured activity, sleep and daily rhythm relate to thinking speed (DSST)?
NHANES 2011-2014 adults 60+ who wore the wrist monitor (features: build_device_features.py).

Pre-specified analysis (written before looking at results):
  * Outcome: DSST score. Survey-weighted least squares (MEC weights), cluster-robust SEs
    (strata x PSU), same as train_cognition.py.
  * Model A, one feature at a time + age, sex, education, income. Holm correction over the
    7 features (6 from the wrist monitor + exam resting pulse). This is the number the app shows.
  * Model B, all 6 features together + demographics (are they independent?).
  * Model C, Model B + health status (diabetes, stroke, heart disease, depression score, BMI,
    smoking, systolic BP). Poor health lowers both activity and test scores (reverse causation),
    so an effect that vanishes here is shown as "explained by health" in the app.
  * Contrast: 25th -> 75th percentile of each feature (direction: toward the healthier end).
  * Cross-validated R^2 gain of the device features over demographics (5 x 5-fold, linear).
  * Reference percentiles (all, and by age band) so a user's watch values can be placed.

Associations in a one-time survey are not proof of cause; the app says so.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurotrail.nhanes import DEMOGRAPHIC, build_table  # noqa: E402

ART = ROOT / "models" / "artifacts"
log = logging.getLogger("train_device")
HEALTH = ["diabetes", "stroke", "heart_disease", "phq9", "bmi", "current_smoker", "sbp", "bp_med"]

# name -> (label, healthier direction: +1 higher is better / -1 lower is better)
FEATS = {
    "mims_day": ("Daily movement (wrist monitor)", +1),
    "sri": ("Sleep regularity (same sleep hours each day)", +1),
    "is": ("Steady daily rhythm (same pattern each day)", +1),
    "iv": ("Broken-up rhythm (rest and activity switching)", -1),
    "sleep_dev": ("Sleep length away from 8 h", -1),
    "mid_dev": ("Sleep timing away from about 3 AM midpoint", -1),
    # Seated 60-second pulse at the NHANES exam (BPXPLS), same people. Watches report resting
    # heart rate in the same unit (beats per minute), measured differently (lowest steady rate).
    "rhr": ("Resting heart rate", -1),
}


def load() -> pd.DataFrame:
    t = build_table()
    dev = pd.read_csv(ART / "nhanes_device_features.csv")
    df = t.merge(dev, on="SEQN", how="inner")
    pulse = pd.concat([pd.read_sas(Path(__file__).resolve().parents[1] / "data" / "raw" / f"BPX_{c}.xpt")[["SEQN", "BPXPLS"]]
                       for c in ("G", "H")])
    df = df.merge(pulse, on="SEQN", how="left")
    df["rhr"] = df["BPXPLS"].where(df["BPXPLS"].between(30, 200))
    df["sleep_dev"] = (df["sleep_wear_h"] - 8).abs()
    mid = df["sleep_mid"]
    df["mid_dev"] = np.minimum((mid - 3) % 24, (3 - mid) % 24)
    return df


def wls(d: pd.DataFrame, cols: list[str]):
    d = d.dropna(subset=cols + ["dsst", "weight"])
    d = d[d["weight"] > 0]
    X = sm.add_constant(d[cols].astype(float))
    w = d["weight"] / d["weight"].mean()
    return sm.WLS(d["dsst"], X, weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["cluster"])[0]})


def contrast(fit, f: str, lo: float, hi: float, sign: int) -> dict:
    a, b = (lo, hi) if sign > 0 else (hi, lo)     # move toward the healthier end
    pts = fit.params[f] * (b - a)
    ci = fit.conf_int().loc[f].to_numpy() * (b - a)
    return {"points": float(pts), "ci": sorted(float(x) for x in ci), "p": float(fit.pvalues[f]),
            "from": float(a), "to": float(b), "n": int(fit.nobs)}


def cv_gain(df: pd.DataFrame, extra: list[str]) -> dict:
    d = df.dropna(subset=DEMOGRAPHIC + extra + ["dsst"])
    y = d["dsst"].to_numpy()
    gains = []
    for seed in range(5):
        p0, p1 = np.zeros(len(d)), np.zeros(len(d))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(d):
            for cols, p in ((DEMOGRAPHIC, p0), (DEMOGRAPHIC + extra, p1)):
                m = LinearRegression().fit(d.iloc[tr][cols], y[tr])
                p[te] = m.predict(d.iloc[te][cols])
        gains.append((r2_score(y, p0), r2_score(y, p1)))
    g = np.array(gains)
    return {"r2_demographics": float(g[:, 0].mean()), "r2_with_device": float(g[:, 1].mean()),
            "gain_mean": float((g[:, 1] - g[:, 0]).mean()),
            "gain_min": float((g[:, 1] - g[:, 0]).min()), "n": int(len(d))}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = load()
    q = {f: df[f].quantile([.25, .75]).tolist() for f in FEATS}
    res = {"n_device": int(len(df)), "features": {}}

    A = {f: wls(df, DEMOGRAPHIC + [f]) for f in FEATS}
    holm = multipletests([A[f].pvalues[f] for f in FEATS], method="holm")[1]
    B = wls(df, DEMOGRAPHIC + list(FEATS))
    C = wls(df, DEMOGRAPHIC + list(FEATS) + HEALTH)
    C1 = {f: wls(df, DEMOGRAPHIC + [f] + HEALTH) for f in FEATS}
    for (f, (label, sign)), ph in zip(FEATS.items(), holm):
        lo, hi = q[f]
        res["features"][f] = {
            "label": label, "healthier": "higher" if sign > 0 else "lower",
            "p25": lo, "p75": hi,
            "alone": contrast(A[f], f, lo, hi, sign) | {"p_holm": float(ph)},
            "together": contrast(B, f, lo, hi, sign),
            "health_adjusted": contrast(C1[f], f, lo, hi, sign),
            "together_health_adjusted": contrast(C, f, lo, hi, sign),
        }
    res["age_coef"] = float(A["sri"].params["age"])
    res["cv"] = cv_gain(df, list(FEATS))
    res["dsst_sd"] = float(df["dsst"].std())

    # reference percentiles for placing a user's watch values (all 60+, and by age band)
    pct = [5, 10, 25, 50, 75, 90, 95]
    df["band"] = pd.cut(df["age"], [59, 69, 79, 200], labels=["60-69", "70-79", "80+"])
    ref = {}
    for f in ("sri", "is", "iv", "sleep_wear_h", "sleep_mid", "mims_day", "rhr"):
        ref[f] = {"all": [float(df[f].quantile(p / 100)) for p in pct]}
        for b, g in df.groupby("band", observed=True):
            ref[f][str(b)] = [float(g[f].quantile(p / 100)) for p in pct]
    res["reference"] = {"percentiles": pct, "values": ref}

    (ART / "device_associations.json").write_text(json.dumps(res, indent=2))
    log.info("n=%d  CV R2 demographics %.3f -> +device %.3f (gain %.3f, min %.3f)", len(df),
             res["cv"]["r2_demographics"], res["cv"]["r2_with_device"], res["cv"]["gain_mean"],
             res["cv"]["gain_min"])
    for f, r in res["features"].items():
        a, t, h = r["alone"], r["together"], r["health_adjusted"]
        log.info("%-10s p25-p75 %.2f-%.2f | alone %+.2f [%+.2f,%+.2f] p=%.2g holm=%.2g | "
                 "together %+.2f p=%.2g | health-adj %+.2f [%+.2f,%+.2f] p=%.2g", f, r["p25"], r["p75"],
                 a["points"], *a["ci"], a["p"], a["p_holm"], t["points"], t["p"],
                 h["points"], *h["ci"], h["p"])


if __name__ == "__main__":
    main()
