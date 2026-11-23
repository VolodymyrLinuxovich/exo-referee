"""Data for the explorer (site/explore.html): every Kepler Object of Interest at its real sky position.

Columns: n kepoi name, k Kepler name, d NASA disposition (0 confirmed, 1 false positive, 2 candidate),
ra/dec, P period days, R radius Earth radii, T equilibrium K, S insolation Earth=1, Ts star K, Rs star Sun=1,
dep transit depth ppm, dur hours, p Redshift's planet score (unresolved candidates only).
"""
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
koi = pd.read_csv(ROOT / "exo/data/raw/koi.csv",
                  usecols=["kepoi_name", "kepler_name", "koi_disposition", "ra", "dec", "koi_period", "koi_prad",
                           "koi_teq", "koi_insol", "koi_steff", "koi_srad", "koi_depth", "koi_duration"])
score = pd.read_csv(ROOT / "exo/runs/unresolved_ranked.csv", usecols=["kepoi_name", "p_planet"]).set_index("kepoi_name").p_planet
code = {"CONFIRMED": 0, "FALSE POSITIVE": 1, "CANDIDATE": 2}


def num(x, nd):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


rows = []
for r in koi.itertuples(index=False):
    rows.append([r.kepoi_name, r.kepler_name if isinstance(r.kepler_name, str) else "", code[r.koi_disposition],
                 round(r.ra, 4), round(r.dec, 4), num(r.koi_period, 3), num(r.koi_prad, 2), num(r.koi_teq, 0),
                 num(r.koi_insol, 2), num(r.koi_steff, 0), num(r.koi_srad, 2), num(r.koi_depth, 0),
                 num(r.koi_duration, 2), num(score.get(r.kepoi_name), 3)])
out = {"cols": ["n", "k", "d", "ra", "dec", "P", "R", "T", "S", "Ts", "Rs", "dep", "dur", "p"], "rows": rows}
(ROOT / "site/explore.json").write_text(json.dumps(out, separators=(",", ":")))
print(len(rows), "objects,", round((ROOT / "site/explore.json").stat().st_size / 1e6, 2), "MB")
