"""Rank the unresolved Kepler candidates with the current promoted incumbent.

The Referee's work is done, so this trains on every labelled object (train + val + test),
averages 3 seeds, and scores the 1977 KOIs NASA still lists as CANDIDATE.
Output: runs/unresolved_ranked.csv. A ranking for follow-up, not a discovery claim.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE / "data"
os.environ["EXO_DATA"] = str(D)
inc = HERE / "runs/incumbent_exo.py"
spec = importlib.util.spec_from_file_location("inc", inc)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

parts = [pd.read_parquet(D / "train.parquet")]
for f in ("val", "test"):
    x = pd.read_parquet(D / f"{f}_inputs.parquet")
    x["label"] = pd.read_parquet(D / f"hidden/{f}_labels.parquet").label.values
    parts.append(x)
lab = pd.concat(parts, ignore_index=True)
unres = pd.read_parquet(D / "unresolved.parquet")

p = np.mean([m.predict(m.train(lab, s), unres) for s in (0, 1, 2)], axis=0)
out = unres[["kepoi_name", "kepid", "koi_period", "koi_prad", "koi_teq", "koi_insol", "koi_model_snr",
             "koi_count", "koi_score"]].assign(p_planet=p).sort_values("p_planet", ascending=False)
out.to_csv(HERE / "runs/unresolved_ranked.csv", index=False)
print(out.head(15).to_string(index=False))
print("\nn =", len(out), "| p>0.9:", (p > 0.9).sum(), "| p<0.1:", (p < 0.1).sum())
print("Spearman with NASA robovetter koi_score (model never saw it):",
      round(out[["p_planet", "koi_score"]].corr("spearman").iloc[0, 1], 3))
hz = out[(out.koi_prad < 2) & (out.koi_insol.between(0.25, 2)) & (out.p_planet > 0.8)]
print("\nSmall (<2 Earth radii), temperate (0.25-2x Earth insolation), p>0.8:", len(hz))
print(hz.head(10).to_string(index=False))
