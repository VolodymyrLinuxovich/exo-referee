"""What each loop's final model is worth in practice: follow up its top picks on new candidates.

Hidden test stars, answer columns removed (as a genuinely new candidate arrives), mean of 3 seeds.
Writes demo/followup.json, which the site reads.
"""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXO = ROOT / "exo"
os.environ["EXO_DATA"] = str(EXO / "data")
TRAP = [
    "kepler_name",
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_pdisposition",
    "koi_comment",
    "koi_disp_prov",
    "koi_vet_stat",
]


def scores(path):
    spec = importlib.util.spec_from_file_location("m", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    tr = pd.read_parquet(EXO / "data/train.parquet")
    X = pd.read_parquet(EXO / "data/test_inputs.parquet").assign(**{c: np.nan for c in TRAP})
    return np.mean([np.asarray(m.predict(m.train(tr, s), X)) for s in (0, 1, 2)], axis=0)


y = pd.read_parquet(EXO / "data/hidden/test_labels.parquet").label.values
out = {"n": int(len(y)), "real": int(y.sum()), "k": 100, "loops": {}}
for name, f in [
    ("naive", EXO / "runs/incumbent_exo-naive.py"),
    ("guarded", EXO / "runs/incumbent_exo.py"),
    ("start", EXO / "candidates/baseline.py"),
]:
    p = scores(f)
    order = np.argsort(-p)
    row = {}
    for k in (100, 300, int(y.sum())):
        top = y[order[:k]]
        row[str(k)] = {"real": int(top.sum()), "false": int(k - top.sum())}
    row["picks"] = y[order[: int(y.sum())]].astype(int).tolist()  # every pick, best first, 1 = real planet
    out["loops"][name] = row
    print(name, {k: v for k, v in row.items() if k != "picks"})
(ROOT / "demo/followup.json").write_text(json.dumps(out))
