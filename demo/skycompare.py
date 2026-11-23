"""Which hidden test stars each loop's final model would send for follow-up, for the sky view.

Same setup as demo/followup.py: hidden test stars, answer columns removed, mean of 3 seeds.
Each loop picks as many stars as there are real planets (539). Writes site/skycompare.json:
for every test star its name, whether it is a real planet, and whether each loop picked it.
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

tr = pd.read_parquet(EXO / "data/train.parquet")
X = pd.read_parquet(EXO / "data/test_inputs.parquet")
y = pd.read_parquet(EXO / "data/hidden/test_labels.parquet").label.values.astype(int)
blind = X.assign(**{c: np.nan for c in TRAP})
k = int(y.sum())


def picked(path):
    spec = importlib.util.spec_from_file_location("m", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    p = np.mean([np.asarray(m.predict(m.train(tr, s), blind)) for s in (0, 1, 2)], axis=0)
    mask = np.zeros(len(p), dtype=int)
    mask[np.argsort(-p)[:k]] = 1
    return mask


naive = picked(EXO / "runs/incumbent_exo-naive.py")
guarded = picked(EXO / "runs/incumbent_exo.py")

out = {
    "n": int(len(y)),
    "k": k,
    "cols": ["n", "real", "naive", "guarded"],
    "rows": [[name, int(a), int(b), int(c)] for name, a, b, c in zip(X.kepoi_name, y, naive, guarded)],
}
(ROOT / "site/skycompare.json").write_text(json.dumps(out, separators=(",", ":")))
for name, mask in (("naive", naive), ("guarded", guarded)):
    print(name, "false picks", int(((mask == 1) & (y == 0)).sum()), "missed planets", int(((mask == 0) & (y == 1)).sum()))
