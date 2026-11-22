"""Runs one candidate inside the sandbox. Never sees val/test labels.

  python runner.py CANDIDATE.py SEED MODE OUTDIR
MODE: plain | ablate (every transit measurement replaced by its train median at inference)
      | shuffle (train labels shuffled)
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
cand, seed, mode, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
TRANSIT = json.load(open(DATA / "meta.json"))["transit"]
os.environ["EXO_DATA"] = str(DATA)

spec = importlib.util.spec_from_file_location("candidate", cand)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

tr = pd.read_parquet(DATA / "train.parquet")
if mode == "shuffle":
    tr["label"] = np.random.default_rng(1000 + seed).permutation(tr["label"].values)
med = tr[TRANSIT].median()
t0 = time.time()
state = m.train(tr, seed)
train_s = time.time() - t0
for split in ("val", "test"):
    X = pd.read_parquet(DATA / f"{split}_inputs.parquet")
    if mode == "ablate":
        for c in TRANSIT:
            X[c] = med[c]
    p = np.asarray(m.predict(state, X), dtype=np.float64).reshape(-1)
    assert len(p) == len(X), f"predict returned {len(p)} values for {len(X)} rows"
    assert np.isfinite(p).all(), "non-finite predictions"
    np.save(out / f"{split}.npy", p)
json.dump({"train_s": train_s}, open(out / "meta.json", "w"))
