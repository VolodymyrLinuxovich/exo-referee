"""Runs one candidate inside the sandbox.

  python runner.py WORKDIR SEED

The Referee prepares WORKDIR before launch: candidate.py, train.parquet, val_inputs.parquet,
test_inputs.parquet and meta.json, already shuffled or ablated for this run. The runner never
learns which mode it is in, and the sandbox denies the real data folder, the run logs and the
cache, so a candidate cannot look up answers or undo an ablation by re-reading the originals.
It writes only val.npy, test.npy and run_meta.json back into WORKDIR.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

work, seed = Path(sys.argv[1]), int(sys.argv[2])
os.environ["EXO_DATA"] = str(work)          # candidates read meta.json from here

spec = importlib.util.spec_from_file_location("candidate", work / "candidate.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

t0 = time.time()
state = m.train(pd.read_parquet(work / "train.parquet"), seed)
train_s = time.time() - t0
for split in ("val", "test"):
    X = pd.read_parquet(work / f"{split}_inputs.parquet")
    p = np.asarray(m.predict(state, X), dtype=np.float64).reshape(-1)
    assert len(p) == len(X), f"predict returned {len(p)} values for {len(X)} rows"
    assert np.isfinite(p).all(), "non-finite predictions"
    np.save(work / f"{split}.npy", p)
json.dump({"train_s": train_s}, open(work / "run_meta.json", "w"))
