"""Guarded loop vs naive loop on the exoplanet task. Writes demo/compare.html and demo/compare.json.

For each loop: the incumbent's validation AUC after every claim (what the loop believes),
then a reality check on the final incumbents: AUC on the hidden stars with the columns that
encode the answer (kepler_name, koi_score, koi_fpflag_*, koi_pdisposition ...) removed,
the way a genuinely new candidate arrives. Run from anywhere:

  .venv/bin/python demo/compare.py
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOOPS = Path(__file__).resolve().parents[1]
EXO = LOOPS / "exo"
sys.path.insert(0, str(EXO))
from referee import auc  # noqa: E402

os.environ["EXO_DATA"] = str(EXO / "data")
TRAP = ["kepler_name", "koi_score", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
        "koi_pdisposition", "koi_comment", "koi_disp_prov", "koi_vet_stat"]


def series(tag, kind):
    """Incumbent val AUC after each claim the loop judged."""
    evs = [json.loads(l) for l in open(EXO / "runs/events.jsonl")]
    vs = [e for e in evs if e.get("tag") == tag and e["kind"] == kind and e.get("val")
          and e.get("proposer") != "proposer-live"]
    if not vs:
        return []
    cur = vs[0]["base_val"]["auc"]
    pts = [{"i": 0, "auc": cur, "verdict": "start", "claim": "Baseline: transit shape + host star", "who": ""}]
    for i, e in enumerate(vs, 1):
        moved = e["verdict"] in ("PROMOTED", "ACCEPTED")
        if moved:
            cur = e["val"]["auc"]
        pts.append({"i": i, "auc": cur, "verdict": e["verdict"], "claim": e.get("claim", "")[:140],
                    "who": e.get("proposer", ""), "moved": moved, "gain": e.get("val_gain")})
    return pts


def reality(inc_file):
    spec = importlib.util.spec_from_file_location("m", inc_file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    tr = pd.read_parquet(EXO / "data/train.parquet")
    X = pd.read_parquet(EXO / "data/test_inputs.parquet")
    y = pd.read_parquet(EXO / "data/hidden/test_labels.parquet").label.values
    out = {}
    for name, df in (("as_scored", X), ("answers_removed", X.assign(**{c: np.nan for c in TRAP}))):
        out[name] = float(np.mean([auc(y, np.asarray(m.predict(m.train(tr, s), df))) for s in (0, 1, 2)]))
    return out


data = {"guarded": series("exo", "verdict"), "naive": series("exo-naive", "naive_verdict"),
        "reality": {"guarded": reality(EXO / "runs/incumbent_exo.py"),
                    "naive": reality(EXO / "runs/incumbent_exo-naive.py")},
        "baseline": reality(EXO / "candidates/baseline.py")}
(LOOPS / "demo/compare.json").write_text(json.dumps(data, indent=1))
html = (LOOPS / "demo/compare.tmpl.html").read_text().replace("/*DATA*/null", json.dumps(data))
(LOOPS / "demo/compare.html").write_text(html)
r = data["reality"]
print(f"guarded: believes {data['guarded'][-1]['auc']:.4f} | hidden stars, answers removed {r['guarded']['answers_removed']:.4f}")
print(f"naive:   believes {data['naive'][-1]['auc']:.4f} | hidden stars, answers removed {r['naive']['answers_removed']:.4f}")
print(f"baseline hidden stars {data['baseline']['answers_removed']:.4f}")
print("wrote", LOOPS / "demo/compare.html")
