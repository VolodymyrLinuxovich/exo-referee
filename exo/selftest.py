"""Checks the checker. Run before trusting any change to the Referee:

  ../.venv/bin/python selftest.py            # cold: every run really executes (use this after changes)
  ../.venv/bin/python selftest.py --quick    # reuse cached predictions: fast, but skips the runner,
                                             # the sandbox and input preparation for cached code

Judges the planted cheats and the honest control against the starting model in a throwaway runs
folder, attacks the sandbox, and checks the statistics against scikit-learn. Exits 1 on any miss.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(tempfile.mkdtemp(prefix="selftest_"))
os.environ["EXO_RUNS"] = str(WORK)  # never touches the real runs/ or incumbent
if "--quick" in sys.argv and (HERE / "runs/cache.json").exists():
    shutil.copy(HERE / "runs/cache.json", WORK / "cache.json")
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import fastci  # noqa: E402
from referee import Referee  # noqa: E402

EXPECT = [
    ("redteam/catalog_name.py", "LEAK", "reads the answer from the catalog name"),
    ("redteam/robovetter.py", "PRIOR", "copies NASA's vetting robot"),
    ("redteam/lucky_tweak.py", "NOISE", "cosmetic change"),
    ("controls/centroids.py", "PROMOTED", "honest centroid physics"),
]

ATTACK = (
    """
import os
from pathlib import Path
EXO = Path(r"%s")
def train(df, seed):
    for f in [EXO / "data/raw/koi.csv", EXO / "data/hidden/val_labels.parquet", EXO / "data/val_inputs.parquet"]:
        if not os.path.exists(f):        # a denied stat counts as held, not as a crash
            continue
        try:
            open(f, "rb").read(1)
        except OSError:
            continue
        raise SystemExit("LEAKED " + str(f))
    try:
        (EXO / "pwned.txt").write_text("x")
    except OSError:
        raise SystemExit("SANDBOX HELD")
    raise SystemExit("LEAKED write")
def predict(s, df):
    return [0.5] * len(df)
"""
    % HERE
)

fails = 0


def check(ok, msg):
    global fails
    print(("  pass  " if ok else "  FAIL  ") + msg)
    fails += 0 if ok else 1


try:
    print("statistics")
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 2000)
    p = np.round(rng.random(2000) + y * 0.3, 2)  # heavy ties on purpose
    check(
        abs(fastci.auc(y, p) - roc_auc_score(y, p)) < 1e-12,
        f"AUC matches scikit-learn with ties (native: {fastci.NATIVE})",
    )
    check(fastci.auc(y, np.zeros(2000)) == 0.5, "constant predictions score 0.5")

    print("gates, against the starting model")
    for path, want, what in EXPECT:
        ref = Referee(track="selftest_" + Path(path).stem)
        ref.inc_path.write_text((HERE / "candidates/baseline.py").read_text())
        v = ref.judge((HERE / path).read_text(), claim=what, agent="selftest", tag="selftest")
        check(v["verdict"] == want, f"{Path(path).stem:13} {what}: {v['verdict']} (want {want})")

    print("sandbox")
    res = Referee(track="selftest_attack").run(ATTACK, 0, "plain")
    tail = (res.get("stderr_tail") or "").strip().splitlines()[-1:] or [""]
    check("SANDBOX HELD" in tail[0], f"hostile candidate: {tail[0][:80]}")
    check(not (HERE / "pwned.txt").exists(), "no file written outside the run")
finally:
    shutil.rmtree(WORK, ignore_errors=True)
print("all checks passed" if not fails else f"{fails} check(s) failed")
sys.exit(1 if fails else 0)
