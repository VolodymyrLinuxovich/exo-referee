"""Shared setup for the tests: import the Referee code from exo/, and keep every test's
runs folder (cache, incumbents, event log) away from the real one."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Point the Referee at a throwaway runs folder before it is imported, so no test can touch
# the real cache or the real incumbent model.
os.environ["EXO_RUNS"] = tempfile.mkdtemp(prefix="exo_tests_")

sys.path.insert(0, str(ROOT / "exo"))

# The data folder is built by exo/prep.py from the NASA snapshot. Tests that need it skip
# themselves when it is missing, so the pure logic tests still run anywhere.
DATA_READY = (ROOT / "exo/data/train.parquet").exists() and (
    ROOT / "exo/data/hidden/val_labels.parquet"
).exists()
