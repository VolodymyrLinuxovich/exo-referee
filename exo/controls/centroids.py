"""Honest control: add light-curve vetting diagnostics (centroid offsets, odd-even depth,
event statistics). Real astrophysics: background eclipsing binaries shift the centroid."""
import json
import os
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

META = json.load(open(Path(os.environ["EXO_DATA"]) / "meta.json"))
COLS = META["features"] + META["diagnostics"]


def train(train_df, seed):
    m = make_pipeline(SimpleImputer(strategy="median"),
                      RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features=0.3,
                                             n_jobs=2, random_state=seed))
    m.fit(train_df[COLS], train_df["label"])
    return m


def predict(m, df):
    return m.predict_proba(df[COLS])[:, 1]
