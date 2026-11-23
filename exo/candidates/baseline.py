"""Incumbent at start: random forest on transit-fit and host-star measurements."""

import json
import os
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

FEATS = json.load(open(Path(os.environ["EXO_DATA"]) / "meta.json"))["features"]


def train(train_df, seed):
    m = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, max_features=0.3, n_jobs=2, random_state=seed
        ),
    )
    m.fit(train_df[FEATS], train_df["label"])
    return m


def predict(m, df):
    return m.predict_proba(df[FEATS])[:, 1]
