"""What the Referee shows agents, how it prepares each run, and how it keeps its cache."""

import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import pytest
from conftest import DATA_READY

import referee


# ---------------------------------------------------------------- what agents may see


def test_public_view_hides_every_hidden_test_number():
    verdict = {
        "verdict": "PROMOTED",
        "val_gain": 0.02,
        "test_gain": 0.016,
        "test_ci": [0.011, 0.022],
        "base_test": {"auc": 0.99},
        "evidence": "val +0.0200 AUC, hidden stars +0.0160 CI [+0.0110, +0.0220]",
        "public_evidence": "val gain +0.0200, confirmed on the hidden stars",
    }
    seen = referee.Referee.public_view(verdict)

    assert "test_gain" not in seen and "test_ci" not in seen and "base_test" not in seen
    assert "0.016" not in json.dumps(seen)
    assert seen["evidence"] == "val gain +0.0200, confirmed on the hidden stars"
    assert seen["val_gain"] == 0.02


def test_public_view_keeps_rejection_reasons():
    verdict = {"verdict": "LEAK", "evidence": "trained on shuffled labels it still scores AUC 0.999"}
    assert referee.Referee.public_view(verdict)["evidence"] == verdict["evidence"]


# ---------------------------------------------------------------- the sandbox profile


def test_sandbox_profile_denies_everything_the_referee_keeps(tmp_path):
    text = referee.profile(tmp_path)
    for kept in ("exo/data", "exo/runs", ".env"):
        assert kept in text
    assert "(deny network*)" in text
    assert "(deny file-write*)" in text
    assert str(tmp_path.resolve()) in text


# ---------------------------------------------------------------- the shared cache


def _write_many(i):
    for j in range(40):
        referee._merge_into_cache({f"p{i}:{j}": {"ok": True}})


def test_cache_keeps_every_entry_with_several_writers():
    if referee.CACHE.exists():
        referee.CACHE.unlink()
    # fork, so each writer starts with the same test setup (spawn would re-import from scratch)
    ctx = mp.get_context("fork")
    workers = [ctx.Process(target=_write_many, args=(i,)) for i in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert len(json.loads(referee.CACHE.read_text())) == 160


def test_a_damaged_cache_is_rebuilt_not_wiped():
    referee.CACHE.write_text('{"k0": {"ok": tr')
    memory = {f"k{i}": {"ok": True} for i in range(30)}
    referee._merge_into_cache({"new": {"ok": True}}, fallback=memory)

    assert len(json.loads(referee.CACHE.read_text())) == 31
    assert any("corrupt" in p.name for p in referee.CACHE.parent.iterdir())


# ---------------------------------------------------------------- preparing a run


@pytest.mark.skipif(not DATA_READY, reason="run exo/prep.py first")
def test_shuffle_permutes_labels_and_nothing_else(tmp_path):
    referee.prepare(tmp_path, 0, "shuffle")
    shuffled = pd.read_parquet(tmp_path / "train.parquet")
    original = pd.read_parquet(referee.HERE / "data/train.parquet")

    assert sorted(shuffled.label) == sorted(original.label)
    assert (shuffled.label.values != original.label.values).any()
    features = [c for c in original.columns if c != "label"]
    pd.testing.assert_frame_equal(shuffled[features], original[features])


@pytest.mark.skipif(not DATA_READY, reason="run exo/prep.py first")
def test_ablate_blanks_every_light_curve_measurement(tmp_path):
    referee.prepare(tmp_path, 0, "ablate")
    meta = json.loads((tmp_path / "meta.json").read_text())
    val = pd.read_parquet(tmp_path / "val_inputs.parquet")

    for column in meta["transit"]:
        assert val[column].nunique(dropna=False) == 1


@pytest.mark.skipif(not DATA_READY, reason="run exo/prep.py first")
def test_val_and_test_inputs_carry_no_labels(tmp_path):
    referee.prepare(tmp_path, 0, "plain")
    for split in ("val", "test"):
        assert "label" not in pd.read_parquet(tmp_path / f"{split}_inputs.parquet").columns


@pytest.mark.skipif(not DATA_READY, reason="run exo/prep.py first")
def test_folds_share_no_host_star():
    train = set(pd.read_parquet(referee.HERE / "data/train.parquet").kepid)
    val = set(pd.read_parquet(referee.HERE / "data/val_inputs.parquet").kepid)
    test = set(pd.read_parquet(referee.HERE / "data/test_inputs.parquet").kepid)
    assert not (train & val) and not (train & test) and not (val & test)


def test_summary_counts_recall_at_the_number_of_real_planets():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
    # The top 3 picks hold 2 of the 3 real planets.
    assert referee.summary(y, p)["recall"] == pytest.approx(2 / 3)
