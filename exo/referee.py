"""Exoplanet Referee: the only process that ever reads val/test labels.

Same rule as Spectre: a claim becomes a fact only when the Referee promotes it.
Task: is a Kepler Object of Interest a real planet (CONFIRMED) or a FALSE POSITIVE?
Metric: ROC AUC. Folds are split by host star; the hidden test fold is 1328 stars no proposer sees.

Gates, first failure is the verdict:
  CRASH    every sandbox run must finish
  NOISE    3 seeds; paired bootstrap 95% CI over objects above 0, gain > 2x incumbent seed spread,
           positive on every seed
  LEAK     retrain with the training labels shuffled. If >= half the gain survives, it was not
           learned: the answer is in the inputs
  PRIOR    replace every transit measurement with its train median. If >= half the gain survives,
           the model is not reading the light curve (host-star or catalog priors)
  OVERFIT  hidden test stars, 3 seeds. CI must stay above 0
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PY = HERE.parent / ".venv/bin/python"
RUNS = Path(os.environ.get("EXO_RUNS", HERE / "runs"))
RUNS.mkdir(exist_ok=True)
EVENTS = RUNS / "events.jsonl"
CACHE = RUNS / "cache.json"
SEEDS = (0, 1, 2)
TAG = os.environ.get("EXO_TAG", "exo")
PARALLEL = int(os.environ.get("EXO_PARALLEL", "5"))
PROFILE = (HERE / "sandbox.sb.tmpl").read_text().replace("HIDDEN", str(HERE / "data/hidden"))

_lock = threading.Lock()


def log(who: str, kind: str, **kw):
    if "agent" in kw:
        kw["proposer"] = kw.pop("agent")
    kw.setdefault("tag", TAG)
    with _lock, open(EVENTS, "a") as f:
        f.write(json.dumps({"ts": time.time(), "agent": who, "kind": kind, **kw}, default=float) + "\n")


def sha(src: str) -> str:
    return hashlib.sha256(src.encode()).hexdigest()[:12]


def boot_ci(diff, n=2000, seed=0):
    g = np.random.default_rng(seed)
    m = diff[g.integers(0, len(diff), (n, len(diff)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    r = np.empty(len(p))
    r[np.argsort(p, kind="mergesort")] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class Labels:
    def __init__(self):
        self.y = {s: pd.read_parquet(HERE / f"data/hidden/{s}_labels.parquet").label.values
                  for s in ("val", "test")}


def summary(y, p) -> dict:
    thr = np.sort(p)[::-1][:int(y.sum())].min()          # flag as many as there are real planets
    return {"auc": auc(y, p), "recall": float(((p >= thr) & (y == 1)).sum() / y.sum())}


def paired_ci(y, C, B, n=2000, seed=0):
    """Bootstrap objects; AUC of seed-averaged candidate minus incumbent."""
    g = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = g.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        out.append(auc(y[i], C[i]) - auc(y[i], B[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


class Referee:
    def __init__(self, track: str = "main"):
        self.labels = Labels()
        self.inc_path = RUNS / f"incumbent_{track}.py"
        self.cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    # ------------------------------------------------------------ sandbox
    def run(self, src: str, seed: int, mode: str = "plain") -> dict:
        key = f"{sha(src)}:{seed}:{mode}"
        with _lock:
            if key in self.cache:
                return self.cache[key]
        out = Path(tempfile.mkdtemp(prefix="qrun_"))
        cand = out / "candidate.py"
        cand.write_text(src)
        t0 = time.time()
        try:
            r = subprocess.run(["sandbox-exec", "-p", PROFILE, str(PY), str(HERE / "runner.py"),
                                str(cand), str(seed), mode, str(out)],
                               capture_output=True, text=True, timeout=600, cwd=out,
                               env={**os.environ, "OMP_NUM_THREADS": "2", "EXO_DATA": str(HERE / "data"), "PYTHONPATH": ""})
            ok = r.returncode == 0
            err = r.stderr[-1500:]
        except subprocess.TimeoutExpired:
            ok, err = False, "timeout after 600 s"
        if ok:
            res = {"ok": True, "wall_s": time.time() - t0,
                   "val": np.load(out / "val.npy").tolist(), "test": np.load(out / "test.npy").tolist()}
        else:
            res = {"ok": False, "error": err.strip().splitlines()[-1] if err.strip() else "crashed",
                   "stderr_tail": err, "wall_s": time.time() - t0}
        shutil.rmtree(out, ignore_errors=True)
        with _lock:
            self.cache[key] = res
            tmp = CACHE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache))
            tmp.replace(CACHE)
        return res

    def batch(self, jobs):
        with ThreadPoolExecutor(PARALLEL) as ex:
            return list(ex.map(lambda j: self.run(*j), jobs))

    def incumbent(self) -> str:
        if not self.inc_path.exists():
            self.inc_path.write_text((HERE / "candidates/baseline.py").read_text())
        return self.inc_path.read_text()

    # ------------------------------------------------------------ verdicts
    def judge(self, src: str, claim: str = "", agent: str = "?", guarded: bool = True,
              exp_id: str = "", tag: str = TAG) -> dict:
        t0 = time.time()
        inc = self.incumbent()
        v = {"agent": agent, "claim": claim, "code": sha(src), "incumbent": sha(inc),
             "exp_id": exp_id, "tag": tag, "domain": "exo"}

        yv, yt = self.labels.y["val"], self.labels.y["test"]
        if not guarded:
            c, b = self.batch([(src, 0), (inc, 0)])
            if not (c["ok"] and b["ok"]):
                v.update(verdict="CRASH", error=c.get("error") or b.get("error"))
            else:
                cv, bv = np.array(c["val"]), np.array(b["val"])
                g = auc(yv, cv) - auc(yv, bv)
                v.update(verdict="ACCEPTED" if g > 0 else "REJECTED", val_gain=g,
                         val=summary(yv, cv), base_val=summary(yv, bv))
                if g > 0:
                    self.inc_path.write_text(src)
            v["wall_s"] = time.time() - t0
            log(agent, "naive_verdict", **v)
            return v

        jobs = [(s, sd, "plain") for s in (src, inc) for sd in SEEDS]
        jobs += [(s, sd, "shuffle") for s in (src, inc) for sd in SEEDS]
        jobs += [(s, 0, "ablate") for s in (src, inc)]
        res = dict(zip([(j[0] == src, j[1], j[2]) for j in jobs], self.batch(jobs)))
        bad = [r for r in res.values() if not r["ok"]]
        if bad:
            v.update(verdict="CRASH", error=bad[0].get("error"), stderr_tail=bad[0].get("stderr_tail"))
            return self._done(v, t0)

        Cs = [np.array(res[(True, s, "plain")]["val"]) for s in SEEDS]
        Bs = [np.array(res[(False, s, "plain")]["val"]) for s in SEEDS]
        per_seed = [auc(yv, c) - auc(yv, b) for c, b in zip(Cs, Bs)]
        gain = float(np.mean(per_seed))
        C, B = np.mean(Cs, 0), np.mean(Bs, 0)
        lo, hi = paired_ci(yv, C, B)
        seed_std = float(np.std([auc(yv, b) for b in Bs], ddof=1))
        v.update(val=summary(yv, C), base_val=summary(yv, B), val_gain=gain,
                 val_ci=[lo, hi], per_seed_gain=per_seed, noise_band=2 * seed_std)
        if lo <= 0 or gain <= 2 * seed_std or min(per_seed) <= 0:
            why = []
            if hi < 0:
                why.append(f"worse than the incumbent: 95% CI [{lo:+.4f}, {hi:+.4f}] is entirely below 0")
            else:
                if lo <= 0:
                    why.append(f"95% CI over {len(yv)} objects [{lo:+.4f}, {hi:+.4f}] includes 0")
                if gain <= 2 * seed_std:
                    why.append(f"gain {gain:+.4f} is inside the seed noise band (2 sigma = {2 * seed_std:.4f})")
                if min(per_seed) <= 0:
                    why.append(f"not positive on every seed {[round(x, 4) for x in per_seed]}")
            v.update(verdict="NOISE", evidence="; ".join(why))
            return self._done(v, t0)

        # A model trained on shuffled labels scores about 0.5 +- 0.03 by chance, so average 3 shuffles
        # and require real predictive power (AUC >= 0.6) before calling it a leak.
        sh_c = float(np.mean([auc(yv, np.array(res[(True, sd, "shuffle")]["val"])) for sd in SEEDS]))
        sh_b = float(np.mean([auc(yv, np.array(res[(False, sd, "shuffle")]["val"])) for sd in SEEDS]))
        sh = sh_c - sh_b
        ab_c = auc(yv, np.array(res[(True, 0, "ablate")]["val"]))
        ab = ab_c - auc(yv, np.array(res[(False, 0, "ablate")]["val"]))
        inc_plain = float(np.mean([auc(yv, b) for b in Bs]))
        v.update(shuffle_gain=sh, ablate_gain=ab, shuffle_auc=sh_c, ablate_auc=ab_c)
        if sh_c >= 0.6 and sh >= 0.5 * gain:
            v.update(verdict="LEAK", evidence=f"trained on shuffled labels it still scores AUC {sh_c:.3f} "
                                              f"(chance is 0.5): {"none of the gain was learned" if sh >= gain else f"{sh / gain:.0%} of the gain was not learned"}, "
                                              "the answer is in the inputs")
            return self._done(v, t0)
        # Blanked models sit near chance, where small gaps mean little, so PRIOR also requires that the
        # blanked candidate still roughly matches the incumbent WITH its light curve.
        if ab >= 0.5 * gain and ab_c >= inc_plain - gain:
            v.update(verdict="PRIOR", evidence=f"{"all" if ab >= gain else f"{ab / gain:.0%}"} of the gain survives with every transit "
                                               f"measurement blanked (AUC {ab_c:.3f} with no light curve): "
                                               "the model is not reading the light curve")
            return self._done(v, t0)

        Ct = np.mean([res[(True, s, "plain")]["test"] for s in SEEDS], 0)
        Bt = np.mean([res[(False, s, "plain")]["test"] for s in SEEDS], 0)
        tg = auc(yt, Ct) - auc(yt, Bt)
        tlo, thi = paired_ci(yt, Ct, Bt)
        v.update(test_gain=tg, test_ci=[tlo, thi], test=summary(yt, Ct), base_test=summary(yt, Bt))
        if tlo <= 0:
            v.update(verdict="OVERFIT", evidence=f"val gain {gain:+.4f}, but on {len(yt)} hidden stars "
                                                 f"the CI is [{tlo:+.4f}, {thi:+.4f}]")
        else:
            v.update(verdict="PROMOTED", evidence=f"val {gain:+.4f} AUC, hidden stars {tg:+.4f} "
                                                  f"CI [{tlo:+.4f}, {thi:+.4f}]")
            (RUNS / f"incumbent_{sha(inc)}.py").write_text(inc)
            self.inc_path.write_text(src)
        return self._done(v, t0)

    def _done(self, v, t0):
        v["wall_s"] = time.time() - t0
        log("referee", "verdict", **v)
        return v

    @staticmethod
    def public_view(v: dict) -> dict:
        """What proposers may see: no hidden-era numbers."""
        return {k: x for k, x in v.items() if not k.startswith("test") and not k.startswith("base_test")}


if __name__ == "__main__":
    import sys
    ref = Referee(os.environ.get("EXO_TRACK", "cli"))
    src = Path(sys.argv[1]).read_text()
    out = ref.judge(src, claim=sys.argv[2] if len(sys.argv) > 2 else "", agent="cli",
                    guarded="--naive" not in sys.argv)
    print(json.dumps({k: x for k, x in out.items() if k != "stderr_tail"}, indent=2))
