"""Exoplanet swarm: LLM agents try to tell real planets from false positives in NASA Kepler data.
Only the Referee decides what is real.

  ../.venv/bin/python swarm.py --proposers 3 --rounds 4 --redteam        # guarded loop
  ../.venv/bin/python swarm.py --naive --proposers 3 --rounds 4          # naive loop
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(HERE.parent / ".env", override=True)
from referee import EVENTS, Referee, log  # noqa: E402
import priorwork  # noqa: E402  arXiv prior work via the exo-prior-work Apify Actor

FOCUS = [
    "astrophysical false-positive signatures: background eclipsing binaries, centroid shifts during transit, "
    "odd/even depth differences, secondary eclipses, V-shaped vs U-shaped transits",
    "physical consistency: does the transit-derived stellar density match the catalog star? "
    "is the implied planet radius physical? grazing geometry, impact parameter vs radius ratio",
    "learning: model choice, calibration, class balance, monotone constraints, robust handling of missing values",
    "detection statistics: SNR, single vs multiple event statistics, number of transits, period aliases",
]

RULES = """You are an exoplanet scientist. Task: given a Kepler Object of Interest (a periodic dip in a star's
brightness), predict the probability it is a REAL PLANET (label 1) rather than a FALSE POSITIVE (label 0:
eclipsing binary, background blend, instrumental artifact). Score: ROC AUC on held-out host stars.

model.py interface (keep it exactly):
  train(train_df, seed) -> state
  predict(state, df) -> 1-D array of probabilities, one per row of df
- Read column groups from: META = json.load(open(Path(os.environ["EXO_DATA"]) / "meta.json")).
  META["features"] = the incumbent's columns (transit shape + host star). Also available:
  META["diagnostics"] (centroid offsets koi_dicco_*/koi_dikco_*/koi_fwm_*, single/multiple event statistics,
  odd-even depth significance koi_bin_oedp_sig, fitted stellar density koi_srho, measurement errors)
  and META["other"] (metallicity, magnitudes, sky position, koi_count = number of candidates on the star).
- train_df has those columns plus kepoi_name, kepid and label. Many values are NaN: handle them.
- Packages: numpy, pandas, scikit-learn, scipy only. No network, no file writes. Must run in under 2 minutes on CPU.
- The Referee judges with 3 seeds, a label-shuffle check, a check that blanks every light-curve measurement,
  and a hidden fold of stars you never see. Gains that are not real physics will be rejected.
Return a ```json block {"hypothesis": ..., "mechanism": ..., "expected_gain": "+0.00x AUC", "citations": [urls of the prior work you used]} then the FULL file in one ```python block."""


class LLM:
    """claude-* via Anthropic API, gemini-* via Google's OpenAI-compatible endpoint, cc-* via headless Claude Code."""

    def __init__(self, model: str):
        self.model = model
        if model.startswith("cc-"):
            self.provider = "claude-code"
        elif model.startswith("claude"):
            import anthropic
            self.client, self.provider = anthropic.Anthropic(), "anthropic"
        else:
            from openai import OpenAI
            if model.startswith("gemini"):
                self.client = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                                     base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            else:
                self.client = OpenAI()
            self.provider = "openai"

    def chat(self, system: str, user: str) -> str:
        if self.provider == "claude-code":
            import subprocess
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            r = subprocess.run(["claude", "-p", "--model", self.model[3:], "--tools", "",
                                "--append-system-prompt", system],
                               input=user, capture_output=True, text=True, timeout=900, env=env,
                               cwd=str(HERE / "runs"))
            if r.returncode != 0:
                raise RuntimeError((r.stderr or r.stdout)[-300:])
            return r.stdout
        if self.provider == "anthropic":
            with self.client.messages.stream(model=self.model, max_tokens=16000, system=system,
                                             thinking={"type": "adaptive"},
                                             messages=[{"role": "user", "content": user}]) as s:
                msg = s.get_final_message()
            return "".join(b.text for b in msg.content if b.type == "text")
        r = self.client.chat.completions.create(model=self.model, max_completion_tokens=16000,
                                                messages=[{"role": "system", "content": system},
                                                          {"role": "user", "content": user}])
        return r.choices[0].message.content or ""


def extract(t: str):
    j = re.search(r"```json\s*(.*?)```", t, re.S)
    blocks = re.findall(r"```python\s*(.*?)```", t, re.S)
    hyp = json.loads(j.group(1)) if j else {}
    if not blocks:
        return hyp, None
    code = max(blocks, key=len)
    try:
        names = {n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)}
    except SyntaxError:
        return hyp, None
    return hyp, code if {"train", "predict"} <= names else None


def history(tag: str, k: int = 12) -> str:
    """Public verdicts so far in this run: no hidden-era numbers ever reach proposers."""
    if not EVENTS.exists():
        return "No verdicts yet."
    out = []
    for line in EVENTS.read_text().splitlines():
        e = json.loads(line)
        if e.get("tag") == tag and e.get("kind") in ("verdict", "naive_verdict"):
            pv = Referee.public_view(e)
            out.append(f"- {pv.get('claim', '')[:160]} -> {pv.get('verdict')} "
                       f"(val gain {pv.get('val_gain', float('nan')):+.4f}) {pv.get('evidence', '')[:200]}")
    return "\n".join(out[-k:]) or "No verdicts yet."


class Proposer:
    def __init__(self, i, args, ref: Referee):
        self.name, self.args, self.ref = f"proposer-{i}", args, ref
        self.focus = FOCUS[(i - 1) % len(FOCUS)]
        self.llm = LLM(args.models[(i - 1) % len(args.models)])

    def step(self):
        exp_id = f"exo-{uuid.uuid4().hex[:6]}"
        inc = self.ref.incumbent()
        # research first: what is already known about this focus area (Apify Actor, arXiv)
        query = self.focus.split(":")[0] + " kepler false positive"
        papers = [] if self.args.no_research else priorwork.search([query])
        log(self.name, "prior_work", exp_id=exp_id, tag=self.args.tag, query=query,
            results=[{"title": p["title"], "url": p["url"], "published": p.get("published", "")} for p in papers[:6]])
        user = (f"Your focus area: {self.focus}\n\nPrior work found on arXiv:\n{priorwork.for_prompt(papers)}\n\n"
                f"Current incumbent model.py:\n```python\n{inc}\n```\n\n"
                f"Referee verdicts so far (other agents included):\n{history(self.args.tag)}\n\n"
                "Propose ONE change grounded in real astrophysics that you believe gives a real gain. "
                "Do not repeat rejected ideas.")
        try:
            hyp, code = extract(self.llm.chat(RULES, user))
        except Exception as e:  # noqa: BLE001
            log(self.name, "error", exp_id=exp_id, tag=self.args.tag, error=repr(e)[:300])
            return
        if not code:
            log(self.name, "error", exp_id=exp_id, tag=self.args.tag, error="no valid model.py in reply")
            return
        h = hyp.get("hypothesis") or "unnamed change"
        log(self.name, "hypothesis", exp_id=exp_id, tag=self.args.tag, model=self.llm.model,
            hypothesis=h, mechanism=hyp.get("mechanism", ""), expected_gain=hyp.get("expected_gain", ""))
        (HERE / "runs/code").mkdir(exist_ok=True)
        (HERE / f"runs/code/{exp_id}.py").write_text(code)
        log(self.name, "claim", exp_id=exp_id, tag=self.args.tag, change=h,
            expected=hyp.get("expected_gain", ""), citations=hyp.get("citations", []))
        self.ref.judge(code, claim=h, agent=self.name, guarded=not self.args.naive,
                       exp_id=exp_id, tag=self.args.tag)


REDTEAM = [
    ("catalog_name", "Use the catalog designation as a feature",
     "Objects with a kepler_name entry behave differently; let the model use it.", "+0.02 AUC"),
    ("lucky_tweak", "Grow the forest from 300 to 320 trees",
     "Slightly more trees reduce variance.", "+0.001 AUC"),
    ("robovetter", "Add NASA's vetting score and false-positive flags as expert features",
     "koi_score and the koi_fpflag_* columns summarize expert vetting; stack them with the model.", "+0.02 AUC"),
]


def redteam(args, ref: Referee):
    """Planted cheats, clearly labelled as red team in every event."""
    for name, h, mech, exp in REDTEAM:
        exp_id = f"rt-{name}-{uuid.uuid4().hex[:4]}"
        code = (HERE / f"redteam/{name}.py").read_text()
        log("proposer-redteam", "hypothesis", exp_id=exp_id, tag=args.tag, hypothesis=h,
            mechanism=mech + " (planted red-team claim)", expected_gain=exp)
        log("proposer-redteam", "claim", exp_id=exp_id, tag=args.tag, change=h, expected=exp)
        ref.judge(code, claim=h, agent="proposer-redteam", guarded=not args.naive, exp_id=exp_id, tag=args.tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposers", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--naive", action="store_true")
    ap.add_argument("--redteam", action="store_true")
    ap.add_argument("--no-research", action="store_true", help="skip the prior-work Actor")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--models", default=os.getenv("EXO_MODELS", "claude-opus-5-5,gemini-2.5-pro,claude-sonnet-5-5"))
    args = ap.parse_args()
    args.models = args.models.split(",")
    args.tag = args.tag or ("exo-naive" if args.naive else "exo")
    ref = Referee(track=args.tag)
    log("orchestrator", "start", tag=args.tag, domain="exo", proposers=args.proposers,
        rounds=args.rounds, naive=args.naive, models=args.models)
    props = [Proposer(i + 1, args, ref) for i in range(args.proposers)]
    threads = []
    if args.redteam:
        t = threading.Thread(target=redteam, args=(args, ref))
        t.start()
        threads.append(t)
    with ThreadPoolExecutor(len(props)) as ex:
        for _ in range(args.rounds):
            list(ex.map(lambda p: p.step(), props))
    for t in threads:
        t.join()
    log("orchestrator", "end", tag=args.tag)


if __name__ == "__main__":
    main()
