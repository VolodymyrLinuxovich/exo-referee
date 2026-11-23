"""Watch one real agent cycle as it happens: research, claim, sealed judgment.

  .venv/bin/python demo/live_agent.py

Starts one Claude agent in the guarded loop (its own track, so the main model is untouched)
and prints each event from the run log as the agent and the Referee write it.
"""

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "exo/runs/events.jsonl"
TAG = f"exo-live-{int(time.time())}"
B, DIM, RST = "\033[1m", "\033[2m", "\033[0m"
COL = {
    "PROMOTED": "\033[1;32m",
    "LEAK": "\033[1;31m",
    "PRIOR": "\033[1;35m",
    "NOISE": "\033[1;33m",
    "OVERFIT": "\033[1;36m",
    "CRASH": "\033[1;37m",
}

start = EVENTS.stat().st_size if EVENTS.exists() else 0
proc = subprocess.Popen(
    [
        str(ROOT / ".venv/bin/python"),
        "swarm.py",
        "--proposers",
        "1",
        "--rounds",
        "1",
        "--tag",
        TAG,
        "--models",
        "claude-opus-5-5",
    ],
    cwd=ROOT / "exo",
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
print(f"{B}agent{RST} Claude Opus is starting a cycle on NASA Kepler data\n", flush=True)
t0, pos, done = time.time(), start, False
while not done:
    time.sleep(0.5)
    with open(EVENTS) as f:
        f.seek(pos)
        lines = f.readlines()
        pos = f.tell()
    for line in lines:
        e = json.loads(line)
        if e.get("tag") != TAG:
            continue
        k, s = e["kind"], f"{DIM}{time.time() - t0:5.0f}s{RST}"
        if k == "prior_work":
            print(f"{s}  {B}research{RST} asked the Apify Actor: {e['query']}", flush=True)
            for r in e["results"][:3]:
                print(f"        {DIM}arXiv{RST} {r['title'][:88]}", flush=True)
        elif k == "hypothesis":
            print(f"\n{s}  {B}claim{RST} {e['hypothesis'][:300]}", flush=True)
        elif k == "claim":
            c = e.get("citation_check") or {}
            if c.get("total"):
                print(
                    f"        {DIM}cites {c['from_search']} of {c['total']} papers from the search{RST}",
                    flush=True,
                )
            print(
                f"\n{s}  {B}referee{RST} running it sealed: 3 seeds, 3 shuffled-label runs, light curve blanked, hidden stars ...",
                flush=True,
            )
        elif k == "verdict":
            v = e["verdict"]
            why = e.get("evidence") or (e.get("error") or "")
            print(f"\n{s}  {COL.get(v, '')}{v}{RST}  {why[:200]}", flush=True)
            done = True
        elif k == "error":
            print(f"\n{s}  error: {e.get('error', '')[:200]}", flush=True)
            done = True
    if proc.poll() is not None and not lines:
        done = True
proc.wait()
