"""Run the exo-prior-work Actor on the Apify platform and show what happened, from Apify's own API.

  .venv/bin/python demo/apify_run.py "centroid offset background eclipsing binary kepler"

Prints the run as Apify recorded it: run ID and console link, status, duration, compute units,
the Actor's Crawlee log lines, and the papers in its dataset. Needs APIFY_TOKEN and APIFY_ACTOR in .env.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
TOKEN, ACTOR = os.environ["APIFY_TOKEN"], os.environ.get("APIFY_ACTOR", "exo-prior-work").replace("/", "~")
B, DIM, G, RST = "\033[1m", "\033[2m", "\033[1;32m", "\033[0m"


def api(path, data=None, raw=False):
    req = urllib.request.Request(f"https://api.apify.com/v2/{path}", method="POST" if data is not None else "GET",
                                 data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read().decode()
    return body if raw else json.loads(body)


query = " ".join(sys.argv[1:]) or "centroid offset background eclipsing binary kepler"
print(f"{B}apify{RST} starting exo-prior-work on the Apify platform", flush=True)
print(f"      {DIM}input{RST} {json.dumps({'queries': [query], 'maxPerQuery': 4})}", flush=True)
t0 = time.time()
run = api(f"acts/{ACTOR}/runs?waitForFinish=90", {"queries": [query], "maxPerQuery": 4})["data"]
print(f"      {DIM}run{RST}   {run['id']}  console.apify.com/actors/runs/{run['id']}", flush=True)
while run["status"] in ("READY", "RUNNING"):
    time.sleep(2)
    run = api(f"actor-runs/{run['id']}")["data"]
stats = run.get("stats", {})
print(f"      {DIM}status{RST} {G if run['status'] == 'SUCCEEDED' else ''}{run['status']}{RST} in {stats.get('runTimeSecs', time.time() - t0):.1f} s"
      f", {stats.get('computeUnits', 0):.4f} compute units\n", flush=True)

log = api(f"actor-runs/{run['id']}/log", raw=True)
for line in log.splitlines():
    if any(k in line for k in ("papers for", "Finished!", "All requests")):
        print(f"  {DIM}log{RST} {line.split('INFO')[-1].strip()[:110]}", flush=True)

items = api(f"datasets/{run['defaultDatasetId']}/items?clean=true")
print(f"\n{B}dataset{RST} {len(items)} papers", flush=True)
for p in items:
    print(f"  {p.get('published', '')[:4]}  {p['title'][:92]}", flush=True)
