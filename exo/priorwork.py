"""Prior work for an agent's idea, from the exo-prior-work Apify Actor (../actor).

On the Apify platform when APIFY_TOKEN is set (Actor id from APIFY_ACTOR, default
"<your-username>~exo-prior-work"), otherwise the same Actor runs locally with Crawlee.
Returns a list of {title, authors, published, url, abstract}. Never raises: research is
optional, so a failed search just means the agent proposes without citations.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ACTOR_DIR = Path(__file__).resolve().parents[1] / "actor"


def search(queries: list[str], per_query: int = 4) -> list[dict]:
    payload = {"queries": [q for q in queries if q], "maxPerQuery": per_query}
    if not payload["queries"]:
        return []
    try:
        if os.environ.get("APIFY_TOKEN"):
            return _platform(payload)
        return _local(payload)
    except Exception:  # noqa: BLE001  research is best effort
        return []


def _platform(payload: dict) -> list[dict]:
    actor = os.environ.get("APIFY_ACTOR", "exo-prior-work").replace("/", "~")
    url = f"https://api.apify.com/v2/acts/{actor}/run-sync-get-dataset-items?timeout=120"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ['APIFY_TOKEN']}"})
    with urllib.request.urlopen(req, timeout=150) as r:
        return _dedupe(json.load(r))


def _local(payload: dict) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="priorwork_") as d:
        store = Path(d)
        (store / "key_value_stores/default").mkdir(parents=True)
        (store / "key_value_stores/default/INPUT.json").write_text(json.dumps(payload))
        env = {**os.environ, "CRAWLEE_STORAGE_DIR": str(store), "APIFY_LOCAL_STORAGE_DIR": str(store)}
        subprocess.run(["node", "src/main.js"], cwd=ACTOR_DIR, env=env, capture_output=True, timeout=120)
        items = [json.loads(f.read_text()) for f in sorted((store / "datasets/default").glob("*.json"))]
    return _dedupe(items)


def _dedupe(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for it in items:
        if it.get("url") and it["url"] not in seen:
            seen.add(it["url"])
            out.append(it)
    return out


def for_prompt(papers: list[dict], k: int = 6) -> str:
    if not papers:
        return "No prior work found."
    return "\n".join(f"- {p['title']} ({p.get('published', '')[:4]}) <{p['url']}>: {p.get('abstract', '')[:240]}"
                     for p in papers[:k])


if __name__ == "__main__":
    import sys
    for p in search(sys.argv[1:] or ["centroid offset false positive kepler"]):
        print(p["published"], p["title"][:90], p["url"])
