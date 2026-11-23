"""Build site/data.json from the exoplanet loop's event log.

  .venv/bin/python site/build_data.py        (run from the repo root)

Reads exo/runs/events.jsonl (tags "exo" = guarded Referee, "exo-naive" = naive loop).
Re-run any time; the page only shows what the log contains.
"""

import json
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVENTS = HERE.parent / "exo/runs/events.jsonl"
OUT = HERE / "data.json"

ev = [json.loads(line) for line in EVENTS.read_text().splitlines() if line.strip()]
hyp = {e["exp_id"]: e for e in ev if e["kind"] == "hypothesis"}


def auc(d):
    return (d or {}).get("auc")


def planted(who):
    """Anything not written by an LLM proposer (red team, live demo cheat) is a planted claim."""
    return not re.fullmatch(r"proposer-\d+", who or "")


def claim_row(v):
    h = hyp.get(v.get("exp_id"), {})
    who = v.get("proposer") or v.get("agent") or "?"
    row = {
        "id": v.get("exp_id"),
        "ts": v["ts"],
        "who": ("live demo" if who == "proposer-live" else "red team") if planted(who) else who,
        "model": h.get("model") or ("planted cheat" if planted(who) else ""),
        "claim": (h.get("hypothesis") or v.get("claim") or "")[:400],
        "verdict": v["verdict"],
        "evidence": (v.get("evidence") or v.get("error") or "")[:300],
        "val_gain": v.get("val_gain"),
        "val_auc": auc(v.get("val")),
        "base_auc": auc(v.get("base_val")),
        "test_gain": v.get("test_gain"),
        "test_ci": v.get("test_ci"),
        "redteam": planted(who),
        "note": "",
    }
    if str(v.get("exp_id", "")).endswith("-r"):
        row["note"] = (
            "Re-judged after we fixed a false-positive bug in the PRIOR gate. "
            "The first verdict on this exact code was PRIOR, and it was wrong."
        )
    elif v["verdict"] == "PRIOR" and "ablate_auc" not in v and not row["redteam"]:
        row["note"] = (
            "Judged under the first PRIOR rule, which had a false-positive bug. "
            "Superseded by the re-judged card above, and left in the log on purpose."
        )
    return row


def loop(tag, kind, accept):
    vs = sorted((e for e in ev if e.get("tag") == tag and e["kind"] == kind), key=lambda e: e["ts"])
    vs = [v for v in vs if v.get("proposer") != "cli"]
    # counts: one verdict per claim. Superseded first verdicts (re-judged as <id>-r) and live stage
    # demos stay in the feed but are not counted twice.
    redone = {str(v.get("exp_id"))[:-2] for v in vs if str(v.get("exp_id", "")).endswith("-r")}
    counts = {}
    for v in vs:
        if v.get("exp_id") in redone or v.get("proposer") == "proposer-live":
            continue
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    timeline = []
    first = next((v for v in vs if v.get("base_val")), None)
    if first:
        timeline.append({"ts": first["ts"], "auc": auc(first["base_val"]), "label": "baseline"})
    for v in vs:
        if v["verdict"] == accept and v.get("val"):
            timeline.append(
                {
                    "ts": v["ts"],
                    "auc": auc(v["val"]),
                    "label": (hyp.get(v.get("exp_id"), {}).get("hypothesis") or v.get("claim") or "")[:120],
                    "redteam": planted(v.get("proposer")),
                    "test_gain": v.get("test_gain"),
                }
            )
    claims = [claim_row(v) for v in vs]
    return {"counts": counts, "timeline": timeline, "claims": claims[::-1], "n": sum(counts.values())}


data = {
    "built": time.strftime("%Y-%m-%d %H:%M"),
    "guarded": loop("exo", "verdict", "PROMOTED"),
    "naive": loop("exo-naive", "naive_verdict", "ACCEPTED"),
}
g = data["guarded"]["timeline"]
cmp = Path(__file__).resolve().parents[1] / "demo/compare.json"  # written by demo/compare.py
if cmp.exists():
    c = json.loads(cmp.read_text())
    data["reality"] = {
        "guarded": c["reality"]["guarded"]["answers_removed"],
        "naive": c["reality"]["naive"]["answers_removed"],
        "baseline": c["baseline"]["answers_removed"],
    }
data["hidden_gain_total"] = sum(p.get("test_gain") or 0 for p in g)
OUT.write_text(json.dumps(data, indent=1))
print(
    f"wrote {OUT}: guarded {data['guarded']['n']} verdicts {data['guarded']['counts']}, "
    f"naive {data['naive']['n']} verdicts {data['naive']['counts']}"
)
