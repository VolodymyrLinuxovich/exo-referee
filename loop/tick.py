"""One cycle of the Redshift business loop. Runs unattended on a schedule.

  .venv/bin/python loop/tick.py              # full cycle
  .venv/bin/python loop/tick.py --no-agents  # data + report only (no model spend)

Each cycle, in order:
  1. Check the checker. Run exo/selftest.py. If any gate or the sandbox fails, stop: no agent
     claim is judged by a broken referee, and the report says why. A person fixes it.
  2. Process new data. Pull today's dispositions from the NASA Exoplanet Archive and find every
     Kepler candidate that was unresolved in our 2026-11-22 snapshot and has since been confirmed
     or rejected. Score our ranking of those candidates against the new answers.
  3. Run background work. One round of the guarded swarm: agents research prior work through the
     Apify Actor, propose a change, and the Referee judges it. Only promoted code moves the model.
  4. Deploy updates. Rebuild the site data from the new verdicts and run loop/deploy.sh if present.
  5. Report. Write egbe-docs/reports/loop-<time>.md and keep loop/state.json for the next cycle.

Humans stay on the loop: every cycle leaves a report, and a failed self-test halts the loop.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXO = ROOT / "exo"
PY = str(ROOT / ".venv/bin/python")
STATE = ROOT / "loop/state.json"
REPORTS = ROOT / "egbe-docs/reports"
NASA = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
        "select+kepoi_name,koi_disposition+from+cumulative&format=csv")


def sh(args, cwd=ROOT, timeout=3600):
    env = {**os.environ, "OS_AGENT_NAME": "Claude Code"}   # how this agent shows up in the Egbe dashboard
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    return r.returncode, (r.stdout + r.stderr)[-4000:]


def check_the_checker():
    code, out = sh([PY, "selftest.py"], cwd=EXO, timeout=1800)
    lines = [l for l in out.splitlines() if "pass" in l or "FAIL" in l or "checks" in l]
    return code == 0, lines


def new_answers():
    """Candidates unresolved in the snapshot that NASA has resolved since, with our score for each."""
    with urllib.request.urlopen(urllib.request.Request(NASA, headers={"User-Agent": "exo-referee loop"}), timeout=120) as r:
        now = {row["kepoi_name"]: row["koi_disposition"] for row in csv.DictReader(io.StringIO(r.read().decode()))}
    ranked = list(csv.DictReader(open(EXO / "runs/unresolved_ranked.csv")))
    resolved = [(r["kepoi_name"], float(r["p_planet"]), now.get(r["kepoi_name"]))
                for r in ranked if now.get(r["kepoi_name"]) in ("CONFIRMED", "FALSE POSITIVE")]
    out = {"checked": len(ranked), "resolved_since_snapshot": len(resolved),
           "resolved": [{"koi": k, "p_planet": round(p, 3), "now": d} for k, p, d in resolved]}
    pos = [p for _, p, d in resolved if d == "CONFIRMED"]
    neg = [p for _, p, d in resolved if d == "FALSE POSITIVE"]
    if pos and neg:   # AUC of our ranking on answers that did not exist when it was made
        wins = sum((a > b) + .5 * (a == b) for a in pos for b in neg)
        out["auc_on_new_answers"] = round(wins / (len(pos) * len(neg)), 3)
    return out


def agents_round(proposers: int):
    code, out = sh([PY, "swarm.py", "--proposers", str(proposers), "--rounds", "1", "--tag", "exo-loop"], cwd=EXO)
    verdicts = []
    for line in open(EXO / "runs/events.jsonl"):
        e = json.loads(line)
        if e.get("tag") == "exo-loop" and e["kind"] == "verdict" and e["ts"] > START:
            verdicts.append({"verdict": e["verdict"], "claim": e.get("claim", "")[:160],
                             "val_gain": e.get("val_gain"), "evidence": e.get("public_evidence") or e.get("evidence", "")})
        if e.get("tag") == "exo-loop" and e["kind"] == "error" and e["ts"] > START:
            verdicts.append({"verdict": "ERROR", "claim": e.get("error", "")[:160]})
    return code == 0, verdicts


def deploy():
    steps = []
    for args, cwd in (([PY, "site/build_data.py"], ROOT), ([PY, "demo/followup.py"], ROOT)):
        code, _ = sh(args, cwd=cwd)
        steps.append(f"{' '.join(Path(a).name for a in args[1:])}: {'ok' if code == 0 else 'failed'}")
    if (ROOT / "demo/followup.json").exists():
        (ROOT / "site/followup.json").write_text((ROOT / "demo/followup.json").read_text())
    hook = ROOT / "loop/deploy.sh"
    if hook.exists():
        code, out = sh(["bash", str(hook)])
        steps.append(f"deploy.sh: {'ok' if code == 0 else 'failed: ' + out.strip().splitlines()[-1][:120]}")
    else:
        steps.append("deploy.sh: not configured, site rebuilt locally only")
    return steps


def report(r: dict) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    t = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%MZ")
    lines = [f"# Redshift loop, {t}", "", f"Status: **{r['status']}**", ""]
    lines += ["## Check the checker", *[f"- {l.strip()}" for l in r["selftest"]], ""]
    na = r.get("new_answers")
    if na:
        lines += ["## New data from NASA",
                  f"- Unresolved candidates tracked: {na['checked']}",
                  f"- Resolved by NASA since the 2026-11-22 snapshot: {na['resolved_since_snapshot']}"]
        if "auc_on_new_answers" in na:
            lines.append(f"- AUC of our ranking on those new answers: {na['auc_on_new_answers']}")
        lines += [f"  - {x['koi']}: now {x['now']}, we scored {x['p_planet']}" for x in na["resolved"][:20]]
        lines.append("")
    elif "new_answers_error" in r:
        lines += ["## New data from NASA", f"- Not checked: {r['new_answers_error']}", ""]
    if "verdicts" in r:
        judged = [f"- **{v['verdict']}**: {v['claim']}" for v in r["verdicts"]] or ["- none"]
        lines += ["## Agent claims judged", *judged, ""]
    if "deploy" in r:
        lines += ["## Deploy", *[f"- {s}" for s in r["deploy"]], ""]
    p = REPORTS / f"loop-{t}.md"
    p.write_text("\n".join(lines))
    return p


START = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-agents", action="store_true", help="skip the agent round (no model spend)")
    ap.add_argument("--proposers", type=int, default=int(os.environ.get("LOOP_PROPOSERS", "2")))
    args = ap.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else {"cycles": 0}
    r = {"status": "ok"}

    ok, r["selftest"] = check_the_checker()
    if not ok:
        r["status"] = "halted: self-test failed, no claims judged until a person fixes the referee"
    else:
        try:
            r["new_answers"] = new_answers()
        except Exception as e:  # noqa: BLE001  the archive being down must not stop the loop
            r["new_answers_error"] = repr(e)[:200]
        if not args.no_agents:
            _, r["verdicts"] = agents_round(args.proposers)
        r["deploy"] = deploy()

    path = report(r)
    # hand the report to Egbe: the business's loop plan, reports and checkpoints live there
    code, out = sh([PY, ".claude/skills/egbe-os-quickstart/scripts/sync-artifacts.py"], timeout=300) \
        if (ROOT / ".claude/skills/egbe-os-quickstart/scripts/sync-artifacts.py").exists() else (1, "Egbe not installed")
    synced = "synced to Egbe" if code == 0 else "Egbe sync failed: " + out.strip().splitlines()[-1][:160]
    path.write_text(path.read_text() + f"\n## Egbe\n- {synced}\n")
    state.update(cycles=state.get("cycles", 0) + 1, last_run=datetime.now(timezone.utc).isoformat(),
                 last_status=r["status"], last_report=str(path.relative_to(ROOT)))
    STATE.write_text(json.dumps(state, indent=1))
    print(path.read_text())
    sys.exit(0 if r["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
