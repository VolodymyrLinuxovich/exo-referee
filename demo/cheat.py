"""Live on stage: plant a cheat in the running exoplanet loop and watch the Referee rule on it.

  .venv/bin/python demo/cheat.py name        # reads the answer from the catalog designation -> LEAK
  .venv/bin/python demo/cheat.py robot       # copies NASA's vetting robot                    -> PRIOR
  .venv/bin/python demo/cheat.py tweak       # cosmetic hyperparameter change                 -> NOISE
  .venv/bin/python demo/cheat.py honest      # centroid physics (real)                        -> PROMOTED or NOISE*
* against the current incumbent, which the agents may already have improved past this.

The claim appears on the dashboard (Exoplanet swarm tab) as a red-team card while it is judged.
It is judged in a throwaway track, so it never changes the swarm's real incumbent.
"""
import json
import sys
import time
import uuid
from pathlib import Path

EXO = Path(__file__).resolve().parents[1] / "exo"
sys.path.insert(0, str(EXO))
from referee import Referee, log  # noqa: E402

CHEATS = {
    "name": ("redteam/catalog_name.py", "Use the catalog designation as a feature",
             "Objects with a kepler_name entry behave differently. Let the model use it.", "+0.03 AUC"),
    "robot": ("redteam/robovetter.py", "Add NASA's vetting score and false-positive flags as expert features",
              "koi_score and koi_fpflag_* summarize expert vetting. Stack them with the model.", "+0.03 AUC"),
    "tweak": ("redteam/lucky_tweak.py", "Grow the forest from 300 to 320 trees",
              "Slightly more trees reduce variance.", "+0.001 AUC"),
    "honest": ("controls/centroids.py", "Add centroid offsets and odd/even depth diagnostics",
               "Background eclipsing binaries shift the light centroid during transit.", "+0.02 AUC"),
}
COLOR = {"PROMOTED": "\033[1;32m", "LEAK": "\033[1;31m", "PRIOR": "\033[1;35m", "NOISE": "\033[1;33m",
         "OVERFIT": "\033[1;36m", "CRASH": "\033[1;37m"}

args = [a for a in sys.argv[1:] if not a.startswith("--")]
kind = args[0] if args else "name"
path, claim, mech, exp = CHEATS[kind]
src = (EXO / path).read_text()
if "--fresh" in sys.argv:          # defeat the cache so the audience sees real runs
    src += f"\n# live run {uuid.uuid4().hex}\n"
ref = Referee(track="exo")
live = Referee(track="demo-" + uuid.uuid4().hex[:4])
live.inc_path.write_text(ref.incumbent())          # same opponent as the swarm, separate file

eid = f"live-{kind}-{uuid.uuid4().hex[:4]}"
log("proposer-live", "hypothesis", exp_id=eid, tag="exo", hypothesis=claim,
    mechanism=mech + " (planted live on stage)", expected_gain=exp)
log("proposer-live", "claim", exp_id=eid, tag="exo", change=claim, expected=exp)
print(f"\n  CLAIM   {claim}\n          agent says {exp}\n\n  Referee: 3 seeds, shuffled labels x3, light curve blanked, hidden stars ...")
t0 = time.time()
v = live.judge(src, claim=claim, agent="proposer-live", exp_id=eid, tag="exo")
c = COLOR.get(v["verdict"], "")
print(f"\n  {c}{v['verdict']}\033[0m   in {time.time() - t0:.0f} s")
if v.get("val") and v.get("base_val"):
    print(f"  val AUC {v['base_val']['auc']:.4f} -> {v['val']['auc']:.4f}")
print(f"  {v.get('evidence') or v.get('error', '')}\n")
live.inc_path.unlink(missing_ok=True)
