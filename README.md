# Exo Referee

**Self-improving AI grades its own work. This is the referee.**

Claude agents try to improve a model that tells real exoplanets from false positives in NASA Kepler
data. Every change they claim goes to a Referee that runs it sealed, checks it with five gates, and
scores it on 1,328 host stars no agent ever sees. A change becomes the new model only if it is promoted.

Live site: https://spectre-planets.vercel.app

## Result

Same agents, same data, same afternoon.

| Loop | Believes (validation AUC) | On new candidates |
|---|---|---|
| Starting model | | 0.978 |
| Naive loop, keeps anything that scores higher | 0.9999 | **0.902** |
| Guarded loop, keeps only what the Referee promotes | 0.9957 | **0.995** |

"New candidates" means the hidden stars, with the archive columns that encode the answer removed
(`kepler_name`, `koi_score`, `koi_fpflag_*`), the way a genuinely new candidate arrives.
The naive loop accepted a catalog-name cheat in its first minute and then rejected every honest idea.

What that means in practice: the hidden stars hold 539 real planets. Give each final model 539
follow-up observations, its top picks.

| Model | Real planets found | False alarms followed up |
|---|---|---|
| Naive loop | 416 | 123 |
| Starting model | 484 | 55 |
| Guarded loop | **521** | **18** |

Same telescope time: the guarded loop finds 105 more planets with about 7x fewer false alarms
(`demo/followup.py`).

An Opus agent independently rediscovered centroid vetting, the way astronomers catch background
eclipsing binaries, and it was promoted at +0.016 AUC on hidden stars.

## The five gates

The first one a claim fails is its verdict.

- **CRASH** every run in a macOS `sandbox-exec` sandbox that cannot read the label files must finish
- **NOISE** three seeds; paired bootstrap CI above zero, beyond twice the seed spread, positive on every seed
- **LEAK** retrain on shuffled labels three times; if it still scores AUC 0.6 or more, the answer is in the inputs
- **PRIOR** blank every light-curve measurement; if it still matches the incumbent, it reads a catalog, not the planet
- **OVERFIT** hidden test stars; the CI must stay above zero

We found and fixed two false-positive bugs in these gates during the build, and re-judged the
affected claims in the open.

## Run it

```
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python numpy pandas pyarrow scikit-learn scipy anthropic openai python-dotenv
echo "ANTHROPIC_API_KEY=..." > .env

cd exo
mkdir -p data/raw data/hidden && curl -o data/raw/koi.csv \
  "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"
../.venv/bin/python prep.py
../.venv/bin/python swarm.py --proposers 3 --rounds 4 --redteam     # guarded loop
../.venv/bin/python swarm.py --naive --proposers 3 --rounds 4        # naive loop
cd ..
.venv/bin/python demo/cheat.py name --fresh                          # plant a cheat live, watch LEAK
.venv/bin/python demo/compare.py                                     # believed vs real chart
.venv/bin/python site/build_data.py                                  # site data
```

## Where it comes from

Built at Sundai on November 22, 2026. The gate design follows my earlier project
[Spectre](https://github.com/VolodymyrLinuxovich/spectre), which applied the same idea to
mass spectra at the SF AI Research Hackathon. This repo is new code for a new domain.

## Limits

The exoplanet classifier itself works on a well-known public table. `exo/rank.py` ranks the 1,977
unresolved Kepler candidates, but that ranking could not be validated: only one has been confirmed
since, and the model gave it 0.25. Treat it as a follow-up list, not a discovery.

Data: NASA Exoplanet Archive. Not affiliated with NASA.
