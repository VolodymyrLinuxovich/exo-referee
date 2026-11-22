#!/usr/bin/env bash
# Deploy step of the Redshift loop: publish the site on Egbe hosting (<name>.egbe.app).
#
# Copies the built site into loop/publish (its own git repo), commits if anything changed,
# pushes it to the business's Egbe git repo, and asks Egbe to build and serve it.
# Credentials come from the project's git-ignored .env.local and are never printed.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
SLUG=redshift
REPO=redshift-site
PUB=$ROOT/loop/publish
HELPER=$ROOT/.claude/skills/egbe-os-quickstart/scripts/egbe-curl.py
PY=$ROOT/.venv/bin/python
export OS_AGENT_NAME="Claude Code"
egbe() { "$PY" "$HELPER" --egbe-request "$@"; }

OS_URL=$(grep '^OS_URL=' .env.local | cut -d= -f2-)

# 1. the site, plus a tiny non-root web server on 8080
mkdir -p "$PUB"
cp site/index.html site/data.json site/followup.json "$PUB"/
cp demo/compare.html "$PUB"/
cat > "$PUB/Dockerfile" <<'EOF'
FROM nginxinc/nginx-unprivileged:alpine
COPY index.html data.json followup.json compare.html /usr/share/nginx/html/
EXPOSE 8080
EOF
printf 'Dockerfile\n.git\n' > "$PUB/.dockerignore"

# 2. commit only when the site actually changed
cd "$PUB"
[ -d .git ] || { git init -q -b main; git config user.name "Redshift loop"; git config user.email "loop@exo-referee.local"; }
git add -A
if git diff --cached --quiet && git rev-parse -q --verify HEAD >/dev/null; then
  echo "site unchanged"
else
  git commit -qm "Site update $(date -u +%Y-%m-%dT%H:%MZ)"
fi

# 3. the Egbe repo (created once) and the push, authenticated by header only
egbe "/git/repos" -s -X POST -H "Content-Type: application/json" -d "{\"name\":\"$REPO\",\"private\":true}" >/dev/null 2>&1 || true
REMOTE="$OS_URL/git/http/$REPO.git"
git remote get-url egbe >/dev/null 2>&1 || git remote add egbe "$REMOTE"
OS_GIT_AUTH="Authorization: Bearer $(grep '^OS_API_KEY=' "$ROOT/.env.local" | cut -d= -f2-)" \
  git --config-env="http.$REMOTE.extraHeader=OS_GIT_AUTH" push -q -u egbe HEAD:main
SHA=$(git rev-parse HEAD)

# 4. build and serve
cd "$ROOT"
egbe "/deploy" -s -X POST -H "Content-Type: application/json" -d "{\"slug\":\"$SLUG\",\"business_name\":\"Redshift\",\"repo\":\"$REPO\",\"ref\":\"$SHA\"}"
echo
echo "deployed $SHA"
