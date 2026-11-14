#!/bin/sh
# Run one sealed practice trial (0, 1 or 2) against the kit's env and scorer.
# Usage: ./run-trial.sh [n] — needs the kit at $MAKEFUN_KIT.
set -eu
TRIAL="${1:-0}"
KIT="${MAKEFUN_KIT:-$HOME/workspace/makefun/kits/free-model-aggregator-01/free-model-aggregator-01-kit}"
ENTRY="$(cd "$(dirname "$0")" && pwd)"
STATE="$(mktemp -d)/state"
mkdir -p "$STATE"
ENV_PORT=$((18100 + TRIAL))
ENV_URL="http://127.0.0.1:$ENV_PORT"
trap 'kill "$ENV_PID" 2>/dev/null || true' EXIT INT TERM
MAKEFUN_ENV_DATA_DIR="$KIT/practice/trial-$TRIAL/input/env" \
MAKEFUN_STATE_DIR="$STATE" \
MAKEFUN_ENV_PORT="$ENV_PORT" \
  node "$KIT/program/env/server.mjs" >"$STATE/env.log" 2>&1 &
ENV_PID=$!
for _ in $(seq 1 100); do
  [ -f "$STATE/ready" ] && break
  sleep 0.1
done
[ -f "$STATE/ready" ] || { echo "env did not start"; cat "$STATE/env.log"; exit 1; }
MAKEFUN_ENV_URL="$ENV_URL" \
MAKEFUN_PROMPT_FILE="$KIT/practice/trial-$TRIAL/input/prompt.txt" \
MAKEFUN_TRIAL_SEED="practice-$TRIAL" \
  node "$ENTRY/main.ts"
ENTRY_CODE=$?
kill -TERM "$ENV_PID" 2>/dev/null || true
wait "$ENV_PID" 2>/dev/null || true
trap - EXIT INT TERM
echo "entry exit: $ENTRY_CODE"
[ -f "$STATE/export/state.json" ] || { echo "no state export"; cat "$STATE/env.log"; exit 1; }
OUT="$(mktemp -d)"
MAKEFUN_OUTPUT_DIR="$STATE/export" \
MAKEFUN_REFERENCE_DIR="$KIT/practice/trial-$TRIAL/reference" \
MAKEFUN_SCORES_DIR="$OUT" \
  node "$KIT/program/scorer/score.mjs"
cat "$OUT/scores.json"
python3 -c "
import json
s=json.load(open('$STATE/export/state.json'))
for r in s['requests']:
    bad=[c for c in r['checks'] if not c['pass']]
    print(r['id'], 'valid' if r['valid'] else 'INVALID', r['model'], 'calls:', r['upstream_calls'], 'failures:', [(f['provider'], f['status']) for f in r['failures']], 'failed checks:', [(c['name'], c['detail']) for c in bad] if bad else '-')
print('totals:', s['totals'])
print('catalog_route:', s['catalog_route'])
"
