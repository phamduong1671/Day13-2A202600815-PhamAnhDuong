#!/usr/bin/env sh
set -eu

TEAM="${TEAM:-PhamAnhDuong}"
SIM="${SIM:-./observathon-sim}"
SCORE="${SCORE:-./observathon-score}"

if [ -z "${LOCAL_BASE_URL:-}" ]; then
  if grep -qi microsoft /proc/version 2>/dev/null; then
    HOST_IP="$(awk '/nameserver/ {print $2; exit}' /etc/resolv.conf)"
    export LOCAL_BASE_URL="http://${HOST_IP}:11434/v1"
  else
    export LOCAL_BASE_URL="http://127.0.0.1:11434/v1"
  fi
fi

chmod +x "$SIM" "$SCORE" 2>/dev/null || true
"$SIM" --config solution/config.json --wrapper solution/wrapper.py --out run_output.json --concurrency "${CONCURRENCY:-2}"
"$SCORE" --run run_output.json --findings solution/findings.json --team "$TEAM" --out score.json
