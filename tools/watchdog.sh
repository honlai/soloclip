#!/usr/bin/env bash
# Turn silence into a signal.
#
# A `tail -F | grep` monitor only fires when new lines appear, so a dead process
# is indistinguishable from a busy one - both produce nothing. This polls the
# things that actually prove liveness and emits a line only when the state
# changes, so it stays quiet while healthy but speaks up the moment it is not.
#
# Emits on: process gone, log gone stale, recovery, supervisor finishing.
set -u
cd "$(dirname "$0")/.."
SUP_OUT=${1:?usage: watchdog.sh <supervisor-output> [stale_seconds]}
STALE=${2:-600}

# Match the interpreter path, not the words "soloclip run" - the latter also
# matches the shell running this very check, which is how a dead run once got
# reported as healthy.
alive() { pgrep -f "bin/python -u -m soloclip" >/dev/null; }
clips() { ls out/*.mp4 2>/dev/null | wc -l; }
newest_log() { ls -t logs/*-run.log logs/*.log 2>/dev/null | head -1; }

state=""
while :; do
  log=$(newest_log)
  age=$(( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || date +%s) ))

  if grep -q "=== SUPERVISOR" "$SUP_OUT" 2>/dev/null; then
    echo "SUPERVISOR FINISHED: $(grep '=== SUPERVISOR' "$SUP_OUT" | tail -1) | clips=$(clips)"
    break
  fi

  if alive; then
    if [ "$age" -gt "$STALE" ]; then
      now="stalled"
      msg="STALLED: no log write for ${age}s though the process is alive | clips=$(clips)"
    else
      now="ok"
      msg="RECOVERED: log advancing again | clips=$(clips)"
    fi
  else
    now="dead"
    msg="DEAD: no soloclip process running | clips=$(clips) | last log line: $(tail -1 "$log" 2>/dev/null | cut -c1-100)"
  fi

  # only speak when something changed, and never announce the initial healthy state
  if [ "$now" != "$state" ]; then
    [ "$now" = "ok" ] && [ -z "$state" ] || echo "$msg"
    state="$now"
  fi
  sleep 60
done
