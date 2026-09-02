#!/usr/bin/env bash
# Keep `soloclip run` going across native crashes.
#
# onnxruntime/insightface occasionally dies with SIGILL (exit 132) part-way
# through the ASD stage. There is no Python traceback because the crash is below
# Python, so it cannot be caught in-process. It is transient rather than
# content-specific: a video that crashed at frame 2160 completed normally on the
# next attempt. Restarting is cheap because `run` skips videos that already have
# a clip, so the supervisor simply relaunches.
#
# Guard against looping forever: if a restart produces no new clips, count it as
# a stall and give up after MAX_STALLS in a row.
#
#   tools/supervise.sh <output-log> [config.yaml]
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}   # activated env by default; export PY to override
OUT=${1:-supervise.out}
CONFIG=${2:-config.yaml}
MAX_STALLS=3

# The stall guard counts finished clips, so it has to look where *this* config
# writes them - otherwise a second list would appear to make no progress and be
# killed off after three restarts.
outdirs() {
  $PY - "$CONFIG" <<'EOF'
import sys, pathlib, yaml
cfg = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text()) or {}
paths = cfg.get("paths", {})
root = pathlib.Path(sys.argv[1]).resolve().parent
out = paths.get("out_dir", "out")
aud = paths.get("audio_out_dir") or (out + "_audio")
print(root / out)
print(root / aud)
EOF
}
DIRS=$(outdirs)
META=$($PY - "$CONFIG" <<'EOF'
import sys, pathlib, yaml
cfg = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text()) or {}
root = pathlib.Path(sys.argv[1]).resolve().parent
print(root / (cfg.get("paths", {}).get("work_dir", "work")) / "meta")
EOF
)

clips() { ls $(echo "$DIRS" | sed 's|$|/*|') 2>/dev/null | wc -l; }

# Progress, for the stall guard. Counting finished clips alone is wrong in
# per_stage mode: nothing is rendered until the whole list has been through
# every stage, so three crashes early on look like three stalls and the
# supervisor quits while real work is being done. Stage cache files advance in
# both modes, so count those too.
progress() { echo $(( $(clips) + $(ls "$META"/*.json 2>/dev/null | wc -l) )); }

stalls=0
attempt=0
while :; do
  attempt=$((attempt + 1))
  before=$(progress)
  echo "--- attempt $attempt starting, $(clips) clip(s), progress=$before" >> "$OUT"

  PYTHONPATH=src $PY -u -m soloclip -c "$CONFIG" run >> "$OUT" 2>&1
  rc=$?
  after=$(progress)
  echo "--- attempt $attempt exited rc=$rc, progress $before -> $after ($(clips) clips)" >> "$OUT"

  # rc 3 is a deliberate refusal to start (e.g. no usable GPU); restarting
  # would just fail the same way, so surface it instead of burning attempts.
  if [ $rc -eq 3 ]; then
    echo "=== SUPERVISOR STOPPING: run refused to start, see the error above" >> "$OUT"
    break
  fi
  if [ $rc -eq 0 ]; then
    echo "=== SUPERVISOR DONE: run completed cleanly after $attempt attempt(s)" >> "$OUT"
    break
  fi
  # rc 2 means "ran fine, produced nothing new" - not a crash, so stop.
  if [ $rc -eq 2 ] && [ "$after" = "$before" ]; then
    echo "=== SUPERVISOR DONE: nothing left to produce" >> "$OUT"
    break
  fi

  if [ "$after" = "$before" ]; then
    stalls=$((stalls + 1))
    echo "--- no progress at all (stall $stalls/$MAX_STALLS)" >> "$OUT"
    if [ $stalls -ge $MAX_STALLS ]; then
      echo "=== SUPERVISOR GIVING UP: $MAX_STALLS restarts with no progress at all" >> "$OUT"
      break
    fi
  else
    stalls=0
  fi
  sleep 5
done
