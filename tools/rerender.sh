#!/usr/bin/env bash
# Re-apply thresholds and rebuild every clip, one video at a time.
# Uses the cached per-frame ASD measurements, so no GPU and no video decode.
set -u
PY=${PY:-python}   # activated env by default; export PY to override
cd "$(dirname "$0")/.."
export PYTHONPATH=src

mapfile -t VIDS < <(ls work/cache/*.asd_frames.npz | sed 's|.*/||;s|\.asd_frames\.npz||')
echo "${#VIDS[@]} video(s) with cached measurements"

for vid in "${VIDS[@]}"; do
  # note the = form: ids such as -ABCDEFGHIJ start with a dash and would
  # otherwise be parsed as an option flag
  echo "=== $vid"
  $PY -m soloclip asd    --rescore --video-id="$vid" 2>&1 | grep -E "INFO|ERROR" | sed 's/^/    /'
  $PY -m soloclip select --force   --video-id="$vid" 2>&1 | grep -E "INFO|WARNING|ERROR" | sed 's/^/    /'
  $PY -m soloclip render --force   --video-id="$vid" 2>&1 | grep -E "INFO|ERROR" | sed 's/^/    /'
done
