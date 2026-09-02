#!/usr/bin/env bash
# Run several lists one after another under the crash supervisor.
#
# They must not overlap: 4GB of VRAM cannot hold two copies of
# pyannote + whisper + insightface, so running two lists at once would either
# OOM or thrash. Sequential also keeps the logs readable.
#
#   tools/run_queue.sh configs/interviews.yaml configs/talks.yaml
set -u
cd "$(dirname "$0")/.."
mkdir -p var
QUEUE_LOG=var/queue.out
: > "$QUEUE_LOG"

for cfg in "$@"; do
    name=$(basename "$cfg" .yaml)
    out="var/sup.${name}.out"
    echo "=== $(date '+%F %T') starting $cfg -> $out" >> "$QUEUE_LOG"
    bash tools/supervise.sh "$out" "$cfg"
    echo "=== $(date '+%F %T') finished $cfg" >> "$QUEUE_LOG"
    tail -2 "$out" >> "$QUEUE_LOG"
done
echo "=== $(date '+%F %T') QUEUE DONE" >> "$QUEUE_LOG"
