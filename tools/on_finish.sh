#!/usr/bin/env bash
# Wait for the supervisor to exit, then leave a marker on disk.
#
# Monitoring tools that live inside the conversation get torn down with it, so
# "no news" from them means nothing. This runs detached and records the outcome
# where it can be read back at any time.
set -u
cd "$(dirname "$0")/.."
pidfile=${1:-var/sup.pid}
marker=${2:-var/finished}
rm -f "$marker"
while [ -e "$pidfile" ] && kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null; do
    sleep 60
done
{
    echo "finished_at $(date '+%F %T')"
    echo "clips $(ls out/*.mp4 2>/dev/null | wc -l)"
    echo "last $(grep -oE '\[[0-9]+/[0-9]+\]' var/sup.out 2>/dev/null | tail -1)"
    tail -3 var/sup.out 2>/dev/null
} > "$marker"
