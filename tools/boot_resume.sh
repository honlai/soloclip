#!/usr/bin/env bash
# Resume the queue after a WSL restart.
#
# The supervisor survives a crashed *process*, but not a crashed *VM*: when the
# GPU passthrough takes WSL down (dxgk ioctl failures, then a reboot), every
# process dies including the supervisor, and the run sits idle until a human
# notices - one such gap cost eight hours.
#
# Resuming is safe and cheap: finished videos are skipped in about a second
# each, so this can fire on every boot without checking anything.
set -u
cd "$(dirname "$0")/.."
# cron runs with a minimal PATH, so the env that owns `soloclip` has to be named
# explicitly. Put `SOLOCLIP_ENV=/path/to/env/bin` in tools/local.env.
[ -f tools/local.env ] && . tools/local.env
[ -n "${SOLOCLIP_ENV:-}" ] && export PATH="$SOLOCLIP_ENV:$PATH"

# Only resume while a queue is meant to be running.
[ -e var/queue.wanted ] || exit 0
# Do not stack a second queue on top of a live one.
if [ -e var/queue.pid ] && kill -0 "$(cat var/queue.pid 2>/dev/null)" 2>/dev/null; then
    exit 0
fi

sleep 30   # let the GPU passthrough settle before loading CUDA
mkdir -p var
echo "=== $(date '+%F %T') boot resume" >> var/queue.out
setsid bash tools/run_queue.sh $(cat var/queue.wanted) >/dev/null 2>&1 </dev/null &
echo $! > var/queue.pid
