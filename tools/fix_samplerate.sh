#!/usr/bin/env bash
# Re-mux existing clips to a sane audio sample rate.
#
# Clips rendered before the loudnorm/aresample ordering fix carry 96 kHz audio.
# The picture is untouched (stream copy) and only the audio is re-encoded, so
# this does not need the original download - which cleanup has already deleted.
set -euo pipefail
cd "$(dirname "$0")/.."
rate=${1:-48000}
fixed=0; skipped=0
for f in out/*.mp4 out_audio/*.m4a; do
    [ -e "$f" ] || continue
    have=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 "$f" 2>/dev/null || echo "")
    if [ "$have" = "$rate" ]; then skipped=$((skipped+1)); continue; fi
    tmp="${f%.*}.fixing.${f##*.}"
    if ffmpeg -v error -y -i "$f" -c:v copy -c:a aac -b:a 128k -ar "$rate" -movflags +faststart "$tmp" 2>/dev/null; then
        mv -f "$tmp" "$f"; fixed=$((fixed+1))
    else
        rm -f "$tmp"; echo "FAILED $f" >&2
    fi
done
echo "resampled $fixed, already correct $skipped"
