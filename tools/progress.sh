#!/usr/bin/env bash
# True progress across restarts.
#
# The "=== [n/229]" counter in the log restarts from 1 on every supervisor
# relaunch, so it measures the current attempt, not the job. What actually
# persists is the per-video stage cache: a video that reached `select` has been
# processed, whichever attempt did it.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}   # activated env by default; export PY to override
PYTHONPATH=src $PY - <<'PYEOF'
import datetime, glob, os, sys
sys.path.insert(0, "src")
from soloclip.config import load_config
from soloclip.utils import read_url_lists

cfg = load_config("config.yaml")
urls = read_url_lists(cfg.url_lists())
ids = [u.rstrip("/").rsplit("/", 1)[-1].split("?")[0] for u in urls]

processed = [v for v in ids if (cfg.meta_dir / f"{v}.select.json").exists()]
clips = [v for v in ids if (cfg.out_dir / f"{v}.mp4").exists()]
failed = len(processed) - len(clips)

mins = n = 0.0
for f in sorted(glob.glob("logs/*-run.log"), key=os.path.getmtime)[-4:]:
    lines = open(f, errors="replace").read().splitlines()
    if len(lines) < 2:
        continue
    ts = lambda l: datetime.datetime.strptime(l.split(",")[0], "%Y-%m-%d %H:%M:%S")
    done = sum(1 for l in lines if "rendered " in l or "selection failed" in l)
    if done:
        mins += (ts(lines[-1]) - ts(lines[0])).total_seconds() / 60
        n += done
rate = mins / n if n else 0.0
left = len(ids) - len(processed)

print(f"  time       {datetime.datetime.now():%H:%M:%S}")
print(f"  processed  {len(processed)}/{len(ids)}   clips {len(clips)}  failed {failed}"
      f"  ({100*len(clips)/max(len(processed),1):.0f}% produced)")
print(f"  rate       {rate:.1f} min/video")
print(f"  remaining  {left} videos, about {left*rate/60:.1f} h")
PYEOF
