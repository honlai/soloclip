"""Throughput of the most recent run logs.

A cumulative clip count on its own is misleading - it says nothing about when
those clips appeared. Always pair it with elapsed time and a per-video rate.
Failed videos count towards throughput too: they still pay for download, asr,
diarize and asd before selection rejects them.
"""

from __future__ import annotations

import datetime
import glob
import os
import sys


def main() -> int:
    logs = sorted(glob.glob("logs/*-run.log"), key=os.path.getmtime)[-2:]
    if not logs:
        print("rate       (no run logs yet)")
        return 0
    for path in logs:
        rows = open(path, errors="replace").read().splitlines()
        if not rows:
            continue
        stamp = lambda line: datetime.datetime.strptime(line.split(",")[0], "%Y-%m-%d %H:%M:%S")
        try:
            minutes = (stamp(rows[-1]) - stamp(rows[0])).total_seconds() / 60
        except ValueError:
            continue
        made = sum(1 for l in rows if "rendered " in l)
        failed = sum(1 for l in rows if "selection failed" in l)
        handled = max(made + failed, 1)
        print(f"rate       {os.path.basename(path)} {minutes:.0f}min "
              f"made={made} failed={failed} -> {minutes / handled:.1f} min/video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
