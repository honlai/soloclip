"""Threshold sweep over cached ASD measurements.

Re-detecting faces costs ~2.5 min of GPU per video; re-judging cached
measurements costs milliseconds. This walks a grid of thresholds and reports
what each one would actually produce - clip length and join count - rather than
just a pass rate, because a higher pass rate that still cannot assemble 20
seconds is not an improvement.

    python tools/sweep.py --yaw 30 35 40 --size 0.06 0.05 0.04
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soloclip import asd, select  # noqa: E402
from soloclip.config import Config, load_config  # noqa: E402
from soloclip.utils import read_stage, write_stage  # noqa: E402


def variant(cfg: Config, **overrides) -> Config:
    """A copy of the config with asd.* / select.* keys overridden."""
    data = copy.deepcopy(cfg._data)
    for dotted, value in overrides.items():
        node = data
        parts = dotted.split(".")
        for key in parts[:-1]:
            node = node.setdefault(key, {})
        node[parts[-1]] = value
    return Config(data, cfg.root)


def evaluate(cfg: Config, video_ids: list[str]) -> dict[str, dict]:
    """Score frames and run selection under this config, touching no caches."""
    out = {}
    for vid in video_ids:
        frames = asd.load_frames(cfg, vid)
        if frames is None:
            continue
        record = asd.score_frames(cfg, vid, frames)
        # Selection reads the asd stage file and writes its own, so both have to be
        # put back afterwards. Restoring only asd leaves select.json holding the
        # last trial's thresholds, which then silently disagrees with out/.
        saved = {stage: read_stage(cfg.meta_dir, vid, stage) for stage in ("asd", "select")}
        write_stage(cfg.meta_dir, vid, "asd", record)
        try:
            sel = select.select_one(cfg, vid, force=True)
        except Exception as exc:
            sel = {"status": "error", "reason": str(exc), "total": 0.0, "joins": 0}
        finally:
            for stage, previous in saved.items():
                if previous is not None:
                    write_stage(cfg.meta_dir, vid, stage, previous)
                else:
                    stage_path = cfg.meta_dir / f"{vid}.{stage}.json"
                    stage_path.unlink(missing_ok=True)
        out[vid] = {
            "pass": record["criteria"]["all_ok"],
            "spans": len(record["good_spans"]),
            "status": sel.get("status"),
            "total": sel.get("total", 0.0) or 0.0,
            "joins": sel.get("joins", 0) or 0,
            "strategy": sel.get("strategy"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--yaw", type=float, nargs="+", default=[30.0])
    ap.add_argument("--size", type=float, nargs="+", default=[0.06])
    args = ap.parse_args()

    base = load_config(args.config)
    ids = sorted(p.name[: -len(".asd_frames.npz")]
                 for p in base.cache_dir.glob("*.asd_frames.npz"))
    if not ids:
        print("no cached ASD measurements; run `soloclip asd` first")
        return 1
    print(f"{len(ids)} video(s) with cached measurements\n")

    for size in args.size:
        for yaw in args.yaw:
            cfg = variant(base, **{"asd.max_yaw_deg": yaw, "asd.min_face_ratio": size})
            res = evaluate(cfg, ids)
            ok = [r for r in res.values() if r["status"] == "ok"]
            good = [r for r in ok if r["total"] >= float(base.get("select.min_seconds", 8))
                    and r["total"] >= 15.0]
            zero_join = [r for r in ok if r["joins"] == 0]
            mean_len = sum(r["total"] for r in ok) / len(ok) if ok else 0.0
            mean_join = sum(r["joins"] for r in ok) / len(ok) if ok else 0.0
            print(f"yaw<={yaw:<5g} size>={size:<6g}  "
                  f"clips {len(ok)}/{len(ids)}  >=15s {len(good)}  0-join {len(zero_join)}  "
                  f"avg {mean_len:5.1f}s  avg joins {mean_join:.2f}")
            for vid in ids:
                r = res[vid]
                mark = "ok " if r["status"] == "ok" else "FAIL"
                print(f"    {vid:<13} {mark} {r['total']:5.1f}s {r['joins']}j  "
                      f"pass {r['pass']:4.0%} spans {r['spans']:>3}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
