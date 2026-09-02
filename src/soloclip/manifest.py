"""Per-run summary of what came out, and why anything did not."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .config import Config
from .utils import LOG, read_stage


def build_row(cfg: Config, video_id: str) -> dict[str, Any]:
    dl = read_stage(cfg.meta_dir, video_id, "download") or {}
    diar = read_stage(cfg.meta_dir, video_id, "diarize") or {}
    asd = read_stage(cfg.meta_dir, video_id, "asd") or {}
    sel = read_stage(cfg.meta_dir, video_id, "select") or {}
    ren = read_stage(cfg.meta_dir, video_id, "render") or {}

    return {
        "video_id": video_id,
        "url": dl.get("url"),
        "title": dl.get("title"),
        "source_duration": dl.get("duration"),
        "num_speakers": diar.get("num_speakers"),
        "audio_clean_seconds": diar.get("clean_total"),
        "asd_spans": len(asd.get("good_spans", [])),
        "status": ren.get("status") or sel.get("status") or "incomplete",
        "mode": ren.get("mode") or sel.get("mode"),
        "reason": ren.get("reason") or sel.get("reason"),
        "strategy": sel.get("strategy"),
        "clip_seconds": sel.get("total"),
        "joins": sel.get("joins"),
        "source_spans": sel.get("pieces"),
        "out_path": ren.get("out_path"),
    }


def append_row(cfg: Config, video_id: str) -> dict[str, Any]:
    """Record one video's outcome as soon as it is finished, not at the end."""
    row = build_row(cfg, video_id)
    row["run_at"] = datetime.now().isoformat(timespec="seconds")
    path = cfg.meta_dir / "manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    detail = f"{row['clip_seconds']:.1f}s/{row['joins']}j" if row.get("clip_seconds") else "-"
    LOG.info("[%s] manifest: %s %s", video_id, row["status"], detail)
    return row


def write_manifest(cfg: Config, video_ids: list[str]) -> tuple[int, int]:
    path = cfg.meta_dir / "manifest.jsonl"
    stamp = datetime.now().isoformat(timespec="seconds")
    ok = 0
    with path.open("a", encoding="utf-8") as fh:
        for vid in video_ids:
            row = build_row(cfg, vid)
            row["run_at"] = stamp
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if row["status"] == "ok":
                ok += 1
    LOG.info("manifest: %d/%d produced a clip -> %s", ok, len(video_ids), path)
    return ok, len(video_ids)
