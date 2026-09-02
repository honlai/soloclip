"""Stage 2: extract mono 16 kHz WAV for the speech models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
from .utils import (LOG, ffprobe, load_path, read_stage, require_stage, run,
                    store_path, write_stage)

STAGE = "audio"


def scan_window(cfg: Config, duration: float) -> float:
    """How much of the source we are willing to analyse, in seconds."""
    limit = float(cfg.get("download.max_scan_seconds", 0) or 0)
    return min(duration, limit) if limit > 0 else duration


def extract_one(cfg: Config, video_id: str, force: bool = False) -> dict[str, Any]:
    dl = require_stage(cfg.meta_dir, video_id, "download")
    video_path = load_path(cfg.data_root, dl["video_path"])
    wav_path = cfg.audio_dir / f"{video_id}.wav"

    cached = None if force else read_stage(cfg.meta_dir, video_id, STAGE)
    if cached and wav_path.exists():
        LOG.info("[%s] audio already extracted", video_id)
        return cached

    probe = ffprobe(video_path)
    limit = scan_window(cfg, probe["duration"])

    cfg.audio_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video_path)]
    if limit < probe["duration"]:
        cmd += ["-t", f"{limit:.3f}"]
    cmd += [
        "-vn",
        "-ac", str(cfg.get("audio.channels", 1)),
        "-ar", str(cfg.get("audio.sample_rate", 16000)),
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    run(cmd)

    record = {
        "video_id": video_id,
        "wav_path": store_path(cfg.data_root, wav_path),
        "video_path": store_path(cfg.data_root, video_path),
        "duration": probe["duration"],
        "scan_seconds": limit,
        "has_video": bool(probe["width"] and probe["height"]),
        "width": probe["width"],
        "height": probe["height"],
        "fps": probe["fps"],
    }
    write_stage(cfg.meta_dir, video_id, STAGE, record)
    LOG.info("[%s] audio %.1fs -> %s", video_id, limit, wav_path.name)
    return record
