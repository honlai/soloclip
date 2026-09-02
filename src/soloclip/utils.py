"""Shared helpers: logging, subprocess wrapping, ffprobe, JSON stage cache."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

LOG = logging.getLogger("soloclip")


def setup_logging(log_dir: Path, command: str = "", verbose: bool = False) -> None:
    """One log file per invocation, named after the subcommand.

    delay=True means the file is only created once something is actually logged,
    so read-only commands like `status` stop littering the directory with empty
    files. The name carries the command because a bare timestamp tells you
    nothing about which of a dozen files is the run you care about.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{command}" if command else ""
    LOG.setLevel(logging.DEBUG)
    LOG.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    LOG.addHandler(console)

    fileh = logging.FileHandler(log_dir / f"{stamp}{suffix}.log", encoding="utf-8", delay=True)
    fileh.setLevel(logging.DEBUG)
    fileh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s"))
    LOG.addHandler(fileh)


def run(cmd: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Run an external command, logging the full argv and never swallowing stderr."""
    LOG.debug("exec: %s", " ".join(map(str, cmd)))
    proc = subprocess.run(
        [str(c) for c in cmd],
        capture_output=capture,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-25:]
        LOG.error("command failed (%d): %s", proc.returncode, " ".join(map(str, cmd)))
        for line in tail:
            LOG.error("  | %s", line)
        if check:
            raise RuntimeError(f"command failed: {cmd[0]}")
    return proc


def ffprobe(path: Path) -> dict[str, Any]:
    """Return duration/fps/size for a media file."""
    proc = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    data = json.loads(proc.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    fps = 0.0
    if video.get("avg_frame_rate", "0/0") not in ("0/0", "", None):
        num, _, den = video["avg_frame_rate"].partition("/")
        fps = float(num) / float(den) if float(den or 0) else 0.0
    return {
        "duration": float(data.get("format", {}).get("duration", 0.0)),
        "width": int(video.get("width", 0) or 0),
        "height": int(video.get("height", 0) or 0),
        "fps": fps,
    }


# -- portable paths ----------------------------------------------------------

def store_path(data_root: Path, path: Path | str) -> str:
    """How a path is written into a stage record: relative to data_root.

    Absolute paths here would defeat `paths.data_root` entirely - move the data
    directory (or rename it) and every cached record still points at the old
    location, even though the files came along.
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(data_root.resolve()))
    except ValueError:
        return str(path)          # genuinely outside the data dir; keep as-is


def load_path(data_root: Path, value: str) -> Path:
    """Resolve a stored path. Absolute values are honoured so that records
    written before paths became relative keep working."""
    path = Path(value)
    return path if path.is_absolute() else (data_root / path)


# -- stage cache -------------------------------------------------------------

def stage_file(meta_dir: Path, video_id: str, stage: str) -> Path:
    return meta_dir / f"{video_id}.{stage}.json"


def read_stage(meta_dir: Path, video_id: str, stage: str) -> dict[str, Any] | None:
    path = stage_file(meta_dir, video_id, stage)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOG.warning("corrupt stage cache, ignoring: %s", path)
        return None


def write_stage(meta_dir: Path, video_id: str, stage: str, payload: dict[str, Any]) -> Path:
    path = stage_file(meta_dir, video_id, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def require_stage(meta_dir: Path, video_id: str, stage: str) -> dict[str, Any]:
    data = read_stage(meta_dir, video_id, stage)
    if data is None:
        raise RuntimeError(f"[{video_id}] stage '{stage}' has not been run yet")
    return data


def free_gpu() -> None:
    """Release VRAM between stages - 4GB cannot hold two models at once."""
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def read_url_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"url list not found: {path}")
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def read_url_lists(paths: list[Path]) -> list[str]:
    """Concatenate several url lists, keeping order and dropping duplicates.

    Two lists naming the same video in different URL forms would otherwise be
    downloaded twice, so the last path segment is used as the identity.
    """
    seen: set[str] = set()
    urls: list[str] = []
    missing: list[Path] = []
    for path in paths:
        if not path.exists():
            missing.append(path)
            continue
        for url in read_url_list(path):
            key = url.rstrip("/").rsplit("/", 1)[-1].split("?")[0]
            if key in seen:
                continue
            seen.add(key)
            urls.append(url)
    if missing:
        # a list that is not there yet should not abort the ones that are
        for path in missing:
            LOG.warning("url list not found, skipping: %s", path)
    if not urls and missing:
        raise FileNotFoundError(f"none of the configured url lists exist: {paths}")
    return urls
