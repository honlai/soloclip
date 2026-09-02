"""Stage 1: fetch source videos with yt-dlp.

Uses the Python API rather than the CLI so we get the info dict (id, duration,
title) back directly instead of having to re-parse it off disk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import Config
from .utils import LOG, load_path, read_stage, store_path, write_stage

STAGE = "download"


def cookie_opts(cfg: Config) -> dict[str, Any]:
    """yt-dlp cookie settings, used by both the probe and the real download.

    YouTube answers some requests with "Sign in to confirm you're not a bot";
    cookies from a signed-in session are what gets past it. A cookies.txt file
    wins over a browser profile because it is explicit and does not depend on a
    browser being installed where this runs.
    """
    d = cfg.get("download", {}) or {}
    opts: dict[str, Any] = {}

    jar = str(d.get("cookies_file") or "").strip()
    if jar:
        path = Path(jar)
        if not path.is_absolute():
            path = cfg.root / path
        if path.exists():
            opts["cookiefile"] = str(path)
            return opts
        LOG.warning("cookies_file does not exist, ignoring: %s", path)

    browser = str(d.get("cookies_from_browser") or "").strip()
    if browser:
        # "firefox" or "firefox:/path/to/profile"
        name, _, profile = browser.partition(":")
        opts["cookiesfrombrowser"] = (name.strip(), profile.strip() or None, None, None)
    return opts


def _ydl_opts(cfg: Config) -> dict[str, Any]:
    d = cfg.get("download", {}) or {}
    opts: dict[str, Any] = {
        "format": d.get("format", "b"),
        "merge_output_format": d.get("merge_output_format", "mp4"),
        "outtmpl": {"default": str(cfg.raw_dir / "%(id)s.%(ext)s")},
        "paths": {"home": str(cfg.raw_dir)},
        "writeinfojson": bool(d.get("write_info_json", True)),
        "writeautomaticsub": bool(d.get("write_auto_subs", False)),
        "writesubtitles": bool(d.get("write_auto_subs", False)),
        "subtitleslangs": [s.strip() for s in str(d.get("sub_langs", "en")).split(",") if s.strip()],
        "concurrent_fragment_downloads": int(d.get("concurrent_fragments", 1)),
        "retries": int(d.get("retries", 5)),
        "ignoreerrors": False,
        "noprogress": True,
        "quiet": True,
        "no_warnings": False,
        "noplaylist": True,
        "overwrites": False,
    }
    opts.update(cookie_opts(cfg))
    opts.update(extractor_opts(cfg))
    opts.update(js_opts(cfg))
    if d.get("match_filter"):
        from yt_dlp.utils import match_filter_func

        opts["match_filter"] = match_filter_func(str(d["match_filter"]))
    if d.get("archive"):
        archive = cfg.data_root / str(d["archive"])
        archive.parent.mkdir(parents=True, exist_ok=True)
        opts["download_archive"] = str(archive)
    return opts


def _resolve_filepath(ydl, info: dict[str, Any]) -> Path | None:
    """Find the merged output on disk; requested_downloads is authoritative."""
    for entry in info.get("requested_downloads") or []:
        candidate = entry.get("filepath") or entry.get("_filename")
        if candidate and Path(candidate).exists():
            return Path(candidate)
    guess = Path(ydl.prepare_filename(info))
    if guess.exists():
        return guess
    # after a merge the extension changes; probe the usual containers
    for ext in ("mp4", "mkv", "webm"):
        alt = guess.with_suffix("." + ext)
        if alt.exists():
            return alt
    return None


# youtu.be/ID, watch?v=ID, shorts/ID, embed/ID, live/ID - ids are 11 chars
_YT_ID = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:[^#]*&)?v=|shorts/|embed/|live/|v/))"
    r"([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])"
)


def _id_cache_path(cfg: Config) -> Path:
    return cfg.meta_dir / "url_ids.json"


def _read_id_cache(cfg: Config) -> dict[str, str]:
    path = _id_cache_path(cfg)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOG.warning("corrupt url id cache, starting over: %s", path)
        return {}


def _remember_id(cfg: Config, url: str, video_id: str) -> None:
    cache = _read_id_cache(cfg)
    if cache.get(url) == video_id:
        return
    cache[url] = video_id
    path = _id_cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")


def extractor_opts(cfg: Config) -> dict[str, Any]:
    """Per-extractor arguments, notably the YouTube player client.

    YouTube's default player response sometimes comes back as "The page needs
    to be reloaded", which no amount of retrying fixes; asking for a different
    player client does. Kept configurable because which clients work shifts
    over time.
    """
    clients = cfg.get("download.youtube_player_client") or []
    if isinstance(clients, str):
        clients = [c.strip() for c in clients.split(",") if c.strip()]
    if not clients:
        return {}
    return {"extractor_args": {"youtube": {"player_client": list(clients)}}}


def js_opts(cfg: Config) -> dict[str, Any]:
    """JavaScript runtime settings for YouTube's signature challenges.

    Without a JS runtime *and* the EJS solver script, YouTube withholds every
    real format and yt-dlp reports only storyboards - which surfaces as
    "Requested format is not available" or an apparent SABR-only response.
    Both halves are needed: the runtime alone still fails the n-challenge.

    The solver script is fetched from the yt-dlp project's own distribution,
    which is why it is opt-in here rather than silently enabled.
    """
    d = cfg.get("download", {}) or {}
    opts: dict[str, Any] = {}
    runtimes = d.get("js_runtimes")
    if runtimes:
        if isinstance(runtimes, str):
            runtimes = [r.strip() for r in runtimes.split(",") if r.strip()]
        opts["js_runtimes"] = {r: {} for r in runtimes}
    components = d.get("remote_components")
    if components:
        if isinstance(components, str):
            components = [c.strip() for c in components.split(",") if c.strip()]
        opts["remote_components"] = set(components)
    return opts


def probe_video_id(url: str, cfg: Config | None = None) -> str:
    """Resolve a URL to its video id, over the network only as a last resort.

    This runs for *every* url on *every* run, including ones already finished,
    so a network round trip here costs ~90s per already-done video - hours of
    pure skipping on a long list after a restart. A standard YouTube url already
    contains the id, and anything resolved the hard way is remembered on disk.

    The network path deliberately uses a bare YoutubeDL: with the download
    archive or a match filter attached, extract_info returns None for anything
    already fetched or filtered out, and we would lose the id we need.
    """
    match = _YT_ID.search(url)
    if match:
        return match.group(1)

    if cfg is not None:
        cached = _read_id_cache(cfg).get(url)
        if cached:
            return cached

    import yt_dlp

    opts: dict[str, Any] = {"quiet": True, "no_warnings": True, "noplaylist": True}
    if cfg is not None:
        # the bot check and the player-client problem both fire on this call too
        opts.update(cookie_opts(cfg))
        opts.update(extractor_opts(cfg))
        opts.update(js_opts(cfg))
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False, process=False)
    if not info or not info.get("id"):
        raise RuntimeError(f"could not resolve a video id for {url}")
    video_id = str(info["id"])
    if cfg is not None:
        _remember_id(cfg, url, video_id)
    return video_id


def download_one(cfg: Config, url: str, force: bool = False) -> dict[str, Any]:
    """Download a single URL. Returns the stage record (also written to cache)."""
    import yt_dlp

    video_id = probe_video_id(url, cfg)
    cached = None if force else read_stage(cfg.meta_dir, video_id, STAGE)
    if cached and load_path(cfg.data_root, cached.get("video_path", "")).exists():
        LOG.info("[%s] already downloaded, skipping", video_id)
        return cached

    opts = _ydl_opts(cfg)
    with yt_dlp.YoutubeDL(opts) as ydl:
        LOG.info("[%s] downloading %s", video_id, url)
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError(
                f"[{video_id}] yt-dlp produced nothing: it is in the download archive "
                "but the file is gone, or match_filter excluded it, or it is unavailable"
            )
        path = _resolve_filepath(ydl, info)
        if path is None:
            raise RuntimeError(f"[{video_id}] download finished but output file not found")

    record = {
        "video_id": video_id,
        "url": url,
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "duration": float(info.get("duration") or 0.0),
        "width": info.get("width"),
        "height": info.get("height"),
        "fps": info.get("fps"),
        "video_path": store_path(cfg.data_root, path),
    }
    write_stage(cfg.meta_dir, video_id, STAGE, record)
    LOG.info("[%s] saved %s (%.1fs)", video_id, path.name, record["duration"])
    return record


def download_all(cfg: Config, urls: list[str], force: bool = False) -> list[dict[str, Any]]:
    """Download every URL, recording failures instead of aborting the batch."""
    results: list[dict[str, Any]] = []
    for url in urls:
        try:
            results.append(download_one(cfg, url, force=force))
        except Exception as exc:  # one bad URL must not kill the run
            LOG.error("download failed for %s: %s", url, exc)
            results.append({"url": url, "error": str(exc)})
    return results
