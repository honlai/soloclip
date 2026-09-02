"""End-to-end processing of one video at a time.

Running the whole list stage-by-stage means nothing lands in out/ until the
slowest stage has chewed through every video, and every raw download sits on
disk the entire time. Going video-by-video gives a finished clip within minutes
and lets each video's intermediates be reclaimed as soon as it is done.

The cost is reloading each model once per video (~16s total). That is a fair
trade for early output on a long list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import asd, asr, audio, diarize, download, host, manifest, render, select
from .config import Config
from .utils import LOG, free_gpu, load_path, read_stage

STAGES_AFTER_DOWNLOAD = ("audio", "diarize", "asr", "asd", "select", "render")


def _drop_from_archive(cfg: Config, video_id: str) -> None:
    """Keep the download archive honest when we delete the file it vouches for."""
    entry = cfg.get("download.archive")
    if not entry:
        return
    path = cfg.data_root / str(entry)
    if not path.exists():
        return
    kept = [line for line in path.read_text(encoding="utf-8").splitlines()
            if video_id not in line]
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def cleanup(cfg: Config, video_id: str, rendered: bool) -> None:
    """Reclaim intermediates for a finished video, per runtime.cleanup.

    none  - keep everything
    audio - drop the wav (regenerates from the raw file in about a second)
    all   - also drop the raw download, once a clip actually came out of it
    """
    level = str(cfg.get("runtime.cleanup", "audio")).lower()
    if level == "none":
        return

    freed = 0
    wav = cfg.audio_dir / f"{video_id}.wav"
    if wav.exists():
        freed += wav.stat().st_size
        wav.unlink()

    if level == "all" and rendered:
        dl = read_stage(cfg.meta_dir, video_id, "download") or {}
        raw = load_path(cfg.data_root, dl.get("video_path", ""))
        if raw.exists():
            freed += raw.stat().st_size
            raw.unlink()
            # the archive would otherwise silently block ever fetching it again
            _drop_from_archive(cfg, video_id)
        for extra in cfg.raw_dir.glob(f"{video_id}.*"):
            if extra.suffix in (".vtt", ".srt", ".json"):
                freed += extra.stat().st_size
                extra.unlink()

    if freed:
        LOG.info("[%s] reclaimed %.0f MB", video_id, freed / 1e6)


def already_finished(cfg: Config, video_id: str) -> dict[str, Any] | None:
    """A video with a clip on disk is done - do not re-download it.

    Matters once runtime.cleanup deletes raw files: without this check, resuming a
    long list would re-fetch every finished video just because its source is gone.
    """
    record = read_stage(cfg.meta_dir, video_id, "render")
    if not record or record.get("status") != "ok":
        return None
    if not load_path(cfg.data_root, record.get("out_path", "")).exists():
        return None
    return record


def process_one(cfg: Config, url: str | None = None, video_id: str | None = None,
                force: bool = False) -> dict[str, Any]:
    """Carry a single video all the way from URL to rendered clip."""
    if url is not None:
        record = download.download_one(cfg, url, force=force)
        video_id = record["video_id"]
    if video_id is None:
        raise ValueError("process_one needs either a url or a video_id")

    media = audio.extract_one(cfg, video_id, force=force)

    diar = diarize.diarize_one(cfg, video_id, force=force,
                               host_profile=host.load_profile(cfg))
    if not diar.get("clean_spans"):
        raise RuntimeError("no clean single-speaker audio in the scan window")

    asr.transcribe_one(cfg, video_id, force=force)

    # A podcast has no picture to judge, so there is nothing for ASD to do and
    # the speaker rules alone decide the clip. Config says what we asked for;
    # the probe says what we actually got - either one being audio is enough.
    audio_only = bool(cfg.get("runtime.audio_only", False)) or not media.get("has_video", True)
    if audio_only:
        LOG.info("[%s] audio-only source, skipping ASD", video_id)
    else:
        asd.analyse_one(cfg, video_id, force=force)
    select.select_one(cfg, video_id, force=force, audio_only=audio_only)
    result = render.render_one(cfg, video_id, force=force)
    free_gpu()
    return result


def require_gpu(cfg: Config) -> None:
    """Abort before starting if the GPU was asked for but is not usable.

    A Windows driver update pulls the GPU out from under a running WSL VM, and
    every video then dies at the first model load with "CUDA error: unknown
    error". Because each failure is per-video, the run happily marches through
    the whole list downloading and discarding - 24 videos went that way once.
    Failing fast keeps the list intact for a real retry.
    """
    if str(cfg.get("runtime.device", "cuda")).lower() != "cuda":
        return
    import torch

    problem = None
    if not torch.cuda.is_available():
        problem = "torch reports no usable CUDA device"
    else:
        try:
            torch.zeros(8, device="cuda").sum().item()
            torch.cuda.synchronize()
        except Exception as exc:
            problem = f"{type(exc).__name__}: {exc}"
    if problem:
        raise RuntimeError(
            f"GPU unusable ({problem}). On WSL this usually means the Windows "
            "driver changed under the running VM - run `wsl --shutdown` in "
            "Windows and start it again. Set runtime.device: cpu to run anyway."
        )
    LOG.info("GPU ready: %s", torch.cuda.get_device_name(0))


def run_stages(cfg: Config, urls: list[str], force: bool = False) -> tuple[int, int]:
    """Whole list through one stage before starting the next.

    The point is CUDA context churn: per-video processing builds and tears down
    three contexts (pyannote, whisper, insightface) for *every* video, which is
    ~690 contexts over a 230-video list. Stage batching does it three times in
    total. Native crashes appeared only after the switch to per-video, so this
    mode exists to test whether that churn is what destabilises the WSL GPU
    layer - and to fall back to if it is.

    The costs are real and are why this is not the default: no clip lands until
    the slowest stage has chewed through everything, and every raw download sits
    on disk the whole time.
    """
    require_gpu(cfg)

    video_ids: list[str] = []
    for index, url in enumerate(urls, 1):
        LOG.info("=== download [%d/%d] %s", index, len(urls), url)
        try:
            video_ids.append(download.download_one(cfg, url, force=force)["video_id"])
        except Exception as exc:
            LOG.error("download failed for %s: %s", url, exc)

    ready: list[str] = []
    for vid in video_ids:
        try:
            audio.extract_one(cfg, vid, force=force)
            ready.append(vid)
        except Exception as exc:
            LOG.error("[%s] audio extraction failed: %s", vid, exc)

    diarize.diarize_batch(cfg, ready, force=force)
    ready = [v for v in ready if (read_stage(cfg.meta_dir, v, "diarize") or {}).get("clean_spans")]
    asr.transcribe_batch(cfg, ready, force=force)

    # audio-only sources have no picture for ASD to judge
    with_video = [v for v in ready
                  if not bool(cfg.get("runtime.audio_only", False))
                  and (read_stage(cfg.meta_dir, v, "audio") or {}).get("has_video", True)]
    asd.analyse_batch(cfg, with_video, force=force)

    ok = 0
    for vid in ready:
        rendered = False
        try:
            select.select_one(cfg, vid, force=force, audio_only=vid not in with_video)
            rendered = render.render_one(cfg, vid, force=force).get("status") == "ok"
            ok += rendered
        except Exception as exc:
            LOG.error("[%s] selection/render failed: %s", vid, exc)
        finally:
            manifest.append_row(cfg, vid)
            cleanup(cfg, vid, rendered)
    LOG.info("finished: %d/%d produced a clip", ok, len(urls))
    return ok, len(urls)


def run_list(cfg: Config, urls: list[str], force: bool = False) -> tuple[int, int]:
    """Process each URL to completion in turn, never letting one failure stop the rest."""
    require_gpu(cfg)
    ok = 0
    for index, url in enumerate(urls, 1):
        LOG.info("=== [%d/%d] %s", index, len(urls), url)
        rendered = False
        try:
            # resolve the id first so even a download failure gets a manifest row
            video_id = download.probe_video_id(url, cfg)
        except Exception as exc:
            LOG.error("could not resolve %s: %s", url, exc)
            continue
        done = None if force else already_finished(cfg, video_id)
        if done:
            LOG.info("[%s] clip already exists, skipping", video_id)
            ok += 1
            continue
        try:
            result = process_one(cfg, url=url, force=force)
            rendered = result.get("status") == "ok"
            ok += rendered
        except Exception as exc:
            LOG.error("[%s] pipeline failed: %s", video_id, exc)
        finally:
            free_gpu()
            manifest.append_row(cfg, video_id)
            cleanup(cfg, video_id, rendered)
        LOG.info("=== [%d/%d] done, %d clip(s) so far", index, len(urls), ok)
    LOG.info("finished: %d/%d produced a clip", ok, len(urls))
    return ok, len(urls)
