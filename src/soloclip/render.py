"""Stage 7: cut and splice the chosen pieces into the final clip.

Everything happens in one ffmpeg pass over the source: trim filters give us
frame-accurate cuts, concat joins them, and loudnorm evens out the result.
Video joins are hard cuts on purpose - a dissolve between two shots of the same
person talking reads as an edit, a hard cut usually does not. Audio gets a
20 ms fade on each side of a join purely to kill the click.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
from .utils import (LOG, load_path, read_stage, require_stage, run, store_path,
                    write_stage)

STAGE = "render"


def _loudnorm_chain(cfg: Config) -> str:
    """Loudness pass plus a final resample.

    loudnorm runs its filter at 192 kHz internally and passes that rate on, so
    an aresample placed before it is silently overridden and the file ends up at
    an odd 96 kHz. The rate has to be forced after it.
    """
    rate = int(cfg.get("render.sample_rate", 48000))
    if cfg.get("render.loudnorm", True):
        return f"loudnorm=I=-16:TP=-1.5:LRA=11,aresample={rate}"
    return f"aresample={rate}"


def _build_filter(cfg: Config, pieces: list[tuple[float, float]]) -> tuple[str, str, str]:
    height = int(cfg.get("render.height", 720))
    fps = cfg.get("render.fps", 25)
    fade = float(cfg.get("render.audio_xfade_ms", 20)) / 1000.0

    parts: list[str] = []
    labels: list[str] = []
    for i, (start, end) in enumerate(pieces):
        dur = end - start
        parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,"
            f"scale=-2:{height}:flags=bicubic,fps={fps},setsar=1,format=yuv420p[v{i}]"
        )
        afilters = [
            f"atrim=start={start:.3f}:end={end:.3f}",
            "asetpts=PTS-STARTPTS",
            "aresample=48000",
        ]
        if fade > 0 and dur > 2 * fade:
            afilters.append(f"afade=t=in:st=0:d={fade:.3f}")
            afilters.append(f"afade=t=out:st={dur - fade:.3f}:d={fade:.3f}")
        parts.append(f"[0:a]{','.join(afilters)}[a{i}]")
        labels.append(f"[v{i}][a{i}]")

    parts.append(f"{''.join(labels)}concat=n={len(pieces)}:v=1:a=1[vcat][acat]")
    parts.append(f"[acat]{_loudnorm_chain(cfg)}[aout]")
    return ";".join(parts), "[vcat]", "[aout]"


def _build_audio_filter(cfg: Config, pieces: list[tuple[float, float]]) -> tuple[str, str]:
    """Same cuts, sound only - used when the picture failed but the voice is clean."""
    fade = float(cfg.get("render.audio_xfade_ms", 20)) / 1000.0
    parts: list[str] = []
    labels: list[str] = []
    for i, (start, end) in enumerate(pieces):
        dur = end - start
        filters = [f"atrim=start={start:.3f}:end={end:.3f}", "asetpts=PTS-STARTPTS",
                   "aresample=48000"]
        if fade > 0 and dur > 2 * fade:
            filters.append(f"afade=t=in:st=0:d={fade:.3f}")
            filters.append(f"afade=t=out:st={dur - fade:.3f}:d={fade:.3f}")
        parts.append(f"[0:a]{','.join(filters)}[a{i}]")
        labels.append(f"[a{i}]")
    parts.append(f"{''.join(labels)}concat=n={len(pieces)}:v=0:a=1[acat]")
    parts.append(f"[acat]{_loudnorm_chain(cfg)}[aout]")
    return ";".join(parts), "[aout]"


def render_one(cfg: Config, video_id: str, force: bool = False) -> dict[str, Any]:
    sel = require_stage(cfg.meta_dir, video_id, "select")
    if sel.get("status") != "ok" or not sel.get("pieces"):
        LOG.info("[%s] nothing to render (%s)", video_id, sel.get("reason", "selection failed"))
        return {"video_id": video_id, "status": "skipped",
                "reason": sel.get("reason", "selection failed")}

    audio = require_stage(cfg.meta_dir, video_id, "audio")
    audio_only = sel.get("mode") == "audio"
    out_dir = cfg.audio_out_dir if audio_only else cfg.out_dir
    out_path = out_dir / f"{video_id}.{'m4a' if audio_only else 'mp4'}"

    cached = None if force else read_stage(cfg.meta_dir, video_id, STAGE)
    if cached and out_path.exists():
        LOG.info("[%s] clip already rendered", video_id)
        return cached

    pieces = [(float(s), float(e)) for s, e in sel["pieces"]]
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-v", "error", "-i",
           str(load_path(cfg.data_root, audio["video_path"]))]
    if audio_only:
        filtergraph, alabel = _build_audio_filter(cfg, pieces)
        cmd += ["-filter_complex", filtergraph, "-map", alabel, "-vn"]
    else:
        filtergraph, vlabel, alabel = _build_filter(cfg, pieces)
        cmd += [
            "-filter_complex", filtergraph,
            "-map", vlabel, "-map", alabel,
            "-c:v", str(cfg.get("render.vcodec", "libx264")),
            "-crf", str(cfg.get("render.crf", 20)),
            "-preset", str(cfg.get("render.preset", "medium")),
            "-pix_fmt", "yuv420p",
        ]
    cmd += [
        "-c:a", str(cfg.get("render.acodec", "aac")),
        "-b:a", str(cfg.get("render.abit", "128k")),
        "-movflags", "+faststart",
        str(out_path),
    ]
    run(cmd)

    record = {
        "video_id": video_id,
        "status": "ok",
        "mode": "audio" if audio_only else "video",
        "out_path": store_path(cfg.data_root, out_path),
        "pieces": sel["pieces"],
        "total": sel["total"],
        "joins": sel["joins"],
        "strategy": sel.get("strategy"),
        "reason": sel.get("reason"),
    }
    write_stage(cfg.meta_dir, video_id, STAGE, record)
    LOG.info("[%s] rendered %s %.1fs (%d join(s)) -> %s",
             video_id, "audio" if audio_only else "video",
             sel["total"], sel["joins"], out_path.name)
    return record


def render_batch(cfg: Config, video_ids: list[str], force: bool = False) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for vid in video_ids:
        try:
            results[vid] = render_one(cfg, vid, force=force)
        except Exception as exc:
            LOG.error("[%s] render failed: %s", vid, exc)
            results[vid] = {"video_id": vid, "status": "error", "reason": str(exc)}
    return results


def pair_audio(cfg: Config, force: bool = False) -> tuple[int, int]:
    """Write an audio-only twin of every finished clip.

    The track is stream-copied out of the rendered clip rather than re-selected
    from the source: a pair is only useful if both halves are the *same* moment,
    and running the selector again would land somewhere else entirely. Copying
    is also lossless, and the audio was loudness-normalised at render time.
    """
    src_dir = cfg.out_dir
    # sibling of the clips, not of the config file - a config living in configs/
    # must not scatter output next to itself
    entry = cfg.get("paths.pair_dir")
    dst_dir = ((cfg.data_root / str(entry)).resolve() if entry
               else src_dir.parent / f"{src_dir.name}_pairs")
    dst_dir.mkdir(parents=True, exist_ok=True)

    made = skipped = failed = 0
    for clip in sorted(src_dir.glob("*.mp4")):
        target = dst_dir / f"{clip.stem}.m4a"
        if not force and target.exists() and target.stat().st_mtime >= clip.stat().st_mtime:
            skipped += 1
            continue
        try:
            run(["ffmpeg", "-y", "-v", "error", "-i", str(clip),
                 "-vn", "-c:a", "copy", "-movflags", "+faststart", str(target)])
            made += 1
        except Exception as exc:
            target.unlink(missing_ok=True)
            failed += 1
            LOG.error("[%s] pair extraction failed: %s", clip.stem, exc)
    LOG.info("pairs: %d written, %d already current, %d failed -> %s",
             made, skipped, failed, dst_dir)
    return made, failed
