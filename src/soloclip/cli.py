"""Command line entry point.

Every stage is also its own subcommand so a single problem video can be
re-run from the failing stage without redoing the expensive work before it.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from . import asd, asr, audio, diarize, download, host, manifest, pipeline, render, select
from .config import load_config
from .utils import LOG, read_stage, read_url_lists, setup_logging

STAGES = ["download", "audio", "diarize", "asr", "asd", "select", "render"]


def _known_ids(cfg) -> list[str]:
    """Video ids that have at least been downloaded, in url-list order."""
    ids: list[str] = []
    for path in sorted(cfg.meta_dir.glob("*.download.json")):
        ids.append(path.name[: -len(".download.json")])
    return ids


def _target_ids(cfg, args) -> list[str]:
    if args.video_id:
        return list(args.video_id)
    return _known_ids(cfg)


def _run_download(cfg, args) -> list[str]:
    urls = read_url_lists(cfg.url_lists())
    if not urls:
        LOG.error("url lists are empty: %s", cfg.url_lists())
        return []
    LOG.info("%d url(s) queued", len(urls))
    results = download.download_all(cfg, urls, force=args.force)
    return [r["video_id"] for r in results if "video_id" in r]


def _run_audio(cfg, ids, force) -> list[str]:
    ok: list[str] = []
    for vid in ids:
        try:
            audio.extract_one(cfg, vid, force=force)
            ok.append(vid)
        except Exception as exc:
            LOG.error("[%s] audio extraction failed: %s", vid, exc)
    return ok


def cmd_run(cfg, args) -> int:
    """One video at a time: a finished clip lands in out/ before the next starts."""
    if args.video_id:
        ok = 0
        for vid in args.video_id:
            try:
                result = pipeline.process_one(cfg, video_id=vid, force=args.force)
                ok += result.get("status") == "ok"
            except Exception as exc:
                LOG.error("[%s] pipeline failed: %s", vid, exc)
            finally:
                manifest.append_row(cfg, vid)
                pipeline.cleanup(cfg, vid, rendered=bool(ok))
        return 0 if ok else 2

    urls = read_url_lists(cfg.url_lists())
    if not urls:
        LOG.error("url lists are empty: %s", cfg.url_lists())
        return 1
    mode = str(cfg.get("runtime.mode", "per_video")).lower()
    LOG.info("%d url(s) queued, mode=%s", len(urls), mode)
    runner = pipeline.run_stages if mode == "per_stage" else pipeline.run_list
    ok, _ = runner(cfg, urls, force=args.force)
    return 0 if ok else 2


def cmd_stage(cfg, args) -> int:
    stage = args.stage
    if stage == "download":
        return 0 if _run_download(cfg, args) else 1

    ids = _target_ids(cfg, args)
    if not ids:
        LOG.error("no videos found; run `download` first")
        return 1

    if stage == "audio":
        _run_audio(cfg, ids, args.force)
    elif stage == "diarize":
        if getattr(args, "retarget", False):
            profile = host.load_profile(cfg)
            if profile is None:
                LOG.warning("no host profile; targets fall back to the dominant speaker")
            for vid in ids:
                try:
                    diarize.retarget_one(cfg, vid, profile)
                except Exception as exc:
                    LOG.error("[%s] retarget failed: %s", vid, exc)
        else:
            diarize.diarize_batch(cfg, ids, force=args.force)
    elif stage == "asr":
        asr.transcribe_batch(cfg, ids, force=args.force)
    elif stage == "asd":
        if getattr(args, "rescore", False):
            for vid in ids:
                try:
                    asd.rescore_one(cfg, vid)
                except Exception as exc:
                    LOG.error("[%s] rescore failed: %s", vid, exc)
        else:
            asd.analyse_batch(cfg, ids, force=args.force)
    elif stage == "select":
        select.select_batch(cfg, ids, force=args.force)
    elif stage == "render":
        render.render_batch(cfg, ids, force=args.force)
        manifest.write_manifest(cfg, ids)
    return 0


def cmd_pair(cfg, args) -> int:
    """Optional last step: an audio twin of each finished clip."""
    made, failed = render.pair_audio(cfg, force=args.force)
    return 0 if not failed else 2


def cmd_host(cfg, args) -> int:
    """Build the host profile from whatever diarization results exist."""
    ids = _known_ids(cfg)
    if not ids:
        LOG.error("no videos found; run `diarize` first")
        return 1
    profile = host.build_profile(cfg, ids)
    if profile is None:
        return 2
    print(f"host voice found in {profile['videos']}/{profile['total_videos']} videos "
          f"({profile['coverage']:.0%}), {profile['speech_seconds']:.0f}s of speech")
    for ex in profile["examples"]:
        print(f"  {ex}")
    return 0


def cmd_status(cfg, args) -> int:
    ids = _target_ids(cfg, args)
    if not ids:
        print("no videos downloaded yet")
        return 0
    width = max(len(v) for v in ids)
    print(f"{'video':<{width}}  " + "  ".join(s[:5] for s in STAGES) + "   result")
    for vid in ids:
        marks = []
        for stage in STAGES:
            data = read_stage(cfg.meta_dir, vid, stage)
            marks.append(" ok  " if data else "  -  ")
        row = manifest.build_row(cfg, vid)
        detail = row["status"]
        if row.get("clip_seconds"):
            detail += f" {row['clip_seconds']:.1f}s/{row.get('joins', 0)}j"
        if row.get("reason"):
            detail += f" ({row['reason']})"
        print(f"{vid:<{width}}  " + "  ".join(marks) + f"   {detail}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soloclip", description=__doc__)
    parser.add_argument("-c", "--config", default=None,
                        help="path to a config file. Global option, so it goes BEFORE the "
                             "subcommand: `soloclip -c config.audio.yaml run`")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="download -> render for every url in the list")
    run_p.add_argument("--force", action="store_true", help="ignore stage caches")
    run_p.add_argument("--video-id", nargs="*",
                       help="restrict to these ids (skips download); ids beginning with "
                            "'-' need the equals form, e.g. --video-id=-ABCDEFGHIJ")
    run_p.set_defaults(func=cmd_run)

    for stage in STAGES:
        p = sub.add_parser(stage, help=f"run only the {stage} stage")
        p.add_argument("--force", action="store_true")
        p.add_argument("--video-id", nargs="*",
                       help="ids to act on; ids beginning with '-' need the "
                            "equals form, e.g. --video-id=-ABCDEFGHIJ")
        if stage == "diarize":
            p.add_argument("--retarget", action="store_true",
                           help="re-pick the target speaker from cached embeddings using "
                                "the host profile (no GPU)")
        if stage == "asd":
            p.add_argument("--rescore", action="store_true",
                           help="re-apply thresholds to cached per-frame measurements "
                                "(no GPU, no decode) instead of re-detecting faces")
        p.set_defaults(func=cmd_stage, stage=stage)

    pp = sub.add_parser("pair-audio",
                        help="write an audio-only twin of every finished clip (optional)")
    pp.add_argument("--force", action="store_true", help="rewrite pairs that already exist")
    pp.set_defaults(func=cmd_pair, video_id=None)

    hp = sub.add_parser("host", help="identify the recurring host across the list")
    hp.set_defaults(func=cmd_host, force=False, video_id=None)

    st = sub.add_parser("status", help="show per-video stage progress")
    st.add_argument("--video-id", nargs="*")
    st.set_defaults(func=cmd_status, force=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    cfg.ensure_dirs()
    setup_logging(cfg.log_dir, command=args.command, verbose=args.verbose)
    try:
        return int(args.func(cfg, args) or 0)
    except RuntimeError as exc:
        # a precondition failure, not a per-video problem: tell the supervisor
        # to stop rather than restart into the same wall
        LOG.error("%s", exc)
        return 3
    except KeyboardInterrupt:
        LOG.warning("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
