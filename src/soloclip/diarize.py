"""Stage 3: speaker diarization.

Produces the audio-side constraint for selection: intervals where the dominant
speaker - and nobody else - is talking. Any region where a second speaker is
active is removed together with a safety pad on both sides, because a clip that
clips even a syllable of another voice fails the brief.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .config import Config, hf_token
from .intervals import close_gaps, dilate, normalize, subtract, total
from .utils import (LOG, free_gpu, load_path, read_stage, require_stage,
                    write_stage)

STAGE = "diarize"


def _torch_load_patch():
    """torch 2.6 defaults torch.load to weights_only=True, which pyannote's
    lightning checkpoints cannot satisfy. Inside this context manager only, and
    only while loading the pipeline the user explicitly authorised on the HF
    hub, torch.load goes back to a full unpickle.
    """
    import contextlib

    import torch

    @contextlib.contextmanager
    def _patched():
        safe: list = []
        for dotted in ("torch.torch_version.TorchVersion",
                       "omegaconf.listconfig.ListConfig",
                       "omegaconf.dictconfig.DictConfig",
                       "omegaconf.base.ContainerMetadata",
                       "omegaconf.base.Metadata",
                       "omegaconf.nodes.AnyNode",
                       "omegaconf.nodes.ValueNode"):
            module_name, _, attr = dotted.rpartition(".")
            try:
                module = __import__(module_name, fromlist=[attr])
                safe.append(getattr(module, attr))
            except Exception:
                continue
        torch.serialization.add_safe_globals(safe)

        original = torch.load

        def loose_load(*args, **kwargs):
            # Must overwrite, not setdefault: lightning passes weights_only=True
            # explicitly. Retrying after a failure is not an option either -
            # lightning hands torch.load an already-consumed stream, so the
            # second attempt dies on a half-read pickle.
            kwargs["weights_only"] = False
            return original(*args, **kwargs)

        torch.load = loose_load
        try:
            yield
        finally:
            torch.load = original

    return _patched()


def _load_pipeline(cfg: Config):
    from pyannote.audio import Pipeline

    token = hf_token()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. pyannote diarization models are gated: accept the "
            "terms for pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0 "
            "on huggingface.co, then put the token in .env"
        )
    model = str(cfg.get("diarize.model", "pyannote/speaker-diarization-3.1"))
    LOG.info("loading diarization pipeline %s", model)
    with _torch_load_patch():
        pipeline = Pipeline.from_pretrained(model, use_auth_token=token)
    if pipeline is None:
        raise RuntimeError(
            f"pyannote returned no pipeline for {model}: the token may lack access, "
            "or the gated model conditions have not been accepted"
        )

    if cfg.device == "cuda":
        import torch

        pipeline.to(torch.device("cuda"))
    return pipeline


def _speaker_embeddings(annotation, embeddings) -> dict[str, list[float]]:
    """Map pyannote's embedding matrix back onto speaker labels.

    Rows line up with `annotation.labels()`. A speaker with very little speech
    can come back as NaN, which is not an embedding we can compare, so it is
    dropped rather than stored as a poisoned vector.
    """
    import numpy as np

    out: dict[str, list[float]] = {}
    if embeddings is None:
        return out
    for label, vector in zip(annotation.labels(), embeddings):
        vec = np.asarray(vector, dtype=np.float32)
        if vec.size == 0 or not np.isfinite(vec).all():
            continue
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            continue
        out[label] = (vec / norm).astype(float).tolist()
    return out


def choose_target(cfg: Config, totals: dict[str, float],
                  embeddings: dict[str, list[float]],
                  host_profile: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Pick whose voice the clip should be built from.

    Normally that is simply whoever speaks most. When a host profile is known,
    the host is deprioritised: an interview is wanted for the guest, and the
    host recurs across the whole list, so their material is the least valuable.
    The host is still used rather than producing nothing at all.
    """
    dominant = max(totals, key=totals.get)
    info: dict[str, Any] = {"host_speakers": [], "host_matched": False}
    if not host_profile or not host_profile.get("centroid"):
        return dominant, info

    import numpy as np

    centroid = np.asarray(host_profile["centroid"], dtype=np.float32)
    threshold = float(cfg.get("host.distance", 0.55))
    host_speakers = []
    for spk, vec in embeddings.items():
        distance = 1.0 - float(np.asarray(vec, dtype=np.float32) @ centroid)
        if distance <= threshold:
            host_speakers.append(spk)
    info["host_speakers"] = host_speakers
    info["host_matched"] = bool(host_speakers)

    guests = {spk: secs for spk, secs in totals.items() if spk not in host_speakers}
    if guests:
        return max(guests, key=guests.get), info
    # everyone present is the host - better their voice than no clip at all
    info["host_only"] = True
    return dominant, info


def _rebuild(cfg: Config, video_id: str, record: dict[str, Any],
             host_profile: dict[str, Any] | None) -> dict[str, Any]:
    """(Re)compute the target speaker and their clean spans from stored turns."""
    speakers = {spk: normalize([tuple(x) for x in spans])
                for spk, spans in record["speakers"].items()}
    totals = {spk: total(spans) for spk, spans in speakers.items()}
    embeddings = record.get("embeddings") or {}

    target, info = choose_target(cfg, totals, embeddings, host_profile)
    others = normalize([span for spk, spans in speakers.items()
                        if spk != target for span in spans])
    pad = float(cfg.get("diarize.overlap_pad", 0.2))
    bounds = (0.0, float(record["scan_seconds"]))
    # bridge the speaker's own breathing pauses first, then punch out everyone else
    bridged = close_gaps(speakers[target], float(cfg.get("diarize.bridge_pause", 1.0)))
    clean = subtract(bridged, dilate(others, pad, bounds))

    record.update({
        "dominant": max(totals, key=totals.get),
        "target": target,
        "target_is_host": target in info["host_speakers"],
        "host_speakers": info["host_speakers"],
        "speaker_totals": totals,
        "num_speakers": len(speakers),
        "dominant_spans": speakers[target],
        "other_spans": others,
        "clean_spans": clean,
        "clean_total": total(clean),
    })
    return record


def diarize_one(cfg: Config, video_id: str, force: bool = False, pipeline=None,
                host_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    cached = None if force else read_stage(cfg.meta_dir, video_id, STAGE)
    if cached:
        LOG.info("[%s] diarization cached", video_id)
        return cached

    audio = require_stage(cfg.meta_dir, video_id, "audio")
    own_pipeline = pipeline is None
    pipeline = pipeline or _load_pipeline(cfg)

    LOG.info("[%s] diarizing %.1fs of audio", video_id, audio["scan_seconds"])
    result = pipeline(str(load_path(cfg.data_root, audio["wav_path"])),
                      return_embeddings=True)
    annotation, embeddings = result if isinstance(result, tuple) else (result, None)

    min_turn = float(cfg.get("diarize.min_speaker_turn", 0.0))
    by_speaker: dict[str, list[tuple[float, float]]] = defaultdict(list)
    turns: list[dict[str, Any]] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        if segment.duration < min_turn:
            continue
        by_speaker[speaker].append((segment.start, segment.end))
        turns.append({"start": segment.start, "end": segment.end, "speaker": speaker})

    if not by_speaker:
        raise RuntimeError(f"[{video_id}] diarization found no speech")

    record = {
        "video_id": video_id,
        "scan_seconds": float(audio["scan_seconds"]),
        "turns": turns,
        "speakers": {spk: normalize(spans) for spk, spans in by_speaker.items()},
        "embeddings": _speaker_embeddings(annotation, embeddings),
    }
    _rebuild(cfg, video_id, record, host_profile)
    write_stage(cfg.meta_dir, video_id, STAGE, record)
    LOG.info("[%s] %d speakers, target=%s%s (%.1fs), audio-clean %.1fs",
             video_id, record["num_speakers"], record["target"],
             " [host]" if record["target_is_host"] else "",
             record["speaker_totals"][record["target"]], record["clean_total"])
    if own_pipeline:
        del pipeline
        free_gpu()
    return record


def retarget_one(cfg: Config, video_id: str, host_profile: dict[str, Any] | None) -> dict[str, Any]:
    """Re-pick the target speaker from cached turns and embeddings. No GPU."""
    record = require_stage(cfg.meta_dir, video_id, STAGE)
    if "speakers" not in record:
        raise RuntimeError(f"[{video_id}] diarize cache predates speaker embeddings; re-run diarize")
    before = record.get("target")
    _rebuild(cfg, video_id, record, host_profile)
    write_stage(cfg.meta_dir, video_id, STAGE, record)
    LOG.info("[%s] target %s -> %s%s, audio-clean %.1fs", video_id, before, record["target"],
             " [host]" if record["target_is_host"] else "", record["clean_total"])
    return record


def diarize_batch(cfg: Config, video_ids: list[str], force: bool = False) -> dict[str, Any]:
    """Load the pipeline once for the whole batch, then release the VRAM."""
    results: dict[str, Any] = {}
    pending = [
        vid for vid in video_ids
        if force or read_stage(cfg.meta_dir, vid, STAGE) is None
    ]
    pipeline = _load_pipeline(cfg) if pending else None
    from .host import load_profile
    profile = load_profile(cfg)
    try:
        for vid in video_ids:
            try:
                results[vid] = diarize_one(cfg, vid, force=force, pipeline=pipeline,
                                           host_profile=profile)
            except Exception as exc:
                LOG.error("[%s] diarization failed: %s", vid, exc)
                results[vid] = {"error": str(exc)}
    finally:
        del pipeline
        free_gpu()
    return results
